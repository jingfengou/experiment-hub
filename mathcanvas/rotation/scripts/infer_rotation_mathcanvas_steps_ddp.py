#!/usr/bin/env python3
"""
Two-GPU (DDP) inference with rotation step hints (text + images).
Feeds the model: step descriptions -> step images -> combined image -> question prompt.
Prompt template matches Orthus rotation script (<think>/<answer>).
"""
import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from PIL import Image, UnidentifiedImageError

from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MathCanvas rotation inference with step hints (DDP).")
    parser.add_argument("--model-dir", required=True, help="Path with llm_config.json, vit_config.json, tokenizer files.")
    parser.add_argument("--ckpt-dir", required=True, help="Directory containing model.safetensors/ema.safetensors.")
    parser.add_argument("--ckpt-file", default="model.safetensors", help="Checkpoint filename.")
    parser.add_argument("--dataset-file", default="/workspace/oujingfeng/datasets/mydatasets/dataset/data_modified_with_subject.json", help="Rotation dataset json.")
    parser.add_argument("--dataset-root", default="/workspace/oujingfeng/datasets/mydatasets/dataset", help="Dataset root for images.")
    parser.add_argument("--output-dir", default="./outputs/rotation_mathcanvas_steps_ddp", help="Output directory for generated texts.")
    parser.add_argument("--sample-fraction", type=float, default=1.0, help="Fraction (0-1] of dataset to run.")
    parser.add_argument("--sample-region", choices=["head", "tail"], default="head", help="Pick head/tail portion after sampling.")
    parser.add_argument("--max-samples", type=int, default=1000, help="Cap on number of samples (0 means no cap).")
    parser.add_argument("--max-think-tokens", type=int, default=1024, help="Max tokens for text generation.")
    parser.add_argument("--text-temperature", type=float, default=0.3, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--skip-final-image",
        action="store_true",
        help="Do not generate the final image; answer directly after reasoning context.",
    )
    return parser.parse_args()


def init_distributed() -> int:
    if "LOCAL_RANK" not in os.environ:
        return -1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


def get_rank_world() -> (int, int):
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def select_indices(total: int, fraction: float, region: str, max_samples: int) -> List[int]:
    fraction = max(0.0, min(1.0, fraction))
    if fraction == 0.0:
        indices = list(range(total))
    else:
        count = max(1, math.ceil(total * fraction))
        indices = list(range(min(count, total))) if region == "head" else list(range(max(total - count, 0), total))
    if max_samples > 0:
        indices = indices[:max_samples]
    return indices


def chunk_indices(indices: List[int], world_size: int, rank: int) -> List[int]:
    return indices[rank::world_size]


def build_prompt(item: dict) -> str:
    instruction = (
        "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
        "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
        "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
    )
    question = item.get("Question", "")
    choices = item.get("Choices", [])
    choices_text = "\n".join([f"{chr(65 + i)}) {c}" for i, c in enumerate(choices)]) if choices else ""
    if choices_text and not choices_text.startswith("\n"):
        choices_text = "\n" + choices_text
    prompt = instruction + "<image>\n\n" + f"Question: {question}{choices_text}\n\nAnswer: "
    return prompt


def build_step_texts(item: dict) -> List[str]:
    """
    Construct step-wise reasoning text that pairs with each step image.
    """
    steps = item.get("Rotation_steps", []) or []
    texts = []
    for idx, s in enumerate(steps):
        desc = s.get("description") or "rotate the object accordingly"
        axis = s.get("axis")
        angle = s.get("angle")
        lead = "First" if idx == 0 else "Then" if idx == 1 else f"Next (step {s.get('step', idx+1)})"
        axis_part = f" along the {axis} axis" if axis else ""
        angle_part = f" by {angle} degrees" if angle is not None else ""
        text = f"{lead}, {desc}{axis_part}{angle_part}. Let's visualize the state after this step."
        texts.append(text)
    if not texts:
        texts.append("Rotation steps: (none provided)")
    return texts


def resolve_image(dataset_root: str, item: dict) -> Image.Image:
    task = item.get("Task", "")
    level = item.get("Level", "")
    image_id = item.get("Image_id", "")
    combined_image = item.get("Combined_image", "")
    image_path = item.get("image_path", "") or item.get("image", "")

    if image_path and os.path.isabs(image_path) and os.path.exists(image_path):
        path = image_path
    else:
        path = os.path.join(dataset_root, "data", task, level, image_id, combined_image)

    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except FileNotFoundError:
        return Image.new("RGB", (512, 512), (255, 255, 255))


def resolve_step_images(dataset_root: str, item: dict) -> List[Image.Image]:
    task = item.get("Task", "")
    level = item.get("Level", "")
    image_id = item.get("Image_id", "")
    base_dir = os.path.join(dataset_root, "data", task, level, image_id)
    images: List[Image.Image] = []
    for s in item.get("Rotation_steps", []) or []:
        rel = s.get("image", "")
        if not rel:
            continue
        path = os.path.join(base_dir, rel)
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        except (FileNotFoundError, UnidentifiedImageError):
            continue
    return images


def load_model(model_dir: str, ckpt_dir: str, ckpt_file: str, device_index: int) -> InterleaveInferencer:
    llm_config = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_dir, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_dir, "ae.safetensors"))
    vae_model = vae_model.to(f"cuda:{device_index}").eval()

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_dir)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 512, 14)

    device_map = infer_auto_device_map(
        model,
        max_memory={device_index: "80GiB"},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        dtype=torch.bfloat16,
    )
    checkpoint_path = os.path.join(ckpt_dir, ckpt_file)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
    )
    model = model.eval()

    inferencer = InterleaveInferencer(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
    return inferencer


def main() -> None:
    args = parse_args()
    local_rank = init_distributed()
    rank, world_size = get_rank_world()
    device_index = local_rank if local_rank >= 0 else 0

    set_seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)

    if rank == 0:
        print(f"Loading model on cuda:{device_index} | world_size={world_size}")
    inferencer = load_model(args.model_dir, args.ckpt_dir, args.ckpt_file, device_index)

    data = load_json(args.dataset_file)
    total = len(data)
    indices = select_indices(total, args.sample_fraction, args.sample_region, args.max_samples)
    assigned = chunk_indices(indices, world_size, rank)

    if rank == 0:
        print(f"Total samples: {total}, selected: {len(indices)}; per-rank ~{len(assigned)}")
    if not assigned:
        if rank == 0:
            print("No samples assigned, exiting.")
        return

    for idx in assigned:
        item = data[idx]
        prompt = build_prompt(item)
        step_texts = build_step_texts(item)
        step_images = resolve_step_images(args.dataset_root, item)
        combined_image = resolve_image(args.dataset_root, item)

        # Feed order: question prompt + question image, then step-wise text/image pairs
        input_list = [prompt, combined_image]
        paired = zip(step_texts, step_images)
        for text, img in paired:
            input_list.extend([text, img])
        if len(step_texts) > len(step_images):
            input_list.extend(step_texts[len(step_images):])

        outputs = inferencer.interleave_inference(
            input_list,
            think=True,
            understanding_output=True,
            max_think_token_n=args.max_think_tokens,
            do_sample=True,
            text_temperature=args.text_temperature,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            cfg_interval=[0.0, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
            cfg_renorm_type="text_channel",
            enable_taylorseer=False,
            skip_final_image=args.skip_final_image,
        )

        gen_texts = [o for o in outputs if isinstance(o, str)]
        generated = gen_texts[0] if gen_texts else ""
        out_name = f"sample{idx:05d}.txt"
        out_path = os.path.join(args.output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"sample{idx:05d}\n")
            f.write(f"Image_id: {item.get('Image_id', '')}\n")
            f.write(f"Question: {item.get('Question', '')}\n")
            f.write(f"Ground truth: {item.get('Answer', '')}\n")
            f.write("Generated:\n")
            f.write(generated)

        print(f"[Rank {rank}] Processed sample {idx} -> {out_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()

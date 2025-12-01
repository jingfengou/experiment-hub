#!/usr/bin/env python3
"""
用 Hunyuan3D-2.1 将 rotation 题干图生成 3D 网格，并按 Rotation_steps 逐步旋转后渲染单视角 PNG。

- 仅使用 question.png 生成网格（避免 combined 含选项）。
- 轴映射遵循题目定义：X=右，Y=向前，Z=向上（内部映射到 +X,+Z,+Y）。
- 输出：
  sample_xxxx/
    mesh.glb
    projections/base.png           # 原姿态
    projections/rotated.png        # 全部步骤应用后的结果
    projections/steps/step1.png... # 若 --render-steps 开启，则每一步后的中间图

依赖：trimesh pyrender PyOpenGL；无显示需 export PYOPENGL_PLATFORM=egl

示例：
  export PYOPENGL_PLATFORM=egl
  python scripts/inference/generate_rotation_hunyuan3d_projections.py \
    --dataset-json /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json \
    --dataset-root /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data \
    --output-dir /workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas/outputs/hunyuan3d_rotation_proj \
    --model-path tencent/Hunyuan3D-2.1 \
    --max-samples 2 \
    --apply-rotation-steps \
    --render-steps \
    --offset 1,1,1 \
    --dist-scale 2.0 \
    --bg 255,255,255,255 \
    --color 70,70,70,255 \
    --intensity 1.0 \
    --ambient 0.5 \
    --roughness 0.8 \
    --edge-overlay \
    --edge-crease-only \
    --edge-threshold 0.1 \
    --edge-color 0,0,0,255
"""
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import trimesh
import pyrender
from PIL import Image

import sys
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
HUNYUAN3D_ROOT = Path(os.environ.get("HUNYUAN3D_ROOT", PROJECT_ROOT / "Hunyuan3D-2.1"))
HY3D_SHAPE_PATH = HUNYUAN3D_ROOT / "hy3dshape"
if not HY3D_SHAPE_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find Hunyuan3D-2.1 hy3dshape at {HY3D_SHAPE_PATH}. "
        f"Set HUNYUAN3D_ROOT env if path differs."
    )
sys.path.insert(0, str(HY3D_SHAPE_PATH))
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


def load_data(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data


def resolve_image_path(item: Dict, dataset_root: Path) -> Path:
    task = item.get("Task", "")
    level = item.get("Level", "")
    image_id = item.get("Image_id", "")
    question = item.get("Question_image") or "question.png"
    path = dataset_root / task / level / image_id / question
    if path.exists():
        return path
    raise FileNotFoundError(f"No question image found for {image_id}: {path}")


def generate_mesh(pipeline, image_path: Path, out_glb: Path) -> Path:
    image = Image.open(image_path).convert("RGBA")
    if image.mode == "RGB":
        rembg = BackgroundRemover()
        image = rembg(image)
    mesh = pipeline(image=str(image_path))[0]
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_glb))
    return out_glb


def apply_rotations(mesh: trimesh.Trimesh, steps: List[Dict]) -> trimesh.Trimesh:
    if not steps:
        return mesh.copy()
    m = mesh.copy()
    for s in steps:
        axis = s.get("axis", "").upper()
        angle = s.get("angle", 0)
        theta = math.radians(angle)
        # Dataset axis: X=right, Y=forward -> internal +Z, Z=up -> internal +Y
        if axis == "X":
            rot = trimesh.transformations.rotation_matrix(theta, [1, 0, 0])
        elif axis == "Y":
            rot = trimesh.transformations.rotation_matrix(theta, [0, 0, 1])
        else:
            rot = trimesh.transformations.rotation_matrix(theta, [0, 1, 0])
        m.apply_transform(rot)
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    if up is None:
        up = np.array([0, 1, 0], dtype=np.float64)
    eye = eye.astype(np.float64)
    target = target.astype(np.float64)
    up = up.astype(np.float64)
    z_axis = eye - target
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = x_axis
    mat[:3, 1] = y_axis
    mat[:3, 2] = z_axis
    mat[:3, 3] = eye
    return mat


def render_single(
    mesh: trimesh.Trimesh,
    out_path: Path,
    offset_dir: np.ndarray,
    dist_scale: float,
    res: int,
    bg_rgba: List[int],
    mesh_color_rgba: List[int],
    intensity: float,
    ambient: float,
    metallic: float,
    roughness: float,
    edge_overlay: bool,
    edge_crease_only: bool,
    edge_threshold: float,
    edge_color_rgba: List[int],
):
    extents = mesh.extents
    max_extent = max(extents.max(), 1e-6)
    fov = np.pi / 3.0
    dist = (max_extent / 2.0) / np.tan(fov / 2.0) * dist_scale

    scene = pyrender.Scene(bg_color=bg_rgba)
    scene.ambient_light = np.array([ambient, ambient, ambient, 1.0])

    offset_dir = offset_dir / (np.linalg.norm(offset_dir) + 1e-8)
    cam_pos = offset_dir * dist
    cam_pose = look_at(cam_pos, target=np.array([0, 0, 0], dtype=np.float64))

    cam = pyrender.PerspectiveCamera(yfov=fov, znear=0.01, zfar=100.0)
    scene.add(cam, pose=cam_pose)

    gray = np.array([0.85, 0.85, 0.85])
    light_dir = pyrender.DirectionalLight(color=gray, intensity=intensity)
    light_pt = pyrender.PointLight(color=gray, intensity=max(intensity * 5, 0.1))
    scene.add(light_dir, pose=cam_pose)
    scene.add(light_pt, pose=cam_pose)

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[c / 255.0 for c in mesh_color_rgba],
        metallicFactor=metallic,
        roughnessFactor=roughness,
        doubleSided=True,
    )
    mesh_node = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene.add(mesh_node)

    if edge_overlay:
        edge_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[c / 255.0 for c in edge_color_rgba]
        )
        if edge_crease_only:
            try:
                edges = mesh.face_adjacency_edges
                angles = mesh.face_adjacency_angles
                mask = angles > edge_threshold
                edges_sel = edges[mask]
                if edges_sel.size == 0:
                    edges_sel = mesh.edges_boundary
                if edges_sel is None or len(edges_sel) == 0:
                    edges_sel = mesh.edges_sorted
                path = trimesh.load_path(mesh.vertices[edges_sel])
                edge_node = pyrender.Mesh.from_trimesh(path, material=edge_mat, smooth=False)
                scene.add(edge_node)
            except Exception:
                pass
        else:
            try:
                edge_node = pyrender.Mesh.from_trimesh(mesh, material=edge_mat, wireframe=True)
                scene.add(edge_node)
            except Exception:
                pass

    r = pyrender.OffscreenRenderer(res, res)
    try:
        color_img, _ = r.render(scene)
        Image.fromarray(color_img).save(out_path)
    finally:
        r.delete()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True, help="Rotation JSON")
    parser.add_argument("--dataset-root", required=True, help="Rotation image root (data/)")
    parser.add_argument("--output-dir", required=True, help="Output dir for meshes and projections")
    parser.add_argument("--model-path", default="tencent/Hunyuan3D-2.1", help="Hunyuan3D model id/path")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples; 0=all")
    parser.add_argument("--apply-rotation-steps", action="store_true", help="Apply Rotation_steps before rendering")
    parser.add_argument("--render-steps", action="store_true", help="Render intermediate images after each rotation step")
    parser.add_argument("--render-res", type=int, default=512, help="Render resolution (square)")
    parser.add_argument("--offset", type=str, default="1.0,1.0,1.0", help="Camera offset direction, e.g., 1,1,1")
    parser.add_argument("--dist-scale", type=float, default=2.0, help="Camera distance scale multiplier")
    parser.add_argument("--bg", type=str, default="255,255,255,255", help="Background RGBA")
    parser.add_argument("--color", type=str, default="104,109,114,255", help="Mesh RGBA color")
    parser.add_argument("--intensity", type=float, default=8.0, help="Directional light intensity")
    parser.add_argument("--ambient", type=float, default=0.5, help="Ambient light strength (0-1)")
    parser.add_argument("--metallic", type=float, default=0.0, help="Material metallic factor (0-1)")
    parser.add_argument("--roughness", type=float, default=0.8, help="Material roughness factor (0-1)")
    parser.add_argument("--edge-overlay", action="store_true", help="Add wireframe overlay")
    parser.add_argument("--edge-threshold", type=float, default=0.1, help="Crease threshold (radians) when edge-overlay is on")
    parser.add_argument("--edge-color", type=str, default="0,0,0,255", help="Edge color RGBA when overlay enabled")
    parser.add_argument("--edge-crease-only", action="store_true", help="Only draw crease/boundary edges when overlay enabled")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_json).resolve()
    data_root = Path(args.dataset_root).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading Hunyuan3D-Shape pipeline...")
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_path)

    records = load_data(dataset_path)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    print(f"Loaded {len(records)} samples")

    offset_dir = np.array([float(x) for x in args.offset.split(",")], dtype=np.float64)
    bg_rgba = [int(x) for x in args.bg.split(",")]
    mesh_color_rgba = [int(x) for x in args.color.split(",")]
    edge_color_rgba = [int(x) for x in args.edge_color.split(",")]

    for idx, item in enumerate(records):
        image_id = item.get("Image_id", f"sample_{idx:05d}")
        sample_dir = out_root / image_id
        mesh_path = sample_dir / "mesh.glb"
        proj_dir = sample_dir / "projections"
        proj_dir.mkdir(parents=True, exist_ok=True)

        try:
            img_path = resolve_image_path(item, data_root)
        except FileNotFoundError as e:
            print(f"[skip] {image_id}: {e}")
            continue

        print(f"[{idx+1}/{len(records)}] {image_id} | img={img_path}")
        mesh_path = generate_mesh(pipeline_shapegen, img_path, mesh_path)

        base_mesh = trimesh.load(mesh_path, force="mesh")
        base_mesh.apply_translation(-base_mesh.centroid)

        # Render base
        render_single(
            base_mesh,
            proj_dir / "base.png",
            offset_dir,
            args.dist_scale,
            args.render_res,
            bg_rgba,
            mesh_color_rgba,
            args.intensity,
            args.ambient,
            args.metallic,
            args.roughness,
            args.edge_overlay,
            args.edge_crease_only,
            args.edge_threshold,
            edge_color_rgba,
        )

        steps = item.get("Rotation_steps") or []
        if args.apply_rotation_steps and steps:
            mesh_work = base_mesh.copy()
            for si, s in enumerate(steps, 1):
                mesh_work = apply_rotations(mesh_work, [s])
                if args.render_steps:
                    step_dir = proj_dir / "steps"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    render_single(
                        mesh_work,
                        step_dir / f"step{si}.png",
                        offset_dir,
                        args.dist_scale,
                        args.render_res,
                        bg_rgba,
                        mesh_color_rgba,
                        args.intensity,
                        args.ambient,
                        args.metallic,
                        args.roughness,
                        args.edge_overlay,
                        args.edge_crease_only,
                        args.edge_threshold,
                        edge_color_rgba,
                    )
            # final after all steps
            render_single(
                mesh_work,
                proj_dir / "rotated.png",
                offset_dir,
                args.dist_scale,
                args.render_res,
                bg_rgba,
                mesh_color_rgba,
                args.intensity,
                args.ambient,
                args.metallic,
                args.roughness,
                args.edge_overlay,
                args.edge_crease_only,
                args.edge_threshold,
                edge_color_rgba,
            )

    print(f"Done. Outputs in {out_root}")


if __name__ == "__main__":
    main()

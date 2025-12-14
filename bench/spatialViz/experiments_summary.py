import pandas as pd

# 数据录入 (包含 Bagel-7B)
data = {
    'Model': ['Orthus-7B-Instruct', 'Anole-7B', 'Mirage', 'Math-Canvas', 'Bagel-7B'],
    
    # Missing Rate (Lower is better)
    'Missing Rate': [0.00, 76.53, 3.98, 0.34, 0.17],
    
    # Overall Accuracy (Higher is better)
    'Overall': [25.51, 7.12, 26.61, 33.34, 29.32],
    
    # Mental Rotation
    'MR (2DR)': [25.00, 6.25, 17.50, 31.25, 32.50],
    'MR (3DR)': [21.25, 12.50, 23.75, 41.77, 27.50],
    'MR (3VP)': [29.00, 3.00, 23.00, 42.00, 38.00],
    
    # Mental Folding
    'MF (PF)': [27.50, 7.50, 29.17, 24.37, 34.17],
    'MF (CU)': [26.67, 5.00, 30.00, 23.33, 25.83],
    'MF (CR)': [22.50, 5.83, 29.17, 29.41, 26.67],
    
    # Visual Penetration
    'VP (CS)': [30.83, 10.83, 14.17, 25.83, 19.17],
    'VP (CC)': [25.00, 3.33, 35.00, 44.54, 23.33],
    'VP (CA)': [25.00, 11.25, 35.00, 41.25, 22.50],
    
    # Mental Animation
    'MA (AM)': [15.00, 6.25, 17.50, 30.00, 25.00],
    'MA (BM)': [27.50, 7.50, 26.25, 38.75, 35.00],
    'MA (MS)': [27.50, 8.75, 37.50, 35.00, 48.75]
}

df = pd.DataFrame(data)
df.set_index('Model', inplace=True)

# 样式函数：最大值/最小值加粗
def highlight_best(s):
    is_best = pd.Series(data=[False]*len(s), index=s.index)
    
    if s.name == 'Missing Rate':
        # Missing Rate 找最小
        is_best = s == s.min()
    else:
        # 精度找最大
        is_best = s == s.max()
        
    return ['font-weight: bold; background-color: #e6f3ff' if v else '' for v in is_best]

# 生成样式表格
styled_df = df.style.apply(highlight_best, axis=0)\
            .format("{:.2f}")

# # Jupyter Notebook 中显示
# styled_df
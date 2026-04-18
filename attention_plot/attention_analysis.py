import argparse
import glob
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# 配置matplotlib以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 自定义颜色列表
my_extracted_colors = ['#49026a', '#4f076b', '#53096a', '#560c6b', '#5c106b', '#5f136b', '#65176b', '#691a6b', '#6f1c6a', '#73216d', '#77246c', '#7d286c', '#812b6c', '#852c6c', '#8c336d', '#8f356b', '#94386b', '#993b6d', '#9e3e6c', '#a3426d', '#a6466c', '#ad496f', '#b14e6d', '#b44f6d', '#bb546f', '#be566d', '#c3596f', '#c55b6f', '#c35c6f', '#c55e71', '#c55e71', '#c65f74', '#c76274', '#c86176', '#c96476', '#c96476', '#ca6577', '#cb6678', '#ca677b', '#cb687c', '#cc697d', '#cd6a7e', '#ce6b7f', '#cf6c80', '#d06d7f', '#cf6e81', '#d06f82', '#d27184', '#d27184', '#d47386', '#d57487', '#d37587', '#d47688', '#d57789', '#d7798b', '#d7798b', '#d87a8c', '#d97c8e', '#d97c8e', 
'#db7e90', '#dc7f91', '#db8092', '#dc8193', '#dd8294', '#dd8496', '#df8496', '#de8597', '#de8798', '#e08799', '#e1889a', '#e18a9b', '#e28b9c', '#e38c9d', '#e48d9e', '#e38ea1', '#e48fa2', '#e48fa2', '#e691a4', '#e792a5', '#e794a6', '#e895a7', '#e996a8', '#e998a9', '#e998ab', '#eb9aad', '#ea9bae', '#eb9caf', '#eb9caf', '#ec9db0', '#eda0b2', '#eda0b4', '#eca1b5', '#eda2b6', '#eea3b7', '#eda4b7', '#efa6b9', '#f0a7ba', '#f0a9bb', '#f0a8bc', '#f2aabe', '#f0abbe', '#f1acbf', '#f1aebf', '#f2afc1', '#f3b0c2', '#f4b1c3', '#f2b2c3', '#f3b3c4', '#f3b3c4', '#f4b4c5', '#f4b5c6', '#f4b5c6', '#f5b6c7', '#f5b8c7', '#f6b9c8', '#f4bac8', '#f5bbc9', '#f6bcca', '#f6becb', '#f6becb', '#f7bfcc', '#f7bfcc', '#f6c0cd', '#f6c1cb', '#f7c1ce', '#f7c4cd', '#f7c4cd', '#f7c5d0', '#f8c6cf', '#f9c7d0', '#f7c8d0', '#f8c9d1', '#f8cbd2', '#f9ccd3', '#f9ccd3', '#facdd4', '#facdd4', '#f9ced5', '#f9ced5', '#fad0d4', '#fad1d7', '#fbd2d8', '#fad3d6', '#fbd4d9', '#fbd4d9', '#fad6d8', '#fbd7db', '#fcd8da', '#fbd8dc', '#fcdadb', '#fcdadb', '#fbdbdc', '#fcdcdf', '#fbdddd', '#fcdede', 
'#fddfdf', '#fde1e0', '#fde1e0', '#fde1e0', '#fde1e0', '#fee2df', '#fee2df', '#fce3df', '#fce3df', '#fce3df', '#fde4e0', '#fde4e0', '#fee5e1', '#fde6e0', '#fde6e0', '#fde6e0', '#fde6e0', '#fee7e1', '#fee7e1', '#fee7e1', '#fce8df', '#fce8df', '#fde9e0', '#fde9e0', '#fde9e0', '#fdebe1', '#fdebe1', '#fdebdf']

def load_model_and_tokenizer(model_path):
    """加载模型和分词器"""
    print("正在加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        output_attentions=True  # 重要: 需要输出注意力权重
    ).to('cuda')
    model.eval()
    print("模型加载完成!")
    return model, tokenizer

def read_jsonl(file_path, max_samples=None):
    """读取 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
    return data

def find_token_positions(full_text, tokenizer):
    """
    找到 <think> 和 </think> 以及步骤分隔的 token 位置
    返回: {
        'think_start': int,
        'think_end': int,
        'steps': [(start, end), ...]
    }
    """
    positions = {
        'think_start': None,
        'think_end': None,
        'steps': []
    }
    
    # 查找 <think> 和 </think> 的位置
    think_start_str = '<think>'
    think_end_str = '</think>'
    
    # 在文本中找到这些标记
    if think_start_str in full_text:
        think_start_idx = full_text.index(think_start_str)
        # 找到对应的 token 位置
        prefix_text = full_text[:think_start_idx]
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
        positions['think_start'] = len(prefix_tokens)
    
    if think_end_str in full_text:
        think_end_idx = full_text.index(think_end_str) + len(think_end_str)
        prefix_text = full_text[:think_end_idx]
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
        positions['think_end'] = len(prefix_tokens)
    
    # 查找步骤分隔 (使用 \n\n 作为分隔符)
    if positions['think_start'] is not None and positions['think_end'] is not None:
        # 提取 <think> 和 </think> 之间的内容
        think_content_start = full_text.index(think_start_str) + len(think_start_str)
        think_content_end = full_text.index(think_end_str)
        think_content = full_text[think_content_start:think_content_end]
        
        # 按 \n\n 分割步骤
        steps_text = [s.strip() for s in think_content.split('\n\n') if s.strip()]
        
        current_pos = think_content_start
        for step_text in steps_text:
            step_start = full_text.index(step_text, current_pos)
            step_end = step_start + len(step_text)
            
            # 转换为 token 位置
            prefix_tokens_start = tokenizer.encode(full_text[:step_start], add_special_tokens=False)
            prefix_tokens_end = tokenizer.encode(full_text[:step_end], add_special_tokens=False)
            
            positions['steps'].append((len(prefix_tokens_start), len(prefix_tokens_end)))
            current_pos = step_end
    
    return positions

def plot_attention_heatmap(attention_weights, tokens, tokenizer, positions,
                          layer_idx=0, save_path=None, sample_id=0):
    """
    绘制注意力热力图并标注关键位置

    参数:
        attention_weights: 注意力权重张量 (num_layers, batch, num_heads, seq_len, seq_len)
        tokens: token IDs
        tokenizer: 分词器
        positions: 关键位置字典
        layer_idx: 要可视化的层索引
        save_path: 保存路径
        sample_id: 样本ID
    """
    # 提取指定层所有头的平均注意力权重
    attn = attention_weights[layer_idx][0].mean(dim=0).float().cpu().numpy()
    seq_len = attn.shape[0]
    
    # 不对上三角进行mask，完整显示注意力图
    # mask = np.triu_indices(seq_len, k=1)  # k=1表示不包括对角线
    # attn[mask] = np.nan

    # 归一化：将注意力权重归一化到[0, 1]区间
    attn_min = np.min(attn)
    attn_max = np.max(attn)
    if attn_max > attn_min:
        attn = (attn - attn_min) / (attn_max - attn_min)
    else:
        attn = np.zeros_like(attn)  # 如果所有值相同，设为0
    
    # 使用自定义颜色列表
    custom_cmap = LinearSegmentedColormap.from_list('custom', my_extracted_colors, N=256)

    # 创建图形
    fig, ax = plt.subplots(figsize=(20, 18))

    # 归一化后的颜色范围固定为[0, 1]
    vmin = 0.0
    vmax = 0.04
    im = ax.imshow(attn, cmap=custom_cmap, aspect='auto', interpolation='nearest',
                   vmin=vmin, vmax=vmax)
    
    # 设置标题和标签（字号大幅放大）
    title_text = f'Average Attention Map in Layer {layer_idx}'
    ax.set_title(title_text, fontsize=48, pad=30, fontweight='bold')
    ax.set_xlabel('Token Key Position', fontsize=36)
    ax.set_ylabel('Token Query Position', fontsize=36)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontsize=32)
    cbar.ax.tick_params(labelsize=28)
    
    # 只设置关键位置的刻度：最后一个token、<think> 和 </think>（不显示0，避免与<think>重叠）
    all_ticks = [seq_len - 1]  # 只显示最后一个token
    all_labels = [str(seq_len - 1)]
    
    # 添加 <think> 和 </think> 的特殊标记
    if positions['think_start'] is not None:
        all_ticks.append(positions['think_start'])
        all_labels.append('<think>')
    
    if positions['think_end'] is not None:
        all_ticks.append(positions['think_end'])
        all_labels.append('</think>')
    
    # 按位置排序
    sorted_indices = sorted(range(len(all_ticks)), key=lambda i: all_ticks[i])
    all_ticks = [all_ticks[i] for i in sorted_indices]
    all_labels = [all_labels[i] for i in sorted_indices]
    
    # 设置刻度和标签（字号放大）
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels, fontsize=28)
    ax.set_yticks(all_ticks)
    ax.set_yticklabels(all_labels, fontsize=28)
    
    # 为 <think> 和 </think> 标签设置特殊样式（字号减小）
    for tick_label in ax.get_xticklabels():
        if tick_label.get_text() in ['<think>', '</think>']:
            tick_label.set_fontsize(22)
            tick_label.set_color('red')
            tick_label.set_fontweight('bold')
    
    for tick_label in ax.get_yticklabels():
        if tick_label.get_text() in ['<think>', '</think>']:
            tick_label.set_fontsize(22)
            tick_label.set_color('red')
            tick_label.set_fontweight('bold')
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"热力图已保存至: {save_path}")
    
    plt.close()

def plot_multi_layer_attention_heatmap(attention_weights, tokens, tokenizer, positions,
                                       layer_indices=[0, 6, 10, 15, 18, 24], 
                                       save_path=None, sample_id=0):
    """
    绘制多层注意力热力图的水平拼接图
    
    参数:
        attention_weights: 注意力权重张量
        tokens: token IDs
        tokenizer: 分词器
        positions: 关键位置字典
        layer_indices: 要可视化的层索引列表
        save_path: 保存路径
        sample_id: 样本ID
    """
    num_layers = len(layer_indices)
    
    # 提取第一层的注意力来获取序列长度
    attn_sample = attention_weights[layer_indices[0]][0].mean(dim=0).float().cpu().numpy()
    seq_len = attn_sample.shape[0]
    
    # 准备所有层的归一化注意力图
    attn_list = []
    for layer_idx in layer_indices:
        attn = attention_weights[layer_idx][0].mean(dim=0).float().cpu().numpy()
        
        # 归一化
        attn_min = np.min(attn)
        attn_max = np.max(attn)
        if attn_max > attn_min:
            attn = (attn - attn_min) / (attn_max - attn_min)
        else:
            attn = np.zeros_like(attn)
        
        attn_list.append(attn)
    
    # 使用自定义颜色列表
    custom_cmap = LinearSegmentedColormap.from_list('custom', my_extracted_colors, N=256)
    
    # 创建图形：使用GridSpec来更精细地控制布局
    from matplotlib.gridspec import GridSpec
    
    # 计算总宽度：每层图的宽度 + colorbar的宽度，增大间距避免拥挤
    fig_width = num_layers * 8 + 2.5  # 每层8英寸，colorbar 2.5英寸
    fig_height = 8
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # 创建GridSpec：num_layers个子图 + 1个colorbar，增加间距避免拥挤
    gs = GridSpec(1, num_layers + 1, figure=fig, 
                  width_ratios=[1]*num_layers + [0.05],
                  wspace=0.25)  # 进一步增加子图之间的间距
    
    # 绘制每一层的注意力图
    axes = []
    ims = []
    for i, (layer_idx, attn) in enumerate(zip(layer_indices, attn_list)):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        
        # 绘制热力图
        vmin = 0.0
        vmax = 0.04
        im = ax.imshow(attn, cmap=custom_cmap, aspect='auto', interpolation='nearest',
                       vmin=vmin, vmax=vmax)
        ims.append(im)
        
        # 设置标题
        ax.set_title(f'Layer {layer_idx}', fontsize=32, pad=15, fontweight='bold')
        
        # 只在最左边的图显示y轴标题
        if i == 0:
            ax.set_ylabel('Token Query Position', fontsize=32, fontweight='bold')
        
        # 设置刻度（不显示0，避免与<think>重叠）
        all_ticks = [seq_len - 1]  # 只显示最后一个token
        all_labels = [str(seq_len - 1)]
        
        if positions['think_start'] is not None:
            all_ticks.append(positions['think_start'])
            all_labels.append('<think>')
        
        if positions['think_end'] is not None:
            all_ticks.append(positions['think_end'])
            all_labels.append('</think>')
        
        # 按位置排序
        sorted_indices = sorted(range(len(all_ticks)), key=lambda j: all_ticks[j])
        all_ticks = [all_ticks[j] for j in sorted_indices]
        all_labels = [all_labels[j] for j in sorted_indices]
        
        # 设置刻度（水平显示，不倾斜）
        ax.set_xticks(all_ticks)
        ax.set_xticklabels(all_labels, fontsize=24, rotation=0, ha='center', fontweight='bold')
        
        if i == 0:  # 只在第一个图显示y刻度标签
            ax.set_yticks(all_ticks)
            ax.set_yticklabels(all_labels, fontsize=24, fontweight='bold')
            
            # 为 <think> 和 </think> 设置特殊样式（字号减小，竖着显示）
            for tick_label in ax.get_yticklabels():
                if tick_label.get_text() in ['<think>', '</think>']:
                    tick_label.set_fontsize(22)
                    tick_label.set_color('red')
                    tick_label.set_rotation(90)  # 竖着写
                    tick_label.set_verticalalignment('center')
        else:
            # 其他图不显示y轴刻度
            ax.set_yticks([])
        
        # 为 x轴的 <think> 和 </think> 设置特殊样式（字号减小）
        for tick_label in ax.get_xticklabels():
            if tick_label.get_text() in ['<think>', '</think>']:
                tick_label.set_fontsize(22)
                tick_label.set_color('red')
    
    # 添加colorbar（最右边）
    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = plt.colorbar(ims[0], cax=cbar_ax, ticks=np.linspace(0, 0.04, 5))  # 只显示5个刻度
    # 不显示标签
    cbar.ax.tick_params(labelsize=24, width=2)  # 增加刻度线宽度
    
    # 在整个图形的底部中心添加x轴标签
    fig.text(0.5, -0.02, 'Token Key Position', ha='center', fontsize=32, fontweight='bold')
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"多层拼接热力图已保存至: {save_path}")
    
    plt.close()

def analyze_sample(model, tokenizer, sample, output_dir, sample_id, 
                   layers_to_visualize=None):
    """
    分析单个样本的注意力模式
    """
    # 构建完整文本
    full_text = sample['prompt'] + sample['full_output']
    
    print(f"\n正在分析样本 {sample_id}...")
    print(f"文本长度: {len(full_text)} 字符")
    
    # 分词
    inputs = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=4096)
    input_ids = inputs['input_ids'].to(model.device)
    
    print(f"Token 数量: {input_ids.shape[1]}")
    
    # 获取注意力权重
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        attentions = outputs.attentions  # Tuple of (num_layers, batch, num_heads, seq_len, seq_len)
    
    # 查找关键位置
    positions = find_token_positions(full_text, tokenizer)
    
    print(f"<think> 位置: {positions['think_start']}")
    print(f"</think> 位置: {positions['think_end']}")
    print(f"步骤数量: {len(positions['steps'])}")
    for i, (start, end) in enumerate(positions['steps']):
        print(f"  Step {i+1}: tokens {start}-{end} (长度: {end-start})")
    
    # 创建输出目录
    sample_output_dir = os.path.join(output_dir, f'sample_{sample_id}')
    os.makedirs(sample_output_dir, exist_ok=True)

    # 确定要可视化的层
    num_layers = len(attentions)
    
    if layers_to_visualize is None:
        # 默认可视化所有层
        layers_to_visualize = list(range(num_layers))
    
    # 创建img文件夹
    img_output_dir = os.path.join(sample_output_dir, 'img')
    os.makedirs(img_output_dir, exist_ok=True)

    # 清理旧版本留下的非注意力图产物，避免复用输出目录时混入历史文件。
    legacy_positions_path = os.path.join(sample_output_dir, 'positions.json')
    if os.path.exists(legacy_positions_path):
        os.remove(legacy_positions_path)

    for legacy_region_stats_path in glob.glob(os.path.join(img_output_dir, 'region_stats_layer*.png')):
        os.remove(legacy_region_stats_path)
    
    # 生成热力图 - 每层一张图（所有头的平均）
    for layer_idx in layers_to_visualize:
        if layer_idx >= num_layers:
            continue
        
        save_path = os.path.join(img_output_dir, 
                                f'attention_layer{layer_idx}.png')
        plot_attention_heatmap(attentions, input_ids[0].tolist(), tokenizer, 
                             positions, layer_idx, save_path, sample_id)

    # 生成多层拼接的注意力热力图（指定层：0, 6, 18, 24）
    multi_layer_indices = [0, 6, 18, 24]
    # 确保所有指定的层都存在
    valid_multi_layer_indices = [idx for idx in multi_layer_indices if idx < num_layers]
    if valid_multi_layer_indices:
        multi_layer_save_path = os.path.join(img_output_dir, 
                                            f'attention_multi_layers.png')
        plot_multi_layer_attention_heatmap(attentions, input_ids[0].tolist(), tokenizer,
                                          positions, valid_multi_layer_indices,
                                          multi_layer_save_path, sample_id)
    
    print(f"样本 {sample_id} 分析完成!")

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize attention patterns for a generated CoT sample.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the causal LM used to generate the CoT.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to a JSONL file generated by step1_cot_generation.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for attention visualizations.",
    )
    parser.add_argument(
        "--sample_id",
        type=int,
        default=0,
        help="Sample index in the JSONL file. Defaults to the first sample.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Optional comma-separated layer indices. Leave empty to visualize all layers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model_path:
        raise ValueError("model_path is required. Please provide --model_path.")
    if not args.data_path:
        raise ValueError("data_path is required. Please provide --data_path.")
    if not args.output_dir:
        raise ValueError("output_dir is required. Please provide --output_dir.")
    layers_to_visualize = None
    if args.layers:
        layers_to_visualize = [int(layer.strip()) for layer in args.layers.split(",") if layer.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    data = read_jsonl(args.data_path, max_samples=args.sample_id + 1)
    print(f"共读取 {len(data)} 个样本")

    if len(data) > args.sample_id:
        sample = data[args.sample_id]
        try:
            analyze_sample(model, tokenizer, sample, args.output_dir, args.sample_id, layers_to_visualize)
        except Exception as e:
            print(f"分析样本 {args.sample_id} 时出错: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(f"错误: 数据文件中没有足够的样本（需要至少 {args.sample_id + 1} 个样本）")

    print(f"\n分析完成! 结果保存在: {args.output_dir}")

if __name__ == '__main__':
    main()

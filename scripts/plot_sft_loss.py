"""
SFT 训练损失可视化脚本
=====================================================================
从训练日志中提取损失值并绘制损失曲线。

使用方法：
    python scripts/plot_sft_loss.py
    python scripts/plot_sft_loss.py --log_file sft_training_20260130_163249.log
=====================================================================
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def parse_training_log(log_file: str):
    """
    解析训练日志，提取损失值
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        包含训练信息的字典
    """
    train_losses = []
    eval_losses = []
    steps = []
    epochs = []
    
    current_epoch = 0
    step_count = 0
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if 'Epoch' in line and '/' in line:
                match = re.search(r'Epoch (\d+)/(\d+)', line)
                if match:
                    current_epoch = int(match.group(1))
            
            if 'Step' in line and 'loss' in line.lower():
                match = re.search(r'Step (\d+): loss = ([\d.]+)', line)
                if match:
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    steps.append(step)
                    train_losses.append(loss)
                    epochs.append(current_epoch)
                    step_count = step
            
            if '训练损失' in line or 'train loss' in line.lower():
                match = re.search(r'[:=] ([\d.]+)', line)
                if match:
                    loss = float(match.group(1))
                    if loss not in train_losses[-5:]:
                        train_losses.append(loss)
                        steps.append(step_count)
                        epochs.append(current_epoch)
            
            if '评估损失' in line or 'eval loss' in line.lower():
                match = re.search(r'[:=] ([\d.]+)', line)
                if match:
                    eval_losses.append(float(match.group(1)))
    
    return {
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'steps': steps,
        'epochs': epochs,
        'log_file': log_file
    }


def find_latest_log(log_dir: str = "."):
    """
    查找最新的训练日志
    
    Args:
        log_dir: 日志目录
        
    Returns:
        最新日志文件路径
    """
    log_files = list(Path(log_dir).glob("sft_training_*.log"))
    if not log_files:
        return None
    
    return str(sorted(log_files)[-1])


def plot_loss_curves(data: dict, save_path: str = None, show: bool = True):
    """
    绘制损失曲线
    
    Args:
        data: 解析的训练数据
        save_path: 图片保存路径
        show: 是否显示图片
    """
    train_losses = data['train_losses']
    eval_losses = data['eval_losses']
    
    if not train_losses:
        print("❌ 未找到损失数据")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.plot(range(len(train_losses)), train_losses, 'b-', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss over Steps', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    if len(train_losses) > 10:
        window = min(10, len(train_losses) // 5)
        if window > 1:
            smoothed = []
            for i in range(len(train_losses)):
                start = max(0, i - window)
                smoothed.append(sum(train_losses[start:i+1]) / (i - start + 1))
            ax1.plot(range(len(smoothed)), smoothed, 'r-', linewidth=2, label='Smoothed')
            ax1.legend()
    
    ax2 = axes[1]
    if eval_losses:
        ax2.plot(range(1, len(eval_losses) + 1), eval_losses, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Evaluation Loss', fontsize=12)
        ax2.set_title('Evaluation Loss over Epochs', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        min_eval_idx = eval_losses.index(min(eval_losses)) + 1
        ax2.axvline(x=min_eval_idx, color='r', linestyle='--', alpha=0.5)
        ax2.annotate(f'Best: Epoch {min_eval_idx}', 
                    xy=(min_eval_idx, eval_losses[min_eval_idx-1]),
                    xytext=(min_eval_idx + 0.5, eval_losses[min_eval_idx-1] + 0.1),
                    fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No evaluation data', ha='center', va='center', fontsize=14)
        ax2.set_title('Evaluation Loss', fontsize=14)
    
    fig.suptitle(f"SFT Training Loss - {Path(data['log_file']).stem}", fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 损失曲线已保存: {save_path}")
    
    if show:
        plt.show()


def plot_combined_loss(data: dict, save_path: str = None, show: bool = True):
    """
    绘制综合损失曲线（训练 + 评估）
    
    Args:
        data: 解析的训练数据
        save_path: 图片保存路径
        show: 是否显示图片
    """
    train_losses = data['train_losses']
    eval_losses = data['eval_losses']
    
    if not train_losses:
        print("❌ 未找到损失数据")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = data['steps'] if data['steps'] else list(range(len(train_losses)))
    
    ax.plot(steps, train_losses, 'b-', linewidth=1, alpha=0.6, label='Training Loss')
    
    if len(train_losses) > 20:
        window = min(20, len(train_losses) // 5)
        if window > 2:
            smoothed = []
            for i in range(len(train_losses)):
                start = max(0, i - window)
                smoothed.append(sum(train_losses[start:i+1]) / (i - start + 1))
            ax.plot(steps, smoothed, 'r-', linewidth=2, label='Smoothed (MAE)')
    
    if eval_losses:
        eval_steps = [steps[max(0, len(steps) * i // len(eval_losses) - 1)] for i in range(1, len(eval_losses) + 1)]
        ax.plot(eval_steps, eval_losses, 'go-', linewidth=2, markersize=10, label='Evaluation Loss')
        
        min_eval_idx = eval_losses.index(min(eval_losses))
        min_eval_step = eval_steps[min_eval_idx]
        ax.scatter([min_eval_step], [eval_losses[min_eval_idx]], 
                  color='red', s=200, zorder=5, marker='*')
        ax.annotate(f'Best Eval\nLoss: {eval_losses[min_eval_idx]:.4f}',
                   xy=(min_eval_step, eval_losses[min_eval_idx]),
                   xytext=(min_eval_step + 50, eval_losses[min_eval_idx] + 0.05),
                   fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('SFT Training Loss Curve', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    stats_text = f"Total Steps: {len(train_losses)}\n"
    stats_text += f"Min Train Loss: {min(train_losses):.4f}\n"
    if eval_losses:
        stats_text += f"Min Eval Loss: {min(eval_losses):.4f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 综合损失曲线已保存: {save_path}")
    
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot SFT training loss')
    parser.add_argument('--log_file', type=str, default=None,
                       help='训练日志文件路径')
    parser.add_argument('--log_dir', type=str, default='.',
                       help='日志目录')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片路径')
    parser.add_argument('--no_show', action='store_true',
                       help='不显示图片，只保存')
    
    args = parser.parse_args()
    
    log_file = args.log_file
    if log_file is None:
        log_file = find_latest_log(args.log_dir)
    
    if log_file is None or not os.path.exists(log_file):
        print("❌ 未找到训练日志文件")
        print("请指定日志文件: python scripts/plot_sft_loss.py --log_file <log_file>")
        return
    
    print(f"📊 解析日志文件: {log_file}")
    
    data = parse_training_log(log_file)
    
    if not data['train_losses']:
        print("❌ 日志中未找到损失数据")
        return
    
    print(f"\n📈 损失统计:")
    print(f"  训练损失记录数: {len(data['train_losses'])}")
    print(f"  评估损失记录数: {len(data['eval_losses'])}")
    print(f"  最小训练损失: {min(data['train_losses']):.4f}")
    if data['eval_losses']:
        print(f"  最小评估损失: {min(data['eval_losses']):.4f}")
    
    output_path = args.output
    if output_path is None:
        log_name = Path(log_file).stem
        output_path = f"loss_curve_{log_name}.png"
    
    print(f"\n🎨 绘制损失曲线...")
    
    plot_combined_loss(data, save_path=output_path, show=not args.no_show)
    
    print(f"\n✅ 完成！")


if __name__ == "__main__":
    main()

"""
GRPO训练脚本
=====================================================================
该脚本提供了GRPO训练的完整入口，支持命令行参数配置。
可用于训练各种问答任务，包括亲属关系推理。

功能特点：
1. 命令行参数解析
2. 多任务支持（kinship, qa, multi_hop）
3. LoRA微调支持
4. 4bit量化支持
5. 模型保存和断点续训
6. 日志记录

使用方法：
    python scripts/train_grpo.py --model_path <model> --task_type kinship --epochs 10
    python scripts/train_grpo.py --task_type qa --batch_size 4 --learning_rate 1e-4
=====================================================================
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from src.config import get_default_config, load_config
from src.grpo import (
    GRPOTrainer,
    KinshipRewardFunction,
    MultiHopRewardFunction,
    QARewardFunction,
    create_reward_function
)
from src.utils.model_utils import (
    load_model_and_tokenizer,
    prepare_model_for_training,
    get_model_memory_usage
)


# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    """
    简单问答数据集
    
    用于快速测试的内存数据集。
    
    Attributes:
        queries: 问题列表
        ground_truths: 答案列表
    """
    
    def __init__(self, queries, ground_truths=None):
        """
        初始化数据集
        
        Args:
            queries: 问题列表
            ground_truths: 答案列表，None则使用占位符
        """
        self.queries = queries
        self.ground_truths = ground_truths or [None] * len(queries)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.queries)
    
    def __getitem__(self, idx):
        """获取数据项"""
        return {
            'query': self.queries[idx],
            'ground_truth': self.ground_truths[idx]
        }


def create_sample_data(task_type: str = "kinship"):
    """
    创建示例训练数据
    
    Args:
        task_type: 任务类型，可选 "kinship", "qa", "other"
        
    Returns:
        (问题, 答案) 元组列表
        
    Example:
        >>> data = create_sample_data("kinship")
        >>> [("Who is the father of Alice?", "Bob"), ...]
    """
    if task_type == "kinship":
        return [
            ("Who is the grandfather of Alice?", "Bob"),
            ("Who is the sister of John's father?", "Mary"),
            ("Who is the aunt of Tom?", "Susan"),
            ("Who is the uncle of Emma?", "James"),
            ("Who is the grandmother of baby?", "Nancy"),
            ("What is the relationship of Sarah to her brother's son?", "nephew"),
            ("Who is the cousin of Mike?", "Lisa"),
            ("What do you call the daughter of your aunt?", "cousin"),
            ("Who is the nephew of Mrs. Smith?", "Tommy"),
            ("What is the grandfather's granddaughter called?", "granddaughter"),
        ]
    elif task_type == "qa":
        return [
            ("What is the capital of France?", "Paris"),
            ("Who wrote Romeo and Juliet?", "William Shakespeare"),
            ("What is the largest planet?", "Jupiter"),
            ("When did World War II end?", "1945"),
            ("What is the chemical symbol for gold?", "Au"),
        ]
    else:
        return [
            ("Solve: 2 + 3 = ?", "5"),
            ("What color is the sky?", "blue"),
        ]


def parse_args():
    """
    解析命令行参数
    
    Returns:
        包含所有命令行参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description='GRPO Training Script')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, 
                       default="/Users/xry/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct",
                       help='模型路径，可以是HuggingFace模型名或本地路径')
    
    # 任务参数
    parser.add_argument('--task_type', type=str, default='kinship',
                       choices=['kinship', 'qa', 'multi_hop'],
                       help='任务类型')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./outputs/grpo_output',
                       help='模型输出目录')
    
    # GRPO参数
    parser.add_argument('--group_size', type=int, default=3,
                       help='每组轨迹数量，用于计算组内相对优势')
    
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='学习率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=2,
                       help='批次大小')
    
    parser.add_argument('--epochs', type=int, default=5,
                       help='训练轮数')
    
    # 生成参数
    parser.add_argument('--max_new_tokens', type=int, default=256,
                       help='最大生成token数')
    
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='生成温度')
    
    # LoRA参数
    parser.add_argument('--use_lora', action='store_true', default=True,
                       help='是否使用LoRA微调')
    
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA秩')
    
    # 量化参数
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                       help='是否使用4bit量化加载模型')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='auto',
                       help='设备，"auto"表示自动选择')
    
    # 随机种子
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子，用于复现结果')
    
    return parser.parse_args()


def main():
    """
    主训练函数
    
    执行完整的训练流程：
    1. 解析参数
    2. 设置随机种子
    3. 加载模型和分词器
    4. 准备模型（应用LoRA）
    5. 初始化奖励函数
    6. 初始化训练器
    7. 训练循环
    8. 保存模型
    """
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 打印训练配置
    logger.info("=" * 60)
    logger.info("GRPO Training Started")
    logger.info("=" * 60)
    
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Task Type: {args.task_type}")
    logger.info(f"Output Dir: {args.output_dir}")
    logger.info(f"Group Size: {args.group_size}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Use LoRA: {args.use_lora}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和分词器
    logger.info("加载模型和分词器...")
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        load_in_4bit=args.load_in_4bit,
        device=args.device,
        torch_dtype="bfloat16",
        pad_token=None,
        trust_remote_code=True
    )
    
    # 打印内存使用
    mem_info = get_model_memory_usage(model)
    logger.info(f"模型内存使用: {mem_info}")
    
    # 准备模型
    logger.info("准备模型...")
    if args.use_lora:
        model = prepare_model_for_training(
            model,
            use_lora=True,
            lora_config={
                'lora_r': args.lora_r,
                'lora_alpha': 32,
                'lora_dropout': 0.05,
                'bias': 'none'
            }
        )
        mem_info = get_model_memory_usage(model)
        logger.info(f"LoRA 模型内存使用: {mem_info}")
    
    # 配置GRPO参数
    grpo_config = {
        'group_size': args.group_size,
        'clip_epsilon': 0.1,
        'kl_coeff': 0.05,
        'beta': 0.01,
        'learning_rate': args.learning_rate,
        'normalize_reward': True,
        'reward_scale': 0.1,
        'max_grad_norm': 1.0
    }
    
    # 配置训练参数
    training_config = {
        'output_dir': args.output_dir,
        'logging_dir': os.path.join(args.output_dir, 'logs'),
        'logging_steps': 10,
        'save_steps': 200,
        'eval_steps': 200,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'buffer_size': 500,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': 0.9,
        'exploration_rate': 0.1
    }
    
    # 根据任务类型选择奖励函数
    if args.task_type == "kinship":
        reward_function = KinshipRewardFunction(
            path_length_penalty=0.1,
            correct_answer_bonus=1.0,
            wrong_answer_penalty=0.0
        )
    elif args.task_type == "qa":
        reward_function = QARewardFunction(
            em_weight=1.0,
            f1_weight=0.5,
            length_penalty=0.01
        )
    else:
        reward_function = MultiHopRewardFunction(
            path_length_penalty=0.1,
            correct_answer_bonus=1.0,
            wrong_answer_penalty=0.0
        )
    
    # 初始化训练器
    logger.info("初始化训练器...")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        grpo_config=grpo_config,
        training_config=training_config,
        reward_function=reward_function,
        device=args.device
    )
    
    # 准备训练数据
    logger.info("准备训练数据...")
    sample_data = create_sample_data(args.task_type)
    queries = [item[0] for item in sample_data]
    ground_truths = [item[1] for item in sample_data]
    
    # 创建数据加载器
    dataset = SimpleDataset(queries, ground_truths)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: {
            'query': [item['query'] for item in x],
            'ground_truth': [item['ground_truth'] for item in x]
        }
    )
    
    # 准备优化器
    num_training_steps = len(dataloader) * args.epochs
    trainer.prepare_optimizer(num_training_steps)
    
    # 开始训练
    logger.info("开始训练...")
    logger.info(f"总训练步数: {num_training_steps}")
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # 训练一个epoch
        metrics = trainer.train_epoch(dataloader)
        
        # 打印指标
        logger.info(f"Epoch {epoch + 1} Metrics:")
        logger.info(f"  Mean Loss: {metrics['mean_loss']:.4f}")
        logger.info(f"  Mean Policy Loss: {metrics['mean_policy_loss']:.4f}")
        logger.info(f"  Mean Reward: {metrics['mean_reward']:.4f}")
        
        # 每2个epoch保存一次
        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}')
            trainer.save_model(save_path)
    
    # 训练完成
    logger.info("\n" + "=" * 60)
    logger.info("Training Completed!")
    logger.info("=" * 60)
    
    # 保存最终模型
    final_save_path = os.path.join(args.output_dir, 'final_model')
    trainer.save_model(final_save_path)
    
    logger.info(f"最终模型已保存到: {final_save_path}")


if __name__ == "__main__":
    main()

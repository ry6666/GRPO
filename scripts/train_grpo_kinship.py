"""
GRPO亲属关系数据集训练脚本
=====================================================================
该脚本专门用于在亲属关系知识图谱数据上训练GRPO模型。
支持多跳推理、知识图谱上下文和LoRA微调。

功能特点：
1. 加载真实亲属关系数据
2. 生成单跳和多跳问答对
3. 使用知识图谱上下文增强推理
4. 完整的训练和评估流程
5. 支持加载SFT训练后的模型继续优化

SFT → GRPO 训练流程：
1. 先运行 SFT 训练：python scripts/train_sft_kinship.py --epochs 3
2. 再运行 GRPO 训练：python scripts/train_grpo_kinship.py --sft_model_path ./outputs/sft_kinship/best_model

使用方法：
    # 直接使用预训练模型
    python scripts/train_grpo_kinship.py --data_path ./kinship/kinship.data
    
    # 使用SFT训练后的模型（推荐）
    python scripts/train_grpo_kinship.py --sft_model_path ./outputs/sft_kinship/best_model --epochs 5
    
    # 多跳问答
    python scripts/train_grpo_kinship.py --multi_hop --epochs 10
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

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_default_config, load_config
from src.grpo import (
    GRPOTrainer,
    KinshipRewardFunction,
    MultiHopRewardFunction,
    create_reward_function
)
from src.data.kinship import (
    KinshipDataset,
    load_kinship_data,
    create_kinship_context,
    format_prompt_with_context
)
from src.utils.model_utils import (
    load_model_and_tokenizer,
    prepare_model_for_training,
    get_model_memory_usage
)


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class KinshipQADataset(Dataset):
    """
    亲属关系问答数据集
    
    封装亲属关系问答对，支持知识图谱上下文。
    
    Attributes:
        queries: 问题列表
        answers: 答案列表
        kinship_context: 知识图谱上下文
        extra_info: 额外信息列表
    """
    
    def __init__(self, qa_pairs, kinship_context: str = None):
        """
        初始化数据集
        
        Args:
            qa_pairs: 问答对列表
            kinship_context: 知识图谱上下文字符串
        """
        self.queries = [pair['query'] for pair in qa_pairs]
        self.answers = [pair['answer'] for pair in qa_pairs]
        self.kinship_context = kinship_context
        self.extra_info = qa_pairs
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.queries)
    
    def __getitem__(self, idx):
        """获取数据项"""
        return {
            'query': self.queries[idx],
            'answer': self.answers[idx],
            'kinship_context': self.kinship_context,
            'extra_info': self.extra_info[idx]
        }


def parse_args():
    """
    解析命令行参数
    
    Returns:
        包含所有参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description='GRPO Training with Kinship Dataset')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, 
                       default="/Users/xry/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct",
                       help='预训练模型路径（SFT训练前）')
    
    parser.add_argument('--sft_model_path', type=str, default=None,
                       help='SFT训练后的模型路径（优先使用，覆盖model_path）')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, 
                       default="./kinship/kinship.data",
                       help='Kinship数据文件路径')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./outputs/grpo_kinship',
                       help='输出目录')
    
    # GRPO参数
    parser.add_argument('--group_size', type=int, default=3,
                       help='组大小')
    
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='学习率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=2,
                       help='批次大小')
    
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    
    # 生成参数
    parser.add_argument('--max_new_tokens', type=int, default=256,
                       help='最大生成token数')
    
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='生成温度')
    
    # LoRA参数
    parser.add_argument('--use_lora', action='store_true', default=True,
                       help='是否使用LoRA')
    
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA秩')
    
    # 量化参数
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                       help='是否使用4bit量化')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='auto',
                       help='设备')
    
    # 随机种子
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 任务参数
    parser.add_argument('--multi_hop', action='store_true', default=False,
                       help='是否使用多跳问答')
    
    parser.add_argument('--use_context', action='store_true', default=True,
                       help='是否使用知识图谱上下文')
    
    return parser.parse_args()


def main():
    """
    主函数
    
    执行亲属关系数据的GRPO训练：
    1. 加载亲属关系数据
    2. 生成问答对
    3. 加载模型
    4. 训练模型
    5. 评估和保存
    """
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("GRPO Training with Kinship Dataset")
    logger.info("=" * 60)
    
    effective_model_path = args.sft_model_path if args.sft_model_path else args.model_path
    logger.info(f"Model Path: {effective_model_path}")
    if args.sft_model_path:
        logger.info(f"  (从SFT模型加载: {args.sft_model_path})")
    logger.info(f"Data Path: {args.data_path}")
    logger.info(f"Output Dir: {args.output_dir}")
    logger.info(f"Group Size: {args.group_size}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Multi-hop: {args.multi_hop}")
    logger.info(f"Use Context: {args.use_context}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载亲属关系数据集
    logger.info("\n加载 kinship 数据集...")
    kinship_dataset = KinshipDataset(args.data_path)
    stats = kinship_dataset.get_stats()
    logger.info(f"数据集统计: {stats}")
    
    # 生成问答对
    queries, answers, qa_pairs = load_kinship_data(args.data_path, args.multi_hop)
    logger.info(f"加载了 {len(qa_pairs)} 个问答对")
    
    # 创建知识图谱上下文
    kinship_context = None
    if args.use_context:
        kinship_context = create_kinship_context(kinship_dataset)
        logger.info(f"知识图谱上下文已创建，共 {len(kinship_context)} 字符")
    
    # 加载模型和分词器
    logger.info("\n加载模型和分词器...")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(
        effective_model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.sft_model_path:
        logger.info(f"从SFT模型加载: {args.sft_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if args.load_in_4bit else torch.float32,
            load_in_4bit=args.load_in_4bit,
            device_map=args.device,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, args.sft_model_path)
        logger.info("已加载SFT训练的LoRA权重")
    else:
        model, tokenizer = load_model_and_tokenizer(
            model_path=effective_model_path,
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
    
    # 配置GRPO
    grpo_config = {
        'group_size': args.group_size,
        'clip_epsilon': 0.1,
        'kl_coeff': 0.05,
        'beta': 0.01,
        'learning_rate': args.learning_rate,
        'normalize_reward': True,
        'reward_scale': 0.1,
        'max_grad_norm': 1.0,
        'path_length_penalty': 0.1
    }
    
    # 配置训练
    training_config = {
        'output_dir': args.output_dir,
        'logging_dir': os.path.join(args.output_dir, 'logs'),
        'logging_steps': 10,
        'save_steps': 500,
        'eval_steps': 500,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'buffer_size': 500,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': 0.9,
        'exploration_rate': 0.1
    }
    
    # 创建奖励函数
    reward_function = KinshipRewardFunction(
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
    
    # 准备数据
    logger.info("准备训练数据...")
    dataset = KinshipQADataset(qa_pairs, kinship_context)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: {
            'query': [item['query'] for item in x],
            'ground_truth': [item['answer'] for item in x],
            'kinship_context': [item['kinship_context'] for item in x],
            'extra_info': [item['extra_info'] for item in x]
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
        
        metrics = trainer.train_epoch(dataloader)
        
        logger.info(f"Epoch {epoch + 1} Metrics:")
        logger.info(f"  Mean Loss: {metrics['mean_loss']:.4f}")
        logger.info(f"  Mean Policy Loss: {metrics['mean_policy_loss']:.4f}")
        logger.info(f"  Mean Reward: {metrics['mean_reward']:.4f}")
        
        # 保存检查点
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
    
    # 评估模型
    logger.info("\n评估模型...")
    eval_results = trainer.evaluate(dataloader)
    logger.info(f"评估结果: {eval_results}")


if __name__ == "__main__":
    main()

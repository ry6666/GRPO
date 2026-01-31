"""
SFT（有监督微调）训练脚本 - 亲属关系任务
=====================================================================
该脚本用于在亲属关系知识图谱数据上进行SFT训练，
为后续GRPO训练奠定基础。

核心目标：
1. 让模型掌握kinship任务的基础亲属关系规则
2. 统一答案格式（短句回答、无冗余表述）
3. 适配模型的指令跟随能力

使用方法：
    python scripts/train_sft_kinship.py --epochs 3
    python scripts/train_sft_kinship.py --data_path ./dataset/kinship.data --epochs 5
=====================================================================
"""

import os
import sys
import argparse
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_default_config
from src.data.kinship import (
    KinshipDataset,
    load_kinship_data,
    create_kinship_context
)
from src.data.kinship_augment import augment_dataset, load_augmented_data, get_sft_format_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'sft_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class SFTKinshipDataset(Dataset):
    """
    SFT亲属关系数据集
    
    将问答对格式化为指令微调的训练格式。
    
    Attributes:
        conversations: 会话列表
        tokenizer: 分词器
        max_length: 最大序列长度
    """
    
    def __init__(
        self,
        queries: List[str],
        answers: List[str],
        tokenizer,
        max_length: int = 512
    ):
        """
        初始化数据集
        
        Args:
            queries: 问题列表
            answers: 答案列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.conversations = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for query, answer in zip(queries, answers):
            conversation = self._format_conversation(query, answer)
            self.conversations.append(conversation)
    
    def _format_conversation(self, query: str, answer: str) -> List[Dict]:
        """
        格式化对话
        
        遵循Qwen的对话格式。
        
        Args:
            query: 问题
            answer: 答案
            
        Returns:
            格式化后的对话列表
        """
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant specialized in kinship relationships. Answer the questions about family relationships accurately and concisely."
            },
            {
                "role": "user",
                "content": query
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.conversations)
    
    def __getitem__(self, idx):
        """获取数据项"""
        conversation = self.conversations[idx]
        
        try:
            text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            logger.warning(f"模板应用失败，使用默认格式: {e}")
            text = f"User: {conversation[1]['content']}\nAssistant: {conversation[2]['content']}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def create_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05) -> LoraConfig:
    """
    创建LoRA配置
    
    Args:
        r: LoRA矩阵的秩
        alpha: 缩放因子
        dropout: Dropout比率
        
    Returns:
        LoRA配置对象
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False
    )


def load_model_and_tokenizer(
    model_path: str,
    torch_dtype: str = "float16",
    device: str = "auto"
):
    """
    加载模型和分词器
    
    Args:
        model_path: 模型路径
        torch_dtype: 数据类型 ('bfloat16', 'float16', 'float32')
        device: 设备 ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        tuple: (模型, 分词器)
    """
    logger.info(f"加载模型: {model_path}")
    logger.info(f"数据类型: {torch_dtype}, 设备: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32
    }
    torch_dtype_value = dtype_map.get(torch_dtype, torch.float16)
    
    if device == "mps":
        device_map = None
    else:
        device_map = "auto"
    
    logger.info(f"加载参数: dtype={torch_dtype_value}, device_map={device_map}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype_value,
        device_map=device_map,
        trust_remote_code=True
    )
    
    if device == "mps":
        model = model.to("mps")
    
    return model, tokenizer


def prepare_model_for_sft(model, lora_config: LoraConfig):
    """
    准备模型用于SFT训练
    
    Args:
        model: 原始模型
        lora_config: LoRA配置
        
    Returns:
        添加了LoRA的模型
    """
    logger.info("应用LoRA配置...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    gradient_accumulation_steps: int = 1
):
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        gradient_accumulation_steps: 梯度累积步数
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        
    Returns:
        平均损失
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(dataloader)
    logger.info(f"评估损失: {avg_loss:.4f}")
    return avg_loss


def save_model(model, tokenizer, output_dir: str, epoch: int = None):
    """
    保存模型
    
    Args:
        model: 模型
        tokenizer: 分词器
        output_dir: 输出目录
        epoch: 当前epoch（用于区分保存文件夹）
    """
    save_path = Path(output_dir)
    
    if epoch is not None:
        save_path = save_path / f"epoch_{epoch}"
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"保存模型到: {save_path}")
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    logger.info("模型保存完成！")


def parse_args():
    """
    解析命令行参数
    
    Returns:
        包含所有参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description='SFT Training for Kinship Task')
    
    parser.add_argument('--model_path', type=str,
                       default="/Users/xry/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct",
                       help='预训练模型路径')
    
    parser.add_argument('--data_path', type=str,
                       default="./dataset/kinship.data",
                       help='Kinship数据文件路径')
    
    parser.add_argument('--output_dir', type=str,
                       default='./outputs/sft_kinship',
                       help='输出目录')
    
    parser.add_argument('--epochs', type=int, default=3,
                       help='训练轮数')
    
    parser.add_argument('--batch_size', type=int, default=2,
                       help='批次大小')
    
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='学习率')
    
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大序列长度')
    
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='学习率预热比例')
    
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='梯度累积步数')
    
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA秩')
    
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA缩放因子')
    
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='LoRA Dropout')
    
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                       help='是否使用4bit量化')
    
    parser.add_argument('--save_every_epoch', action='store_true',
                       help='是否每个epoch都保存模型')
    
    parser.add_argument('--test_size', type=float, default=0.1,
                       help='测试集比例')
    
    parser.add_argument('--use_augmented_data', action='store_true',
                       help='是否使用增强后的数据集')
    
    parser.add_argument('--augmented_train_path', type=str,
                       default="./dataset/augmented/train.json",
                       help='增强后的训练集路径')
    
    parser.add_argument('--augmented_test_path', type=str,
                       default="./dataset/augmented/test.json",
                       help='增强后的测试集路径')
    
    parser.add_argument('--augment_factor', type=int, default=3,
                       help='数据增强倍数（仅在未预生成时使用）')
    
    parser.add_argument('--use_m4_optimized', action='store_true',
                       help='是否使用 M4 Apple Silicon 优化配置')
    
    parser.add_argument('--torch_dtype', type=str, default='float16',
                       choices=['bfloat16', 'float16', 'float32'],
                       help='PyTorch 数据类型（MPS 建议用 float16）')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='训练设备')
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("SFT 训练 - 亲属关系任务")
    logger.info("=" * 60)
    logger.info(f"模型: {args.model_path}")
    logger.info(f"数据: {args.data_path}")
    logger.info(f"使用增强数据: {args.use_augmented_data}")
    logger.info(f"使用 M4 优化: {args.use_m4_optimized}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"数据类型: {args.torch_dtype}")
    logger.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == 'cpu':
        device = torch.device("cpu")
    elif args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
    
    logger.info(f"使用设备: {device}")
    
    logger.info("加载数据...")
    
    if args.use_augmented_data:
        logger.info(f"加载增强后的训练集: {args.augmented_train_path}")
        logger.info(f"加载增强后的测试集: {args.augmented_test_path}")
        
        train_data, test_data = load_augmented_data(
            args.augmented_train_path,
            args.augmented_test_path
        )
        queries, answers = get_sft_format_data(train_data)
        
        logger.info(f"加载了 {len(queries)} 个训练问答对")
        logger.info(f"测试集: {len(test_data)} 个问答对")
    else:
        queries, answers, qa_pairs = load_kinship_data(args.data_path, multi_hop=False)
        logger.info(f"加载了 {len(queries)} 个问答对")
        
        test_data = None
    
    logger.info("加载模型和分词器...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        torch_dtype=args.torch_dtype,
        device=str(device)
    )
    
    logger.info("创建SFT数据集...")
    dataset = SFTKinshipDataset(
        queries=queries,
        answers=answers,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    if args.use_augmented_data and test_data:
        test_queries, test_answers = get_sft_format_data(test_data)
        eval_dataset = SFTKinshipDataset(
            queries=test_queries,
            answers=test_answers,
            tokenizer=tokenizer,
            max_length=args.max_length
        )
        train_size = len(dataset) - len(eval_dataset)
        
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        
        logger.info(f"训练集: {len(train_dataset)} 样本")
        logger.info(f"验证集: {len(eval_dataset)} 样本")
    else:
        train_size = int(len(dataset) * (1 - args.test_size))
        train_dataset, eval_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, len(dataset) - train_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"训练集: {len(train_dataset)} 样本")
        logger.info(f"验证集: {len(eval_dataset)} 样本")
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    logger.info("配置LoRA...")
    lora_config = create_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )
    
    model = prepare_model_for_sft(model, lora_config)
    model.to(device)
    
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("开始训练...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            args.gradient_accumulation_steps
        )
        logger.info(f"训练损失: {train_loss:.4f}")
        
        eval_loss = evaluate(model, eval_loader, device)
        
        if eval_loss < best_loss:
            best_loss = eval_loss
            save_model(model, tokenizer, str(output_dir / "best_model"), epoch + 1)
            logger.info(f"保存最佳模型 (损失: {eval_loss:.4f})")
        
        if args.save_every_epoch:
            save_model(model, tokenizer, str(output_dir), epoch + 1)
    
    logger.info("\n训练完成！")
    logger.info(f"最佳验证损失: {best_loss:.4f}")
    logger.info(f"模型已保存到: {output_dir}")
    logger.info("\n后续GRPO训练时，使用以下命令加载SFT模型:")
    logger.info(f"  python scripts/train_grpo_kinship.py --model_path {output_dir}/best_model")


if __name__ == "__main__":
    main()

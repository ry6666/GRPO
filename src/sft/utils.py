"""
SFT 工具函数模块
=====================================================================
提供 SFT 训练所需的工具函数，包括：
1. LoRA 配置创建
2. 模型和分词器加载
3. 模型保存
4. 优化器和调度器设置

使用示例：
    from src.sft.utils import (
        create_lora_config,
        load_model_and_tokenizer,
        save_model
    )
    
    lora_config = create_lora_config(r=16, alpha=32)
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct")
=====================================================================
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig
)

logger = logging.getLogger(__name__)


def create_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    bias: str = "none",
    target_modules: Optional[list] = None,
    task_type: str = "CAUSAL_LM"
) -> LoraConfig:
    """
    创建 LoRA 配置
    
    Args:
        r: LoRA 矩阵的秩（低秩分解的维度）
        alpha: 缩放因子，通常设置为 r 的 2 倍
        dropout: Dropout 比率
        bias: 偏置项处理方式
        target_modules: 目标模块列表
        task_type: 任务类型
        
    Returns:
        LoraConfig 实例
    
    Example:
        >>> config = create_lora_config(r=16, alpha=32)
        >>> print(config.r)
        16
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ2SEQ_LM": TaskType.SEQ2SEQ_LM,
        "TOKEN_CLASSIFICATION": TaskType.TOKEN_CLASSIFICATION,
    }
    
    return LoraConfig(
        task_type=task_type_map.get(task_type, TaskType.CAUSAL_LM),
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules=target_modules,
        inference_mode=False
    )


def load_model_and_tokenizer(
    model_path: str,
    torch_dtype: str = "bfloat16",
    load_in_4bit: bool = True,
    device_map: str = "auto",
    pad_token: Optional[str] = None,
    trust_remote_code: bool = True
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    加载模型和分词器
    
    Args:
        model_path: 模型路径
        torch_dtype: PyTorch 数据类型
        load_in_4bit: 是否使用 4bit 量化
        device_map: 设备映射策略
        pad_token: 填充 token
        trust_remote_code: 是否信任远程代码
        
    Returns:
        (模型, 分词器) 元组
    
    Example:
        >>> model, tokenizer = load_model_and_tokenizer(
        ...     "Qwen/Qwen2.5-7B-Instruct",
        ...     load_in_4bit=True
        ... )
    """
    logger.info(f"加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if pad_token is not None:
        tokenizer.pad_token = pad_token
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    
    torch_dtype_value = dtype_map.get(torch_dtype, torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype_value,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
        trust_remote_code=trust_remote_code
    )
    
    logger.info("模型加载完成！")
    
    return model, tokenizer


def load_sft_model(
    base_model_path: str,
    sft_model_path: str,
    load_in_4bit: bool = True,
    device_map: str = "auto"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    加载 SFT 训练后的模型
    
    加载预训练基础模型，然后合并 LoRA 权重。
    
    Args:
        base_model_path: 基础模型路径
        sft_model_path: SFT 模型路径
        load_in_4bit: 是否使用 4bit 量化
        device_map: 设备映射策略
        
    Returns:
        (模型, 分词器) 元组
    
    Example:
        >>> model, tokenizer = load_sft_model(
        ...     "Qwen/Qwen2.5-7B-Instruct",
        ...     "./outputs/sft/best_model"
        ... )
    """
    logger.info(f"加载基础模型: {base_model_path}")
    logger.info(f"加载 SFT 模型: {sft_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        sft_model_path,
        trust_remote_code=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if load_in_4bit else torch.float32,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, sft_model_path)
    
    logger.info("SFT 模型加载完成！")
    
    return model, tokenizer


def prepare_model_for_training(
    model: nn.Module,
    lora_config: Optional[LoraConfig] = None,
    use_gradient_checkpointing: bool = True
) -> nn.Module:
    """
    准备模型用于训练
    
    应用 LoRA 配置（如果提供），并启用梯度检查点（如果需要）。
    
    Args:
        model: 原始模型
        lora_config: LoRA 配置
        use_gradient_checkpointing: 是否使用梯度检查点
        
    Returns:
        准备好的模型
    
    Example:
        >>> model = prepare_model_for_training(
        ...     model,
        ...     lora_config=lora_config
        ... )
        >>> model.print_trainable_parameters()
    """
    if lora_config is not None:
        logger.info("应用 LoRA 配置...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        logger.info("启用梯度检查点...")
        model.gradient_checkpointing_enable()
    
    return model


def save_model(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    save_full_model: bool = False
) -> str:
    """
    保存模型
    
    Args:
        model: 模型
        tokenizer: 分词器
        output_dir: 输出目录
        save_full_model: 是否保存完整模型（否则只保存 LoRA 权重）
        
    Returns:
        保存路径
    """
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"保存模型到: {save_path}")
    
    if isinstance(model, PeftModel) and not save_full_model:
        model.save_pretrained(save_path)
    else:
        model.save_pretrained(save_path)
    
    tokenizer.save_pretrained(save_path)
    
    logger.info("模型保存完成！")
    
    return str(save_path)


def save_lora_adapter(
    model: nn.Module,
    output_dir: str
) -> str:
    """
    只保存 LoRA 适配器权重
    
    Args:
        model: 包含 LoRA 的模型
        output_dir: 输出目录
        
    Returns:
        保存路径
    """
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if isinstance(model, PeftModel):
        model.save_pretrained(save_path)
        logger.info(f"LoRA 适配器已保存到: {save_path}")
    else:
        logger.warning("模型不是 PeftModel，无法保存 LoRA 适配器")
    
    return str(save_path)


def get_model_memory_usage(model: nn.Module) -> Dict:
    """
    获取模型显存使用情况
    
    Args:
        model: 模型
        
    Returns:
        显存使用信息字典
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        
        return {
            "allocated_GB": round(allocated, 2),
            "reserved_GB": round(reserved, 2)
        }
    else:
        return {
            "message": "CUDA 不可用",
            "allocated_GB": "N/A",
            "reserved_GB": "N/A"
        }


def get_model_size(model: nn.Module) -> Dict:
    """
    获取模型参数量信息
    
    Args:
        model: 模型
        
    Returns:
        模型大小信息字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_size(num_params: int) -> str:
        if num_params >= 1e9:
            return f"{num_params / 1e9:.2f}B"
        elif num_params >= 1e6:
            return f"{num_params / 1e6:.2f}M"
        elif num_params >= 1e3:
            return f"{num_params / 1e3:.2f}K"
        else:
            return str(num_params)
    
    return {
        "total_params": get_size(total_params),
        "trainable_params": get_size(trainable_params),
        "trainable_ratio": f"{trainable_params / total_params * 100:.2f}%"
    }


def setup_optimizer(
    model: nn.Module,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01
) -> torch.optim.AdamW:
    """
    设置优化器
    
    Args:
        model: 模型
        learning_rate: 学习率
        weight_decay: 权重衰减
        
    Returns:
        AdamW 优化器实例
    """
    no_decay = ['bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]
    
    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate
    )


def setup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    设置学习率调度器
    
    Args:
        optimizer: 优化器
        num_training_steps: 总训练步数
        warmup_ratio: 预热比例
        
    Returns:
        学习率调度器实例
    """
    from transformers import get_linear_schedule_with_warmup
    
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )


def get_default_sft_config() -> Dict:
    """
    获取默认 SFT 训练配置
    
    Returns:
        默认配置字典
    """
    return {
        'learning_rate': 5e-5,
        'batch_size': 2,
        'epochs': 3,
        'max_length': 512,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 4,
        'max_grad_norm': 1.0,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'load_in_4bit': False,
        'save_best': True,
        'test_size': 0.1,
        'torch_dtype': 'float16',
        'device': 'mps'
    }


def get_m4_optimized_config() -> Dict:
    """
    获取 M4 Apple Silicon 优化配置
    
    针对 M4 芯片的最佳配置：
    - 使用 MPS 后端
    - float16 精度
    - 较小的 batch_size
    - 较大的梯度累积
    
    Returns:
        优化配置字典
    """
    return {
        'learning_rate': 1e-4,
        'batch_size': 1,
        'epochs': 3,
        'max_length': 512,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 8,
        'max_grad_norm': 1.0,
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'load_in_4bit': False,
        'save_best': True,
        'test_size': 0.1,
        'torch_dtype': 'float16',
        'device': 'mps'
    }


def get_gpu_config(device: Optional[str] = None) -> Dict:
    """
    根据设备类型获取优化配置
    
    Args:
        device: 设备类型 ('cuda', 'mps', 'cpu')
        
    Returns:
        配置字典
    """
    if device == 'mps' or (device is None and torch.backends.mps.is_available()):
        return get_m4_optimized_config()
    elif device == 'cuda' or (device is None and torch.cuda.is_available()):
        return get_default_sft_config()
    else:
        return {
            'learning_rate': 5e-5,
            'batch_size': 1,
            'epochs': 2,
            'max_length': 256,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'gradient_accumulation_steps': 16,
            'max_grad_norm': 1.0,
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'load_in_4bit': False,
            'save_best': True,
            'test_size': 0.1,
            'torch_dtype': 'float32',
            'device': 'cpu'
        }

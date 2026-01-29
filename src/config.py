"""
配置模块
=====================================================================
该模块定义了项目所有的配置类，包括模型配置、GRPO算法配置、
LoRA微调配置和训练配置。支持从文件加载和保存配置，以及
获取默认配置字典。
=====================================================================
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """
    模型配置类
    
    负责管理语言模型的所有配置参数，包括模型路径、数据类型、
    量化设置等。支持4bit量化以减少显存占用。
    
    Attributes:
        model_name_or_path: 模型路径，可以是本地路径或HuggingFace模型名
        trust_remote_code: 是否信任远程代码（某些模型需要）
        torch_dtype: PyTorch数据类型，支持bfloat16/float16/float32
        load_in_4bit: 是否使用4bit量化加载模型
        bnb_4bit_use_double_quant: 是否使用双重量化
        bnb_4bit_quant_type: 量化类型，支持nf4
        device_map: 设备映射策略，"auto"表示自动分配
        pad_token: 填充token，None则使用eos_token
    """
    model_name_or_path: str = "/Users/xry/.cache/modelscope/hub/models/Qwen/Qwen2_5-7B-Instruct"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    device_map: str = "auto"
    pad_token: Optional[str] = None


@dataclass
class GRPOConfig:
    """
    GRPO (Group Relative Policy Optimization) 算法超参数配置
    
    GRPO是一种强化学习优化算法，通过组内相对优势来更新策略。
    该配置类管理所有与GRPO算法相关的超参数。
    
    Attributes:
        group_size: 每个问题生成的轨迹数量，用于计算组内相对优势
        clip_epsilon: PPO裁剪系数，控制策略更新幅度
        kl_coeff: KL散度系数，用于防止策略过度偏离旧策略
        learning_rate: 学习率
        batch_size: 批次大小
        epochs: 训练轮数
        max_grad_norm: 梯度裁剪最大值
        warmup_ratio: 学习率预热比例
        weight_decay: 权重衰减系数
        beta: GAE (Generalized Advantage Estimation) 参数
        gamma: 折扣因子
        lam: GAE的lambda参数
        normalize_reward: 是否对奖励进行标准化
        reward_scale: 奖励缩放因子
    """
    group_size: int = 3
    clip_epsilon: float = 0.1
    kl_coeff: float = 0.05
    learning_rate: float = 5e-5
    batch_size: int = 8
    epochs: int = 10
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    beta: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    normalize_reward: bool = True
    reward_scale: float = 0.1


@dataclass
class LoRAConfig:
    """
    LoRA (Low-Rank Adaptation) 微调配置
    
    LoRA是一种参数高效的微调方法，通过低秩分解来更新预训练模型
    的权重，大大减少可训练参数数量。
    
    Attributes:
        lora_r: LoRA矩阵的秩，决定低秩分解的维度
        lora_alpha: 缩放因子，通常设置为r的2倍
        lora_dropout: Dropout比率
        lora_target_modules: 应用LoRA的目标模块列表
        bias: 偏置项处理方式
        task_type: 任务类型，CAUSAL_LM表示因果语言建模
    """
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """
    训练配置类
    
    管理训练过程中的各种配置，包括输出目录、日志记录、
    模型保存策略等。
    
    Attributes:
        output_dir: 模型输出目录
        logging_dir: 日志目录
        logging_steps: 日志记录步数间隔
        save_steps: 模型保存步数间隔
        eval_steps: 评估步数间隔
        save_total_limit: 保存的检查点数量限制
        no_cuda: 是否禁用CUDA
        seed: 随机种子
        report_to: 日志报告目标
        ddp_find_unused_parameters: DDP中是否查找未使用参数
    """
    output_dir: str = "./outputs/grpo_training"
    logging_dir: str = "./logs"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    no_cuda: bool = False
    seed: int = 42
    report_to: str = "tensorboard"
    ddp_find_unused_parameters: bool = False


def get_default_config() -> dict:
    """
    获取默认配置字典
    
    将所有配置类整合为一个嵌套字典，便于序列化和使用。
    该函数主要用于获取项目的默认配置。
    
    Returns:
        dict: 包含model、grpo、lora、training四个主要部分的配置字典
        
    Example:
        >>> config = get_default_config()
        >>> config['model']['model_name_or_path']
        '/Users/xry/.cache/modelscope/hub/models/Qwen/Qwen2_5-7B-Instruct'
    """
    return {
        "model": {
            "model_name_or_path": "/Users/xry/.cache/modelscope/hub/models/Qwen/Qwen2_5-7B-Instruct",
            "trust_remote_code": True,
            "torch_dtype": "bfloat16",
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "device_map": "auto",
            "pad_token": None
        },
        "grpo": {
            "group_size": 3,
            "clip_epsilon": 0.1,
            "kl_coeff": 0.05,
            "learning_rate": 5e-5,
            "batch_size": 8,
            "epochs": 10,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "beta": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "normalize_reward": True,
            "reward_scale": 0.1
        },
        "lora": {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "training": {
            "output_dir": "./outputs/grpo_training",
            "logging_dir": "./logs",
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "save_total_limit": 3,
            "no_cuda": False,
            "seed": 42,
            "report_to": "tensorboard",
            "ddp_find_unused_parameters": False
        }
    }


def load_config(config_path: str = None) -> dict:
    """
    加载配置文件
    
    从JSON文件加载配置。如果未指定路径，则返回默认配置。
    
    Args:
        config_path: JSON配置文件路径
        
    Returns:
        dict: 配置字典
        
    Raises:
        FileNotFoundError: 如果配置文件不存在
        
    Example:
        >>> config = load_config("./config.json")
        >>> config['grpo']['learning_rate']
        5e-5
    """
    if config_path is None:
        return get_default_config()
    
    import json
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(config: dict, config_path: str):
    """
    保存配置到文件
    
    将配置字典保存为JSON格式的配置文件。
    
    Args:
        config: 配置字典
        config_path: 保存路径
        
    Example:
        >>> config = get_default_config()
        >>> save_config(config, "./my_config.json")
    """
    import json
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

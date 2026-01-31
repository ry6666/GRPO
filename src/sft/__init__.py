"""
SFT 训练模块
=====================================================================
提供完整的 SFT（有监督微调）训练功能，支持 LoRA 微调。

主要组件：
- SFTDataset: SFT 数据集类
- SFTTrainer: SFT 训练器类
- 工具函数: 模型加载、配置等

使用方法：
    from src.sft import SFTDataset, SFTTrainer, create_lora_config
    
    dataset = SFTDataset(queries, answers, tokenizer)
    trainer = SFTTrainer(model, tokenizer, config)
    trainer.train(dataset)
=====================================================================
"""

from .dataset import SFTDataset
from .trainer import SFTTrainer
from .utils import (
    create_lora_config,
    load_model_and_tokenizer,
    save_model,
    setup_optimizer,
    setup_scheduler
)

__all__ = [
    'SFTDataset',
    'SFTTrainer',
    'create_lora_config',
    'load_model_and_tokenizer',
    'save_model',
    'setup_optimizer',
    'setup_scheduler'
]

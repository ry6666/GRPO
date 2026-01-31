import os
import sys
import json
import random
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# 设置工作目录
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
print(f"Project root: {PROJECT_ROOT}")

# Cell 1: 环境检测
print("\n" + "=" * 60)
print("🍎 Cell 1: 环境检测")
print("=" * 60)

import torch
mps_available = torch.backends.mps.is_available()
mps_built = torch.backends.mps.is_built()
device = torch.device("mps" if mps_available else "cpu")

print(f"📱 MPS 可用: {mps_available}")
print(f"📱 MPS 已构建: {mps_built}")
print(f"🔵 CUDA 可用: {torch.cuda.is_available()}")
print(f"📦 PyTorch 版本: {torch.__version__}")
print(f"✅ 使用设备: {device}")

# Cell 2: 导入依赖
print("\n" + "=" * 60)
print("📦 Cell 2: 导入依赖")
print("=" * 60)

try:
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        DataCollatorForSeq2Seq,
        get_linear_schedule_with_warmup
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from src.sft import create_lora_config
    from src.data.kinship_augment import load_augmented_data
    print("✅ 所有依赖导入成功")
except Exception as e:
    print(f"❌ 导入错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cell 3: 配置参数
print("\n" + "=" * 60)
print("📋 Cell 3: 配置参数")
print("=" * 60)

config = {
    "model_path": "/Users/xry/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct",
    "train_data_path": "./dataset/augmented/train.json",
    "test_data_path": "./dataset/augmented/test.json",
    "output_dir": "./outputs/sft_kinship",
    "epochs": 3,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "max_length": 512,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 8,
    "max_grad_norm": 1.0,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "torch_dtype": torch.float16,
    "device": str(device)
}

os.makedirs(config["output_dir"], exist_ok=True)
print("✅ 配置完成")

# Cell 4: 加载数据
print("\n" + "=" * 60)
print("📊 Cell 4: 加载数据")
print("=" * 60)

try:
    train_data, test_data = load_augmented_data(
        config["train_data_path"],
        config["test_data_path"]
    )
    queries = [item['query'] for item in train_data]
    answers = [item['answer'] for item in train_data]
    test_queries = [item['query'] for item in test_data]
    test_answers = [item['answer'] for item in test_data]

    print(f"✅ 数据加载成功")
    print(f"  训练集: {len(queries)} 条")
    print(f"  测试集: {len(test_queries)} 条")
except Exception as e:
    print(f"❌ 数据加载错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cell 5: 加载模型
print("\n" + "=" * 60)
print("🤖 Cell 5: 加载模型")
print("=" * 60)

try:
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=config["torch_dtype"],
        device_map=None,
        trust_remote_code=True
    ).to(config["device"])

    print(f"✅ 模型加载成功")
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"❌ 模型加载错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cell 6: 配置 LoRA
print("\n" + "=" * 60)
print("⚡ Cell 6: 配置 LoRA")
print("=" * 60)

try:
    lora_config = create_lora_config(
        r=config["lora_r"],
        alpha=config["lora_alpha"],
        dropout=config["lora_dropout"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✅ LoRA 配置成功")
except Exception as e:
    print(f"❌ LoRA 配置错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 所有前置检查通过！")
print("=" * 60)
print("可以继续运行训练循环")

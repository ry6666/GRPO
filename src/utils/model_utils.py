"""
模型工具模块
=====================================================================
该模块提供了模型加载和处理的实用工具函数。
包括模型加载、LoRA配置、内存管理等核心功能。

主要功能：
1. 加载模型和分词器 (load_model_and_tokenizer)
2. 创建LoRA配置 (create_lora_config)
3. 应用LoRA (apply_lora)
4. 准备模型训练 (prepare_model_for_training)
5. 获取内存使用 (get_model_memory_usage)

设计特点：
- 支持4bit量化减少显存
- 支持自动设备映射
- 支持LoRA参数高效微调
=====================================================================
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Tuple
import logging


logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_path: str,
    load_in_4bit: bool = True,
    device: str = "auto",
    torch_dtype: str = "bfloat16",
    pad_token: Optional[str] = None,
    trust_remote_code: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载模型和分词器
    
    从指定路径加载预训练语言模型及其分词器。
    支持4bit量化以减少显存占用。
    
    Args:
        model_path: 模型路径（HuggingFace hub名称或本地路径）
        load_in_4bit: 是否使用4bit量化
        device: 设备映射策略
        torch_dtype: 模型权重数据类型
        pad_token: 填充token，None则使用eos_token
        trust_remote_code: 是否信任远程代码
        
    Returns:
        (模型, 分词器) 元组
        
    Example:
        >>> model, tokenizer = load_model_and_tokenizer(
        ...     model_path="Qwen/Qwen2.5-7B-Instruct",
        ...     load_in_4bit=True
        ... )
    """
    logger.info(f"加载模型从: {model_path}")
    
    # 解析数据类型
    if torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side="right"
    )
    
    # 设置pad_token
    if pad_token is not None:
        tokenizer.pad_token = pad_token
    else:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("分词器加载完成")
    
    # 构建模型参数字典
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": device,
        "torch_dtype": dtype
    }
    
    # 如果启用4bit量化
    if load_in_4bit:
        model_kwargs.update({
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        })
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    logger.info("模型加载完成")
    
    return model, tokenizer


def create_lora_config(
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: list = None,
    bias: str = "none"
) -> LoraConfig:
    """
    创建LoRA配置
    
    LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法。
    它通过低秩分解来更新预训练模型的权重。
    
    Args:
        lora_r: LoRA矩阵的秩（rank），决定低秩矩阵的维度
        lora_alpha: 缩放因子，通常设置为r的2倍
        lora_dropout: Dropout比率
        lora_target_modules: 应用LoRA的目标模块列表
        bias: 偏置处理方式，可选"none", "all", "lora_only"
        
    Returns:
        LoraConfig配置对象
        
    Note:
        LoRA的核心思想是将权重更新表示为:
        W_new = W + BA
        其中B和A是低秩矩阵，r是秩的维度
        
    Example:
        >>> config = create_lora_config(
        ...     lora_r=16,
        ...     lora_alpha=32,
        ...     lora_target_modules=["q_proj", "v_proj"]
        ... )
    """
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias=bias
    )


def apply_lora(
    model: AutoModelForCausalLM,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: list = None,
    bias: str = "none"
) -> AutoModelForCausalLM:
    """
    应用LoRA到模型
    
    将LoRA适配器添加到预训练模型中。
    
    Args:
        model: 预训练语言模型
        lora_r: LoRA秩
        lora_alpha: LoRA缩放因子
        lora_dropout: Dropout比率
        lora_target_modules: 目标模块列表
        bias: 偏置处理方式
        
    Returns:
        应用LoRA后的模型
        
    Example:
        >>> model = apply_lora(
        ...     model,
        ...     lora_r=16,
        ...     lora_alpha=32
        ... )
    """
    # 创建配置
    peft_config = create_lora_config(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        bias=bias
    )
    
    # 应用LoRA
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    return model


def prepare_model_for_training(
    model: AutoModelForCausalLM,
    use_lora: bool = True,
    lora_config: dict = None
) -> AutoModelForCausalLM:
    """
    准备模型用于训练
    
    整合模型准备的所有步骤，包括：
    1. 应用LoRA（如果启用）
    2. 启用梯度检查点（如果未使用LoRA）
    3. 禁用缓存（用于训练）
    
    Args:
        model: 预训练模型
        use_lora: 是否使用LoRA
        lora_config: LoRA配置字典
        
    Returns:
        准备好的模型
        
    Example:
        >>> model = prepare_model_for_training(
        ...     model,
        ...     use_lora=True,
        ...     lora_config={'lora_r': 16}
        ... )
    """
    if use_lora:
        if lora_config is None:
            lora_config = {}
        
        model = apply_lora(
            model,
            lora_r=lora_config.get('lora_r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            lora_target_modules=lora_config.get('lora_target_modules'),
            bias=lora_config.get('bias', 'none')
        )
    
    # 如果不使用LoRA，启用梯度检查点
    if not use_lora:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    return model


def get_model_memory_usage(model: AutoModelForCausalLM) -> dict:
    """
    获取模型内存使用情况
    
    查询GPU内存使用情况，包括已分配、保留和最大分配内存。
    
    Args:
        model: 语言模型
        
    Returns:
        包含内存使用信息的字典：
        - memory_allocated_gb: 已分配内存（GB）
        - memory_reserved_gb: 保留内存（GB）
        - max_memory_allocated_gb: 最大分配内存（GB）
        - total_parameters: 总参数量
        - trainable_parameters: 可训练参数量
        
    Note:
        此函数仅在CUDA可用时工作
        
    Example:
        >>> mem_info = get_model_memory_usage(model)
        >>> print(f"已使用: {mem_info['memory_allocated_gb']} GB")
    """
    import gc
    
    # 清理缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 获取内存信息（转换为GB）
    mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    mem_max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    # 统计参数量
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "memory_allocated_gb": round(mem_allocated, 2),
        "memory_reserved_gb": round(mem_reserved, 2),
        "max_memory_allocated_gb": round(mem_max_allocated, 2),
        "total_parameters": param_count,
        "trainable_parameters": trainable_param_count
    }

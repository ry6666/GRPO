"""
SFT 训练器模块
=====================================================================
提供完整的 SFT（有监督微调）训练器，支持 LoRA 微调。

功能特点：
1. 完整的训练循环（train_epoch）
2. 评估功能（evaluate）
3. 模型保存（save_model）
4. 学习率调度
5. 梯度裁剪
6. 日志记录
7. 早停支持

主要类：
- SFTTrainer: SFT 训练器主类

使用示例：
    from src.sft import SFTTrainer, create_lora_config
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config
    )
    trainer.train(train_dataset, eval_dataset)
=====================================================================
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Union, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq
)
from peft import PeftModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SFTTrainer:
    """
    SFT（有监督微调）训练器
    
    整合所有组件，执行完整的 SFT 训练流程。
    
    Attributes:
        model: 语言模型
        tokenizer: 分词器
        device: 训练设备
        config: 训练配置
        optimizer: 优化器
        scheduler: 学习率调度器
        global_step: 全局步数
        current_epoch: 当前 epoch
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: Optional[Dict] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        初始化训练器
        
        Args:
            model: 语言模型
            tokenizer: 分词器
            config: 训练配置字典
            device: 训练设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or self._get_default_device()
        
        self.config = self._default_config()
        if config:
            self.config.update(config)
        
        self.model.to(self.device)
        
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
    
    def _get_default_device(self) -> torch.device:
        """
        获取默认设备
        
        Returns:
            torch.device
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _default_config(self) -> Dict:
        """
        获取默认配置
        
        Returns:
            默认配置字典
        """
        return {
            'learning_rate': 5e-5,
            'max_grad_norm': 1.0,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'gradient_accumulation_steps': 1,
            'logging_steps': 10,
            'save_steps': 500,
            'eval_steps': 500,
            'output_dir': './outputs/sft',
            'save_best': True,
            'early_stopping_patience': None,
            'seed': 42
        }
    
    def setup_optimizer(
        self,
        num_training_steps: int,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None
    ) -> Optimizer:
        """
        设置优化器
        
        Args:
            num_training_steps: 总训练步数
            learning_rate: 学习率（覆盖配置）
            weight_decay: 权重衰减（覆盖配置）
            
        Returns:
            优化器实例
        """
        lr = learning_rate or self.config['learning_rate']
        wd = weight_decay or self.config['weight_decay']
        
        no_decay = ['bias', 'LayerNorm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': wd,
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr
        )
        
        return self.optimizer
    
    def setup_scheduler(
        self,
        num_training_steps: int,
        warmup_ratio: Optional[float] = None
    ) -> LRScheduler:
        """
        设置学习率调度器
        
        Args:
            num_training_steps: 总训练步数
            warmup_ratio: 预热比例（覆盖配置）
            
        Returns:
            调度器实例
        """
        warmup_ratio = warmup_ratio or self.config['warmup_ratio']
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return self.scheduler
    
    def _create_data_collator(self) -> DataCollatorForSeq2Seq:
        """
        创建数据收集器
        
        Returns:
            DataCollatorForSeq2Seq 实例
        """
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
    
    def _create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否打乱
            
        Returns:
            DataLoader 实例
        """
        collator = self._create_data_collator()
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            pin_memory=True
        )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        gradient_accumulation_steps: Optional[int] = None
    ) -> Dict:
        """
        训练一个 epoch
        
        Args:
            dataloader: 训练数据加载器
            gradient_accumulation_steps: 梯度累积步数
            
        Returns:
            训练指标字典
        """
        gacc = gradient_accumulation_steps or self.config['gradient_accumulation_steps']
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        if self.optimizer is None:
            raise ValueError("Optimizer not set up. Call setup_optimizer() first.")
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / gacc
            loss.backward()
            
            total_loss += loss.item() * gacc
            num_batches += 1
            
            self.global_step += 1
            
            if (step + 1) % gacc == 0:
                if self.config['max_grad_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['max_grad_norm']
                    )
                
                self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
            
            current_loss = loss.item() * gacc
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}' if self.scheduler else f'{self.config["learning_rate"]:.2e}'
            })
            
            if self.global_step % self.config['logging_steps'] == 0:
                logger.info(f"Step {self.global_step}: loss = {current_loss:.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            'loss': avg_loss,
            'num_samples': len(dataloader.dataset),
            'num_batches': num_batches
        }
    
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict:
        """
        评估模型
        
        Args:
            dataloader: 评估数据加载器
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        logger.info(f"评估损失: {avg_loss:.4f}")
        
        return {
            'loss': avg_loss,
            'num_samples': len(dataloader.dataset),
            'num_batches': num_batches
        }
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        epochs: int = 3,
        batch_size: int = 2,
        learning_rate: Optional[float] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        执行完整训练流程
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 评估数据集（可选）
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            output_dir: 输出目录
            
        Returns:
            训练指标字典
        """
        output_dir = output_dir or self.config['output_dir']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        train_loader = self._create_dataloader(
            train_dataset,
            batch_size,
            shuffle=True
        )
        
        num_training_steps = len(train_loader) * epochs // self.config['gradient_accumulation_steps']
        
        self.setup_optimizer(num_training_steps, learning_rate)
        self.setup_scheduler(num_training_steps)
        
        logger.info("=" * 60)
        logger.info("SFT 训练开始")
        logger.info(f"训练样本: {len(train_dataset)}")
        logger.info(f"评估样本: {len(eval_dataset) if eval_dataset else 0}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"学习率: {learning_rate or self.config['learning_rate']}")
        logger.info(f"总步数: {num_training_steps}")
        logger.info("=" * 60)
        
        history = {
            'train_losses': [],
            'eval_losses': [],
            'best_loss': float('inf')
        }
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(f"{'='*60}")
            
            train_metrics = self.train_epoch(train_loader)
            history['train_losses'].append(train_metrics['loss'])
            
            logger.info(f"训练损失: {train_metrics['loss']:.4f}")
            
            if eval_dataset is not None:
                eval_loader = self._create_dataloader(
                    eval_dataset,
                    batch_size,
                    shuffle=False
                )
                eval_metrics = self.evaluate(eval_loader)
                history['eval_losses'].append(eval_metrics['loss'])
                
                if eval_metrics['loss'] < history['best_loss']:
                    history['best_loss'] = eval_metrics['loss']
                    
                    if self.config['save_best']:
                        self._save_best_model(output_dir)
            
            if (epoch + 1) % 2 == 0:
                self._save_checkpoint(output_dir, epoch + 1)
        
        self._save_final_model(output_dir)
        
        logger.info("\n训练完成！")
        logger.info(f"最佳评估损失: {history['best_loss']:.4f}")
        logger.info(f"模型已保存到: {output_dir}")
        
        return history
    
    def _save_best_model(self, output_dir: str):
        """
        保存最佳模型
        
        Args:
            output_dir: 输出目录
        """
        save_path = Path(output_dir) / "best_model"
        self._save_model(save_path)
        logger.info(f"保存最佳模型到: {save_path}")
    
    def _save_checkpoint(self, output_dir: str, epoch: int):
        """
        保存检查点
        
        Args:
            output_dir: 输出目录
            epoch: 当前 epoch
        """
        save_path = Path(output_dir) / f"checkpoint_epoch_{epoch}"
        self._save_model(save_path)
        logger.info(f"保存检查点到: {save_path}")
    
    def _save_final_model(self, output_dir: str):
        """
        保存最终模型
        
        Args:
            output_dir: 输出目录
        """
        save_path = Path(output_dir) / "final_model"
        self._save_model(save_path)
        logger.info(f"保存最终模型到: {save_path}")
    
    def _save_model(self, save_path: Path):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        save_path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)
        
        self.tokenizer.save_pretrained(save_path)
    
    def predict(
        self,
        query: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        使用训练好的模型进行预测
        
        Args:
            query: 输入问题
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            do_sample: 是否采样
            
        Returns:
            生成的答案
        """
        self.model.eval()
        
        conversation = [
            {"role": "user", "content": query}
        ]
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            prompt = f"User: {query}\nAssistant:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response


class SFTTrainerBuilder:
    """
    SFT 训练器构建器
    
    提供便捷的训练器构建方法。
    
    Attributes:
        model: 语言模型
        tokenizer: 分词器
        config: 训练配置
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer
    ):
        """
        初始化构建器
        
        Args:
            model: 语言模型
            tokenizer: 分词器
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = {}
    
    def with_config(self, config: Dict) -> 'SFTTrainerBuilder':
        """
        设置配置
        
        Args:
            config: 配置字典
            
        Returns:
            自身（支持链式调用）
        """
        self.config.update(config)
        return self
    
    def with_learning_rate(self, lr: float) -> 'SFTTrainerBuilder':
        """
        设置学习率
        
        Args:
            lr: 学习率
            
        Returns:
            自身
        """
        self.config['learning_rate'] = lr
        return self
    
    def with_output_dir(self, output_dir: str) -> 'SFTTrainerBuilder':
        """
        设置输出目录
        
        Args:
            output_dir: 输出目录
            
        Returns:
            自身
        """
        self.config['output_dir'] = output_dir
        return self
    
    def with_device(self, device: str) -> 'SFTTrainerBuilder':
        """
        设置设备
        
        Args:
            device: 设备字符串
            
        Returns:
            自身
        """
        self.config['device'] = device
        return self
    
    def build(self, device: Optional[torch.device] = None) -> SFTTrainer:
        """
        构建训练器
        
        Args:
            device: 设备
            
        Returns:
            SFTTrainer 实例
        """
        return SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            device=device
        )

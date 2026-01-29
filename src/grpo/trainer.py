"""
GRPO训练器模块
=====================================================================
该模块提供了完整的GRPO训练流程实现。
训练器负责协调模型、数据、奖励函数和优化器，
执行完整的训练循环。

主要类：
- GRPODataset: 训练数据集类
- GRPOTrainer: GRPO训练器主类

核心功能：
1. 训练循环 (train_epoch)
2. 策略更新 (_update_policy)
3. 模型保存 (save_model)
4. 模型评估 (evaluate)

设计特点：
- 支持LoRA微调
- 支持学习率调度
- 支持梯度裁剪
- 支持日志记录
- 支持断点续训
=====================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import logging
import os
from datetime import datetime

from .core import GRPOCore, Trajectory, TrajectoryBuffer
from .reward import BaseRewardFunction, create_reward_function
from .generator import TrajectoryGenerator


class GRPODataset(Dataset):
    """
    GRPO数据集
    
    简单封装训练数据，支持PyTorch DataLoader使用。
    
    Attributes:
        queries: 问题列表
        ground_truths: 真实答案列表
    """
    
    def __init__(self, queries: List[str], ground_truths: List[str] = None):
        """
        初始化数据集
        
        Args:
            queries: 问题列表
            ground_truths: 真实答案列表，None则使用占位符
        """
        self.queries = queries
        self.ground_truths = ground_truths or [None] * len(queries)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.queries)
    
    def __getitem__(self, idx):
        """获取单个数据项"""
        return {
            'query': self.queries[idx],
            'ground_truth': self.ground_truths[idx]
        }


class GRPOTrainer:
    """
    GRPO训练器
    
    整合所有组件，执行完整的GRPO训练流程。
    
    Attributes:
        model: 语言模型
        tokenizer: 分词器
        grpo_config: GRPO配置
        training_config: 训练配置
        reward_function: 奖励函数
        grpo_core: GRPO核心算法实例
        trajectory_buffer: 轨迹缓冲区
        generator: 轨迹生成器
        optimizer: 优化器
        scheduler: 学习率调度器
        global_step: 全局步数
        logger: 日志记录器
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        grpo_config: dict,
        training_config: dict,
        reward_function: BaseRewardFunction,
        lora_config: dict = None,
        device: str = "auto"
    ):
        """
        初始化训练器
        
        Args:
            model: 预训练语言模型
            tokenizer: 分词器
            grpo_config: GRPO算法配置
            training_config: 训练过程配置
            reward_function: 奖励函数实例
            lora_config: LoRA配置
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.grpo_config = grpo_config
        self.training_config = training_config
        self.reward_function = reward_function
        self.device = device
        
        # 初始化GRPO核心
        self.grpo_core = GRPOCore(
            group_size=grpo_config.get('group_size', 3),
            clip_epsilon=grpo_config.get('clip_epsilon', 0.1),
            kl_coeff=grpo_config.get('kl_coeff', 0.05),
            beta=grpo_config.get('beta', 0.01),
            normalize_reward=grpo_config.get('normalize_reward', True),
            reward_scale=grpo_config.get('reward_scale', 0.1)
        )
        
        # 初始化轨迹缓冲区
        self.trajectory_buffer = TrajectoryBuffer(
            max_size=training_config.get('buffer_size', 1000)
        )
        
        # 初始化轨迹生成器
        self.generator = TrajectoryGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=training_config.get('max_new_tokens', 256),
            temperature=training_config.get('temperature', 0.7),
            top_p=training_config.get('top_p', 0.9),
            exploration_rate=training_config.get('exploration_rate', 0.1),
            device=device
        )
        
        # 优化器和调度器（延迟初始化）
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """
        设置日志记录
        
        配置日志格式和处理器，支持控制台和文件输出。
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    os.path.join(
                        self.training_config.get('logging_dir', './logs'),
                        f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                    )
                )
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_optimizer(self, num_training_steps: int):
        """
        准备优化器和学习率调度器
        
        Args:
            num_training_steps: 总训练步数
        """
        # 定义不应用权重衰减的参数
        no_decay = ['bias', 'LayerNorm.weight']
        
        # 分组优化参数
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': self.training_config.get('weight_decay', 0.01)
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0
            }
        ]
        
        # 创建AdamW优化器
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.grpo_config.get('learning_rate', 5e-5),
            weight_decay=self.training_config.get('weight_decay', 0.01)
        )
        
        # 计算预热步数
        warmup_steps = int(num_training_steps * self.training_config.get('warmup_ratio', 0.1))
        
        # 创建学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        prompt_template: str = None
    ) -> Dict:
        """
        训练一个epoch
        
        执行完整的训练流程：
        1. 生成轨迹
        2. 计算奖励
        3. 更新策略
        4. 更新学习率
        
        Args:
            dataloader: 数据加载器
            prompt_template: 提示模板
            
        Returns:
            训练指标字典
        """
        self.model.train()
        
        epoch_losses = []
        epoch_metrics = {
            'policy_loss': [],
            'kl_loss': [],
            'total_loss': [],
            'mean_reward': [],
            'mean_advantage': []
        }
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            queries = batch['query']
            ground_truths = batch['ground_truth']
            
            group_size = self.grpo_config.get('group_size', 3)
            
            # 存储所有轨迹和优势
            all_trajectories = []
            group_advantages = []
            
            # 为每个查询生成轨迹
            for query, ground_truth in zip(queries, ground_truths):
                trajectories = self.generator.generate_group(
                    query=query,
                    ground_truth=ground_truth,
                    group_size=group_size,
                    prompt_template=prompt_template
                )
                
                # 计算每个轨迹的奖励
                for traj in trajectories:
                    final_answer = self._extract_final_answer(traj)
                    
                    reward = self.reward_function.compute_reward(
                        query=query,
                        states=traj.states,
                        actions=traj.actions,
                        final_answer=final_answer,
                        ground_truth=ground_truth
                    )
                    
                    traj.total_reward = reward
                    traj.is_correct = (ground_truth is not None and 
                                      reward > self.grpo_config.get('path_length_penalty', 0.1))
                
                all_trajectories.extend(trajectories)
                
                # 计算组内优势
                rewards = [traj.total_reward for traj in trajectories]
                advantages = self.grpo_core.compute_group_relative_advantages(rewards)
                group_advantages.extend(advantages)
            
            # 添加到缓冲区
            self.trajectory_buffer.add_group(all_trajectories)
            
            # 更新策略
            loss_info = self._update_policy(all_trajectories, group_advantages)
            
            epoch_losses.append(loss_info['total_loss'])
            
            # 收集指标
            for key in epoch_metrics:
                if key in loss_info:
                    epoch_metrics[key].append(loss_info[key])
            
            rewards = [traj.total_reward for traj in all_trajectories]
            epoch_metrics['mean_reward'].append(sum(rewards) / len(rewards))
            
            # 更新优化器和调度器
            if self.optimizer is not None:
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            self.global_step += 1
            
            # 记录日志
            if self.global_step % self.training_config.get('logging_steps', 10) == 0:
                self.logger.info(f"Step {self.global_step}: Loss = {loss_info['total_loss']:.4f}")
        
        # 计算平均指标
        avg_metrics = {
            'mean_loss': sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0,
            'mean_policy_loss': sum(epoch_metrics['policy_loss']) / len(epoch_metrics['policy_loss']) if epoch_metrics['policy_loss'] else 0.0,
            'mean_kl_loss': sum(epoch_metrics['kl_loss']) / len(epoch_metrics['kl_loss']) if epoch_metrics['kl_loss'] else 0.0,
            'mean_reward': sum(epoch_metrics['mean_reward']) / len(epoch_metrics['mean_reward']) if epoch_metrics['mean_reward'] else 0.0
        }
        
        return avg_metrics
    
    def _update_policy(
        self,
        trajectories: List[Trajectory],
        advantages: List[float]
    ) -> Dict:
        """
        更新策略
        
        执行GRPO策略更新，包括：
        1. 前向传播获取logits
        2. 计算新策略的log probabilities
        3. 计算损失
        4. 反向传播
        
        Args:
            trajectories: 轨迹列表
            advantages: 优势列表
            
        Returns:
            更新信息字典
        """
        self.model.train()
        
        queries = [traj.query for traj in trajectories]
        
        # 分词
        inputs = self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 开启梯度计算
        with torch.set_grad_enabled(True):
            outputs = self.model(**inputs, use_cache=False)
            
            logits = outputs.logits
            
            # 计算log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # 保存旧策略的log probabilities
            old_log_probs = log_probs.detach()
            
            # 准备action mask
            shift_labels = inputs.get('labels', inputs['input_ids'])
            shift_labels = shift_labels[:, 1:]
            shift_log_probs = log_probs[:, :-1, :].reshape(-1, log_probs.size(-1))
            
            action_masks = shift_labels != -100
            selected_log_probs = shift_log_probs[action_masks].reshape(len(trajectories), -1)
            
            # Padding if necessary
            if selected_log_probs.size(1) > 0:
                selected_log_probs = selected_log_probs[:, :min(selected_log_probs.size(1), len(advantages))]
                if selected_log_probs.size(1) < len(advantages):
                    pad_size = len(advantages) - selected_log_probs.size(1)
                    selected_log_probs = torch.cat([
                        selected_log_probs,
                        torch.zeros(selected_log_probs.size(0), pad_size).to(self.device)
                    ], dim=1)
            
            new_log_probs = selected_log_probs
            
            # 转换为张量
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
            
            # 确保维度正确
            if len(new_log_probs.shape) == 1:
                new_log_probs = new_log_probs.unsqueeze(0)
            
            # 计算概率比
            ratios = torch.exp(new_log_probs - old_log_probs[:, :new_log_probs.size(1)].detach())
            
            clip_epsilon = self.grpo_config.get('clip_epsilon', 0.1)
            
            # 计算裁剪损失
            unclipped = ratios * advantages_tensor.unsqueeze(1)
            clipped = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_tensor.unsqueeze(1)
            
            policy_loss = -torch.min(unclipped, clipped).mean()
            
            # 计算KL散度损失
            if self.grpo_config.get('kl_coeff', 0.05) > 0:
                kl_div = torch.kl_div(
                    torch.log_softmax(old_log_probs[:, :new_log_probs.size(1)].detach(), dim=-1),
                    torch.softmax(new_log_probs, dim=-1),
                    reduction='batchmean'
                )
                kl_loss = self.grpo_config.get('kl_coeff', 0.05) * kl_div
                total_loss = policy_loss + kl_loss
            else:
                total_loss = policy_loss
                kl_loss = torch.tensor(0.0)
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            if self.grpo_config.get('max_grad_norm', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grpo_config.get('max_grad_norm', 1.0)
                )
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            'mean_ratio': ratios.mean().item()
        }
    
    def _extract_final_answer(self, trajectory: Trajectory) -> str:
        """
        从轨迹中提取最终答案
        
        Args:
            trajectory: 轨迹对象
            
        Returns:
            最终答案文本
        """
        if trajectory.states:
            return trajectory.states[-1] if trajectory.states else ""
        if trajectory.actions:
            return trajectory.actions[-1] if trajectory.actions else ""
        return ""
    
    def save_model(self, output_dir: str):
        """
        保存模型和分词器
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"模型已保存到 {output_dir}")
    
    def evaluate(
        self,
        dataloader: DataLoader,
        prompt_template: str = None
    ) -> Dict:
        """
        评估模型
        
        在验证集上评估模型性能。
        
        Args:
            dataloader: 验证数据加载器
            prompt_template: 提示模板
            
        Returns:
            评估结果字典
        """
        self.model.eval()
        
        total_correct = 0
        total_samples = 0
        total_reward = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                queries = batch['query']
                ground_truths = batch['ground_truth']
                
                for query, ground_truth in zip(queries, ground_truths):
                    trajectories = self.generator.generate_trajectory(
                        query=query,
                        prompt_template=prompt_template,
                        num_candidates=1
                    )
                    
                    for traj in trajectories:
                        final_answer = self._extract_final_answer(traj)
                        
                        reward = self.reward_function.compute_reward(
                            query=query,
                            states=traj.states,
                            actions=traj.actions,
                            final_answer=final_answer,
                            ground_truth=ground_truth
                        )
                        
                        traj.total_reward = reward
                        total_reward += reward
                        
                        if ground_truth is not None:
                            if reward > 0:
                                total_correct += 1
                            total_samples += 1
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_reward = total_reward / (total_samples or 1)
        
        return {
            'accuracy': accuracy,
            'average_reward': avg_reward,
            'total_correct': total_correct,
            'total_samples': total_samples
        }


def create_lora_config(lora_config: dict) -> LoraConfig:
    """
    创建LoRA配置
    
    Args:
        lora_config: LoRA配置字典
        
    Returns:
        LoraConfig对象
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get('lora_r', 16),
        lora_alpha=lora_config.get('lora_alpha', 32),
        lora_dropout=lora_config.get('lora_dropout', 0.05),
        target_modules=lora_config.get('lora_target_modules', 
                                      ["q_proj", "k_proj", "v_proj", "o_proj"])
    )


def apply_lora(model: nn.Module, lora_config: dict) -> nn.Module:
    """
    应用LoRA到模型
    
    Args:
        model: 基础模型
        lora_config: LoRA配置
        
    Returns:
        应用LoRA后的模型
    """
    peft_config = create_lora_config(lora_config)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

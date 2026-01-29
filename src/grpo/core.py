"""
GRPO核心算法模块
=====================================================================
该模块实现了Group Relative Policy Optimization (GRPO) 算法的核心功能。
GRPO是一种强化学习优化算法，通过组内相对优势来更新策略，
避免了传统PPO中需要值函数的问题。

主要类：
- Trajectory: 轨迹数据类，存储单次推理过程的所有信息
- GRPOCore: GRPO算法核心实现
- TrajectoryBuffer: 轨迹缓冲区，用于管理历史轨迹

核心算法：
1. 组内相对优势计算 (compute_group_relative_advantages)
2. PPO概率比计算 (compute_ppo_ratio)
3. 裁剪损失计算 (compute_clipped_loss)
4. KL散度计算 (compute_kl_divergence)
5. GRPO总损失计算 (compute_grpo_loss)
=====================================================================
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Trajectory:
    """
    轨迹数据类
    
    用于存储单次推理过程的完整信息，包括查询、状态序列、
    动作序列、奖励等。
    
    Attributes:
        query: 输入的问题查询
        states: 推理过程中的状态序列
        actions: 推理过程中的动作序列
        rewards: 每一步的奖励列表
        total_reward: 总奖励
        is_correct: 答案是否正确
        path_length: 轨迹长度（状态数量）
        
    Example:
        >>> traj = Trajectory(
        ...     query="Who is the father of Arthur?",
        ...     states=["Arthur's father is Christopher"],
        ...     actions=["reasoning"],
        ...     rewards=[0.8],
        ...     total_reward=0.8,
        ...     is_correct=True
        ... )
    """
    query: str
    states: List[str]
    actions: List[str]
    rewards: List[float]
    total_reward: float
    is_correct: bool = False
    path_length: int = 0
    
    def __post_init__(self):
        """初始化后处理，计算轨迹长度"""
        self.path_length = len(self.states)


class GRPOCore:
    """
    GRPO算法核心实现类
    
    该类实现了GRPO算法的主要计算逻辑，包括：
    - 组内相对优势计算：将绝对奖励转换为相对优势
    - PPO概率比计算：新旧策略的概率比
    - 裁剪损失计算：限制策略更新幅度
    - KL散度计算：防止策略过度偏离
    
    GRPO的核心思想是对每个问题生成一组回答，然后计算
    组内的相对优势来更新策略，而不是使用全局奖励。
    
    Attributes:
        group_size: 每组轨迹数量
        clip_epsilon: PPO裁剪系数
        kl_coeff: KL散度系数
        beta: 奖励归一化参数
        normalize_reward: 是否标准化奖励
        reward_scale: 奖励缩放因子
    """
    
    def __init__(
        self,
        group_size: int = 3,
        clip_epsilon: float = 0.1,
        kl_coeff: float = 0.05,
        beta: float = 0.01,
        normalize_reward: bool = True,
        reward_scale: float = 0.1
    ):
        """
        初始化GRPO核心
        
        Args:
            group_size: 每组轨迹数量
            clip_epsilon: PPO裁剪系数，控制策略更新幅度
            kl_coeff: KL散度系数，用于正则化
            beta: 奖励归一化参数
            normalize_reward: 是否对奖励进行标准化
            reward_scale: 奖励缩放因子
        """
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.kl_coeff = kl_coeff
        self.beta = beta
        self.normalize_reward = normalize_reward
        self.reward_scale = reward_scale
    
    def compute_group_relative_advantages(
        self,
        rewards: List[float]
    ) -> List[float]:
        """
        计算组内相对优势
        
        这是GRPO的核心操作之一。它将组内每个轨迹的绝对奖励
        转换为相对于组平均值的相对优势。
        
        计算公式：
        advantage_i = (reward_i - mean(rewards)) / std(rewards)
        
        Args:
            rewards: 组内所有轨迹的绝对奖励列表
            
        Returns:
            相对优势列表，长度与输入相同
            
        Raises:
            ValueError: 如果奖励数量不等于组大小
            
        Example:
            >>> grpo = GRPOCore(group_size=3)
            >>> rewards = [0.8, 0.5, 0.2]
            >>> advantages = grpo.compute_group_relative_advantages(rewards)
            >>> # advantages 大约等于 [1.2, 0.0, -1.2]
        """
        if len(rewards) != self.group_size:
            raise ValueError(f"奖励数量 {len(rewards)} 不等于组大小 {self.group_size}")
        
        # 转换为PyTorch张量
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_reward = rewards_tensor.mean()
        
        if self.normalize_reward:
            std_reward = rewards_tensor.std()
            # 避免除零
            if std_reward < 1e-8:
                std_reward = 1.0
            advantages = (rewards_tensor - mean_reward) / std_reward
        else:
            advantages = rewards_tensor - mean_reward
        
        # 缩放优势
        advantages = advantages * self.reward_scale
        
        return advantages.tolist()
    
    def compute_ppo_ratio(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        计算PPO概率比
        
        计算新旧策略在相同动作上的对数概率比值，
        用于评估策略变化的程度。
        
        公式：ratio = exp(new_log_prob - old_log_prob)
        
        Args:
            old_log_probs: 更新前的对数概率
            new_log_probs: 更新后的对数概率
            
        Returns:
            概率比张量
            
        Example:
            >>> old_probs = torch.tensor([0.1, 0.9])
            >>> new_probs = torch.tensor([0.2, 0.8])
            >>> ratio = grpo.compute_ppo_ratio(old_probs.log(), new_probs.log())
        """
        return torch.exp(new_log_probs - old_log_probs)
    
    def compute_clipped_loss(
        self,
        ratios: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float = None
    ) -> torch.Tensor:
        """
        计算裁剪损失
        
        这是PPO/GRPO中防止策略过度更新的关键机制。
        当概率比偏离1太远时，损失会被裁剪。
        
        裁剪逻辑：
        - 如果 ratio * advantage > 0（正确方向），裁剪上限
        - 如果 ratio * advantage < 0（错误方向），裁剪下限
        
        Args:
            ratios: 概率比
            advantages: 优势值
            clip_epsilon: 裁剪系数，默认为实例的clip_epsilon
            
        Returns:
            裁剪后的损失张量
            
        Formula:
            loss = min(ratio * advantage, clamp(ratio) * advantage)
        """
        if clip_epsilon is None:
            clip_epsilon = self.clip_epsilon
        
        # 未裁剪的损失
        unclipped_loss = ratios * advantages
        
        # 裁剪后的损失
        clipped_ratios = torch.clamp(
            ratios,
            1.0 - clip_epsilon,
            1.0 + clip_epsilon
        )
        clipped_loss = clipped_ratios * advantages
        
        # 取两者的较小值
        return torch.min(unclipped_loss, clipped_loss)
    
    def compute_kl_divergence(
        self,
        old_probs: torch.Tensor,
        new_probs: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        计算KL散度
        
        计算两个概率分布之间的KL散度，用于衡量策略变化程度。
        KL散度作为正则项加入损失，防止策略更新过快。
        
        KL(P||Q) = sum(P[i] * log(P[i] / Q[i]))
        
        Args:
            old_probs: 更新前的概率分布
            new_probs: 更新后的概率分布
            eps: 防止log(0)的最小值
            
        Returns:
            KL散度张量
            
        Note:
            KL散度是不对称的，这里计算的是 old || new
        """
        # 避免log(0)
        old_probs = old_probs.clamp(min=eps)
        new_probs = new_probs.clamp(min=eps)
        
        return (old_probs * torch.log(old_probs / new_probs)).sum(dim=-1)
    
    def compute_grpo_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        old_probs: Optional[torch.Tensor] = None,
        new_probs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算GRPO总损失
        
        整合策略损失和KL散度损失，计算最终的GRPO损失。
        
        损失组成：
        1. Policy Loss: 基于概率比和优势的裁剪损失
        2. KL Loss: KL散度正则项
        
        Args:
            old_log_probs: 更新前的对数概率
            new_log_probs: 更新后的对数概率
            advantages: 相对优势列表
            old_probs: 更新前的概率分布（可选，用于KL计算）
            new_probs: 更新后的概率分布（可选，用于KL计算）
            
        Returns:
            tuple: (总损失, 损失信息字典)
            
        Example:
            >>> loss, info = grpo.compute_grpo_loss(
            ...     old_log_probs, new_log_probs, advantages
            ... )
            >>> info['total_loss']
            0.5
        """
        # 转换为张量
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # 计算概率比
        ratios = self.compute_ppo_ratio(old_log_probs, new_log_probs)
        
        # 计算策略损失（负号因为是梯度上升）
        policy_loss = -self.compute_clipped_loss(ratios, advantages).mean()
        
        # 收集损失信息
        loss_info = {
            "policy_loss": policy_loss.item(),
            "mean_ratio": ratios.mean().item(),
            "std_ratio": ratios.std().item()
        }
        
        # 如果提供了概率分布，计算KL散度损失
        if old_probs is not None and new_probs is not None:
            kl_div = self.compute_kl_divergence(old_probs, new_probs).mean()
            kl_loss = self.kl_coeff * kl_div
            
            total_loss = policy_loss + kl_loss
            
            loss_info["kl_div"] = kl_div.item()
            loss_info["kl_loss"] = kl_loss.item()
            loss_info["total_loss"] = total_loss.item()
        else:
            total_loss = policy_loss
            loss_info["total_loss"] = total_loss.item()
        
        return total_loss, loss_info
    
    def process_group(
        self,
        trajectories: List[Trajectory]
    ) -> Dict:
        """
        处理单个组的轨迹
        
        对一组轨迹进行完整处理，包括计算奖励和优势。
        
        Args:
            trajectories: 组内的轨迹列表
            
        Returns:
            处理结果字典，包含：
            - trajectories: 原始轨迹
            - rewards: 奖励列表
            - advantages: 优势列表
            - mean_reward: 平均奖励
            - std_reward: 奖励标准差
            
        Raises:
            ValueError: 如果轨迹数量不等于组大小
        """
        if len(trajectories) != self.group_size:
            raise ValueError(f"轨迹数量 {len(trajectories)} 不等于组大小 {self.group_size}")
        
        # 提取奖励
        rewards = [traj.total_reward for traj in trajectories]
        
        # 计算优势
        advantages = self.compute_group_relative_advantages(rewards)
        
        return {
            "trajectories": trajectories,
            "rewards": rewards,
            "advantages": advantages,
            "mean_reward": sum(rewards) / len(rewards),
            "std_reward": (sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)) ** 0.5
        }


class TrajectoryBuffer:
    """
    轨迹缓冲区
    
    用于管理和存储历史轨迹的缓冲区，支持添加、获取和清空操作。
    
    Attributes:
        max_size: 缓冲区最大容量
        trajectories: 存储的轨迹列表
        
    Example:
        >>> buffer = TrajectoryBuffer(max_size=1000)
        >>> buffer.add(trajectory)
        >>> trajectories = buffer[:10]  # 获取前10个轨迹
    """
    
    def __init__(self, max_size: int = 1000):
        """
        初始化轨迹缓冲区
        
        Args:
            max_size: 缓冲区最大容量，超出时自动移除最旧的轨迹
        """
        self.max_size = max_size
        self.trajectories: List[Trajectory] = []
    
    def add(self, trajectory: Trajectory):
        """
        添加单个轨迹
        
        如果缓冲区已满，移除最旧的轨迹。
        
        Args:
            trajectory: 要添加的轨迹
        """
        if len(self.trajectories) >= self.max_size:
            self.trajectories.pop(0)
        self.trajectories.append(trajectory)
    
    def add_group(self, trajectories: List[Trajectory]):
        """
        添加一组轨迹
        
        Args:
            trajectories: 轨迹列表
        """
        for traj in trajectories:
            self.add(traj)
    
    def clear(self):
        """清空缓冲区"""
        self.trajectories.clear()
    
    def __len__(self):
        """返回缓冲区中的轨迹数量"""
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """通过索引获取轨迹"""
        return self.trajectories[idx]

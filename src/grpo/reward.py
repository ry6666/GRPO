"""
奖励函数模块
=====================================================================
该模块提供了多种奖励函数实现，用于评估模型生成轨迹的质量。
奖励函数是强化学习中的核心组件，决定了模型优化方向。

主要类：
- BaseRewardFunction: 奖励函数抽象基类
- MultiHopRewardFunction: 多跳推理奖励函数
- KinshipRewardFunction: 亲属关系任务专用奖励函数
- QARewardFunction: 问答任务奖励函数
- CompositeRewardFunction: 复合奖励函数

设计特点：
- 支持答案正确性判断
- 支持路径长度惩罚
- 支持格式检查
- 支持多种评估指标（EM, F1）
- 支持奖励函数组合
=====================================================================
"""

from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import re


class BaseRewardFunction(ABC):
    """
    奖励函数抽象基类
    
    定义了奖励函数的通用接口，所有具体奖励函数都需要
    实现compute_reward方法。
    
    Attributes:
        无公共属性
        
    Note:
        奖励值通常设计为：
        - 正值：表示好的行为（如正确答案）
        - 负值：表示需要避免的行为（如错误答案）
    """
    
    @abstractmethod
    def compute_reward(
        self,
        query: str,
        states: List[str],
        actions: List[str],
        final_answer: str,
        ground_truth: Optional[str] = None
    ) -> float:
        """
        计算奖励
        
        Args:
            query: 查询问题
            states: 状态序列
            actions: 动作序列
            final_answer: 最终答案
            ground_truth: 真实答案
            
        Returns:
            奖励值
            
        Returns:
            奖励值浮点数
        """
        pass


class MultiHopRewardFunction(BaseRewardFunction):
    """
    多跳推理奖励函数
    
    专为多跳推理任务设计的奖励函数，综合考虑：
    1. 答案正确性
    2. 路径长度（越短越好）
    
    奖励设计：
    - 正确答案：correct_answer_bonus - path_length_penalty * path_length
    - 错误答案：wrong_answer_penalty - path_length_penalty * path_length
    
    Attributes:
        path_length_penalty: 每步路径的惩罚系数
        correct_answer_bonus: 正确答案的奖励
        wrong_answer_penalty: 错误答案的惩罚
        max_path_length: 最大路径长度限制
    """
    
    def __init__(
        self,
        path_length_penalty: float = 0.1,
        correct_answer_bonus: float = 1.0,
        wrong_answer_penalty: float = 0.0,
        max_path_length: int = 10
    ):
        """
        初始化多跳奖励函数
        
        Args:
            path_length_penalty: 路径长度惩罚系数
            correct_answer_bonus: 正确答案奖励
            wrong_answer_penalty: 错误答案惩罚
            max_path_length: 最大路径长度
        """
        self.path_length_penalty = path_length_penalty
        self.correct_answer_bonus = correct_answer_bonus
        self.wrong_answer_penalty = wrong_answer_penalty
        self.max_path_length = max_path_length
    
    def compute_reward(
        self,
        query: str,
        states: List[str],
        actions: List[str],
        final_answer: str,
        ground_truth: Optional[str] = None
    ) -> float:
        """
        计算多跳推理奖励
        
        综合考虑路径长度和答案正确性。
        
        Args:
            query: 查询问题
            states: 推理状态序列
            actions: 动作序列
            final_answer: 最终答案
            ground_truth: 真实答案
            
        Returns:
            奖励值
        """
        path_length = len(states)
        
        # 限制最大路径长度
        if path_length > self.max_path_length:
            path_length = self.max_path_length
        
        # 如果没有真实答案，只返回路径长度惩罚
        if ground_truth is None:
            return -self.path_length_penalty * path_length
        
        # 检查答案是否正确
        is_correct = self._check_answer(final_answer, ground_truth)
        
        if is_correct:
            reward = self.correct_answer_bonus - self.path_length_penalty * path_length
        else:
            reward = self.wrong_answer_penalty - self.path_length_penalty * path_length
        
        return reward
    
    def _check_answer(self, pred: str, ground_truth: str) -> bool:
        """
        检查答案是否正确
        
        支持多种匹配方式：
        1. 精确匹配
        2. 实体集合完全匹配
        3. 预测实体是真实答案的子集
        
        Args:
            pred: 预测答案
            ground_truth: 真实答案
            
        Returns:
            是否正确
        """
        pred = pred.lower().strip()
        ground_truth = ground_truth.lower().strip()
        
        # 1. 精确匹配
        if pred == ground_truth:
            return True
        
        # 2. 提取实体进行比较
        pred_entities = set(re.findall(r'\b\w+\b', pred))
        gt_entities = set(re.findall(r'\b\w+\b', ground_truth))
        
        # 完全匹配
        if gt_entities and pred_entities == gt_entities:
            return True
        
        # 子集匹配
        if gt_entities and pred_entities.issubset(gt_entities):
            return True
        
        return False


class KinshipRewardFunction(MultiHopRewardFunction):
    """
    亲属关系任务专用奖励函数
    
    继承自MultiHopRewardFunction，针对亲属关系推理任务
    进行了专门的优化，使用大写首字母命名实体匹配。
    
    Attributes:
        path_length_penalty: 路径长度惩罚系数
        correct_answer_bonus: 正确答案奖励
        wrong_answer_penalty: 错误答案惩罚
    """
    
    def __init__(
        self,
        path_length_penalty: float = 0.1,
        correct_answer_bonus: float = 1.0,
        wrong_answer_penalty: float = 0.0
    ):
        """
        初始化亲属关系奖励函数
        
        Args:
            path_length_penalty: 路径长度惩罚
            correct_answer_bonus: 正确答案奖励
            wrong_answer_penalty: 错误答案惩罚
        """
        super().__init__(
            path_length_penalty=path_length_penalty,
            correct_answer_bonus=correct_answer_bonus,
            wrong_answer_penalty=wrong_answer_penalty
        )
    
    def compute_reward(
        self,
        query: str,
        states: List[str],
        actions: List[str],
        final_answer: str,
        ground_truth: Optional[str] = None
    ) -> float:
        """
        计算亲属关系任务奖励
        
        Args:
            query: 查询问题
            states: 推理状态序列
            actions: 动作序列
            final_answer: 最终答案
            ground_truth: 真实答案
            
        Returns:
            奖励值
        """
        if ground_truth is None:
            return -self.path_length_penalty * len(states)
        
        is_correct = self._check_kinship_answer(final_answer, ground_truth)
        
        path_length = len(states)
        length_penalty = self.path_length_penalty * path_length
        
        if is_correct:
            reward = self.correct_answer_bonus - length_penalty
        else:
            reward = self.wrong_answer_penalty - length_penalty
        
        return reward
    
    def _check_kinship_answer(self, pred: str, ground_truth: str) -> bool:
        """
        检查亲属关系答案是否正确
        
        使用大写首字母格式匹配实体名称。
        
        Args:
            pred: 预测答案
            ground_truth: 真实答案
            
        Returns:
            是否正确
        """
        pred = pred.lower().strip()
        ground_truth = ground_truth.lower().strip()
        
        # 清理标点符号
        pred_clean = re.sub(r'[^\w\s]', '', pred).strip()
        gt_clean = re.sub(r'[^\w\s]', '', ground_truth).strip()
        
        # 精确匹配
        if pred_clean == gt_clean:
            return True
        
        # 提取大写命名的实体
        pred_entities = set(re.findall(r'\b[A-Z][a-z]+\b', pred))
        gt_entities = set(re.findall(r'\b[A-Z][a-z]+\b', ground_truth))
        
        # 实体集合匹配
        if pred_entities and gt_entities:
            return pred_entities == gt_entities
        
        return False


class QARewardFunction(BaseRewardFunction):
    """
    问答任务奖励函数
    
    综合考虑多种评估指标：
    1. 精确匹配 (Exact Match)
    2. F1分数
    3. 输出格式
    4. 路径长度惩罚
    
    Attributes:
        em_weight: 精确匹配权重
        f1_weight: F1分数权重
        length_penalty: 长度惩罚系数
        format_bonus: 格式正确奖励
    """
    
    def __init__(
        self,
        em_weight: float = 1.0,
        f1_weight: float = 0.5,
        length_penalty: float = 0.01,
        format_bonus: float = 0.1
    ):
        """
        初始化问答奖励函数
        
        Args:
            em_weight: 精确匹配权重
            f1_weight: F1分数权重
            length_penalty: 长度惩罚系数
            format_bonus: 格式奖励
        """
        self.em_weight = em_weight
        self.f1_weight = f1_weight
        self.length_penalty = length_penalty
        self.format_bonus = format_bonus
    
    def compute_reward(
        self,
        query: str,
        states: List[str],
        actions: List[str],
        final_answer: str,
        ground_truth: Optional[str] = None
    ) -> float:
        """
        计算问答奖励
        
        Args:
            query: 查询问题
            states: 状态序列
            actions: 动作序列
            final_answer: 最终答案
            ground_truth: 真实答案
            
        Returns:
            奖励值
        """
        if ground_truth is None:
            return -self.length_penalty * len(states)
        
        # 计算各项分数
        em_score = self._compute_exact_match(final_answer, ground_truth)
        f1_score = self._compute_f1(final_answer, ground_truth)
        format_score = self._check_format(states, actions)
        
        # 综合奖励
        reward = (
            self.em_weight * em_score +
            self.f1_weight * f1_score +
            self.format_bonus * format_score -
            self.length_penalty * len(states)
        )
        
        return reward
    
    def _compute_exact_match(self, pred: str, ground_truth: str) -> float:
        """
        计算精确匹配分数
        
        检查预测答案和真实答案是否完全一致（按词集合）。
        
        Args:
            pred: 预测答案
            ground_truth: 真实答案
            
        Returns:
            精确匹配分数（0或1）
        """
        pred_tokens = set(pred.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        
        if not gt_tokens:
            return 0.0
        
        return 1.0 if pred_tokens == gt_tokens else 0.0
    
    def _compute_f1(self, pred: str, ground_truth: str) -> float:
        """
        计算F1分数
        
        基于词级别的精确率和召回率计算F1分数。
        
        F1 = 2 * P * R / (P + R)
        
        Args:
            pred: 预测答案
            ground_truth: 真实答案
            
        Returns:
            F1分数 [0, 1]
        """
        pred_tokens = pred.lower().split()
        gt_tokens = ground_truth.lower().split()
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        pred_set = set(pred_tokens)
        gt_set = set(gt_tokens)
        
        # 计算重叠
        overlap = len(pred_set & gt_set)
        
        # 精确率
        precision = overlap / len(pred_set) if pred_set else 0.0
        # 召回率
        recall = overlap / len(gt_set) if gt_set else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def _check_format(self, states: List[str], actions: List[str]) -> float:
        """
        检查输出格式
        
        检查状态和动作序列是否合理。
        
        Args:
            states: 状态序列
            actions: 动作序列
            
        Returns:
            格式分数（0或1）
        """
        if not states or not actions:
            return 0.0
        
        if len(states) != len(actions):
            return 0.0
        
        return 1.0


class CompositeRewardFunction(BaseRewardFunction):
    """
    复合奖励函数
    
    将多个奖励函数组合起来，可以同时考虑多种评估维度。
    
    Attributes:
        reward_functions: 奖励函数列表，每个元素为 (名称, 函数, 权重)
        
    Example:
        >>> composite = CompositeRewardFunction([
        ...     ("accuracy", accuracy_reward, 1.0),
        ...     ("format", format_reward, 0.1)
        ... ])
        >>> reward = composite.compute_reward(...)
    """
    
    def __init__(self, reward_functions: List[Tuple[str, BaseRewardFunction, float]]):
        """
        初始化复合奖励函数
        
        Args:
            reward_functions: 奖励函数元组列表 (名称, 函数, 权重)
        """
        self.reward_functions = reward_functions
    
    def compute_reward(
        self,
        query: str,
        states: List[str],
        actions: List[str],
        final_answer: str,
        ground_truth: Optional[str] = None
    ) -> float:
        """
        计算复合奖励
        
        累加所有子奖励函数的加权分数。
        
        Args:
            query: 查询问题
            states: 状态序列
            actions: 动作序列
            final_answer: 最终答案
            ground_truth: 真实答案
            
        Returns:
            总奖励值
        """
        total_reward = 0.0
        
        for name, reward_func, weight in self.reward_functions:
            reward = reward_func.compute_reward(
                query, states, actions, final_answer, ground_truth
            )
            total_reward += weight * reward
        
        return total_reward
    
    def add_reward_function(
        self,
        name: str,
        reward_func: BaseRewardFunction,
        weight: float = 1.0
    ):
        """
        添加奖励函数
        
        Args:
            name: 名称
            reward_func: 奖励函数
            weight: 权重
        """
        self.reward_functions.append((name, reward_func, weight))


def create_reward_function(config: dict) -> BaseRewardFunction:
    """
    根据配置创建奖励函数
    
    工厂函数，根据配置字典创建相应类型的奖励函数。
    
    Args:
        config: 配置字典，包含type和params字段
        
    Returns:
        奖励函数实例
        
    Example:
        >>> config = {"type": "kinship", "params": {"correct_answer_bonus": 2.0}}
        >>> reward_func = create_reward_function(config)
    """
    reward_type = config.get("type", "multi_hop")
    
    if reward_type == "kinship":
        return KinshipRewardFunction(**config.get("params", {}))
    elif reward_type == "qa":
        return QARewardFunction(**config.get("params", {}))
    elif reward_type == "composite":
        sub_functions = []
        for sub_config in config.get("reward_functions", []):
            sub_func = create_reward_function(sub_config)
            sub_functions.append((
                sub_config.get("name", "default"),
                sub_func,
                sub_config.get("weight", 1.0)
            ))
        return CompositeRewardFunction(sub_functions)
    else:
        return MultiHopRewardFunction(**config.get("params", {}))

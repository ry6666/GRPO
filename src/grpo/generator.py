"""
轨迹生成器模块
=====================================================================
该模块提供了从语言模型生成推理轨迹的功能。
轨迹生成器负责调用模型生成回答，并解析模型输出
提取推理过程中的状态和动作。

主要类：
- TrajectoryGenerator: 基础轨迹生成器
- GuidedTrajectoryGenerator: 引导式轨迹生成器

核心功能：
1. 模型响应生成 (generate_response)
2. 轨迹生成 (generate_trajectory)
3. 组轨迹生成 (generate_group)
4. 响应解析 (_parse_response)

设计特点：
- 支持温度调节控制生成随机性
- 支持探索率增加生成多样性
- 支持自定义提示模板
- 智能解析模型输出格式
=====================================================================
"""

import random
import torch
from typing import List, Dict, Optional, Callable
from transformers import AutoTokenizer, AutoModelForCausalLM
from .core import Trajectory


class TrajectoryGenerator:
    """
    轨迹生成器
    
    负责从语言模型生成推理轨迹。支持单轨迹生成和组轨迹生成，
    可以通过温度和探索率控制生成的多样性。
    
    Attributes:
        model: 语言模型
        tokenizer: 分词器
        max_new_tokens: 最大生成token数
        temperature: 生成温度
        top_p: top-p采样参数
        top_k: top-k采样参数
        exploration_rate: 探索率
        device: 设备
        
    Example:
        >>> generator = TrajectoryGenerator(model, tokenizer)
        >>> trajectories = generator.generate_trajectory(
        ...     query="Who is the father of Arthur?"
        ... )
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        exploration_rate: float = 0.1,
        device: str = "auto"
    ):
        """
        初始化轨迹生成器
        
        Args:
            model: 预训练语言模型
            tokenizer: 对应的分词器
            max_new_tokens: 最大生成token数
            temperature: 温度参数，值越高生成越随机
            top_p: nucleus采样参数
            top_k: top-k采样参数
            exploration_rate: 探索率，用于增加生成多样性
            device: 设备，"auto"表示自动选择
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.exploration_rate = exploration_rate
        self.device = device
    
    def set_temperature(self, temperature: float):
        """
        设置温度参数
        
        温度影响生成文本的随机性：
        - 温度低（接近0）：更确定性，输出更保守
        - 温度高（接近1或更高）：更多样化，输出更随机
        
        Args:
            temperature: 新的温度值
        """
        self.temperature = temperature
    
    def set_exploration_rate(self, rate: float):
        """
        设置探索率
        
        探索率用于控制生成多样性的额外增量。
        在生成多个候选轨迹时，后续轨迹会使用更高的温度。
        
        Args:
            rate: 探索率，0表示不使用探索
        """
        self.exploration_rate = rate
    
    def generate_trajectory(
        self,
        query: str,
        prompt_template: Optional[str] = None,
        ground_truth: Optional[str] = None,
        num_candidates: int = 1
    ) -> List[Trajectory]:
        """
        生成单个轨迹
        
        根据查询生成一个或多个推理轨迹。
        
        Args:
            query: 查询问题
            prompt_template: 提示模板
            ground_truth: 真实答案（用于后续评估）
            num_candidates: 候选轨迹数量
            
        Returns:
            轨迹列表
            
        Example:
            >>> trajectories = generator.generate_trajectory(
            ...     query="Who is the father of Arthur?",
            ...     num_candidates=3
            ... )
        """
        trajectories = []
        
        for i in range(num_candidates):
            # 计算当前候选的温度
            if i > 0 and self.exploration_rate > 0:
                # 后续候选使用更高的温度
                temp = self.temperature * (1 + self.exploration_rate * i)
            else:
                temp = self.temperature
            
            # 生成响应
            response = self._generate_response(query, prompt_template, temp)
            
            # 解析响应
            states, actions = self._parse_response(response)
            
            # 创建轨迹
            trajectory = Trajectory(
                query=query,
                states=states,
                actions=actions,
                rewards=[],
                total_reward=0.0,
                is_correct=False,
                path_length=len(states)
            )
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def _generate_response(
        self,
        query: str,
        prompt_template: Optional[str],
        temperature: float
    ) -> str:
        """
        生成模型响应
        
        调用语言模型生成响应文本。
        
        Args:
            query: 查询问题
            prompt_template: 提示模板
            temperature: 生成温度
            
        Returns:
            模型生成的响应文本
        """
        # 构建提示
        if prompt_template is None:
            prompt = f"问题: {query}\n请逐步推理并给出答案:"
        else:
            prompt = prompt_template.format(query=query)
        
        # 分词
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        
        # 移至指定设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # 移除提示部分
        response = response[len(prompt):].strip()
        
        return response
    
    def _parse_response(self, response: str) -> tuple:
        """
        解析模型响应
        
        从模型生成的响应中提取状态和动作序列。
        支持多种格式的解析：
        - 状态/动作编号格式：状态1: xxx
        - 步骤格式：步骤1: xxx
        - 箭头格式：xxx -> yyy
        - 推理格式：推理: xxx
        
        Args:
            response: 模型原始响应
            
        Returns:
            (states, actions) 元组
            
        Example:
            >>> response = "状态1: Arthur的父亲是Christopher\\n最终答案: Christopher"
            >>> states, actions = generator._parse_response(response)
        """
        states = []
        actions = []
        
        lines = response.split('\n')
        
        # 正则表达式模式
        state_pattern = r'^状态?\s*(\d+)[:：]?\s*(.+)$'
        action_pattern = r'^动作?\s*(\d+)[:：]?\s*(.+)$'
        step_pattern = r'^步?骤?\s*(\d+)[:：]?\s*(.+)$'
        reasoning_pattern = r'^推理[:：]?\s*(.+)$'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 匹配不同格式
            state_match = re.match(state_pattern, line, re.IGNORECASE)
            action_match = re.match(action_pattern, line, re.IGNORECASE)
            step_match = re.match(step_pattern, line, re.IGNORECASE)
            
            if state_match:
                states.append(state_match.group(2).strip())
            elif action_match:
                actions.append(action_match.group(2).strip())
            elif step_match:
                content = step_match.group(2).strip()
                # 处理箭头格式
                if "→" in content or "->" in content:
                    parts = re.split(r'\s*→\s*|\s*->\s*', content)
                    if len(parts) >= 2:
                        states.append(parts[0].strip())
                        actions.append(parts[1].strip())
                else:
                    states.append(content)
                    actions.append(content)
            else:
                # 跳过答案标记行
                if "答:" in line or "答案:" in line or "最终答案:" in line:
                    continue
                # 其他行添加到状态或动作
                if len(states) == len(actions):
                    states.append(line)
                else:
                    actions.append(line)
        
        # 如果没有解析出任何内容，将整个响应作为状态
        if not states and not actions:
            states = [response]
        
        return states, actions
    
    def generate_group(
        self,
        query: str,
        ground_truth: Optional[str] = None,
        group_size: int = 3,
        prompt_template: Optional[str] = None
    ) -> List[Trajectory]:
        """
        生成一组轨迹
        
        生成多个轨迹用于组内相对优势计算。
        策略性地生成正确、部分正确和错误的轨迹以促进学习。
        
        Args:
            query: 查询问题
            ground_truth: 真实答案
            group_size: 组大小
            prompt_template: 提示模板
            
        Returns:
            轨迹列表，数量等于group_size
            
        Note:
            生成的轨迹会按类型分类：
            - 1/3 正确轨迹
            - 1/3 部分正确轨迹
            - 1/3 错误轨迹
            
        Example:
            >>> trajectories = generator.generate_group(
            ...     query="Who is the father of Arthur?",
            ...     ground_truth="Christopher",
            ...     group_size=3
            ... )
        """
        trajectories = []
        
        # 计算各类型数量
        num_correct = max(1, group_size // 3)
        num_partial = max(0, group_size // 3)
        num_wrong = group_size - num_correct - num_partial
        
        # 生成正确轨迹
        if num_correct > 0:
            correct_trajs = self.generate_trajectory(
                query, prompt_template, ground_truth, num_correct
            )
            for traj in correct_trajs:
                traj.is_correct = True
            trajectories.extend(correct_trajs)
        
        # 生成部分正确轨迹
        if num_partial > 0:
            partial_trajs = self.generate_trajectory(
                query, prompt_template, ground_truth, num_partial
            )
            trajectories.extend(partial_trajs)
        
        # 生成错误轨迹
        if num_wrong > 0:
            wrong_trajs = self.generate_trajectory(
                query, prompt_template, ground_truth, num_wrong
            )
            trajectories.extend(wrong_trajs)
        
        # 随机打乱顺序
        random.shuffle(trajectories)
        
        return trajectories[:group_size]


class GuidedTrajectoryGenerator(TrajectoryGenerator):
    """
    引导式轨迹生成器
    
    继承自基础生成器，增加了结构化推理引导功能。
    强制模型按照特定的推理步骤生成回答。
    
    Attributes:
        reasoning_template: 推理步骤模板
        answer_template: 答案格式模板
        
    Example:
        >>> generator = GuidedTrajectoryGenerator(
        ...     model, tokenizer,
        ...     reasoning_template="分析问题：{query}\n步骤：",
        ...     answer_template="\n最终答案："
        ... )
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        reasoning_template: str = None,
        answer_template: str = None,
        **kwargs
    ):
        """
        初始化引导式轨迹生成器
        
        Args:
            model: 语言模型
            tokenizer: 分词器
            reasoning_template: 推理引导模板
            answer_template: 答案格式模板
            **kwargs: 其他传递给父类的参数
        """
        super().__init__(model, tokenizer, **kwargs)
        
        # 默认推理模板
        self.reasoning_template = reasoning_template or (
            "问题: {query}\n"
            "请按照以下步骤进行推理：\n"
            "1. 分析问题\n"
            "2. 搜索相关信息\n"
            "3. 进行推理\n"
            "4. 给出答案\n\n"
            "推理过程："
        )
        
        # 默认答案模板
        self.answer_template = answer_template or (
            "\n\n最终答案："
        )
    
    def _generate_response(
        self,
        query: str,
        prompt_template: Optional[str],
        temperature: float
    ) -> str:
        """
        生成带引导的响应
        
        使用结构化模板引导模型生成更规范的推理过程。
        
        Args:
            query: 查询问题
            prompt_template: 自定义提示模板
            temperature: 生成温度
            
        Returns:
            引导格式的响应
        """
        # 构建提示
        if prompt_template is None:
            prompt = self.reasoning_template.format(query=query)
            prompt += self.answer_template
        else:
            prompt = prompt_template.format(query=query)
        
        # 分词
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        
        # 移至设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # 移除提示部分
        response = response[len(prompt):].strip()
        
        return response


import re

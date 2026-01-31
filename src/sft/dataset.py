"""
SFT 数据集模块
=====================================================================
提供 SFT（有监督微调）数据集类，支持多种数据格式和聊天模板。

功能特点：
1. 支持问答对格式化为对话
2. 自动应用聊天模板（支持 Qwen, LLaMA 等模型）
3. 支持截断和填充
4. 支持多种数据源

主要类：
- SFTDataset: 核心数据集类

使用示例：
    from src.sft import SFTDataset
    
    dataset = SFTDataset(
        queries=["Who is the father of Arthur?"],
        answers=["Christopher"],
        tokenizer=tokenizer
    )
=====================================================================
"""

import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Union
from transformers import PreTrainedTokenizer


class SFTDataset(Dataset):
    """
    SFT（有监督微调）数据集
    
    将问答对格式化为指令微调的训练格式，支持多种聊天模板。
    
    Attributes:
        conversations: 会话列表
        tokenizer: 分词器
        max_length: 最大序列长度
        add_system_prompt: 是否添加系统提示
        system_prompt: 系统提示内容
    """
    
    def __init__(
        self,
        queries: List[str],
        answers: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        add_system_prompt: bool = True,
        system_prompt: Optional[str] = None
    ):
        """
        初始化数据集
        
        Args:
            queries: 问题列表
            answers: 答案列表
            tokenizer: 分词器
            max_length: 最大序列长度
            add_system_prompt: 是否添加系统提示
            system_prompt: 自定义系统提示（None 则使用默认）
        """
        if len(queries) != len(answers):
            raise ValueError(f"Queries and answers must have same length: {len(queries)} != {len(answers)}")
        
        self.conversations = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_system_prompt = add_system_prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        for query, answer in zip(queries, answers):
            conversation = self._format_conversation(query, answer)
            self.conversations.append(conversation)
    
    def _get_default_system_prompt(self) -> str:
        """
        获取默认系统提示
        
        Returns:
            系统提示字符串
        """
        return "You are a helpful assistant specialized in kinship relationships. Answer the questions about family relationships accurately and concisely."
    
    def _format_conversation(
        self,
        query: str,
        answer: str
    ) -> List[Dict[str, str]]:
        """
        格式化对话
        
        遵循标准的对话格式。
        
        Args:
            query: 问题
            answer: 答案
            
        Returns:
            格式化后的对话列表
        """
        conversation = []
        
        if self.add_system_prompt:
            conversation.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        conversation.append({
            "role": "user",
            "content": query
        })
        
        conversation.append({
            "role": "assistant",
            "content": answer
        })
        
        return conversation
    
    def _apply_chat_template(self, conversation: List[Dict]) -> str:
        """
        应用聊天模板
        
        尝试使用 tokenizer 的 chat_template，如果失败则使用默认格式。
        
        Args:
            conversation: 对话列表
            
        Returns:
            格式化的文本字符串
        """
        try:
            text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            return text
        except Exception as e:
            return self._default_format(conversation)
    
    def _default_format(self, conversation: List[Dict]) -> str:
        """
        默认格式化
        
        使用简单的格式，适用于大多数模型。
        
        Args:
            conversation: 对话列表
            
        Returns:
            格式化的文本字符串
        """
        parts = []
        
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        
        return "".join(parts)
    
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        分词处理
        
        Args:
            text: 文本字符串
            
        Returns:
            包含 input_ids, attention_mask 的字典
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoding
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            包含 input_ids, attention_mask, labels 的字典
        """
        conversation = self.conversations[idx]
        
        text = self._apply_chat_template(conversation)
        
        encoding = self._tokenize(text)
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        labels = input_ids.clone()
        
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels[labels == self.tokenizer.eos_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def get_sample(self, idx: int) -> Dict:
        """
        获取原始样本（未分词）
        
        Args:
            idx: 索引
            
        Returns:
            包含 query, answer, conversation 的字典
        """
        conversation = self.conversations[idx]
        
        user_msg = next((m for m in conversation if m["role"] == "user"), None)
        assistant_msg = next((m for m in conversation if m["role"] == "assistant"), None)
        
        return {
            'query': user_msg["content"] if user_msg else "",
            'answer': assistant_msg["content"] if assistant_msg else "",
            'conversation': conversation
        }


class SFTDatasetBuilder:
    """
    SFT 数据集构建器
    
    提供便捷的数据集构建方法，支持多种数据源。
    
    Attributes:
        tokenizer: 分词器
        max_length: 最大序列长度
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        """
        初始化构建器
        
        Args:
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def from_qa_pairs(
        self,
        qa_pairs: List[Dict],
        query_key: str = 'query',
        answer_key: str = 'answer',
        system_prompt: Optional[str] = None
    ) -> SFTDataset:
        """
        从问答对列表构建数据集
        
        Args:
            qa_pairs: 问答对列表
            query_key: 问题字段名
            answer_key: 答案字段名
            system_prompt: 系统提示
            
        Returns:
            SFTDataset 实例
        """
        queries = [pair[query_key] for pair in qa_pairs]
        answers = [pair[answer_key] for pair in qa_pairs]
        
        return SFTDataset(
            queries=queries,
            answers=answers,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            system_prompt=system_prompt
        )
    
    def from_augmented_data(
        self,
        data: List[Dict],
        use_formatted_answer: bool = False
    ) -> SFTDataset:
        """
        从增强数据构建数据集
        
        Args:
            data: 增强后的数据列表
            use_formatted_answer: 是否使用格式化答案
            
        Returns:
            SFTDataset 实例
        """
        queries = []
        answers = []
        
        for item in data:
            queries.append(item['query'])
            
            if use_formatted_answer and 'answer_text' in item:
                answers.append(item['answer_text'])
            else:
                answers.append(item['answer'])
        
        return SFTDataset(
            queries=queries,
            answers=answers,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
    
    def from_separate_lists(
        self,
        queries: List[str],
        answers: List[str],
        system_prompt: Optional[str] = None
    ) -> SFTDataset:
        """
        从分离的问题和答案列表构建数据集
        
        Args:
            queries: 问题列表
            answers: 答案列表
            system_prompt: 系统提示
            
        Returns:
            SFTDataset 实例
        """
        return SFTDataset(
            queries=queries,
            answers=answers,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            system_prompt=system_prompt
        )
    
    def split_train_eval(
        self,
        dataset: SFTDataset,
        test_size: float = 0.1,
        seed: int = 42
    ) -> tuple:
        """
        划分训练集和验证集
        
        Args:
            dataset: 完整数据集
            test_size: 测试集比例
            seed: 随机种子
            
        Returns:
            (训练集, 验证集) 元组
        """
        random.seed(seed)
        
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        test_count = int(len(dataset) * test_size)
        train_count = len(dataset) - test_count
        
        train_indices = indices[:train_count]
        test_indices = indices[train_count:]
        
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        eval_subset = torch.utils.data.Subset(dataset, test_indices)
        
        return train_subset, eval_subset


def create_sft_dataset(
    queries: List[str],
    answers: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    **kwargs
) -> SFTDataset:
    """
    便捷函数：创建 SFT 数据集
    
    Args:
        queries: 问题列表
        answers: 答案列表
        tokenizer: 分词器
        max_length: 最大序列长度
        **kwargs: 其他参数
        
    Returns:
        SFTDataset 实例
    
    Example:
        >>> dataset = create_sft_dataset(
        ...     queries=["Who is the father of Arthur?"],
        ...     answers=["Christopher"],
        ...     tokenizer=tokenizer
        ... )
    """
    return SFTDataset(
        queries=queries,
        answers=answers,
        tokenizer=tokenizer,
        max_length=max_length,
        **kwargs
    )

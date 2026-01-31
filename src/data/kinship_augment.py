"""
Kinship 数据增强模块
=====================================================================
该模块提供 kinship 数据集的增强功能，包括：
1. 同义问题改写（多种问法）
2. 反向问题生成
3. 家庭成员多跳问答对生成
4. 训练集/测试集划分

使用方法：
    from src.data.kinship_augment import augment_dataset, load_augmented_data
    
    # 基础增强（2x）
    train_data, test_data = augment_dataset("./dataset/kinship.data", augment_factor=2)
    
    # 充分增强（4x）
    train_data, test_data = augment_dataset("./dataset/kinship.data", augment_factor=4)
=====================================================================
"""

import random
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from .kinship import KinshipDataset
except ImportError:
    from kinship import KinshipDataset


class KinshipDataAugmentor:
    """
    Kinship 数据增强器
    
    支持多种数据增强策略，扩充训练数据量。
    
    Attributes:
        dataset: 原始 KinshipDataset
        relation_templates: 关系问法模板
        answer_templates: 答案格式模板
    """
    
    def __init__(self, data_path: str):
        """
        初始化增强器
        
        Args:
            data_path: 数据文件路径
        """
        self.dataset = KinshipDataset(data_path)
        self.relation_templates = self._init_relation_templates()
        self.answer_templates = self._init_answer_templates()
    
    def _init_relation_templates(self) -> Dict[str, List[Dict]]:
        """
        初始化关系问法模板
        
        Returns:
            关系到模板的映射
        """
        return {
            'father': [
                {'template': "Who is the father of {person}?", 'paraphrase': True},
                {'template': "Who is {person}'s father?", 'paraphrase': True},
                {'template': "What is the name of {person}'s father?", 'paraphrase': True},
                {'template': "{person}'s dad is who?", 'paraphrase': True},
                {'template': "Find {person}'s father.", 'paraphrase': False},
            ],
            'mother': [
                {'template': "Who is the mother of {person}?", 'paraphrase': True},
                {'template': "Who is {person}'s mother?", 'paraphrase': True},
                {'template': "What is the name of {person}'s mother?", 'paraphrase': True},
                {'template': "{person}'s mom is who?", 'paraphrase': True},
                {'template': "Find {person}'s mother.", 'paraphrase': False},
            ],
            'husband': [
                {'template': "Who is the husband of {person}?", 'paraphrase': True},
                {'template': "Who is {person}'s husband?", 'paraphrase': True},
                {'template': "Who is married to {person}?", 'paraphrase': True},
                {'template': "{person}'s spouse is who?", 'paraphrase': False},
            ],
            'wife': [
                {'template': "Who is the wife of {person}?", 'paraphrase': True},
                {'template': "Who is {person}'s wife?", 'paraphrase': True},
                {'template': "Who is married to {person}?", 'paraphrase': True},
                {'template': "{person}'s spouse is who?", 'paraphrase': False},
            ],
            'son': [
                {'template': "Who is the son of {person}?", 'paraphrase': True},
                {'template': "Who is {person}'s son?", 'paraphrase': True},
                {'template': "{person} has a son. Who is he?", 'paraphrase': False},
                {'template': "Find {person}'s son.", 'paraphrase': False},
            ],
            'daughter': [
                {'template': "Who is the daughter of {person}?", 'paraphrase': True},
                {'template': "Who is {person}'s daughter?", 'paraphrase': True},
                {'template': "{person} has a daughter. Who is she?", 'paraphrase': False},
                {'template': "Find {person}'s daughter.", 'paraphrase': False},
            ],
            'brother': [
                {'template': "Who is the brother of {person}?", 'paraphrase': True},
                {'template': "Who is {person}'s brother?", 'paraphrase': True},
                {'template': "{person}'s male sibling is who?", 'paraphrase': False},
                {'template': "Find {person}'s brother.", 'paraphrase': False},
            ],
            'sister': [
                {'template': "Who is the sister of {person}?", 'paraphrase': True},
                {'template': "Who is {person}'s sister?", 'paraphrase': True},
                {'template': "{person}'s female sibling is who?", 'paraphrase': False},
                {'template': "Find {person}'s sister.", 'paraphrase': False},
            ],
            'uncle': [
                {'template': "Who is the uncle of {person}?", 'paraphrase': True},
                {'template': "Who is {person}'s uncle?", 'paraphrase': True},
                {'template': "{person}'s father's brother is who?", 'paraphrase': False},
                {'template': "{person}'s mother's brother is who?", 'paraphrase': False},
                {'template': "Find {person}'s uncle.", 'paraphrase': False},
            ],
            'aunt': [
                {'template': "Who is the aunt of {person}?", 'paraphrase': True},
                {'template': "Who is {person}'s aunt?", 'paraphrase': True},
                {'template': "{person}'s father's sister is who?", 'paraphrase': False},
                {'template': "{person}'s mother's sister is who?", 'paraphrase': False},
                {'template': "Find {person}'s aunt.", 'paraphrase': False},
            ],
            'nephew': [
                {'template': "Who is the nephew of {person}?", 'paraphrase': True},
                {'template': "Who is {person}'s nephew?", 'paraphrase': True},
                {'template': "{person}'s sibling's son is who?", 'paraphrase': False},
                {'template': "Find {person}'s nephew.", 'paraphrase': False},
            ],
            'niece': [
                {'template': "Who is the niece of {person}?", 'paraphrase': True},
                {'template': "Who is {person}'s niece?", 'paraphrase': True},
                {'template': "{person}'s sibling's daughter is who?", 'paraphrase': False},
                {'template': "Find {person}'s niece.", 'paraphrase': False},
            ],
        }
    
    def _init_answer_templates(self) -> List[str]:
        """
        初始化答案格式模板
        
        Returns:
            答案格式列表
        """
        return [
            "{answer}",
            "The answer is {answer}.",
            "{answer} is the answer.",
        ]
    
    def generate_paraphrases(
        self,
        relation: str,
        person: str,
        answer: str,
        num_variants: int = 2
    ) -> List[Dict]:
        """
        生成同义改写问题
        
        Args:
            relation: 关系类型
            person: 被查询的人
            answer: 正确答案
            num_variants: 每种关系生成的变体数量
            
        Returns:
            改写后的问答对列表
        """
        templates = self.relation_templates.get(relation, [])
        if not templates:
            return []
        
        qa_pairs = []
        
        paraphrase_templates = [t for t in templates if t.get('paraphrase', False)]
        direct_templates = [t for t in templates if not t.get('paraphrase', False)]
        
        if paraphrase_templates:
            selected = random.sample(
                paraphrase_templates,
                min(num_variants, len(paraphrase_templates))
            )
            for t in selected:
                query = t['template'].format(person=person)
                answer_text = random.choice(self.answer_templates).format(answer=answer)
                qa_pairs.append({
                    'query': query,
                    'answer': answer,
                    'answer_text': answer_text,
                    'relation': relation,
                    'entity': person,
                    'augment_type': 'paraphrase'
                })
        
        if direct_templates and len(qa_pairs) < num_variants:
            remaining = num_variants - len(qa_pairs)
            selected = random.sample(
                direct_templates,
                min(remaining, len(direct_templates))
            )
            for t in selected:
                query = t['template'].format(person=person)
                answer_text = random.choice(self.answer_templates).format(answer=answer)
                qa_pairs.append({
                    'query': query,
                    'answer': answer,
                    'answer_text': answer_text,
                    'relation': relation,
                    'entity': person,
                    'augment_type': 'direct'
                })
        
        return qa_pairs
    
    def generate_reverse_questions(
        self,
        relation: str,
        subject: str,
        target: str
    ) -> List[Dict]:
        """
        生成反向问题
        
        从答案角度反推问题，增加问题多样性。
        
        Args:
            relation: 关系类型
            subject: 主体
            target: 客体（答案）
            
        Returns:
            反向问答对列表
        """
        reverse_templates = {
            'father': [
                "Who is the child of {target}?",
                "{target} has children. Who are they?",
            ],
            'mother': [
                "Who is the child of {target}?",
                "{target} has children. Who are they?",
            ],
            'husband': [
                "Who is married to {target}?",
                "Who is {target}'s partner?",
            ],
            'wife': [
                "Who is married to {target}?",
                "Who is {target}'s partner?",
            ],
            'son': [
                "Who are the parents of {target}?",
                "{target}'s father is who?",
                "{target}'s mother is who?",
            ],
            'daughter': [
                "Who are the parents of {target}?",
                "{target}'s father is who?",
                "{target}'s mother is who?",
            ],
            'brother': [
                "Who are the siblings of {target}?",
                "{target} has a brother. Who is he?",
            ],
            'sister': [
                "Who are the siblings of {target}?",
                "{target} has a sister. Who is she?",
            ],
            'uncle': [
                "Who is the niece/nephew of {target}?",
                "{target}'s sibling's child is who?",
            ],
            'aunt': [
                "Who is the niece/nephew of {target}?",
                "{target}'s sibling's child is who?",
            ],
            'nephew': [
                "Who is the aunt/uncle of {target}?",
                "{target}'s uncle is who?",
                "{target}'s aunt is who?",
            ],
            'niece': [
                "Who is the aunt/uncle of {target}?",
                "{target}'s uncle is who?",
                "{target}'s aunt is who?",
            ],
        }
        
        templates = reverse_templates.get(relation, [])
        qa_pairs = []
        
        for template in templates:
            query = template.format(target=target)
            answer_text = random.choice(self.answer_templates).format(answer=subject)
            qa_pairs.append({
                'query': query,
                'answer': subject,
                'answer_text': answer_text,
                'relation': relation,
                'entity': target,
                'augment_type': 'reverse'
            })
        
        return qa_pairs
    
    def generate_multi_hop_questions(
        self,
        entity1: str,
        entity2: str,
        path: List[Tuple[str, str, str]],
        max_hops: int = 2
    ) -> List[Dict]:
        """
        生成多跳问答对
        
        Args:
            entity1: 起始实体
            entity2: 目标实体
            path: 实体间的路径
            max_hops: 最大跳数
            
        Returns:
            多跳问答对列表
        """
        if len(path) < 2:
            return []
        
        qa_pairs = []
        
        if len(path) == 1:
            return []
        
        hop_relation = path[-1][0]
        middle_entity = path[0][2]
        
        query_templates = [
            f"Who is the {hop_relation} of {entity1}'s {path[0][0]}?",
            f"What is the relationship between {entity1} and {entity2}?",
            f"Find the relative of {entity1} through {path[0][0]}.",
        ]
        
        for template in query_templates:
            query = template
            answer_text = random.choice(self.answer_templates).format(answer=entity2)
            path_description = " → ".join([f"{r}({e1},{e2})" for r, e1, e2 in path])
            
            qa_pairs.append({
                'query': query,
                'answer': entity2,
                'answer_text': answer_text,
                'relation': f"multi_hop_{len(path)}",
                'entity': f"{entity1} → {entity2}",
                'augment_type': 'multi_hop',
                'path': path_description,
                'hops': len(path)
            })
        
        return qa_pairs
    
    def augment(
        self,
        augment_factor: int = 2,
        include_reverse: bool = True,
        include_multi_hop: bool = True,
        multi_hop_max_hops: int = 2,
        test_size: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        执行数据增强
        
        Args:
            augment_factor: 增强倍数（每个原始问答对生成的数量）
            include_reverse: 是否包含反向问题
            include_multi_hop: 是否包含多跳问题
            multi_hop_max_hops: 多跳最大跳数
            test_size: 测试集比例
            seed: 随机种子
            
        Returns:
            (训练集, 测试集)
        """
        random.seed(seed)
        
        logger.info(f"开始数据增强 (增强倍数: {augment_factor}x)")
        logger.info(f"包含反向问题: {include_reverse}")
        logger.info(f"包含多跳问题: {include_multi_hop}")
        
        all_qa_pairs = []
        original_pairs = self.dataset.generate_qa_pairs()
        
        logger.info(f"原始问答对数量: {len(original_pairs)}")
        
        for pair in original_pairs:
            relation = pair['relation']
            entity = pair['entity1']
            answer = pair['entity2']
            
            paraphrases = self.generate_paraphrases(
                relation, entity, answer,
                num_variants=augment_factor
            )
            all_qa_pairs.extend(paraphrases)
            
            if include_reverse:
                reverse_qs = self.generate_reverse_questions(
                    relation, entity, answer
                )
                all_qa_pairs.extend(reverse_qs)
        
        if include_multi_hop:
            entities = list(self.dataset.entities)
            for e1 in entities:
                for e2 in entities:
                    if e1 == e2:
                        continue
                    
                    paths = self.dataset.get_entity_path(e1, e2, multi_hop_max_hops)
                    
                    for path in paths:
                        if len(path) >= 2:
                            multi_hop_qs = self.generate_multi_hop_questions(
                                e1, e2, path, multi_hop_max_hops
                            )
                            all_qa_pairs.extend(multi_hop_qs)
        
        random.shuffle(all_qa_pairs)
        
        test_count = int(len(all_qa_pairs) * test_size)
        test_data = all_qa_pairs[:test_count]
        train_data = all_qa_pairs[test_count:]
        
        logger.info(f"增强后问答对数量: {len(all_qa_pairs)}")
        logger.info(f"训练集大小: {len(train_data)}")
        logger.info(f"测试集大小: {len(test_data)}")
        
        augment_types = defaultdict(int)
        for pair in all_qa_pairs:
            augment_types[pair['augment_type']] += 1
        
        logger.info(f"增强类型分布: {dict(augment_types)}")
        
        return train_data, test_data
    
    def save_datasets(
        self,
        train_data: List[Dict],
        test_data: List[Dict],
        output_dir: str = "./dataset/augmented"
    ):
        """
        保存增强后的数据集
        
        Args:
            train_data: 训练数据
            test_data: 测试数据
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_file = output_path / "train.json"
        test_file = output_path / "test.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练集已保存: {train_file}")
        logger.info(f"测试集已保存: {test_file}")
        
        return str(train_file), str(test_file)


def augment_dataset(
    data_path: str,
    augment_factor: int = 2,
    include_reverse: bool = True,
    include_multi_hop: bool = True,
    test_size: float = 0.1,
    seed: int = 42,
    output_dir: str = "./dataset/augmented"
) -> Tuple[List[Dict], List[Dict]]:
    """
    便捷函数：执行数据增强并保存
    
    Args:
        data_path: 原始数据文件路径
        augment_factor: 增强倍数
        include_reverse: 是否包含反向问题
        include_multi_hop: 是否包含多跳问题
        test_size: 测试集比例
        seed: 随机种子
        output_dir: 输出目录
        
    Returns:
        (训练集, 测试集)
    
    Example:
        >>> train_data, test_data = augment_dataset(
        ...     "./dataset/kinship.data",
        ...     augment_factor=3,
        ...     output_dir="./dataset/processed"
        ... )
    """
    augmentor = KinshipDataAugmentor(data_path)
    
    train_data, test_data = augmentor.augment(
        augment_factor=augment_factor,
        include_reverse=include_reverse,
        include_multi_hop=include_multi_hop,
        test_size=test_size,
        seed=seed
    )
    
    augmentor.save_datasets(train_data, test_data, output_dir)
    
    return train_data, test_data


def load_augmented_data(
    train_path: str,
    test_path: str = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    加载增强后的数据集
    
    Args:
        train_path: 训练集文件路径
        test_path: 测试集文件路径（可选）
        
    Returns:
        (训练集, 测试集)
    
    Example:
        >>> train_data, test_data = load_augmented_data(
        ...     "./dataset/augmented/train.json",
        ...     "./dataset/augmented/test.json"
        ... )
    """
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    test_data = []
    if test_path and os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    
    return train_data, test_data


def get_sft_format_data(data: List[Dict], use_formatted_answer: bool = False) -> Tuple[List[str], List[str]]:
    """
    转换为 SFT 格式
    
    Args:
        data: 问答对列表
        use_formatted_answer: 是否使用格式化答案
        
    Returns:
        (问题列表, 答案列表)
    """
    queries = []
    answers = []
    
    for item in data:
        queries.append(item['query'])
        if use_formatted_answer and 'answer_text' in item:
            answers.append(item['answer_text'])
        else:
            answers.append(item['answer'])
    
    return queries, answers


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Kinship 数据增强工具")
    print("=" * 60)
    
    train_data, test_data = augment_dataset(
        data_path="./dataset/kinship.data",
        augment_factor=3,
        output_dir="./dataset/augmented"
    )
    
    print(f"\n增强后的数据统计:")
    print(f"训练集: {len(train_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    
    queries, answers = get_sft_format_data(train_data[:5])
    
    print(f"\n示例数据:")
    for q, a in zip(queries[:5], answers[:5]):
        print(f"Q: {q}")
        print(f"A: {a}")
        print("-" * 40)

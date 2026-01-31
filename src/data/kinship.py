"""
亲属关系数据集模块
=====================================================================
该模块提供了处理亲属关系知识图谱数据的功能，包括：
- 加载和解析亲属关系三元组数据
- 生成单跳和多跳问答对
- 创建知识图谱上下文
- 支持实体间路径搜索

主要类：
- KinshipDataset: 核心数据集类，处理所有亲属关系数据操作
=====================================================================
"""

import re
from typing import List, Dict, Tuple, Set, Optional
from pathlib import Path


class KinshipDataset:
    """
    亲属关系数据集类
    
    该类用于加载、解析和处理亲属关系知识图谱数据。
    支持单跳和多跳问答对生成，以及实体间路径搜索。
    
    Attributes:
        RELATIONS: 支持的亲属关系类型列表
        data_path: 数据文件路径
        relations: 关系字典，存储每种关系的三元组
        entities: 实体集合
        triples: 所有三元组列表
        
    Example:
        >>> dataset = KinshipDataset("./kinship.data")
        >>> stats = dataset.get_stats()
        >>> qa_pairs = dataset.generate_qa_pairs()
    """
    
    # 支持的亲属关系类型
    RELATIONS = [
        'wife', 'husband', 'mother', 'father',
        'daughter', 'son', 'sister', 'brother',
        'aunt', 'uncle', 'niece', 'nephew'
    ]
    
    def __init__(self, data_path: str):
        """
        初始化亲属关系数据集
        
        Args:
            data_path: 亲属关系数据文件路径
            
        Note:
            数据文件格式应为Prolog风格的三元组，如：
            father(arthur, christopher)
            mother(arthur, victoria)
        """
        self.data_path = Path(data_path)
        self.relations: Dict[str, List[Tuple[str, str]]] = {}
        self.entities: Set[str] = set()
        self.triples: List[Tuple[str, str, str]] = []
        
        self._parse_data()
    
    def _parse_data(self):
        """
        解析数据文件
        
        从数据文件中提取亲属关系三元组。
        期望的格式：relation(entity1, entity2)
        
        解析过程：
        1. 逐行读取文件
        2. 使用正则表达式匹配三元组格式
        3. 提取关系类型和两个实体
        4. 更新关系字典、实体集合和三元组列表
        """
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 使用正则表达式匹配格式：relation(entity1, entity2)
                match = re.match(r'(\w+)\((\w+),\s*(\w+)\)', line)
                if match:
                    relation = match.group(1)
                    entity1 = match.group(2)
                    entity2 = match.group(3)
                    
                    if relation not in self.relations:
                        self.relations[relation] = []
                    self.relations[relation].append((entity1, entity2))
                    
                    self.entities.add(entity1)
                    self.entities.add(entity2)
                    self.triples.append((relation, entity1, entity2))
    
    def get_all_triples(self) -> List[Tuple[str, str, str]]:
        """
        获取所有三元组
        
        Returns:
            三元组列表，每个三元组为 (关系类型, 实体1, 实体2)
        """
        return self.triples
    
    def get_relation_triples(self, relation: str) -> List[Tuple[str, str]]:
        """
        获取特定关系的三元组
        
        Args:
            relation: 关系类型，如 'father', 'mother'
            
        Returns:
            该关系下的所有 (主体, 客体) 元组列表
        """
        return self.relations.get(relation, [])
    
    def get_entity_relations(self, entity: str) -> List[Tuple[str, str]]:
        """
        获取实体的所有关系
        
        查询给定实体的所有关系，包括正向和逆向关系。
        
        Args:
            entity: 实体名称
            
        Returns:
            关系列表，每个元素为 (关系类型, 相关实体)
            
        Example:
            >>> dataset.get_entity_relations('arthur')
            [('father', 'christopher'), ('mother', 'victoria')]
        """
        results = []
        for relation, triples in self.relations.items():
            for e1, e2 in triples:
                if e1 == entity:
                    results.append((relation, e2))
                elif e2 == entity:
                    # 对于逆向关系，使用逆关系词
                    inverse = self._get_inverse_relation(relation)
                    results.append((inverse, e1))
        return results
    
    def _get_inverse_relation(self, relation: str) -> str:
        """
        获取逆关系
        
        根据预定义的逆关系映射表，将关系转换为逆关系。
        
        Args:
            relation: 原始关系类型
            
        Returns:
            逆关系类型，如果不存在映射则返回原关系
            
        Example:
            >>> ds._get_inverse_relation('wife')
            'husband'
        """
        inverses = {
            'wife': 'husband', 'husband': 'wife',
            'mother': 'son', 'son': 'mother',
            'daughter': 'father', 'father': 'daughter',
            'sister': 'brother', 'brother': 'sister',
            'aunt': 'nephew', 'nephew': 'aunt',
            'uncle': 'niece', 'niece': 'uncle'
        }
        return inverses.get(relation, relation)
    
    def get_entity_path(self, start: str, end: str, max_hops: int = 3) -> List[List[Tuple[str, str, str]]]:
        """
        获取两个实体之间的所有路径
        
        使用深度优先搜索(DFS)查找两个实体之间的所有路径。
        用于多跳推理任务。
        
        Args:
            start: 起始实体
            end: 目标实体
            max_hops: 最大跳数限制
            
        Returns:
            路径列表，每条路径是一系列三元组
            
        Example:
            >>> paths = dataset.get_entity_path('alice', 'bob', max_hops=2)
        """
        paths = []
        self._dfs_find_path(start, end, [], paths, set(), max_hops)
        return paths
    
    def _dfs_find_path(
        self,
        current: str,
        target: str,
        current_path: List[Tuple[str, str, str]],
        all_paths: List[List[Tuple[str, str]]],
        visited: Set[str],
        max_hops: int
    ):
        """
        深度优先搜索路径
        
        递归DFS算法，用于查找实体间的所有路径。
        
        Args:
            current: 当前实体
            target: 目标实体
            current_path: 当前累积的路径
            all_paths: 存储所有找到的路径
            visited: 已访问实体集合，防止循环
            max_hops: 最大跳数限制
        """
        # 找到目标，保存路径
        if current == target:
            all_paths.append(current_path.copy())
            return
        
        # 达到最大跳数，停止搜索
        if len(current_path) >= max_hops:
            return
        
        # 遍历当前实体的所有邻居
        for relation, neighbor in self.get_entity_relations(current):
            if neighbor not in visited:
                visited.add(neighbor)
                current_path.append((relation, current, neighbor))
                self._dfs_find_path(
                    neighbor, target, current_path,
                    all_paths, visited, max_hops
                )
                current_path.pop()
                visited.remove(neighbor)
    
    def generate_qa_pairs(self) -> List[Dict]:
        """
        生成单跳问答对
        
        根据所有三元组生成问答对，支持正向和逆向查询。
        
        Returns:
            问答对列表，每个问答对包含：
            - query: 自然语言问题
            - answer: 答案
            - relation: 关系类型
            - entity1: 主体实体
            - entity2: 客体实体
            - triples: 相关三元组
            
        Example:
            >>> qa_pairs = dataset.generate_qa_pairs()
            >>> qa_pairs[0]['query']
            "Who is the father of Arthur?"
        """
        qa_pairs = []
        
        for relation, triples in self.relations.items():
            for e1, e2 in triples:
                # 生成正向查询
                query = self._generate_query(relation, e1)
                answer = e2
                
                qa_pairs.append({
                    'query': query,
                    'answer': answer,
                    'relation': relation,
                    'entity1': e1,
                    'entity2': e2,
                    'triples': [(relation, e1, e2)]
                })
                
                # 生成逆向查询（仅对部分关系）
                if relation in ['father', 'mother', 'husband', 'wife']:
                    inverse_relation = self._get_inverse_relation(relation)
                    inverse_query = self._generate_query(inverse_relation, e2)
                    inverse_answer = e1
                    
                    qa_pairs.append({
                        'query': inverse_query,
                        'answer': inverse_answer,
                        'relation': inverse_relation,
                        'entity1': e2,
                        'entity2': e1,
                        'triples': [(relation, e1, e2)]
                    })
        
        return qa_pairs
    
    def _generate_query(self, relation: str, entity: str) -> str:
        """
        生成自然语言查询
        
        根据关系类型和实体生成自然语言问题。
        
        Args:
            relation: 关系类型
            entity: 实体名称
            
        Returns:
            自然语言问题字符串
            
        Example:
            >>> ds._generate_query('father', 'arthur')
            "Who is the father of Arthur?"
        """
        templates = {
            'wife': f"Who is the wife of {entity}?",
            'husband': f"Who is the husband of {entity}?",
            'mother': f"Who is the mother of {entity}?",
            'father': f"Who is the father of {entity}?",
            'daughter': f"Who is the daughter of {entity}?",
            'son': f"Who is the son of {entity}?",
            'sister': f"Who is the sister of {entity}?",
            'brother': f"Who is the brother of {entity}?",
            'aunt': f"Who is the aunt of {entity}?",
            'uncle': f"Who is the uncle of {entity}?",
            'niece': f"Who is the niece of {entity}?",
            'nephew': f"Who is the nephew of {entity}?",
        }
        return templates.get(relation, f"Who is the {relation} of {entity}?")
    
    def generate_multi_hop_qa_pairs(self, max_hops: int = 2) -> List[Dict]:
        """
        生成多跳问答对
        
        通过查找实体间的路径，生成需要多步推理才能回答的问题。
        
        Args:
            max_hops: 最大跳数
            
        Returns:
            多跳问答对列表
            
        Example:
            >>> multi_hop_qa = dataset.generate_multi_hop_qa_pairs(max_hops=2)
        """
        qa_pairs = []
        
        # 遍历所有实体对
        for e1 in self.entities:
            for e2 in self.entities:
                if e1 == e2:
                    continue
                
                # 查找路径
                paths = self.get_entity_path(e1, e2, max_hops)
                
                # 为每条有效路径生成问答对
                for path in paths:
                    if len(path) >= 2:
                        query = self._generate_multi_hop_query(path)
                        answer = e2
                        
                        qa_pairs.append({
                            'query': query,
                            'answer': answer,
                            'path': path,
                            'entity1': e1,
                            'entity2': e2,
                            'hops': len(path)
                        })
        
        return qa_pairs
    
    def _generate_multi_hop_query(self, path: List[Tuple[str, str, str]]) -> str:
        """
        生成多跳查询
        
        根据路径生成需要多步推理的问题。
        
        Args:
            path: 实体间的路径
            
        Returns:
            多跳查询问题
            
        Example:
            >>> path = [('father', 'arthur', 'christopher'), ('wife', 'christopher', 'victoria')]
            >>> ds._generate_multi_hop_query(path)
            "Who is the wife of Arthur's father?"
        """
        start = path[0][1]
        end_relation = path[-1][0]
        end = path[-1][2]
        
        return f"Who is the {end_relation} of {start}'s {path[0][0]}?"
    
    def get_all_entities(self) -> List[str]:
        """
        获取所有实体
        
        Returns:
            实体名称列表
        """
        return list(self.entities)
    
    def get_stats(self) -> Dict:
        """
        获取数据集统计信息
        
        Returns:
            包含数据集各项统计指标的字典
        """
        return {
            'num_entities': len(self.entities),
            'num_triples': len(self.triples),
            'num_relations': len(self.relations),
            'relations': list(self.relations.keys()),
            'qa_pairs': len(self.generate_qa_pairs()),
            'multi_hop_qa_pairs': len(self.generate_multi_hop_qa_pairs())
        }


def load_kinship_data(data_path: str, multi_hop: bool = False) -> Tuple[List[str], List[str]]:
    """
    加载亲属关系数据用于训练
    
    便捷函数，用于加载数据并生成问答对。
    
    Args:
        data_path: 数据文件路径
        multi_hop: 是否生成多跳问答
        
    Returns:
        tuple: (问题列表, 答案列表, 完整问答对列表)
        
    Example:
        >>> queries, answers, qa_pairs = load_kinship_data("./kinship.data")
    """
    dataset = KinshipDataset(data_path)
    
    if multi_hop:
        qa_pairs = dataset.generate_multi_hop_qa_pairs()
    else:
        qa_pairs = dataset.generate_qa_pairs()
    
    queries = [pair['query'] for pair in qa_pairs]
    answers = [pair['answer'] for pair in qa_pairs]
    
    return queries, answers, qa_pairs


def create_kinship_context(dataset: KinshipDataset) -> str:
    """
    创建知识图谱上下文
    
    将数据集转换为自然语言上下文，用于提示学习。
    
    Args:
        dataset: 亲属关系数据集实例
        
    Returns:
        格式化的上下文字符串
        
    Example:
        >>> context = create_kinship_context(dataset)
        >>> print(context)
        Here is the kinship relationship knowledge:
        
        father(arthur, christopher)
        mother(arthur, victoria)
    """
    context = "Here is the kinship relationship knowledge:\n\n"
    
    for relation, triples in dataset.relations.items():
        for e1, e2 in triples:
            context += f"{relation}({e1}, {e2})\n"
    
    return context


def format_prompt_with_context(query: str, context: str) -> str:
    """
    使用上下文格式化提示
    
    创建用于模型推理的完整提示。
    
    Args:
        query: 查询问题
        context: 知识图谱上下文
        
    Returns:
        完整提示字符串
        
    Example:
        >>> prompt = format_prompt_with_context(
        ...     "Who is the father of Arthur?",
        ...     "father(arthur, christopher)"
        ... )
    """
    return f"""Context:
{context}

Question: {query}
Please reason step by step and provide the answer.

Answer:"""

"""
亲属关系数据集使用示例
=====================================================================
该脚本展示了如何完整使用亲属关系数据集和GRPO训练框架。
包括数据集探索、问答对生成、奖励函数演示等功能。

主要功能：
1. 数据集探索 (explore_dataset)
2. 问答对生成演示 (demonstrate_qa_generation)
3. 多跳推理演示 (demonstrate_multi_hop)
4. 上下文创建演示 (demonstrate_context)
5. 奖励函数演示 (demonstrate_reward_function)
6. GRPO核心算法演示 (demonstrate_grpo_core)
7. 完整训练示例 (full_training_example)

使用方法：
    python examples/kinship_example.py

输出：
    - 控制台显示各种演示结果
    - 数据集统计信息
    - 问答对示例
    - 奖励计算示例
=====================================================================
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.kinship import (
    KinshipDataset,
    load_kinship_data,
    create_kinship_context,
    format_prompt_with_context
)
from src.grpo import (
    KinshipRewardFunction,
    GRPOCore,
    TrajectoryGenerator
)


def explore_dataset():
    """
    探索亲属关系数据集
    
    展示如何加载数据集并查看其统计信息。
    
    功能：
    1. 加载数据集
    2. 查看统计信息
    3. 列出所有实体
    4. 显示示例三元组
    
    Returns:
        KinshipDataset: 数据集实例
    """
    print("=" * 60)
    print("Kinship Dataset Exploration")
    print("=" * 60)
    
    # 加载数据集
    dataset = KinshipDataset("./kinship/kinship.data")
    
    # 获取统计信息
    stats = dataset.get_stats()
    
    print(f"\nDataset Statistics:")
    print(f"  - Number of entities: {stats['num_entities']}")
    print(f"  - Number of triples: {stats['num_triples']}")
    print(f"  - Number of relations: {stats['num_relations']}")
    print(f"  - Relations: {stats['relations']}")
    print(f"  - QA pairs: {stats['qa_pairs']}")
    print(f"  - Multi-hop QA pairs: {stats['multi_hop_qa_pairs']}")
    
    # 列出所有实体，按字母顺序排序便于查看
    print(f"\nAll entities: {sorted(dataset.get_all_entities())}")
    
    # 显示示例三元组，每个关系类型展示前3个
    print("\nSample triples:")
    for relation, triples in list(dataset.relations.items())[:3]:
        print(f"  {relation}: {triples[:3]}")
    
    return dataset


def demonstrate_qa_generation(dataset):
    """
    演示问答对生成
    
    展示如何从数据集生成单跳问答对。
    
    Args:
        dataset: 亲属关系数据集
        
    Returns:
        list: 生成的问答对列表
    """
    print("\n" + "=" * 60)
    print("QA Pair Generation")
    print("=" * 60)
    
    # 生成问答对
    qa_pairs = dataset.generate_qa_pairs()
    
    print(f"\nGenerated {len(qa_pairs)} QA pairs")
    # 展示前5个问答对作为示例
    print("\nSample QA pairs:")
    for i, pair in enumerate(qa_pairs[:5]):
        print(f"\n{i+1}. Query: {pair['query']}")
        print(f"   Answer: {pair['answer']}")
        print(f"   Relation: {pair['relation']}")
        print(f"   Entity1: {pair['entity1']}, Entity2: {pair['entity2']}")
    
    return qa_pairs


def demonstrate_multi_hop(dataset):
    """
    演示多跳推理问答对生成
    
    展示如何生成需要多步推理的问题。
    
    Args:
        dataset: 亲属关系数据集
        
    Returns:
        list: 多跳问答对列表
    """
    print("\n" + "=" * 60)
    print("Multi-hop Reasoning")
    print("=" * 60)
    
    # 生成多跳问答对，限制最大跳数为2
    multi_hop_qa = dataset.generate_multi_hop_qa_pairs(max_hops=2)
    
    print(f"\nGenerated {len(multi_hop_qa)} multi-hop QA pairs")
    # 展示前5个多跳问答对
    print("\nSample multi-hop QA pairs:")
    for i, pair in enumerate(multi_hop_qa[:5]):
        print(f"\n{i+1}. Query: {pair['query']}")
        print(f"   Answer: {pair['answer']}")
        print(f"   Hops: {pair['hops']}")
        print(f"   Path: {pair['path']}")
    
    return multi_hop_qa


def demonstrate_context(dataset):
    """
    演示知识图谱上下文创建
    
    展示如何将知识图谱转换为自然语言上下文。
    
    Args:
        dataset: 亲属关系数据集
    """
    print("\n" + "=" * 60)
    print("Context Creation")
    print("=" * 60)
    
    # 创建上下文，将知识图谱转换为自然语言描述
    context = create_kinship_context(dataset)
    print(f"\nContext length: {len(context)} characters")
    # 截取前500个字符进行预览
    print("\nContext preview:")
    print(context[:500] + "...")


def demonstrate_reward_function(dataset):
    """
    演示奖励函数使用
    
    展示如何计算不同答案的奖励值。
    
    Args:
        dataset: 亲属关系数据集
    """
    print("\n" + "=" * 60)
    print("Reward Function Demo")
    print("=" * 60)
    
    # 创建奖励函数实例，配置各参数
    reward_func = KinshipRewardFunction(
        path_length_penalty=0.1,       # 路径长度惩罚系数
        correct_answer_bonus=1.0,      # 正确答案奖励
        wrong_answer_penalty=0.0       # 错误答案惩罚
    )
    
    # 生成测试用的问答对
    qa_pairs = dataset.generate_qa_pairs()
    
    print("\nReward computation for sample queries:")
    for pair in qa_pairs[:3]:
        # 计算正确答案奖励，预期获得正奖励
        reward = reward_func.compute_reward(
            query=pair['query'],
            states=[pair['answer']],
            actions=["reasoning"],
            final_answer=pair['answer'],
            ground_truth=pair['answer']
        )
        
        # 计算错误答案奖励，预期获得较低或负奖励
        wrong_reward = reward_func.compute_reward(
            query=pair['query'],
            states=["wrong_answer"],
            actions=["reasoning"],
            final_answer="wrong_answer",
            ground_truth=pair['answer']
        )
        
        print(f"\nQuery: {pair['query']}")
        print(f"  Correct answer reward: {reward:.2f}")
        print(f"  Wrong answer reward: {wrong_reward:.2f}")


def demonstrate_grpo_core():
    """
    演示GRPO核心算法
    
    展示如何计算组内相对优势和处理轨迹组。
    """
    print("\n" + "=" * 60)
    print("GRPO Core Algorithm Demo")
    print("=" * 60)
    
    # 创建GRPO核心实例，配置算法参数
    grpo = GRPOCore(
        group_size=3,              # 每组轨迹数量
        clip_epsilon=0.1,          # PPO裁剪阈值
        kl_coeff=0.05,             # KL散度系数
        beta=0.01,                 # beta参数
        normalize_reward=True,     # 是否对奖励进行标准化
        reward_scale=0.1           # 奖励缩放因子
    )
    
    # 计算相对优势：将奖励转换为优势值
    # 组内奖励会被标准化，生成相对优势分数
    rewards = [0.8, 0.5, 0.2]
    advantages = grpo.compute_group_relative_advantages(rewards)
    
    print(f"\nInput rewards: {rewards}")
    print(f"Computed advantages: {[f'{a:.4f}' for a in advantages]}")
    
    print("\nGroup processing:")
    # 创建示例轨迹用于演示
    from src.grpo import Trajectory
    trajectories = [
        # 正确答案轨迹 - 预期获得最高优势
        Trajectory(
            query="Who is the father of Arthur?",
            states=["Christopher"],
            actions=["reasoning"],
            rewards=[0.8],
            total_reward=0.8,
            is_correct=True
        ),
        # 错误答案轨迹 - 预期获得中等优势
        Trajectory(
            query="Who is the father of Arthur?",
            states=["Andrew"],
            actions=["reasoning"],
            rewards=[0.5],
            total_reward=0.5,
            is_correct=False
        ),
        # 错误答案轨迹 - 预期获得最低优势
        Trajectory(
            query="Who is the father of Arthur?",
            states=["Robert"],
            actions=["reasoning"],
            rewards=[0.2],
            total_reward=0.2,
            is_correct=False
        )
    ]
    
    # 处理轨迹组，计算组统计信息和优势
    result = grpo.process_group(trajectories)
    print(f"  Mean reward: {result['mean_reward']:.4f}")
    print(f"  Std reward: {result['std_reward']:.4f}")
    print(f"  Advantages: {[f'{a:.4f}' for a in result['advantages']]}")


def full_training_example():
    """
    完整训练示例
    
    提供完整训练流程的代码示例。
    包含从数据准备到模型保存的完整流程。
    """
    print("\n" + "=" * 60)
    print("Full Training Example")
    print("=" * 60)
    
    print("""
# Full training script usage:

# 第1步：导入必要的模块
from src.data.kinship import load_kinship_data, create_kinship_context
from src.grpo import GRPOTrainer, KinshipRewardFunction
from src.utils.model_utils import load_model_and_tokenizer, prepare_model_for_training

# 第2步：加载数据
# - queries: 问题列表
# - answers: 答案列表
# - qa_pairs: 问答对字典列表
queries, answers, qa_pairs = load_kinship_data("./kinship/kinship.data")
kinship_context = create_kinship_context(kinship_dataset)

# 第3步：加载模型和分词器
# - 使用4bit量化减少显存占用
# - 自动设备分配
model, tokenizer = load_model_and_tokenizer(
    model_path="/Users/xry/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct",
    load_in_4bit=True,
    device="auto"
)

# 第4步：应用LoRA进行高效微调
# - lora_r: LoRA矩阵的秩
model = prepare_model_for_training(model, use_lora=True, lora_config={'lora_r': 16})

# 第5步：初始化训练器
# - group_size: 每组样本数
# - learning_rate: 学习率
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    grpo_config={'group_size': 3, 'learning_rate': 5e-5},
    training_config={'output_dir': './outputs', 'epochs': 10},
    reward_function=KinshipRewardFunction()
)

# 第6步：训练模型
# - 传入数据加载器
trainer.train(dataloader)

# 第7步：保存最终模型
trainer.save_model("./outputs/final_model")
""")


def main():
    """
    主函数
    
    运行所有演示功能。
    按顺序执行各个演示模块，展示框架的主要功能。
    """
    print("\n" + "=" * 60)
    print("Kinship GRPO Training Framework Demo")
    print("=" * 60)
    
    # 演示1：探索数据集结构和统计信息
    dataset = explore_dataset()
    
    # 演示2：展示问答对生成功能
    demonstrate_qa_generation(dataset)
    
    # 演示3：展示多跳推理问答对生成
    demonstrate_multi_hop(dataset)
    
    # 演示4：展示知识图谱到上下文的转换
    demonstrate_context(dataset)
    
    # 演示5：展示奖励函数的使用和计算
    demonstrate_reward_function(dataset)
    
    # 演示6：展示GRPO核心算法的实现
    demonstrate_grpo_core()
    
    # 演示7：提供完整的训练代码示例
    full_training_example()
    
    print("\n" + "=" * 60)
    print("Demo Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

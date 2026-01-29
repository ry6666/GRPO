# GRPO Training Framework

基于 Qwen2.5-7B-Instruct 的 GRPO（Grouped Relative Policy Optimization）训练框架。

## 特性

- 🚀 **高效训练**: 支持 LoRA/QLoRA 微调，适配消费级 GPU
- 📊 **组内相对奖励**: 无需 Critic 网络，降低训练成本
- 🎯 **多任务支持**: 亲属关系、多跳推理、问答等场景
- 🔧 **易于使用**: 简洁的配置和训练接口

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 修改模型路径

在 `scripts/train_grpo.py` 中修改本地模型路径：

```python
--model_path "/Users/xry/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct"
```

### 2. 运行训练

```bash
cd /Users/xry/Desktop/python/projects/Agent-R1
python scripts/train_grpo.py \
    --model_path "/Users/xry/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct" \
    --task_type kinship \
    --group_size 3 \
    --learning_rate 5e-5 \
    --batch_size 2 \
    --epochs 5 \
    --use_lora
```

### 3. 使用自定义数据

创建数据文件并修改训练脚本：

```python
from src.grpo import GRPOTrainer

# 准备您的数据
queries = ["问题1", "问题2", "问题3"]
ground_truths = ["答案1", "答案2", "答案3"]

# 创建数据集
dataset = SimpleDataset(queries, ground_truths)

# 自定义奖励函数
from src.grpo import KinshipRewardFunction
reward_function = KinshipRewardFunction(
    path_length_penalty=0.1,
    correct_answer_bonus=1.0
)

# 初始化训练器并训练
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    grpo_config={...},
    training_config={...},
    reward_function=reward_function
)
```

## 配置参数

### GRPO 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `group_size` | 3 | 每组轨迹数量 |
| `clip_epsilon` | 0.1 | PPO 裁剪系数 |
| `kl_coeff` | 0.05 | KL 散度系数 |
| `learning_rate` | 5e-5 | 学习率 |
| `normalize_reward` | True | 是否标准化奖励 |

### LoRA 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora_r` | 16 | LoRA 秩 |
| `lora_alpha` | 32 | LoRA 缩放因子 |
| `lora_dropout` | 0.05 | Dropout 比率 |

## 项目结构

```
Agent-R1/
├── scripts/
│   └── train_grpo.py          # 训练主脚本
├── src/
│   ├── grpo/
│   │   ├── __init__.py
│   │   ├── core.py            # GRPO 核心算法
│   │   ├── reward.py          # 奖励函数
│   │   ├── generator.py       # 轨迹生成器
│   │   └── trainer.py         # 训练器
│   ├── utils/
│   │   ├── __init__.py
│   │   └── model_utils.py     # 模型工具
│   └── config.py              # 配置管理
├── configs/                    # 配置文件
├── outputs/                    # 输出目录
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 参考文献

- **GRPO**: Grouped Relative Policy Optimization
- **PPO**: Proximal Policy Optimization
- **LoRA**: Low-Rank Adaptation

## 许可证

MIT License

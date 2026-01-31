"""
测试 Ollama 本地 Qwen 模型在亲属关系任务上的表现
训练前基准测试
"""

import os
import sys
import json
from pathlib import Path

try:
    import ollama
except ImportError:
    print("安装 ollama 库...")
    os.system("uv pip install ollama -q")
    import ollama

from src.data.kinship_augment import load_augmented_data

def test_ollama_model(model_name="qwen2.5:7b"):
    print("=" * 70)
    print(f"🧪 Ollama {model_name} 模型测试 - 亲属关系任务")
    print("=" * 70)
    
    print("\n📊 加载测试数据...")
    _, test_data = load_augmented_data(
        "./dataset/augmented/train.json",
        "./dataset/augmented/test.json"
    )
    
    test_queries = [item['query'] for item in test_data]
    test_answers = [item['answer'] for item in test_data]
    
    print(f"测试集大小: {len(test_queries)} 条\n")
    
    print("=" * 70)
    print("🧪 开始测试...")
    print("=" * 70)
    
    correct = 0
    wrong = 0
    results = []
    
    for i, (query, true_answer) in enumerate(zip(test_queries[:30], test_answers[:30])):
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": "请简短回答亲属关系问题，直接给出答案，不需要解释。"},
                    {"role": "user", "content": query}
                ],
                options={
                    "temperature": 0.1,
                    "num_predict": 30
                }
            )
            model_answer = response['message']['content'].strip()
        except Exception as e:
            print(f"API 错误: {e}")
            model_answer = ""
        
        is_correct = model_answer == true_answer.strip()
        if is_correct:
            correct += 1
        else:
            wrong += 1
        
        results.append({
            'query': query,
            'true_answer': true_answer,
            'model_answer': model_answer,
            'correct': is_correct
        })
        
        status = "✅" if is_correct else "❌"
        print(f"[{status}] {i+1}/30: {query[:30]}...")
        print(f"   回答: {model_answer}")
        if not is_correct:
            print(f"   正确: {true_answer}")
        print()
        
        if (i + 1) % 10 == 0:
            print(f"--- 进度: {i+1}/30, 正确: {correct}, 错误: {wrong} ---")
            print()
    
    print("\n" + "=" * 70)
    print("📊 测试结果汇总")
    print("=" * 70)
    
    accuracy = correct / len(results) * 100
    print(f"\n🎯 测试准确率: {accuracy:.1f}% ({correct}/{len(results)})")
    print(f"✅ 正确: {correct} 条")
    print(f"❌ 错误: {wrong} 条")
    
    print("\n" + "=" * 70)
    print("📝 样本展示")
    print("=" * 70)
    
    for i, result in enumerate(results[:10]):
        status = "✅" if result['correct'] else "❌"
        print(f"\n[{status}] 样本 {i+1}")
        print(f"Q: {result['query']}")
        print(f"A: {result['model_answer']}")
        if not result['correct']:
            print(f"正确: {result['true_answer']}")
        print("-" * 50)
    
    return results

if __name__ == "__main__":
    print("\n检查 Ollama 服务...")
    try:
        ollama.list()
        print("✅ Ollama 服务正常\n")
    except Exception as e:
        print(f"❌ Ollama 服务异常: {e}")
        print("请确保 Ollama 正在运行: ollama serve")
        sys.exit(1)
    
    results = test_ollama_model("qwen2.5:7b")
    
    print("\n" + "=" * 70)
    print("💾 保存结果...")
    print("=" * 70)
    
    os.makedirs("./outputs", exist_ok=True)
    output_path = "./outputs/ollama_test_results.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_path}")

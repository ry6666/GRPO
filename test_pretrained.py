"""
测试预训练模型在亲属关系任务上的表现
训练前基准测试
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.kinship_augment import load_augmented_data

def test_pretrained_model():
    print("=" * 70)
    print("🧪 预训练模型基准测试 - 亲属关系任务")
    print("=" * 70)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n📱 使用设备: {device}")
    
    model_path = "/Users/xry/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct"
    
    print("\n🔄 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("🔄 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True
    ).to(device)
    
    model.eval()
    print("✅ 模型加载完成\n")
    
    print("📊 加载测试数据...")
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
    
    for i, (query, true_answer) in enumerate(zip(test_queries[:50], test_answers[:50])):
        conversation = [
            {"role": "user", "content": query}
        ]
        
        try:
            prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            prompt = f"User: {query}\nAssistant:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        response = response.strip()
        
        is_correct = response == true_answer.strip()
        if is_correct:
            correct += 1
        else:
            wrong += 1
        
        results.append({
            'query': query,
            'true_answer': true_answer,
            'model_answer': response,
            'correct': is_correct
        })
        
        if (i + 1) % 10 == 0:
            print(f"进度: {i+1}/50, 正确: {correct}, 错误: {wrong}")
    
    print("\n" + "=" * 70)
    print("📊 测试结果")
    print("=" * 70)
    
    accuracy = correct / 50 * 100
    print(f"\n🎯 测试准确率: {accuracy:.1f}% ({correct}/50)")
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
    
    print("\n" + "=" * 70)
    print("📈 错误分析")
    print("=" * 70)
    
    wrong_results = [r for r in results if not r['correct']]
    if wrong_results:
        print(f"\n错误类型统计:")
        
        wrong_types = {}
        for r in wrong_results:
            query_type = r['query'].split()[0] if r['query'].split() else "Unknown"
            if query_type not in wrong_types:
                wrong_types[query_type] = 0
            wrong_types[query_type] += 1
        
        for qtype, count in sorted(wrong_types.items(), key=lambda x: -x[1]):
            print(f"  {qtype}: {count} 个错误")
    
    return results

if __name__ == "__main__":
    results = test_pretrained_model()
    
    print("\n" + "=" * 70)
    print("💾 保存结果...")
    print("=" * 70)
    
    import json
    output_path = "./outputs/pretrained_test_results.json"
    os.makedirs("./outputs", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_path}")

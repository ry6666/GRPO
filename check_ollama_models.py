"""
检查 Ollama 可用的模型列表
"""

import ollama

print("=" * 60)
print("📋 Ollama 可用模型列表")
print("=" * 60)

try:
    models = ollama.list()
    print("\n已下载的模型:")
    for model in models['models']:
        print(f"  📦 {model['name']}")
    
    print("\n" + "=" * 60)
    print("💡 提示")
    print("=" * 60)
    print("如果列表为空，请先拉取模型:")
    print("  ollama pull qwen2.5")
    print("  ollama pull qwen2.5:7b")
    print("  ollama pull llama3")
    
except Exception as e:
    print(f"❌ 获取模型列表失败: {e}")
    print("\n请确保 Ollama 服务正在运行:")
    print("  ollama serve")

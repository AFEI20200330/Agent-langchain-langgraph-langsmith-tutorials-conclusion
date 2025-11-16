from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from model_provider import ModelContext, get_all_model_names

print(f"可用模型列表: {get_all_model_names()}")
print()

system_message = "你是一个专业的助手，能够回答用户的问题。"
AI_message = "你好！有什么可以帮助你的吗？"
human_message = "我想要知道当前的天气情况。"

messages = [
    {"role": "system", "content": system_message},
    {"role": "assistant", "content": AI_message},
    {"role": "user", "content": human_message}
]

def test():
    with ModelContext() as context:
        print("=== 测试模型流式输出 ===")
        try:
            chunks = []
            full_message = None
            for chunk in context.invoke(messages, stream=True):
                chunks.append(chunk)
                # 打印每个分块的内容,flush=True 确保立即打印
                print(chunk.content, end="", flush=True)
                full_message = chunk if full_message is None else full_message + chunk
            
            print("\n\n完整消息:", full_message.content)
            print("=== 测试完成 ===")
        except Exception as e:
            print(f"\n\n测试失败: {e}")

test()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from model_provider import init_model_with_fallback, with_model_fallback, get_all_model_names

# 静态API密钥配置
# 注意：在生产环境中，不建议将API密钥直接硬编码在代码中
OPENROUTER_API_KEY = "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"

@tool
def get_weather(location: str) -> str:
    """获取指定位置的天气信息"""
    # 这里只是一个示例，实际实现需要调用天气API
    return f"在{location}的天气是晴朗的"


tools =[get_weather]

# 初始化主要模型
gpt = init_model_with_fallback("openai/gpt-oss-20b:free")

# 暂时注释掉有问题的中间件
# @wrap_model_call
# def dynamic_llm_call(request, handler):
#     # 获取用户消息内容
#     user_message = request.messages[-1].content if request.messages else ""
#     message_count = len(user_message)
#     
#     # 根据消息长度选择不同的模型
#     if message_count > 1000:
#         # 使用Qwen模型处理长消息
#         return handler(request)
#     else:
#         # 使用gpt模型处理短消息
#         return handler(request)


# @wrap_tool_call
def handle_tool_error(tool_name, error):
    return f"工具 {tool_name} 执行出错: {error}"

# 创建代理，暂时不使用中间件
agent = create_agent(
    model = gpt,
    tools = tools,
    system_prompt = "你是一个智能助手，能够使用工具来回答用户的问题。"
    )

# 恢复greet函数，用于在main.py中调用
def greet():
    """显示OpenRouter配置信息"""
    print("=== OpenRouter 配置信息 ===")
    print(f"Base URL: https://openrouter.ai/api/v1")
    print(f"模型: openai/gpt-oss-20b:free 和 qwen/qwen3-4b:free")
    print(f"API密钥: {OPENROUTER_API_KEY[:5]}...{OPENROUTER_API_KEY[-5:]}")
    print("==========================")

# 使用装饰器实现全局错误处理的测试函数
@with_model_fallback
def test_llm_connection(model):
    """测试与LLM的连接"""
    print("\n=== 测试LLM连接 ===")
    try:
        # 使用简单的请求测试连接
        test_message = ["请简单介绍一下你自己"]
        response = model.invoke(test_message)
        print(f"测试请求: {test_message[0]}")
        print(f"LLM响应: {response.content[:100]}...")
        print("连接测试成功！LLM可以正常工作。")
    except Exception as e:
        print(f"连接测试失败: {e}")
    print("==========================")

# 使用装饰器实现带全局错误处理的代理调用
@with_model_fallback
def run_agent(prompt, model):
    """运行代理并返回结果"""
    # 创建临时代理实例
    temp_agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="你是一个智能助手，能够使用工具来回答用户的问题。"
    )
    return temp_agent.invoke({
        "messages": [{"role": "user", "content": prompt}]
    })

# 当直接运行该文件时，测试代理功能
if __name__ == "__main__":
    print(f"可用模型列表: {get_all_model_names()}")
    print()
    
    # 测试LLM连接
    test_llm_connection()
    
    # 测试代理功能
    print("\n=== 测试代理功能 ===")
    try:
        res = run_agent("在北京的天气怎么样？")
        print("代理执行结果:")
        print(res)
    except Exception as e:
        print(f"代理执行错误: {e}")
    print("==========================")

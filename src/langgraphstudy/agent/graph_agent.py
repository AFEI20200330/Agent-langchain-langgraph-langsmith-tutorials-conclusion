# LangGraph Graph API 实践示例
# 展示如何使用LangGraph的底层Graph API构建Agent

# 导入必要的库
import sys
import os

# 将src目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import operator
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from typing import Annotated, List, Literal, TypedDict, Optional

# 尝试导入模型提供器
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
try:
    from src.langchainstudy.model_provider import ModelContext, call_model
    MODEL_PROVIDER_AVAILABLE = True
    logging.info("成功导入模型提供器")
except ImportError as e:
    MODEL_PROVIDER_AVAILABLE = False
    logging.warning(f"无法导入模型提供器: {e}。将使用默认模型配置。")

# 如果模型提供器不可用，使用基本的OpenAI配置
from langchain_openai import ChatOpenAI

# ============================ 1. 定义状态 ============================
# 状态定义了Agent在执行过程中可以访问和修改的数据
class AgentState(TypedDict):
    # LangGraph 中的状态在代理执行过程中始终存在。
    # 该Annotated类型operator.add确保新消息追加到现有列表，而不是替换现有列表。
    messages: Annotated[List, operator.add]
    llm_calls: int
# ============================ 2. 定义工具 ============================
# 使用@tool装饰器定义工具
@tool
def search(query: str) -> str:
    """Search for information on the internet."""
    return f"Search results for '{query}': This is mock search data about {query}."

@tool
def get_weather(location: str) -> str:
    """Get weather information for a specific location."""
    return f"Weather in {location}: Sunny, 72°F with light breeze."

@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expressions. Only use numbers and +, -, *, / operators."""
    try:
        result = eval(expression)
        return f"Calculation result: {expression} = {result}"
    except Exception as e:
        raise ValueError(f"Invalid calculation: {str(e)}")

# 工具列表
tools = [search, get_weather, calculate]

# 创建工具节点
# ToolNode是LangGraph提供的预构建节点，用于执行工具调用
tool_node = ToolNode(tools)

# ============================ 3. 配置模型 ============================
# 使用模型提供器来获取模型，实现自动切换和错误恢复

# ============================ 4. 定义节点函数 ============================
# 定义模型节点函数：负责调用模型并生成响应
def llm_call(state: dict):
    """LLM decides whether to call a tool or not."""
    # 获取当前消息
    messages = state.get("messages", [])
    
    try:
        # 尝试使用模型提供器
        if MODEL_PROVIDER_AVAILABLE:
            # 使用模型提供器调用模型，自动处理错误和模型切换
            with ModelContext() as context:
                # 获取当前模型并绑定工具
                current_model = context.current_model
                model_with_tools = current_model.bind_tools(tools)
                
                # 调用模型
                response = model_with_tools.invoke(messages)
        else:
            # 使用基本的OpenAI模型
            import os
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            if not OPENAI_API_KEY:
                logging.warning("未找到OPENAI_API_KEY环境变量，使用模拟响应")
                # 模拟响应，避免API调用失败
                from langchain_core.messages import AIMessage
                response = AIMessage(content="I'd like to help you, but I need an API key to proceed.")
            else:
                # 创建基本模型
                model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                model_with_tools = model.bind_tools(tools)
                response = model_with_tools.invoke(messages)
        
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
        }
    except Exception as e:
        logging.error(f"LLM调用失败: {e}")
        # 返回模拟响应以避免整个流程崩溃
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(content=f"I encountered an error: {str(e)}")],
            "llm_calls": state.get("llm_calls", 0) + 1
        }


# ============================ 5. 定义条件边 ============================
# 条件边用于根据状态决定下一个执行的节点
def should_continue(state: AgentState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the last message contains tool calls."""
    # 获取最后一条消息
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果模型响应包含工具调用，则继续执行工具节点
    if last_message.tool_calls:
        return "tool_node"
    # 否则结束会话
    return END

# ============================ 6. 构建图 ============================
# 创建状态图实例
workflow = StateGraph(AgentState)

# 添加节点
# 模型节点：负责调用LLM生成响应
workflow.add_node("llm_call", llm_call)
# 工具节点：负责执行工具调用
workflow.add_node("tool_node", tool_node)

# 添加边
# 起始节点 -> 模型节点
workflow.add_edge(START, "llm_call")
# 模型节点 -> 条件边（根据条件决定下一个节点）
workflow.add_conditional_edges(
    "llm_call",      # 源节点
    should_continue,   # 条件函数
    {
        "tool_node": "tool_node",  # 如果条件返回"tool_node"，则执行工具节点
        END: END                     # 如果条件返回END，则结束会话
    }
)
# 工具节点 -> 模型节点（工具执行后再次调用模型）
workflow.add_edge("tool_node", "llm_call")

# 编译图
# 编译后得到一个可执行的Agent
agent = workflow.compile()

# ============================ 7. 可视化图（可选） ============================
# 可以将图可视化为PNG文件（需要安装graphviz）
# agent.get_graph().draw("agent_graph.png", format="png")

# ============================ 8. 使用Agent ============================
def run_example():
    """运行Agent示例"""
    print("=== LangGraph Graph API Agent 示例 ===\n")
    print("注意: 正确运行方式是在命令行中执行 'python graph_agent.py' 而不是在Python交互式环境中使用PowerShell命令格式")
    
    # 检查API密钥
    import os
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not MODEL_PROVIDER_AVAILABLE and not OPENAI_API_KEY:
        print("警告: 未找到API密钥！")
        print("请设置OPENAI_API_KEY或OPENROUTER_API_KEY环境变量。")
        print("例如: set OPENAI_API_KEY=your-api-key (Windows)")
        print("程序将使用模拟响应进行演示...\n")
    
    # 定义系统提示
    system_prompt = SystemMessage(content="You are a helpful assistant. Be concise and accurate in your responses.")
    
    try:
        # 使用简化的示例，避免API调用失败
        print("1. 简化演示示例:")
        # 使用直接的问题而不是需要工具调用的问题
        user_message1 = HumanMessage(content="Hello, what can you do?")
        result1 = agent.invoke({"messages": [system_prompt, user_message1]})
        # 安全地访问消息内容
        if result1['messages'] and hasattr(result1['messages'][-1], 'content'):
            print(f"Agent Response: {result1['messages'][-1].content}\n")
        else:
            print("Agent Response: (No content available)\n")
        
    except Exception as e:
        print(f"执行错误: {str(e)}")
        print("\n提示:")
        print("1. 确保在命令行中运行此脚本，而不是Python交互式环境")
        print("2. 确保API密钥正确且有足够的权限")
        print("3. 检查网络连接是否正常")
        print("4. 查看错误日志以获取更多详细信息")

if __name__ == "__main__":
    run_example()

# # ============================ 8. 使用Agent ============================
# if __name__ == "__main__":
#     print("=== LangGraph Graph API Agent 示例 ===\n")
    
#     # 定义系统提示
#     system_prompt = SystemMessage(content="You are a helpful assistant. Be concise and accurate in your responses.")
    
#     # 示例1：简单天气查询
#     print("1. 天气查询示例:")
#     user_message1 = HumanMessage(content="What's the weather in New York?")
#     result1 = agent.invoke({"messages": [system_prompt, user_message1]})
#     print(f"Agent Response: {result1['messages'][-1].content}\n")
    
#     # 示例2：使用工具链
#     print("2. 工具链示例（搜索+计算）:")
#     user_message2 = HumanMessage(content="Search for the population of India in 2023, then calculate what 5% of that number is.")
#     result2 = agent.invoke({"messages": [system_prompt, user_message2]})
#     print(f"Agent Response: {result2['messages'][-1].content}\n")
    
#     # 示例3：流式输出
#     print("3. 流式输出示例:")
#     user_message3 = HumanMessage(content="Search for the latest developments in AI and give a brief summary.")
    
#     for chunk in agent.stream({"messages": [system_prompt, user_message3]}):
#         # 检查是否有新的消息
#         if "call_model" in chunk:
#             latest_message = chunk["call_model"]["messages"][-1]
#             if latest_message.content:
#                 print(f"Agent: {latest_message.content}")
#             elif latest_message.tool_calls:
#                 print(f"\nCalling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
#         elif "tool_node" in chunk:
#             tool_result = chunk["tool_node"]["messages"][-1]
#             print(f"Tool Result: {tool_result.content}\n")
    
#     print("\n=== 示例结束 ===")
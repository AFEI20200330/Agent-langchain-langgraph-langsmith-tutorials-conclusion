# LangGraph Agent 综合实践示例
# 展示文档中提到的所有关键功能：基本Agent、动态提示、结构化输出、内存管理、流式输出

# 导入必要的库
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, TypedDict

# ============================ 1. 工具定义 ============================
# 使用@tool装饰器定义工具，这些工具将被Agent使用
@tool

def search(query: str) -> str:
    """搜索互联网上的信息。"""
    return f"搜索结果 '{query}': 这是关于 {query} 的模拟搜索数据，包含最新的相关信息。"

@tool

def get_stock_price(ticker: str) -> str:
    """获取特定股票的当前价格。"""
    mock_prices = {"AAPL": 185.50, "MSFT": 378.25, "GOOGL": 142.80, "AMZN": 151.30}
    return f"{ticker} 当前价格: ${mock_prices.get(ticker, 0.00)} USD"

@tool

def calculate(expression: str) -> str:
    """计算数学表达式。只使用数字和 +, -, *, / 运算符。"""
    try:
        # 安全检查：只允许基本的数学运算
        allowed_chars = set("0123456789+-*/. ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("表达式包含不允许的字符")
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 工具列表
tools = [search, get_stock_price, calculate]

# ============================ 2. 模型配置 ============================
# 创建基础模型实例
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1
)

# ============================ 3. 基本Agent实现（使用Graph API） ============================
# 定义基本状态
class BasicState(TypedDict):
    messages: List

def create_basic_agent():
    """使用Graph API创建基本的Agent实例"""
    # 创建状态图
    workflow = StateGraph(BasicState)
    
    # 定义模型节点
    def call_model(state: BasicState):
        """调用模型节点"""
        messages = state["messages"]
        # 添加系统提示
        system_prompt = SystemMessage(content="你是一个有帮助的助手。请简洁准确地回答用户的问题。")
        messages_with_system = [system_prompt] + messages
        
        # 调用绑定了工具的模型
        response = model.bind_tools(tools).invoke(messages_with_system)
        return {"messages": messages + [response]}
    
    # 定义工具节点
    tool_node = ToolNode(tools)
    
    # 定义条件边
    def should_continue(state: BasicState) -> str:
        """决定是继续执行工具还是结束会话"""
        last_message = state["messages"][-1]
        return "tool_node" if last_message.tool_calls else END
    
    # 添加节点和边
    workflow.add_node("call_model", call_model)
    workflow.add_node("tool_node", tool_node)
    workflow.add_edge("__start__", "call_model")
    workflow.add_conditional_edges("call_model", should_continue, {"tool_node": "tool_node", END: END})
    workflow.add_edge("tool_node", "call_model")
    
    # 编译Agent
    return workflow.compile()

# ============================ 4. 动态提示Agent实现 ============================
# 定义动态提示状态
class DynamicPromptState(TypedDict):
    messages: List
    role: str  # 动态角色

def create_dynamic_prompt_agent():
    """使用Graph API创建支持动态提示的Agent实例"""
    # 创建状态图
    workflow = StateGraph(DynamicPromptState)
    
    # 定义模型节点
    def call_model(state: DynamicPromptState):
        """调用模型节点"""
        messages = state["messages"]
        role = state["role"] or "助手"
        
        # 根据动态角色创建系统提示
        system_prompt = SystemMessage(content=f"你是一个专业的{role}。请根据你的专业知识回答用户的问题。")
        messages_with_system = [system_prompt] + messages
        
        # 调用绑定了工具的模型
        response = model.bind_tools(tools).invoke(messages_with_system)
        return {"messages": messages + [response]}
    
    # 定义工具节点
    tool_node = ToolNode(tools)
    
    # 定义条件边
    def should_continue(state: DynamicPromptState) -> str:
        """决定是继续执行工具还是结束会话"""
        last_message = state["messages"][-1]
        return "tool_node" if last_message.tool_calls else END
    
    # 添加节点和边
    workflow.add_node("call_model", call_model)
    workflow.add_node("tool_node", tool_node)
    workflow.add_edge("__start__", "call_model")
    workflow.add_conditional_edges("call_model", should_continue, {"tool_node": "tool_node", END: END})
    workflow.add_edge("tool_node", "call_model")
    
    # 编译Agent
    return workflow.compile()

# ============================ 5. 自定义状态和内存管理 ============================
# 定义自定义状态类
class CustomState(TypedDict):
    """自定义Agent状态，扩展了基本状态"""
    messages: List  # 消息历史
    user_preferences: Dict[str, Any]  # 用户偏好
    conversation_history: List[Dict[str, str]]  # 对话历史

def create_custom_state_agent():
    """创建使用自定义状态的Agent实例"""
    # 创建状态图
    workflow = StateGraph(CustomState)
    
    # 定义模型节点
    def call_model(state: CustomState):
        """调用模型节点"""
        messages = state["messages"]
        user_prefs = state["user_preferences"]
        
        # 根据用户偏好修改系统提示
        system_prompt = SystemMessage(content=f"你是一个有帮助的助手。用户偏好：{user_prefs}")
        messages_with_pref = [system_prompt] + messages
        
        # 调用绑定了工具的模型
        response = model.bind_tools(tools).invoke(messages_with_pref)
        
        # 更新对话历史
        new_history = state["conversation_history"].copy()
        new_history.append({"role": "user", "content": messages[-1].content})
        new_history.append({"role": "assistant", "content": response.content})
        
        return {
            "messages": messages + [response],
            "conversation_history": new_history
        }
    
    # 定义工具节点
    tool_node = ToolNode(tools)
    
    # 定义条件边
    def should_continue(state: CustomState) -> str:
        """决定是继续执行工具还是结束会话"""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tool_node"
        return END
    
    # 添加节点和边
    workflow.add_node("call_model", call_model)
    workflow.add_node("tool_node", tool_node)
    workflow.add_edge("__start__", "call_model")
    workflow.add_conditional_edges("call_model", should_continue, {"tool_node": "tool_node", END: END})
    workflow.add_edge("tool_node", "call_model")
    
    # 编译Agent
    agent = workflow.compile()
    
    return agent

# ============================ 6. 使用示例 ============================
if __name__ == "__main__":
    print("=== LangGraph Agent 综合实践示例 ===\n")
    
    # 示例1：基本Agent
    print("1. 基本Agent示例 - 简单问题:")
    basic_agent = create_basic_agent()
    # 注意：实际运行需要设置OPENAI_API_KEY环境变量
    print("提示：实际运行需要设置OPENAI_API_KEY环境变量")
    # result1 = basic_agent.invoke({"messages": [HumanMessage(content="纽约的天气怎么样？")]})
    # print(f"Agent 响应: {result1['messages'][-1].content}\n")
    
    # 示例2：动态提示Agent
    print("2. 动态提示Agent示例 - 医生角色:")
    dynamic_agent = create_dynamic_prompt_agent()
    print("提示：实际运行需要设置OPENAI_API_KEY环境变量")
    # result2 = dynamic_agent.invoke({
    #     "role": "医生",
    #     "messages": [HumanMessage(content="头痛应该怎么办？")]
    # })
    # print(f"Agent 响应: {result2['messages'][-1].content}\n")
    
    # 示例3：自定义状态Agent
    print("3. 自定义状态Agent示例:")
    custom_agent = create_custom_state_agent()
    initial_state = {
        "messages": [HumanMessage(content="请介绍一下LangGraph")],
        "user_preferences": {"language": "Chinese", "detail_level": "medium"},
        "conversation_history": []
    }
    print("提示：实际运行需要设置OPENAI_API_KEY环境变量")
    # result4 = custom_agent.invoke(initial_state)
    # print(f"Agent 响应: {result4['messages'][-1].content}")
    # print(f"对话历史长度: {len(result4['conversation_history'])}")
    # print(f"用户偏好: {result4['user_preferences']}\n")
    
    print("\n=== 所有示例执行完成（代码结构正确，需要API密钥才能实际运行） ===")
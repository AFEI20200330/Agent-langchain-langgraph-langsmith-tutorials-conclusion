
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
短期记忆(Short-term memory)示例

本文件演示了LangChain中短期记忆的使用方法，包括：
1. 基本的短期记忆配置
2. 自定义代理状态
3. 消息管理策略(修剪、删除、总结)
4. 访问和修改内存的方法
"""

from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
# 为 PostgresSaver 添加条件导入，避免缺少依赖时程序崩溃
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_AVAILABLE = True
except ImportError:
    PostgresSaver = None
    POSTGRES_AVAILABLE = False
from model_provider import init_model_with_fallback
from langchain.messages import RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import before_model, after_model, dynamic_prompt
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing import Any, TypedDict

# ========================== 1. 基本的短期记忆配置 ==========================

def basic_memory_example():
    """基本的短期记忆示例"""
    print("=== 1. 基本的短期记忆配置 ===")
    
    # 创建一个简单的工具
    @tool
    def get_weather(city: str) -> str:
        """获取指定城市的天气信息"""
        return f"{city}的天气晴朗，温度25°C"
    
    # 创建内存保存器
    memory_saver = InMemorySaver()
    
    # 创建代理并配置短期记忆
    agent = create_agent(
        model=init_model_with_fallback("openai/gpt-oss-20b:free"),  # 使用带回退的模型
        tools=[get_weather],
        checkpointer=memory_saver,  # 配置内存保存器
    )
    
    # 定义线程ID，用于区分不同的对话
    config = {"configurable": {"thread_id": "conversation_1"}}
    
    # 第一次对话
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好！我叫张三。"}]},
        config
    )
    print("用户: 你好！我叫张三。")
    print(f"AI: {response1['messages'][-1].content}")
    
    # 第二次对话，AI应该记得用户的名字
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "上海的天气怎么样？"}]},
        config
    )
    print("\n用户: 上海的天气怎么样？")
    print(f"AI: {response2['messages'][-1].content}")
    
    # 第三次对话，AI应该记得之前的对话内容
    response3 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        config
    )
    print("\n用户: 我叫什么名字？")
    print(f"AI: {response3['messages'][-1].content}")

# ========================== 2. 自定义代理状态 ==========================

def custom_state_example():
    """自定义代理状态示例"""
    print("\n=== 2. 自定义代理状态 ===")
    
    # 扩展AgentState，添加自定义字段
    class CustomAgentState(AgentState):
        user_id: str = Field(description="用户ID")
        preferences: dict = Field(description="用户偏好设置")
    
    # 创建工具
    @tool
    def get_user_preferences(runtime: ToolRuntime) -> str:
        """获取用户偏好设置"""
        preferences = runtime.state["preferences"]
        return f"用户偏好: {preferences}"
    
    # 创建内存保存器
    memory_saver = InMemorySaver()
    
    # 创建代理并配置自定义状态
    agent = create_agent(
        model=init_model_with_fallback("openai/gpt-oss-20b:free"),
        tools=[get_user_preferences],
        state_schema=CustomAgentState,  # 配置自定义状态
        checkpointer=memory_saver,
    )
    
    config = {"configurable": {"thread_id": "conversation_2"}}
    
    # 传入自定义状态
    response = agent.invoke(
        {
            "messages": [{"role": "user", "content": "我的偏好是什么？"}],
            "user_id": "user_123",
            "preferences": {"theme": "dark", "language": "zh-CN"}
        },
        config
    )
    
    print(f"AI: {response['messages'][-1].content}")

# ========================== 3. 消息管理策略 ==========================

def message_management_examples():
    """消息管理策略示例"""
    print("\n=== 3. 消息管理策略 ===")
    
    # 创建工具
    @tool
    def dummy_tool() -> str:
        """一个简单的工具"""
        return "工具执行成功"
    
    # --------------------- 3.1 修剪消息(Trim messages) ---------------------
    print("\n--- 3.1 修剪消息 ---")
    
    @before_model
    def trim_messages(state: AgentState, runtime) -> dict[str, Any] | None:
        """只保留最近的几条消息"""
        messages = state["messages"]
        
        if len(messages) <= 3:
            return None  # 不需要修改
        
        # 保留第一条消息和最近的3条消息
        first_msg = messages[0]
        recent_messages = messages[-3:]
        new_messages = [first_msg] + recent_messages
        
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }
    
    memory_saver = InMemorySaver()
    
    agent_trim = create_agent(
        model=init_model_with_fallback("openai/gpt-oss-20b:free"),
        tools=[dummy_tool],
        middleware=[trim_messages],
        checkpointer=memory_saver,
    )
    
    config = {"configurable": {"thread_id": "conversation_3"}}
    
    # 发送多条消息
    agent_trim.invoke({"messages": [{"role": "user", "content": "消息1"}]}, config)
    agent_trim.invoke({"messages": [{"role": "user", "content": "消息2"}]}, config)
    agent_trim.invoke({"messages": [{"role": "user", "content": "消息3"}]}, config)
    agent_trim.invoke({"messages": [{"role": "user", "content": "消息4"}]}, config)
    
    response = agent_trim.invoke(
        {"messages": [{"role": "user", "content": "我刚才发送了多少条消息？"}]},
        config
    )
    print(f"AI: {response['messages'][-1].content}")
    
    # --------------------- 3.2 删除消息(Delete messages) ---------------------
    print("\n--- 3.2 删除消息 ---")
    
    @after_model
    def delete_old_messages(state: AgentState, runtime) -> dict | None:
        """删除旧消息"""
        messages = state["messages"]
        if len(messages) > 2:
            # 删除最早的两条消息
            return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
        return None
    
    memory_saver2 = InMemorySaver()
    
    agent_delete = create_agent(
        model=init_model_with_fallback("openai/gpt-oss-20b:free"),
        tools=[],
        middleware=[delete_old_messages],
        checkpointer=memory_saver2,
    )
    
    config2 = {"configurable": {"thread_id": "conversation_4"}}
    
    agent_delete.invoke({"messages": [{"role": "user", "content": "你好！"}]}, config2)
    agent_delete.invoke({"messages": [{"role": "user", "content": "我叫李四。"}]}, config2)
    
    response = agent_delete.invoke(
        {"messages": [{"role": "user", "content": "我刚才说我叫什么名字？"}]},
        config2
    )
    print(f"AI: {response['messages'][-1].content}")
    
    # --------------------- 3.3 总结消息(Summarize messages) ---------------------
    print("\n--- 3.3 总结消息 ---")
    
    memory_saver3 = InMemorySaver()
    
    agent_summarize = create_agent(
        model=init_model_with_fallback("openai/gpt-oss-20b:free"),
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model=init_model_with_fallback("openai/gpt-oss-20b:free"),
                max_tokens_before_summary=100,  # 触发总结的令牌数
                messages_to_keep=5,  # 总结后保留的消息数
            )
        ],
        checkpointer=memory_saver3,
    )
    
    config3 = {"configurable": {"thread_id": "conversation_5"}}
    
    # 发送多条长消息
    agent_summarize.invoke(
        {"messages": [{"role": "user", "content": "我正在学习Python编程，这是一种非常流行的编程语言。"}]},
        config3
    )
    agent_summarize.invoke(
        {"messages": [{"role": "user", "content": "Python有很多优点，比如语法简洁、可读性强、库丰富等。"}]},
        config3
    )
    agent_summarize.invoke(
        {"messages": [{"role": "user", "content": "我还学习了如何使用Python进行数据分析，使用的库包括Pandas、NumPy和Matplotlib。"}]},
        config3
    )
    
    response = agent_summarize.invoke(
        {"messages": [{"role": "user", "content": "我刚才提到了哪些Python库？"}]},
        config3
    )
    print(f"AI: {response['messages'][-1].content}")

# ========================== 4. 访问和修改内存 ==========================

def memory_access_examples():
    """访问和修改内存示例"""
    print("\n=== 4. 访问和修改内存 ===")
    
    # --------------------- 4.1 在工具中读取内存 ---------------------
    print("\n--- 4.1 在工具中读取内存 ---")
    
    class UserInfoState(AgentState):
        user_id: str
    
    @tool
    def get_user_details(runtime: ToolRuntime) -> str:
        """获取用户详细信息"""
        user_id = runtime.state["user_id"]
        return f"用户ID: {user_id}, 详细信息: 假设的用户数据"
    
    agent_read = create_agent(
        model=init_model_with_fallback("openai/gpt-oss-20b:free"),
        tools=[get_user_details],
        state_schema=UserInfoState,
    )
    
    response = agent_read.invoke({
        "messages": [{"role": "user", "content": "获取我的详细信息"}],
        "user_id": "user_456"
    })
    print(f"AI: {response['messages'][-1].content}")
    
    # --------------------- 4.2 在工具中修改内存 ---------------------
    print("\n--- 4.2 在工具中修改内存 ---")
    
    class UserProfileState(AgentState):
        user_name: str = ""
        age: int = 0
    
    class UserContext(BaseModel):
        user_id: str
    
    @tool
    def update_user_profile(runtime: ToolRuntime[UserContext, UserProfileState]) -> Command:
        """更新用户资料"""
        user_id = runtime.context.user_id
        
        # 根据用户ID模拟获取用户资料
        if user_id == "user_789":
            name = "王五"
            age = 30
        else:
            name = "未知用户"
            age = 0
        
        return Command(update={
            "user_name": name,
            "age": age,
            "messages": [
                ToolMessage(
                    "成功更新用户资料",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    
    @tool
    def greet_user(runtime: ToolRuntime[UserContext, UserProfileState]) -> str:
        """问候用户"""
        user_name = runtime.state["user_name"]
        age = runtime.state["age"]
        return f"你好，{user_name}！你今年{age}岁。"
    
    agent_write = create_agent(
        model=init_model_with_fallback("openai/gpt-oss-20b:free"),
        tools=[update_user_profile, greet_user],
        state_schema=UserProfileState,
        context_schema=UserContext,
    )
    
    response = agent_write.invoke(
        {"messages": [{"role": "user", "content": "问候我并告诉我我的年龄"}]},
        context=UserContext(user_id="user_789"),
    )
    print(f"AI: {response['messages'][-1].content}")
    
    # --------------------- 4.3 动态提示 ---------------------
    print("\n--- 4.3 动态提示 ---")
    
    class AppContext(TypedDict):
        user_name: str
    
    @dynamic_prompt
    def custom_system_prompt(request):
        user_name = request.runtime.context["user_name"]
        return f"你是一个友好的助手，要称呼用户为{user_name}。"
    
    agent_dynamic = create_agent(
        model=init_model_with_fallback("openai/gpt-oss-20b:free"),
        tools=[],
        middleware=[custom_system_prompt],
        context_schema=AppContext,
    )
    
    response = agent_dynamic.invoke(
        {"messages": [{"role": "user", "content": "你好！"}]},
        context=AppContext(user_name="赵六"),
    )
    print(f"AI: {response['messages'][-1].content}")

# ========================== 5. 生产环境配置 ==========================

def production_memory_example():
    """生产环境中使用数据库作为检查点的示例"""
    print("\n=== 5. 生产环境配置 ===")
    
    if not POSTGRES_AVAILABLE:
        print("提示：需要安装 langgraph-checkpoint-postgres 才能运行此示例。")
        print("安装命令：pip install langgraph-checkpoint-postgres psycopg2-binary")
        return
    
    # 注意：在实际生产环境中使用时，需要先安装依赖：
    # pip install langgraph-checkpoint-postgres
    
    @tool
    def get_user_info(user_id: str) -> str:
        """获取用户信息"""
        return f"用户 {user_id} 的信息"
    
    # 配置数据库连接URI
    # DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
    
    # 注意：这里只是示例，实际运行需要配置正确的数据库连接
    print("生产环境配置示例：")
    print("- 使用 PostgresSaver 作为检查点")
    print("- 配置数据库连接URI")
    print("- 自动创建数据库表")
    print("- 支持多线程和持久化存储")
    
    try:
        # 实际使用代码示例（注释掉，避免运行错误）
        """
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()  # 自动创建数据库表
            agent = create_agent(
                model=init_model_with_fallback("openai/gpt-oss-20b:free"),
                tools=[get_user_info],
                checkpointer=checkpointer,
            )
            
            config = {"configurable": {"thread_id": "production_conversation_1"}}
            response = agent.invoke(
                {"messages": [{"role": "user", "content": "获取用户 user_123 的信息"}]},
                config
            )
            print(f"AI: {response['messages'][-1].content}")
        """
    except Exception as e:
        print(f"生产环境配置示例出错: {e}")
        print("请确保已配置了正确的 PostgreSQL 连接")
        print("安装命令: pip install psycopg2-binary")

# ========================== 主函数 ==========================

if __name__ == "__main__":
    # 运行所有示例
    basic_memory_example()
    custom_state_example()
    message_management_examples()
    memory_access_examples()
    production_memory_example()


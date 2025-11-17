# 导入必要的模块
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
# 注释掉需要外部数据库驱动的导入
# from langgraph.checkpoint.postgres import PostgresSaver
# from langgraph.checkpoint.mongodb import MongoDBSaver
# from langgraph.checkpoint.redis import RedisSaver
from typing import Dict, Any, Optional
from typing_extensions import TypedDict
import logging
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    """
    对话状态定义
    
    扩展了MessagesState，添加了自定义字段用于演示长期内存功能
    """
    messages: list  # 对话消息列表
    user_profile: Optional[Dict[str, Any]] = None  # 用户配置文件（长期内存示例）
    conversation_topic: Optional[str] = None  # 对话主题（短期内存示例）


def mock_llm_response(state: ConversationState) -> Dict[str, Any]:
    """
    模拟LLM响应的节点函数
    
    Args:
        state: 当前对话状态
        
    Returns:
        更新后的状态字典，包含新的消息
    """
    logger.info(f"\n处理消息: {state['messages'][-1]}")
    
    # 获取用户输入，支持字典或对象格式
    last_message = state['messages'][-1]
    if hasattr(last_message, 'content'):
        user_input = last_message.content
    else:
        user_input = last_message['content']
    
    # 模拟LLM响应
    if user_input.lower().startswith('hi') or user_input.lower().startswith('hello'):
        response = "你好！很高兴与你交流。"
        # 初始化对话主题
        if not state.get('conversation_topic'):
            return {
                "messages": [{"role": "assistant", "content": response}],
                "conversation_topic": "问候"
            }
        return {"messages": [{"role": "assistant", "content": response}]}
    
    elif "名字" in user_input or "name" in user_input.lower():
        response = "我是一个AI助手，你可以叫我LangGraph助手。"
        return {"messages": [{"role": "assistant", "content": response}]}
    
    elif "主题" in user_input and state.get('conversation_topic'):
        response = f"当前对话主题是：{state['conversation_topic']}"
        return {"messages": [{"role": "assistant", "content": response}]}
    
    elif "个人资料" in user_input and state.get('user_profile'):
        profile = state['user_profile']
        response = f"你的个人资料：{profile}"
        return {"messages": [{"role": "assistant", "content": response}]}
    
    else:
        response = f"你说：{user_input}，这是一个模拟的AI响应。"
        return {"messages": [{"role": "assistant", "content": response}]}


def create_conversation_workflow() -> StateGraph:
    """
    创建对话工作流
    
    Returns:
        配置好的 StateGraph 对象
    """
    # 创建状态图构建器
    workflow = StateGraph(ConversationState)
    
    # 添加工作流节点
    workflow.add_node("respond", mock_llm_response)
    
    # 定义工作流的执行路径
    workflow.add_edge(START, "respond")
    workflow.add_edge("respond", END)
    
    return workflow


def demonstrate_short_term_memory():
    """
    演示短期内存（线程级持久化）功能
    
    展示如何使用checkpointer实现多轮对话的上下文保持
    """
    logger.info("=" * 80)
    logger.info("演示短期内存功能")
    logger.info("=" * 80)
    
    # 创建工作流
    workflow = create_conversation_workflow()
    
    # 创建内存检查点保存器
    checkpointer = InMemorySaver()
    
    # 编译工作流时添加检查点功能
    graph = workflow.compile(checkpointer=checkpointer)
    
    # 创建线程 ID 并配置
    thread_id = f"conversation_{str(uuid.uuid4())[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    logger.info(f"使用线程 ID: {thread_id} 开始对话")
    
    # 第一轮对话
    logger.info("\n第一轮对话:")
    initial_input = {"messages": [{"role": "user", "content": "你好！"}]}
    result = graph.invoke(initial_input, config)
    # 检查消息是否是字典或对象
    last_msg = result['messages'][-1]
    if hasattr(last_msg, 'content'):
        logger.info(f"助手: {last_msg.content}")
    else:
        logger.info(f"助手: {last_msg['content']}")
    
    # 第二轮对话 - 保持上下文
    logger.info("\n第二轮对话:")
    second_input = {"messages": [{"role": "user", "content": "当前对话主题是什么？"}]}
    result = graph.invoke(second_input, config)
    logger.info(f"助手: {result['messages'][-1]['content']}")
    
    # 第三轮对话 - 继续上下文
    logger.info("\n第三轮对话:")
    third_input = {"messages": [{"role": "user", "content": "你能告诉我关于Python的信息吗？"}]}
    result = graph.invoke(third_input, config)
    logger.info(f"助手: {result['messages'][-1]['content']}")
    
    # 查看完整对话历史
    logger.info("\n完整对话历史:")
    state_history = list(graph.get_state_history(config))
    for i, state_snapshot in enumerate(reversed(state_history)):
            logger.info(f"\n检查点 {i + 1}:")
            if 'messages' in state_snapshot.values:
                for msg in state_snapshot.values['messages']:
                    # 检查消息是否是字典或对象
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        logger.info(f"  {msg.role}: {msg.content}")
                    elif isinstance(msg, dict):
                        logger.info(f"  {msg['role']}: {msg['content']}")
                    else:
                        logger.info(f"  {msg}")
            else:
                logger.info(f"  检查点状态: {state_snapshot.values}")


def demonstrate_long_term_memory_concept():
    """
    演示长期内存概念
    
    展示如何在对话中存储和使用长期内存（如用户配置文件）
    """
    logger.info("\n" + "=" * 80)
    logger.info("演示长期内存概念")
    logger.info("=" * 80)
    
    # 创建工作流
    workflow = create_conversation_workflow()
    
    # 创建内存检查点保存器
    checkpointer = InMemorySaver()
    
    # 编译工作流
    graph = workflow.compile(checkpointer=checkpointer)
    
    # 创建线程 ID 并配置
    thread_id = f"long_term_memory_{str(uuid.uuid4())[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 初始化对话，包含用户配置文件
    logger.info(f"\n使用线程 ID: {thread_id} 开始对话")
    
    # 第一轮对话 - 初始化用户配置文件
    logger.info("\n第一轮对话:")
    initial_input = {
        "messages": [{"role": "user", "content": "你好！我叫张三，是一名程序员。"}],
        "user_profile": {"name": "张三", "occupation": "程序员"}
    }
    result = graph.invoke(initial_input, config)
    logger.info(f"助手: {result['messages'][-1]['content']}")
    
    # 第二轮对话 - 获取用户配置文件
    logger.info("\n第二轮对话:")
    second_input = {"messages": [{"role": "user", "content": "我的个人资料是什么？"}]}
    result = graph.invoke(second_input, config)
    logger.info(f"助手: {result['messages'][-1]['content']}")
    
    # 查看完整状态
    logger.info("\n完整对话状态:")
    latest_state = graph.get_state(config)
    logger.info(f"  用户配置文件: {latest_state.values.get('user_profile')}")
    logger.info(f"  对话主题: {latest_state.values.get('conversation_topic')}")
    logger.info(f"  消息数量: {len(latest_state.values.get('messages', []))}")


def demonstrate_different_checkpointers():
    """
    演示不同的检查点保存器配置
    
    包括内存、Postgres、MongoDB和Redis检查点保存器的配置示例
    """
    logger.info("\n" + "=" * 80)
    logger.info("演示不同的检查点保存器配置")
    logger.info("=" * 80)
    
    # 创建工作流
    workflow = create_conversation_workflow()
    
    # 1. 内存检查点保存器（默认）
    logger.info("\n1. 内存检查点保存器 (InMemorySaver):")
    try:
        memory_checkpointer = InMemorySaver()
        memory_graph = workflow.compile(checkpointer=memory_checkpointer)
        logger.info("   ✓ 内存检查点保存器配置成功")
    except Exception as e:
        logger.error(f"   ✗ 内存检查点保存器配置失败: {e}")
    
    # 2. Postgres检查点保存器（示例配置，需要实际数据库）
    logger.info("\n2. Postgres检查点保存器 (PostgresSaver):")
    try:
        Postgre
        # 注意：这只是配置示例，需要实际的Postgres数据库和psycopg依赖
        logger.info("   ✓ Postgres检查点保存器配置示例（需要实际数据库）")
        logger.info("   安装依赖: pip install -U 'psycopg[binary,pool]' langgraph-checkpoint-postgres")
        logger.info("   配置示例: PostgresSaver.from_conn_string('postgresql://user:password@host:port/dbname')")
    except Exception as e:
        logger.error(f"   ✗ Postgres检查点保存器配置失败: {e}")
    
    # 3. MongoDB检查点保存器（示例配置，需要实际数据库）
    logger.info("\n3. MongoDB检查点保存器 (MongoDBSaver):")
    try:
        # 注意：这只是配置示例，需要实际的MongoDB数据库和pymongo依赖
        logger.info("   ✓ MongoDB检查点保存器配置示例（需要实际数据库）")
        logger.info("   安装依赖: pip install -U pymongo langgraph-checkpoint-mongodb")
        logger.info("   配置示例: MongoDBSaver.from_conn_string('mongodb://host:port')")
    except Exception as e:
        logger.error(f"   ✗ MongoDB检查点保存器配置失败: {e}")
    
    # 4. Redis检查点保存器（示例配置，需要实际数据库）
    logger.info("\n4. Redis检查点保存器 (RedisSaver):")
    try:
        # 注意：这只是配置示例，需要实际的Redis数据库
        logger.info("   ✓ Redis检查点保存器配置示例（需要实际数据库）")
        logger.info("   安装依赖: pip install -U langgraph-checkpoint-redis")
        logger.info("   配置示例: RedisSaver.from_conn_string('redis://host:port')")
    except Exception as e:
        logger.error(f"   ✗ Redis检查点保存器配置失败: {e}")


def demonstrate_messages_state_basic():
    """
    演示基本的MessagesState使用
    
    展示如何使用LangGraph内置的MessagesState
    """
    logger.info("\n" + "=" * 80)
    logger.info("演示基本的MessagesState使用")
    logger.info("=" * 80)
    
    def basic_llm_node(state: MessagesState) -> Dict[str, Any]:
        """简单的LLM节点"""
        logger.info(f"\n处理消息: {state['messages'][-1]}")
        response = f"你好！你说：{state['messages'][-1].content}"
        return {"messages": [{"role": "assistant", "content": response}]}
    
    # 创建工作流
    basic_workflow = StateGraph(MessagesState)
    basic_workflow.add_node("llm", basic_llm_node)
    basic_workflow.add_edge(START, "llm")
    basic_workflow.add_edge("llm", END)
    
    # 编译工作流
    checkpointer = InMemorySaver()
    basic_graph = basic_workflow.compile(checkpointer=checkpointer)
    
    # 执行对话
    thread_id = f"basic_messages_{str(uuid.uuid4())[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    logger.info(f"\n使用线程 ID: {thread_id} 开始对话")
    
    # 发送消息
    input_messages = {"messages": [{"role": "user", "content": "使用MessagesState的基本对话"}]}
    result = basic_graph.invoke(input_messages, config)
    
    # 获取助手响应，支持字典或对象格式
    last_response = result['messages'][-1]
    if hasattr(last_response, 'content'):
        assistant_content = last_response.content
    else:
        assistant_content = last_response['content']
    logger.info(f"助手: {assistant_content}")
    logger.info("\n完整对话历史:")
    state_history = list(basic_graph.get_state_history(config))
    for i, state_snapshot in enumerate(reversed(state_history)):
        logger.info(f"\n检查点 {i + 1}:")
        if 'messages' in state_snapshot.values:
            for msg in state_snapshot.values['messages']:
                # 检查消息是否是字典或对象
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    logger.info(f"  {msg.role}: {msg.content}")
                elif isinstance(msg, dict):
                    logger.info(f"  {msg['role']}: {msg['content']}")
                else:
                    logger.info(f"  {msg}")
        else:
            logger.info(f"  检查点状态: {state_snapshot.values}")


def demonstrate_all_memory_features():
    """
    演示LangGraph内存的所有核心功能
    
    按照官方文档的顺序演示:
    1. 短期内存功能
    2. 长期内存概念
    3. 不同的检查点保存器配置
    4. 基本的MessagesState使用
    """
    logger.info("=" * 80)
    logger.info("LangGraph Memory 完整功能演示")
    logger.info("=" * 80)
    
    try:
        # 1. 短期内存功能
        demonstrate_short_term_memory()
        
        # 2. 长期内存概念
        demonstrate_long_term_memory_concept()
        
        # 3. 不同的检查点保存器配置
        demonstrate_different_checkpointers()
        
        # 4. 基本的MessagesState使用
        demonstrate_messages_state_basic()
        
        logger.info("\n" + "=" * 80)
        logger.info("LangGraph Memory 演示全部完成!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}", exc_info=True)


if __name__ == "__main__":
    """
    主函数，执行所有内存功能演示
    """
    demonstrate_all_memory_features()
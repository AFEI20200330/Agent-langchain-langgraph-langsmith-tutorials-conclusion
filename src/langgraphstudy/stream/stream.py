# -*- coding: utf-8 -*-
"""
LangGraph Streaming 功能演示
基于 LangChain 官方文档: https://docs.langchain.com/oss/python/langgraph/streaming

本示例展示了 LangGraph 中各种流式模式的使用方法，包括：
1. 基本状态流式 (values 模式)
2. 状态更新流式 (updates 模式)
3. 多种模式组合流式
4. 中间节点输出流式
5. LLM tokens 过滤
6. 按 LLM 调用过滤
7. 按节点过滤
8. 流式传输自定义数据
9. 与任何 LLM 一起使用
10. 为特定聊天模型禁用流式传输
11. 在 Python < 3.11 中使用异步
"""

from typing import TypedDict, List, Any, Dict, Optional, Callable
from langgraph.graph import StateGraph, START, END
import logging
import asyncio

# 尝试导入不同版本的异步支持
import sys
PYTHON_VERSION = sys.version_info

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BasicState(TypedDict):
    """基本状态类型定义"""
    topic: str
    joke: str
    processed_topic: str


class MultiStepState(TypedDict):
    """
    多步骤状态类型定义
    """
    step: int
    result: str
    history: List[str]


class LLMState(TypedDict):
    """
    包含LLM交互的状态类型定义
    """
    prompt: str
    response: str
    tokens: List[str]
    llm_calls: int


class CustomDataState(TypedDict):
    """
    用于演示自定义数据流式的状态类型定义
    """
    input_data: str
    processed_data: str
    custom_stream: List[Dict[str, Any]]


# ========== 模拟LLM调用的工具函数 ==========

def mock_llm_call(prompt: str, stream_tokens: bool = True, model_name: str = "mock-llm") -> Any:
    """
    模拟LLM调用，支持流式和非流式输出
    
    Args:
        prompt: 输入提示
        stream_tokens: 是否流式输出tokens
        model_name: 模型名称
        
    Returns:
        流式输出时返回生成器，非流式输出时返回完整响应
    """
    mock_response = f"Mock response to: {prompt}"
    mock_tokens = mock_response.split()
    
    # 为特定模型禁用流式传输
    if model_name == "disable-streaming-model":
        stream_tokens = False
    
    if stream_tokens:
        # 模拟流式输出tokens
        for i, token in enumerate(mock_tokens):
            # 添加一点延迟模拟真实LLM
            import time
            time.sleep(0.1)
            yield token
    else:
        # 非流式输出
        return mock_response


# ========== 节点函数 ==========

def refine_topic(state: BasicState) -> dict:
    """
    优化话题的节点函数
    将原始话题扩展，添加额外内容
    """
    logger.info(f"Refine topic node processing: {state}")
    return {"processed_topic": state["topic"] + " and technology"}


def generate_joke(state: BasicState) -> dict:
    """
    生成笑话的节点函数
    根据优化后的话题生成一个简单的笑话
    """
    logger.info(f"Generate joke node processing: {state}")
    return {"joke": f"Why did {state['processed_topic']} cross the road? To get to the server on the other side!"}


def step1(state: MultiStepState) -> dict:
    """
    步骤1节点函数
    """
    logger.info(f"Step 1 processing: {state}")
    return {"step": 1, "result": "Step 1 completed", "history": state["history"] + ["Step 1"]}


def step2(state: MultiStepState) -> dict:
    """
    步骤2节点函数
    """
    logger.info(f"Step 2 processing: {state}")
    return {"step": 2, "result": "Step 2 completed", "history": state["history"] + ["Step 2"]}


def step3(state: MultiStepState) -> dict:
    """
    步骤3节点函数
    """
    logger.info(f"Step 3 processing: {state}")
    return {"step": 3, "result": "Step 3 completed", "history": state["history"] + ["Step 3"]}


def call_llm_node(state: LLMState) -> dict:
    """
    调用LLM的节点函数
    演示如何在工作流中集成LLM并处理tokens
    """
    logger.info(f"Call LLM node processing: {state}")
    
    # 获取当前LLM调用次数，初始化为0
    llm_calls = state.get("llm_calls", 0) + 1
    
    # 模拟LLM调用，获取tokens
    tokens = []
    for token in mock_llm_call(state["prompt"]):
        tokens.append(token)
    
    response = " ".join(tokens)
    
    return {
        "response": response,
        "tokens": tokens,
        "llm_calls": llm_calls
    }


def call_specific_model_node(state: LLMState) -> dict:
    """
    调用特定LLM模型的节点函数
    演示如何为特定模型禁用流式传输
    """
    logger.info(f"Call specific model node processing: {state}")
    
    # 获取当前LLM调用次数，初始化为0
    llm_calls = state.get("llm_calls", 0) + 1
    
    # 模拟调用禁用流式传输的模型
    result = mock_llm_call(state["prompt"], model_name="disable-streaming-model")
    
    # 非流式输出直接返回结果
    if isinstance(result, str):
        response = result
        tokens = response.split()
    else:
        # 流式输出收集tokens
        tokens = list(result)
        response = " ".join(tokens)
    
    return {
        "response": response,
        "tokens": tokens,
        "llm_calls": llm_calls
    }


def custom_data_process_node(state: CustomDataState) -> dict:
    """
    自定义数据处理节点函数
    演示如何发送自定义流数据
    """
    logger.info(f"Custom data process node processing: {state}")
    
    input_data = state["input_data"]
    processed_data = input_data + " - processed"
    
    # 生成自定义流数据
    custom_stream = []
    for i in range(3):
        import time
        time.sleep(0.2)
        custom_stream.append({
            "progress": ((i + 1) / 3) * 100,
            "step": f"Step {i + 1}/3",
            "timestamp": time.time()
        })
    
    return {
        "processed_data": processed_data,
        "custom_stream": custom_stream
    }


# ========== 工作流构建函数 ==========

def create_basic_graph() -> Any:
    """
    创建基本工作流图
    用于演示 values 和 updates 流式模式
    """
    graph = StateGraph(BasicState)
    graph.add_node("refine_topic", refine_topic)
    graph.add_node("generate_joke", generate_joke)
    graph.add_edge(START, "refine_topic")
    graph.add_edge("refine_topic", "generate_joke")
    graph.add_edge("generate_joke", END)
    return graph.compile()


def create_multi_step_graph() -> Any:
    """
    创建多步骤工作流图
    用于演示中间节点输出流式
    """
    graph = StateGraph(MultiStepState)
    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_node("step3", step3)
    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)
    return graph.compile()


def create_llm_graph() -> Any:
    """
    创建包含LLM调用的工作流图
    用于演示LLM tokens过滤和按LLM调用过滤
    """
    graph = StateGraph(LLMState)
    graph.add_node("call_llm", call_llm_node)
    graph.add_node("call_specific_model", call_specific_model_node)
    graph.add_edge(START, "call_llm")
    graph.add_edge("call_llm", "call_specific_model")
    graph.add_edge("call_specific_model", END)
    return graph.compile()


def create_custom_data_graph() -> Any:
    """
    创建自定义数据处理工作流图
    用于演示自定义数据流式
    """
    graph = StateGraph(CustomDataState)
    graph.add_node("custom_process", custom_data_process_node)
    graph.add_edge(START, "custom_process")
    graph.add_edge("custom_process", END)
    return graph.compile()


# ========== 过滤函数 ==========

def filter_by_llm_calls(chunk: Any, llm_call_threshold: int = 1) -> bool:
    """
    按LLM调用次数过滤流数据
    
    Args:
        chunk: 流数据块
        llm_call_threshold: LLM调用次数阈值
        
    Returns:
        是否保留该数据块
    """
    if isinstance(chunk, dict):
        for node_name, node_data in chunk.items():
            if isinstance(node_data, dict) and "llm_calls" in node_data:
                return node_data["llm_calls"] >= llm_call_threshold
    return True

def filter_by_node(chunk: Any, allowed_nodes: List[str]) -> bool:
    """
    按节点名称过滤流数据
    
    Args:
        chunk: 流数据块
        allowed_nodes: 允许通过的节点名称列表
        
    Returns:
        是否保留该数据块
    """
    if isinstance(chunk, dict):
        for node_name in chunk.keys():
            if node_name in allowed_nodes:
                return True
    return False

def filter_llm_tokens(chunk: Any, token_filter: Callable[[str], bool] = None) -> Any:
    """
    过滤LLM tokens
    
    Args:
        chunk: 流数据块
        token_filter: 过滤函数，返回True的token将被保留
        
    Returns:
        过滤后的数据块
    """
    if not isinstance(chunk, dict):
        return chunk
    
    filtered_chunk = chunk.copy()
    for node_name, node_data in filtered_chunk.items():
        if isinstance(node_data, dict) and "tokens" in node_data:
            if token_filter:
                filtered_tokens = [token for token in node_data["tokens"] if token_filter(token)]
                node_data["tokens"] = filtered_tokens
    
    return filtered_chunk


# ========== 异步支持函数 ==========

async def async_stream_wrapper(graph: Any, input_data: Any, stream_mode: Any = "updates"):
    """
    异步流包装器，兼容Python < 3.11
    
    Args:
        graph: 编译后的工作流图
        input_data: 输入数据
        stream_mode: 流式模式
        
    Yields:
        流数据块
    """
    if PYTHON_VERSION >= (3, 11):
        # Python 3.11+ 支持async for
        async for chunk in graph.astream(input_data, stream_mode=stream_mode):
            yield chunk
    else:
        # Python < 3.11 兼容性处理
        # 使用同步stream并包装为异步生成器
        loop = asyncio.get_event_loop()
        
        def sync_stream():
            return list(graph.stream(input_data, stream_mode=stream_mode))
        
        chunks = await loop.run_in_executor(None, sync_stream)
        for chunk in chunks:
            yield chunk


# ========== 演示函数 ==========

def demonstrate_basic_streaming():
    """
    演示基本的流式功能
    包括 values 和 updates 两种模式
    """
    logger.info("\n===== 演示基本流式功能 =====")
    graph = create_basic_graph()
    
    # 演示 updates 模式
    logger.info("\n1. 使用 updates 模式:")
    for chunk in graph.stream(
        {"topic": "programming", "processed_topic": ""},
        stream_mode="updates"
    ):
        logger.info(f"   更新: {chunk}")
    
    # 演示 values 模式
    logger.info("\n2. 使用 values 模式:")
    for chunk in graph.stream(
        {"topic": "programming", "processed_topic": ""},
        stream_mode="values"
    ):
        logger.info(f"   完整状态: {chunk}")


def demonstrate_multiple_stream_modes():
    """
    演示多种流式模式组合
    """
    logger.info("\n===== 演示多种流式模式组合 =====")
    graph = create_basic_graph()
    
    for mode, chunk in graph.stream(
        {"topic": "programming", "processed_topic": ""},
        stream_mode=["updates", "values"]
    ):
        logger.info(f"   模式: {mode}, 数据: {chunk}")


def demonstrate_multi_step_streaming():
    """
    演示多步骤流式功能
    展示中间节点的输出
    """
    logger.info("\n===== 演示多步骤流式功能 =====")
    graph = create_multi_step_graph()
    
    logger.info("\n使用 updates 模式查看中间节点输出:")
    for chunk in graph.stream(
        {"step": 0, "result": "", "history": []},
        stream_mode="updates"
    ):
        logger.info(f"   更新: {chunk}")
    
    logger.info("\n使用 values 模式查看完整状态变化:")
    for chunk in graph.stream(
        {"step": 0, "result": "", "history": []},
        stream_mode="values"
    ):
        logger.info(f"   完整状态: {chunk}")


def demonstrate_llm_streaming():
    """
    演示LLM tokens过滤和按LLM调用过滤
    """
    logger.info("\n===== 演示LLM流式功能 =====")
    graph = create_llm_graph()
    
    logger.info("\n1. 基本LLM流式输出:")
    for chunk in graph.stream(
        {"prompt": "Hello LLM", "response": "", "tokens": [], "llm_calls": 0},
        stream_mode="updates"
    ):
        logger.info(f"   更新: {chunk}")
    
    logger.info("\n2. 按LLM调用次数过滤 (仅显示第2次调用):")
    for chunk in graph.stream(
        {"prompt": "Hello LLM", "response": "", "tokens": [], "llm_calls": 0},
        stream_mode="updates"
    ):
        if filter_by_llm_calls(chunk, llm_call_threshold=2):
            logger.info(f"   过滤后更新: {chunk}")
    
    logger.info("\n3. LLM tokens过滤 (仅保留长度>4的token):")
    for chunk in graph.stream(
        {"prompt": "Hello LLM", "response": "", "tokens": [], "llm_calls": 0},
        stream_mode="updates"
    ):
        filtered_chunk = filter_llm_tokens(chunk, token_filter=lambda x: len(x) > 4)
        logger.info(f"   过滤后tokens: {filtered_chunk}")


def demonstrate_node_filtering():
    """
    演示按节点过滤
    """
    logger.info("\n===== 演示按节点过滤 =====")
    graph = create_multi_step_graph()
    
    logger.info("\n仅显示step2和step3节点的输出:")
    for chunk in graph.stream(
        {"step": 0, "result": "", "history": []},
        stream_mode="updates"
    ):
        if filter_by_node(chunk, allowed_nodes=["step2", "step3"]):
            logger.info(f"   过滤后更新: {chunk}")


def demonstrate_custom_data_streaming():
    """
    演示自定义数据流式
    """
    logger.info("\n===== 演示自定义数据流式 =====")
    graph = create_custom_data_graph()
    
    logger.info("\n自定义数据流式输出:")
    for chunk in graph.stream(
        {"input_data": "test data", "processed_data": "", "custom_stream": []},
        stream_mode="updates"
    ):
        logger.info(f"   自定义数据更新: {chunk}")


def demonstrate_specific_model_streaming():
    """
    演示为特定聊天模型禁用流式传输
    """
    logger.info("\n===== 演示为特定模型禁用流式传输 =====")
    graph = create_llm_graph()
    
    logger.info("\n调用禁用流式传输的模型:")
    for chunk in graph.stream(
        {"prompt": "Hello specific model", "response": "", "tokens": [], "llm_calls": 0},
        stream_mode="updates"
    ):
        if "call_specific_model" in chunk:
            logger.info(f"   特定模型输出: {chunk}")


async def demonstrate_async_streaming():
    """
    演示异步流式功能，兼容Python < 3.11
    """
    logger.info("\n===== 演示异步流式功能 =====")
    graph = create_basic_graph()
    
    logger.info(f"\nPython版本: {PYTHON_VERSION.major}.{PYTHON_VERSION.minor}.{PYTHON_VERSION.micro}")
    logger.info("异步流式输出:")
    
    async for chunk in async_stream_wrapper(
        graph, 
        {"topic": "async programming", "processed_topic": ""}, 
        stream_mode="updates"
    ):
        logger.info(f"   异步更新: {chunk}")


# ========== 主函数 ==========
if __name__ == "__main__":
    try:
        logger.info("开始 LangGraph Streaming 功能演示")
        
        # 演示各种流式功能
        demonstrate_basic_streaming()
        demonstrate_multiple_stream_modes()
        demonstrate_multi_step_streaming()
        demonstrate_llm_streaming()
        demonstrate_node_filtering()
        demonstrate_custom_data_streaming()
        demonstrate_specific_model_streaming()
        
        # 演示异步流式功能
        asyncio.run(demonstrate_async_streaming())
        
        logger.info("\n所有演示完成!")
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}", exc_info=True)
# 导入必要的模块
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import Checkpoint
from langchain_core.runnables import RunnableConfig
from typing import Annotated, Dict, Any, Optional
from typing_extensions import TypedDict
from operator import add
import logging
import uuid
import random
import time
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class State(TypedDict):
    """
    工作流状态定义
    
    这是一个类型化字典，定义了工作流中传递的数据结构。
    包含两个字段：
    - foo: 存储单个字符串值
    - bar: 使用 Annotated 和 add 操作符进行累加的字符串列表
    """
    foo: str
    bar: Annotated[list[str], add]


def node_a(state: State) -> Dict[str, Any]:
    """
    工作流中的第一个节点
    
    Args:
        state: 当前工作流的状态
        
    Returns:
        更新后的状态字典，包含新的 foo 值和添加到 bar 列表的元素
    """
    logger.info(f"执行节点 A，收到状态: {state}")
    return {"foo": "a", "bar": ["a"]}


def node_b(state: State) -> Dict[str, Any]:
    """
    工作流中的第二个节点
    
    Args:
        state: 当前工作流的状态（已包含 node_a 的更新）
        
    Returns:
        更新后的状态字典，包含新的 foo 值和添加到 bar 列表的元素
    """
    logger.info(f"执行节点 B，收到状态: {state}")
    return {"foo": "b", "bar": ["b"]}


def create_workflow() -> StateGraph:
    """
    创建并配置工作流
    
    Returns:
        配置好的 StateGraph 对象
    """
    # 创建状态图构建器
    workflow = StateGraph(State)
    
    # 添加工作流节点
    workflow.add_node("node_a", node_a)  # 添加节点 A
    workflow.add_node("node_b", node_b)  # 添加节点 B
    
    # 定义工作流的执行路径
    workflow.add_edge(START, "node_a")    # 从开始到节点 A
    workflow.add_edge("node_a", "node_b")  # 从节点 A 到节点 B
    workflow.add_edge("node_b", END)     # 从节点 B 到结束
    
    return workflow


def demonstrate_persistence():
    """
    演示 LangGraph 的持久化功能
    
    包括：
    1. 创建带有检查点的工作流
    2. 执行工作流并保存状态
    3. 获取最新状态
    4. 获取状态历史
    5. 创建新线程执行不同的工作流
    """
    logger.info("开始演示 LangGraph 持久化功能...")
    
    # 1. 创建工作流
    workflow = create_workflow()
    
    # 2. 创建内存检查点保存器
    # InMemorySaver 是一个简单的内存检查点实现，适用于演示
    # 实际生产环境可能使用更持久的存储后端
    checkpointer = InMemorySaver()
    
    # 3. 编译工作流时添加检查点功能
    # 编译后的工作流会在每个步骤保存状态
    graph = workflow.compile(checkpointer=checkpointer)
    
    # 4. 创建线程 ID 并配置
    # 线程 ID 是一个唯一标识符，用于跟踪和恢复特定的工作流执行
    thread_id = str(uuid.uuid4())  # 使用 UUID 生成唯一的线程 ID
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    
    logger.info(f"使用线程 ID: {thread_id} 执行工作流")
    
    # 5. 执行工作流，初始状态为 {"foo": ""}
    # bar 字段会自动初始化为空列表 []
    initial_input = {"foo": ""}
    logger.info(f"输入初始状态: {initial_input}")
    result = graph.invoke(initial_input, config)
    logger.info(f"工作流执行结果: {result}")
    
    # 6. 获取最新状态
    # 使用相同的配置获取最新保存的状态
    logger.info("\n获取最新状态:")
    latest_state = graph.get_state(config)
    logger.info(f"最新状态值: {latest_state.values}")
    logger.info(f"最新状态配置: {latest_state.config}")
    logger.info(f"最新状态元数据: {latest_state.metadata}")
    
    # 7. 获取状态历史
    # 工作流执行过程中创建的所有检查点
    logger.info("\n获取状态历史:")
    state_history = list(graph.get_state_history(config))
    logger.info(f"总检查点数量: {len(state_history)}")
    
    # 打印每个检查点的信息
    for i, state_snapshot in enumerate(reversed(state_history)):  # 按时间顺序打印
        logger.info(f"\n检查点 {i + 1}:")
        logger.info(f"  状态值: {state_snapshot.values}")
        logger.info(f"  下一步执行节点: {state_snapshot.next}")
        logger.info(f"  元数据: {state_snapshot.metadata}")
    
    # 8. 演示新线程
    # 创建一个新的线程 ID 来执行相同的工作流但保持独立的状态
    logger.info("\n演示使用新线程:")
    new_thread_id = str(uuid.uuid4())
    new_config = {"configurable": {"thread_id": new_thread_id}}
    logger.info(f"使用新线程 ID: {new_thread_id} 执行工作流")
    
    # 使用不同的初始输入
    new_initial_input = {"foo": "initial"}
    new_result = graph.invoke(new_initial_input, new_config)
    logger.info(f"新线程工作流执行结果: {new_result}")
    
    # 验证两个线程的状态是独立的
    logger.info("\n验证线程状态独立性:")
    original_state = graph.get_state(config)
    new_thread_state = graph.get_state(new_config)
    logger.info(f"原始线程最终状态: {original_state.values}")
    logger.info(f"新线程最终状态: {new_thread_state.values}")
    
    # 9. 从特定检查点恢复
    # 获取第一个检查点 ID 并尝试从该点恢复
    if state_history:
        first_checkpoint_id = state_history[-1].config["configurable"]["checkpoint_id"]
        logger.info(f"\n从第一个检查点恢复: {first_checkpoint_id}")
        
        # 创建一个指向特定检查点的配置
        restore_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": first_checkpoint_id
            }
        }
        
        # 获取该检查点的状态
        restored_state = graph.get_state(restore_config)
        logger.info(f"恢复的状态: {restored_state.values}")
        logger.info(f"恢复点的下一步执行节点: {restored_state.next}")


def demonstrate_durability_modes():
    """
    演示 LangGraph 的耐久性模式
    
    包括：
    1. exit: 仅在工作流执行完成或出错时保存状态
    2. async: 异步保存状态，提供较好的性能和耐用性
    3. sync: 同步保存状态，提供最高的耐用性但性能稍低
    """
    logger.info("\n\n演示耐久性模式...")
    
    # 创建工作流和检查点
    workflow = create_workflow()
    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    
    # 测试不同的耐久性模式
    for durability_mode in ["exit", "async", "sync"]:
        logger.info(f"\n1. 使用 {durability_mode} 模式执行工作流:")
        thread_id = f"durability_{durability_mode}_{str(uuid.uuid4())[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # 记录开始时间
        start_time = time.time()
        
        # 使用指定的耐久性模式执行工作流
        result = graph.invoke({"foo": "test"}, config, durability=durability_mode)
        
        # 记录结束时间
        end_time = time.time()
        
        logger.info(f"   执行结果: {result}")
        logger.info(f"   执行时间: {end_time - start_time:.6f} 秒")
        
        # 验证状态是否正确保存
        state = graph.get_state(config)
        logger.info(f"   保存的状态: {state.values}")
    
    logger.info("\n耐久性模式演示完成!")


def demonstrate_determinism():
    """
    演示 LangGraph 的确定性和一致重放
    
    展示如何通过任务确保工作流在恢复时的行为一致性
    """
    logger.info("\n\n演示确定性和一致重放...")
    
    # 定义一个包含非确定性操作的状态
    class DeterminismState(TypedDict):
        value: int
        operations: Annotated[list[str], add]
        random_values: Annotated[list[int], add]
    
    # 注意：在LangGraph中，要实现真正的确定性重放，
    # 非确定性操作需要通过LangGraph的任务机制来处理
    # 这里我们使用一个简单的示例来演示核心概念
    
    # 方式一：使用种子确保随机性的可重复性（简单但有效）
    def deterministic_node(state: DeterminismState) -> Dict[str, Any]:
        """包含确定性操作的节点（使用种子确保随机数可重复）"""
        # 从状态中获取或创建种子
        seed = state.get("seed", 42)  # 使用固定种子确保可重复性
        
        # 使用种子初始化随机数生成器
        rng = random.Random(seed)
        random_value = rng.randint(0, 100)
        
        logger.info(f"   使用种子 {seed} 生成随机值: {random_value}")
        
        # 确定性操作：递增value
        new_value = state.get("value", 0) + 1
        
        return {
            "value": new_value,
            "operations": [f"操作 {new_value}"],
            "random_values": [random_value],
            "seed": seed  # 保持种子不变以确保可重复性
        }
    
    # 创建工作流
    workflow = StateGraph(DeterminismState)
    workflow.add_node("deterministic_node", deterministic_node)
    workflow.add_edge(START, "deterministic_node")
    workflow.add_edge("deterministic_node", END)
    
    # 编译工作流
    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    
    # 执行工作流
    thread_id = "determinism_demo"
    config = {"configurable": {"thread_id": thread_id}}
    
    logger.info("第一次执行工作流:")
    result1 = graph.invoke({"value": 0, "operations": [], "random_values": [], "seed": 42}, config)
    logger.info(f"结果: {result1}")
    
    # 方式二：LangGraph的任务机制（更高级的方式）
    logger.info("\n\n方式二：演示LangGraph的任务机制（核心概念）")
    logger.info("在实际应用中，您应该将非确定性操作包装在LangGraph的任务中：")
    logger.info("1. 定义任务函数来执行非确定性操作")
    logger.info("2. 在节点中调用这些任务")
    logger.info("3. LangGraph会自动确保任务结果的确定性重放")
    
    # 从检查点恢复工作流（使用种子方式）
    logger.info("\n从检查点恢复工作流:")
    # 获取第一个检查点
    history = list(graph.get_state_history(config))
    if history and len(history) >= 2:
        # 获取执行前的检查点
        checkpoint_before_exec = history[-2]  # 倒数第二个检查点
        restore_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_before_exec.config["configurable"]["checkpoint_id"]
            }
        }
        
        # 从检查点恢复执行
        result2 = graph.invoke(None, restore_config)  # 使用None表示继续执行
        logger.info(f"恢复后的结果: {result2}")
        
        # 比较结果，验证确定性
        if result1["random_values"] == result2["random_values"]:
            logger.info("✓ 成功: 恢复后的随机值与原始执行相同")
        else:
            logger.info("✗ 失败: 恢复后的随机值与原始执行不同")
    
    logger.info("\n确定性和一致重放演示完成!")
    logger.info("\n注意事项：")
    logger.info("1. 要实现真正的确定性，所有非确定性操作必须可重复")
    logger.info("2. 使用固定种子的随机数生成器是一种简单有效的方法")
    logger.info("3. 对于更复杂的场景，应使用LangGraph的任务机制")
    logger.info("4. 外部API调用等操作需要特殊处理来确保确定性")


def demonstrate_workflow_resumption():
    """
    演示工作流恢复功能
    
    包括：
    1. 中断工作流执行
    2. 从检查点恢复工作流
    3. 不同的恢复起点
    """
    logger.info("\n\n演示工作流恢复功能...")
    
    # 定义一个包含多个节点的状态
    class ResumeState(TypedDict):
        value: str
        steps: Annotated[list[str], add]
    
    # 定义多个节点
    def step1(state: ResumeState) -> Dict[str, Any]:
        """第一步"""
        logger.info("   执行步骤 1")
        return {"value": "step1", "steps": ["step1"]}
    
    def step2(state: ResumeState) -> Dict[str, Any]:
        """第二步"""
        logger.info("   执行步骤 2")
        return {"value": "step2", "steps": ["step2"]}
    
    def step3(state: ResumeState) -> Dict[str, Any]:
        """第三步"""
        logger.info("   执行步骤 3")
        return {"value": "step3", "steps": ["step3"]}
    
    # 创建工作流
    workflow = StateGraph(ResumeState)
    workflow.add_node("step1", step1)
    workflow.add_node("step2", step2)
    workflow.add_node("step3", step3)
    
    workflow.add_edge(START, "step1")
    workflow.add_edge("step1", "step2")
    workflow.add_edge("step2", "step3")
    workflow.add_edge("step3", END)
    
    # 编译工作流
    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    
    # 执行工作流的前两步
    thread_id = "resume_demo"
    config = {"configurable": {"thread_id": thread_id}}
    
    logger.info("1. 执行工作流的前两步:")
    # 首先执行到 step1
    # 我们需要手动控制执行以模拟中断
    for i, chunk in enumerate(graph.stream({"value": "init", "steps": []}, config)):
        logger.info(f"   执行块 {i+1}: {chunk}")
        # 只执行两步后中断
        if i == 1:  # 已经执行了 step1 和 step2
            break
    
    # 获取当前状态
    current_state = graph.get_state(config)
    logger.info(f"   中断后状态: {current_state.values}")
    logger.info(f"   下一步节点: {current_state.next}")
    
    # 从当前状态恢复执行
    logger.info("\n2. 从当前状态恢复执行:")
    result = graph.invoke(None, config)  # 使用 None 表示继续执行
    logger.info(f"   恢复后的结果: {result}")
    
    # 从特定检查点恢复
    logger.info("\n3. 从特定检查点恢复:")
    history = list(graph.get_state_history(config))
    if len(history) >= 3:
        # 从 step1 之后的检查点恢复
        checkpoint_after_step1 = history[-2]  # 倒数第二个检查点（step1 完成后）
        restore_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_after_step1.config["configurable"]["checkpoint_id"]
            }
        }
        
        logger.info(f"   从 step1 完成后的检查点恢复: {checkpoint_after_step1.config['configurable']['checkpoint_id']}")
        result_from_step1 = graph.invoke(None, restore_config)
        logger.info(f"   恢复后的结果: {result_from_step1}")
    
    logger.info("\n工作流恢复功能演示完成!")


def demonstrate_persistence_features():
    """
    演示持久化的高级特性
    
    包括：
    1. 状态累加（reducer 功能）
    2. 检查点元数据分析
    3. 动态线程管理
    """
    logger.info("\n\n演示持久化高级特性...")
    
    # 创建工作流和检查点
    workflow = create_workflow()
    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    
    # 创建线程并执行
    thread_id = "feature_demo_thread"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 执行工作流
    graph.invoke({"foo": "start"}, config)
    
    # 1. 演示状态累加（reducer）
    logger.info("\n1. 演示状态累加（reducer）功能:")
    # 因为我们在 State 中为 bar 字段使用了 Annotated[list[str], add]
    # 所以每个节点返回的 bar 值会被累加而不是替换
    final_state = graph.get_state(config)
    logger.info(f"最终的 bar 列表 (累加结果): {final_state.values['bar']}")
    logger.info("说明: 'a' 和 'b' 被累加在一起，而不是后一个替换前一个")
    
    # 2. 分析检查点元数据
    logger.info("\n2. 检查点元数据分析:")
    history = list(graph.get_state_history(config))
    for state_snapshot in reversed(history):
        if 'writes' in state_snapshot.metadata:
            node_name = next(iter(state_snapshot.metadata['writes']))
            logger.info(f"节点 '{node_name}' 写入的内容: {state_snapshot.metadata['writes'][node_name]}")
    
    # 3. 动态线程管理示例
    logger.info("\n3. 动态线程管理:")
    # 在实际应用中，您可以为每个用户会话或请求创建不同的线程
    user_threads = {
        "user1": str(uuid.uuid4()),
        "user2": str(uuid.uuid4())
    }
    
    # 为不同用户执行工作流
    for username, user_thread_id in user_threads.items():
        user_config = {"configurable": {"thread_id": user_thread_id}}
        result = graph.invoke({"foo": username}, user_config)
        logger.info(f"用户 {username} 的工作流结果: {result}")
    
    logger.info("\n持久化功能演示完成!")


def demonstrate_all_features():
    """
    演示 LangGraph 持久化的所有核心功能
    
    按照官方文档的顺序演示:
    1. 基本持久化功能
    2. 耐久性模式
    3. 确定性和一致重放
    4. 工作流恢复
    5. 高级持久化特性
    """
    logger.info("=" * 80)
    logger.info("LangGraph Persistence 完整功能演示")
    logger.info("=" * 80)
    
    try:
        # 1. 基本持久化功能
        demonstrate_persistence()
        
        # 2. 耐久性模式
        demonstrate_durability_modes()
        
        # 3. 确定性和一致重放
        demonstrate_determinism()
        
        # 4. 工作流恢复
        demonstrate_workflow_resumption()
        
        # 5. 高级持久化特性
        demonstrate_persistence_features()
        
        logger.info("\n" + "=" * 80)
        logger.info("LangGraph Persistence 演示全部完成!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}", exc_info=True)


if __name__ == "__main__":
    """
    主函数，执行所有演示
    """
    demonstrate_all_features()
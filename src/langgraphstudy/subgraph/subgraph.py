# 导入必要的模块
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import logging
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_simple_subgraph_invoke():
    """
    演示1：从节点中调用子图 - 简单示例
    """
    logger.info("\n" + "=" * 80)
    logger.info("演示1：从节点中调用子图 - 简单示例")
    logger.info("=" * 80)
    
    # 定义子图状态
    class SubgraphState(TypedDict):
        bar: str
    
    # 子图节点函数
    def subgraph_node_1(state: SubgraphState):
        logger.info(f"   子图节点执行，输入: {state['bar']}")
        return {"bar": "hi! " + state["bar"]}
    
    # 构建并编译子图
    subgraph_builder = StateGraph(SubgraphState)
    subgraph_builder.add_node("subgraph_node_1", subgraph_node_1)
    subgraph_builder.add_edge(START, "subgraph_node_1")
    subgraph = subgraph_builder.compile()
    
    # 定义父图状态
    class ParentState(TypedDict):
        foo: str
    
    # 父图节点函数 - 调用子图
    def call_subgraph(state: ParentState):
        logger.info(f"父图节点执行，输入: {state['foo']}")
        # 转换父图状态为子图状态
        subgraph_output = subgraph.invoke({"bar": state["foo"]})
        # 转换子图输出回父图状态
        logger.info(f"子图执行结果: {subgraph_output['bar']}")
        return {"foo": subgraph_output["bar"]}
    
    # 构建并编译父图
    parent_builder = StateGraph(ParentState)
    parent_builder.add_node("node_1", call_subgraph)
    parent_builder.add_edge(START, "node_1")
    parent_graph = parent_builder.compile()
    
    # 执行父图
    result = parent_graph.invoke({"foo": "world"})
    logger.info(f"父图最终结果: {result['foo']}")


def demonstrate_different_state_schemas():
    """
    演示2：不同状态模式的子图调用
    """
    logger.info("\n" + "=" * 80)
    logger.info("演示2：不同状态模式的子图调用")
    logger.info("=" * 80)
    
    # 定义子图状态（与父图状态完全不同）
    class SubgraphState(TypedDict):
        bar: str
        baz: str
    
    # 子图节点函数
    def subgraph_node_1(state: SubgraphState):
        logger.info(f"   子图节点1执行，输入: {state}")
        return {"baz": "baz"}
    
    def subgraph_node_2(state: SubgraphState):
        logger.info(f"   子图节点2执行，输入: {state}")
        return {"bar": state["bar"] + state["baz"]}
    
    # 构建并编译子图
    subgraph_builder = StateGraph(SubgraphState)
    subgraph_builder.add_node("subgraph_node_1", subgraph_node_1)
    subgraph_builder.add_node("subgraph_node_2", subgraph_node_2)
    subgraph_builder.add_edge(START, "subgraph_node_1")
    subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
    subgraph = subgraph_builder.compile()
    
    # 定义父图状态
    class ParentState(TypedDict):
        foo: str
    
    # 父图节点函数
    def node_1(state: ParentState):
        logger.info(f"父图节点1执行，输入: {state['foo']}")
        return {"foo": "hi! " + state["foo"]}
    
    def node_2(state: ParentState):
        logger.info(f"父图节点2执行，输入: {state['foo']}")
        # 转换父图状态为子图状态
        subgraph_input = {"bar": state["foo"]}
        logger.info(f"   调用子图，输入: {subgraph_input}")
        subgraph_output = subgraph.invoke(subgraph_input)
        logger.info(f"   子图执行结果: {subgraph_output['bar']}")
        # 转换子图输出回父图状态
        return {"foo": subgraph_output["bar"]}
    
    # 构建并编译父图
    parent_builder = StateGraph(ParentState)
    parent_builder.add_node("node_1", node_1)
    parent_builder.add_node("node_2", node_2)
    parent_builder.add_edge(START, "node_1")
    parent_builder.add_edge("node_1", "node_2")
    parent_graph = parent_builder.compile()
    
    # 执行父图并流式输出
    logger.info("\n执行父图并流式输出:")
    for chunk in parent_graph.stream({"foo": "foo"}, subgraphs=True):
        logger.info(f"   {chunk}")
    
    # 获取最终结果
    result = parent_graph.invoke({"foo": "foo"})
    logger.info(f"父图最终结果: {result['foo']}")


def demonstrate_multi_level_subgraphs():
    """
    演示3：多级子图调用（父图 -> 子图 -> 孙图）
    """
    logger.info("\n" + "=" * 80)
    logger.info("演示3：多级子图调用（父图 -> 子图 -> 孙图）")
    logger.info("=" * 80)
    
    # 孙图定义
    class GrandChildState(TypedDict):
        my_grandchild_key: str
    
    def grandchild_node(state: GrandChildState) -> GrandChildState:
        logger.info(f"      孙图节点执行，输入: {state['my_grandchild_key']}")
        return {"my_grandchild_key": state["my_grandchild_key"] + ", how are you?"}
    
    grandchild_builder = StateGraph(GrandChildState)
    grandchild_builder.add_node("grandchild_node", grandchild_node)
    grandchild_builder.add_edge(START, "grandchild_node")
    grandchild_builder.add_edge("grandchild_node", END)
    grandchild_graph = grandchild_builder.compile()
    
    # 子图定义
    class ChildState(TypedDict):
        my_child_key: str
    
    def call_grandchild_graph(state: ChildState) -> ChildState:
        logger.info(f"   子图节点执行，输入: {state['my_child_key']}")
        # 转换子图状态为孙图状态
        grandchild_input = {"my_grandchild_key": state["my_child_key"]}
        logger.info(f"      调用孙图，输入: {grandchild_input}")
        grandchild_output = grandchild_graph.invoke(grandchild_input)
        logger.info(f"      孙图执行结果: {grandchild_output['my_grandchild_key']}")
        # 转换孙图输出回子图状态
        return {"my_child_key": grandchild_output["my_grandchild_key"]}
    
    child_builder = StateGraph(ChildState)
    child_builder.add_node("call_grandchild", call_grandchild_graph)
    child_builder.add_edge(START, "call_grandchild")
    child_builder.add_edge("call_grandchild", END)
    child_graph = child_builder.compile()
    
    # 父图定义
    class ParentState(TypedDict):
        my_parent_key: str
    
    def call_child_graph(state: ParentState) -> ParentState:
        logger.info(f"父图节点执行，输入: {state['my_parent_key']}")
        # 转换父图状态为子图状态
        child_input = {"my_child_key": state["my_parent_key"]}
        logger.info(f"   调用子图，输入: {child_input}")
        child_output = child_graph.invoke(child_input)
        logger.info(f"   子图执行结果: {child_output['my_child_key']}")
        # 转换子图输出回父图状态
        return {"my_parent_key": child_output["my_child_key"]}
    
    parent_builder = StateGraph(ParentState)
    parent_builder.add_node("call_child", call_child_graph)
    parent_builder.add_edge(START, "call_child")
    parent_builder.add_edge("call_child", END)
    parent_graph = parent_builder.compile()
    
    # 执行父图
    logger.info("\n执行多级子图:")
    result = parent_graph.invoke({"my_parent_key": "Hello Bob"})
    logger.info(f"\n父图最终结果: {result['my_parent_key']}")


def demonstrate_subgraph_as_node():
    """
    演示4：将子图作为节点添加到父图（共享状态键）
    """
    logger.info("\n" + "=" * 80)
    logger.info("演示4：将子图作为节点添加到父图（共享状态键）")
    logger.info("=" * 80)
    
    # 定义共享状态
    class SharedState(TypedDict):
        shared_key: str
        parent_specific: str = ""
        child_specific: str = ""
    
    # 子图定义
    def child_node_1(state: SharedState) -> SharedState:
        logger.info(f"   子图节点1执行，共享状态: {state['shared_key']}")
        return {
            "shared_key": state["shared_key"] + " (子图处理1)",
            "child_specific": "子图特有数据"
        }
    
    def child_node_2(state: SharedState) -> SharedState:
        logger.info(f"   子图节点2执行，共享状态: {state['shared_key']}")
        return {
            "shared_key": state["shared_key"] + " (子图处理2)",
            "child_specific": state["child_specific"] + " - 更新"
        }
    
    # 构建并编译子图
    child_builder = StateGraph(SharedState)
    child_builder.add_node("child_node_1", child_node_1)
    child_builder.add_node("child_node_2", child_node_2)
    child_builder.add_edge(START, "child_node_1")
    child_builder.add_edge("child_node_1", "child_node_2")
    child_builder.add_edge("child_node_2", END)
    child_graph = child_builder.compile()
    
    # 父图定义
    def parent_node_1(state: SharedState) -> SharedState:
        logger.info(f"父图节点1执行，初始状态: {state['shared_key']}")
        return {
            "shared_key": state["shared_key"] + " (父图处理1)",
            "parent_specific": "父图特有数据"
        }
    
    def parent_node_2(state: SharedState) -> SharedState:
        logger.info(f"父图节点2执行，状态: {state['shared_key']}")
        return {
            "shared_key": state["shared_key"] + " (父图处理2)",
            "parent_specific": state["parent_specific"] + " - 更新"
        }
    
    # 构建并编译父图，将子图作为节点添加
    parent_builder = StateGraph(SharedState)
    parent_builder.add_node("parent_node_1", parent_node_1)
    parent_builder.add_node("child_subgraph", child_graph)  # 直接添加子图作为节点
    parent_builder.add_node("parent_node_2", parent_node_2)
    
    parent_builder.add_edge(START, "parent_node_1")
    parent_builder.add_edge("parent_node_1", "child_subgraph")
    parent_builder.add_edge("child_subgraph", "parent_node_2")
    parent_builder.add_edge("parent_node_2", END)
    
    parent_graph = parent_builder.compile()
    
    # 执行父图
    logger.info("\n执行包含子图节点的父图:")
    result = parent_graph.invoke({"shared_key": "初始值"})
    
    logger.info(f"\n父图最终结果:")
    logger.info(f"   shared_key: {result['shared_key']}")
    logger.info(f"   parent_specific: {result['parent_specific']}")
    logger.info(f"   child_specific: {result['child_specific']}")


def demonstrate_all_subgraph_features():
    """
    演示所有子图功能
    """
    logger.info("\n" + "=" * 80)
    logger.info("LangGraph 子图功能完整演示")
    logger.info("=" * 80)
    
    try:
        demonstrate_simple_subgraph_invoke()
        demonstrate_different_state_schemas()
        demonstrate_multi_level_subgraphs()
        demonstrate_subgraph_as_node()
        
        logger.info("\n" + "=" * 80)
        logger.info("LangGraph 子图功能演示全部完成!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_all_subgraph_features()
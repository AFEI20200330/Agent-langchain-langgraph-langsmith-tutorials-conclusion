

# 导入操作系统模块，用于获取环境变量
import os
# 导入日志模块，用于记录运行信息和错误
import logging

# 导入TypedDict用于定义类型化的字典结构
from typing import Literal, TypedDict, Optional
# 导入ChatOpenAI用于与OpenAI兼容的API进行交互
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
# 导入LangGraph的核心组件，用于构建状态图工作流
from langgraph.graph import START, StateGraph, END
# 导入Pydantic的BaseModel和Field用于定义结构化输出模型
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 从环境变量中获取OpenRouter API密钥
api_key = os.getenv("OPENROUTER_API_KEY")

# 初始化ChatOpenAI实例，配置模型参数
try:
    llm = ChatOpenAI(
        model = "meituan/longcat-flash-chat:free",  # 使用美团的免费模型
        base_url = "https://openrouter.ai/api/v1" if api_key else None,  # 如果有API密钥则使用OpenRouter的API地址
        api_key = api_key,  # 设置API密钥
        temperature = 0.7,  # 设置生成文本的创造性程度，0.7为中等创造性
        timeout = 30,  # 设置请求超时时间为30秒
    )
    logger.info("成功初始化ChatOpenAI模型")
except Exception as e:
    logger.error(f"初始化ChatOpenAI模型失败: {e}")
    raise

class Route(BaseModel):
    """路由决策模型，用于将用户请求路由到相应的处理节点"""
    step: Literal["poem", "story", "joke"] = Field(None, description="要路由到的步骤")

try:
    router = llm.with_structured_output(Route)
    logger.info("成功创建结构化输出路由器")
except Exception as e:
    logger.error(f"创建结构化输出路由器失败: {e}")
    raise

class State(TypedDict):
    """工作流状态字典，用于在节点间传递数据"""
    input: str  # 用户输入内容
    decision: str  # 路由决策结果
    output: str  # 最终输出结果

# 定义工作流节点函数
def llm_call_1(state: State):
    """创建故事的节点函数
    
    Args:
        state: 当前工作流状态，包含用户输入
        
    Returns:
        更新后的状态，包含生成的故事内容
    """
    try:
        logger.info("开始生成故事...")
        result = llm.invoke(state["input"])
        logger.info("成功生成故事")
        return {"output": result.content}
    except Exception as e:
        logger.error(f"生成故事失败: {e}")
        return {"output": f"生成故事失败: {str(e)}"}


def llm_call_2(state: State):
    """创建笑话的节点函数
    
    Args:
        state: 当前工作流状态，包含用户输入
        
    Returns:
        更新后的状态，包含生成的笑话内容
    """
    try:
        logger.info("开始生成笑话...")
        result = llm.invoke(state["input"])
        logger.info("成功生成笑话")
        return {"output": result.content}
    except Exception as e:
        logger.error(f"生成笑话失败: {e}")
        return {"output": f"生成笑话失败: {str(e)}"}


def llm_call_3(state: State):
    """创建诗歌的节点函数
    
    Args:
        state: 当前工作流状态，包含用户输入
        
    Returns:
        更新后的状态，包含生成的诗歌内容
    """
    try:
        logger.info("开始生成诗歌...")
        result = llm.invoke(state["input"])
        logger.info("成功生成诗歌")
        return {"output": result.content}
    except Exception as e:
        logger.error(f"生成诗歌失败: {e}")
        return {"output": f"生成诗歌失败: {str(e)}"}


def llm_call_router(state: State):
    """路由决策节点函数，将用户请求路由到相应的处理节点
    
    Args:
        state: 当前工作流状态，包含用户输入
        
    Returns:
        更新后的状态，包含路由决策结果
    """
    try:
        logger.info("开始路由决策...")
        decision = router.invoke(
            [
                SystemMessage(
                    content="You are a helpful assistant that routes user requests to the appropriate node."
                ),
                HumanMessage(
                    content=state["input"]
                )
            ]
        )
        logger.info(f"路由决策结果: {decision.step}")
        return {"decision": decision.step}
    except Exception as e:
        logger.error(f"路由决策失败: {e}")
        raise


def route_decision(state: State):
    """路由条件函数，根据决策结果选择下一个节点
    
    Args:
        state: 当前工作流状态，包含路由决策结果
        
    Returns:
        下一个要执行的节点名称
    """
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"
    else:
        logger.warning(f"未知的路由决策: {state['decision']}")
        return "llm_call_1"  # 默认选择生成故事


# 构建工作流图
try:
    logger.info("开始构建工作流图...")
    router_builder = StateGraph(State)
    
    # 添加工作流节点
    router_builder.add_node("llm_call_1", llm_call_1)  # 故事生成节点
    router_builder.add_node("llm_call_2", llm_call_2)  # 笑话生成节点
    router_builder.add_node("llm_call_3", llm_call_3)  # 诗歌生成节点
    router_builder.add_node("llm_call_router", llm_call_router)  # 路由决策节点
    
    # 添加工作流边
    router_builder.add_edge(START, "llm_call_router")  # 从开始节点到路由决策节点
    
    # 添加条件边，根据路由决策结果选择下一个节点
    router_builder.add_conditional_edges(
        "llm_call_router",
        route_decision,
        {
            "llm_call_1": "llm_call_1",
            "llm_call_2": "llm_call_2",
            "llm_call_3": "llm_call_3"
        },
    )
    
    # 从各个生成节点到结束节点
    router_builder.add_edge("llm_call_1", END)
    router_builder.add_edge("llm_call_2", END)
    router_builder.add_edge("llm_call_3", END)
    
    # 编译工作流
    router_workflow = router_builder.compile()
    logger.info("成功构建工作流图")
    
    # 尝试可视化工作流图
    logger.info("生成工作流可视化图...")
    try:
        # 生成工作流图的PNG数据
        graph_png = router_workflow.get_graph().draw_mermaid_png()
        
        # 保存图片到文件
        with open("./router_workflow.png", "wb") as f:
            f.write(graph_png)
        logger.info("工作流图已保存到: router_workflow.png")
        
        # 尝试在IPython环境中显示图片
        try:
            from IPython.display import Image, display
            display(Image(graph_png))
            logger.info("工作流图已在IPython环境中显示")
        except ImportError:
            logger.info("未安装IPython，图片已保存到文件，可在当前目录查看")
    except Exception as e:
        logger.error(f"生成工作流可视化图失败: {e}")
    
    # 执行工作流
    logger.info("开始执行工作流...")
    state = router_workflow.invoke({"input": "Write a story about a cat"})
    logger.info("工作流执行完成")
    
    # 输出结果
    print("\n" + "="*50)
    print("工作流执行结果：")
    print("="*50)
    print(state["output"])
except Exception as e:
    logger.error(f"工作流执行失败: {e}")
    print(f"执行失败: {str(e)}")




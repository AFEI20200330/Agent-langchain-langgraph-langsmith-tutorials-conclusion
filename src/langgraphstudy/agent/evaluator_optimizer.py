

# -*- coding: utf-8 -*-
"""
笑话优化工作流模块

此模块实现了一个基于LangGraph的笑话优化工作流，包含以下主要功能：
1. 笑话生成：根据主题生成笑话
2. 笑话评估：对生成的笑话进行评分（有趣/不有趣）
3. 优化循环：如果笑话不有趣，根据反馈重新生成，直到生成有趣的笑话

使用LangGraph的状态图实现循环工作流，结合结构化输出功能实现笑话评估。
"""

# 导入类型注解相关模块
from typing import Annotated, TypedDict, Literal
# 导入操作系统模块，用于获取环境变量
import os
# 导入日志模块，用于记录运行信息和错误
import logging

# 导入Pydantic的BaseModel和Field，用于定义结构化数据模型
from pydantic import BaseModel, Field
# 导入LangChain的ChatOpenAI，用于与OpenAI兼容的API交互
from langchain_openai import ChatOpenAI
# 导入LangGraph的核心组件，用于构建状态图工作流
from langgraph.graph import START, StateGraph, END

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


# 工作流状态定义
class State(TypedDict):
    """
    笑话优化工作流状态字典
    
    用于在工作流节点间传递数据，包含笑话生成和评估的所有关键信息。
    """
    joke: str  # 生成的笑话内容
    topic: str  # 笑话主题
    feedback: str  # 笑话评估反馈
    funny_or_not: str  # 笑话评估结果（"funny"或"not funny"）


# 笑话评估结果数据模型
class Feedback(BaseModel):
    """
    笑话评估结果数据模型
    
    定义了笑话评估的结构化输出格式，包括评分和反馈内容。
    用于指导LLM生成符合预期结构的评估结果。
    """
    grade: Literal["funny", "not funny"] = Field(
        description="判断笑话是否有趣。",
    )
    feedback: str = Field(
        description="如果笑话不有趣，请提供改进建议。",
    )


# 增强LLM，使其能够生成结构化的笑话评估结果
evaluator = llm.with_structured_output(Feedback)
logger.info("成功初始化结构化输出评估器")


# 定义工作流节点函数

def llm_call_generator(state: State):
    """笑话生成节点函数
    
    根据主题和反馈（如果有）生成笑话。
    如果有反馈，会根据反馈内容改进笑话；否则，直接根据主题生成笑话。
    
    Args:
        state: 当前工作流状态，包含笑话主题和可能的反馈
        
    Returns:
        更新后的状态，包含生成的笑话内容
    """
    try:
        logger.info(f"开始生成关于'{state['topic']}'的笑话...")
        if state.get("feedback"):
            logger.info(f"使用反馈改进笑话: {state['feedback']}")
            msg = llm.invoke(
                f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
            )
        else:
            msg = llm.invoke(f"Write a joke about {state['topic']}")
        logger.info("成功生成笑话")
        return {"joke": msg.content}
    except Exception as e:
        logger.error(f"生成笑话失败: {e}")
        raise


def llm_call_evaluator(state: State):
    """笑话评估节点函数
    
    对生成的笑话进行评估，判断是否有趣，并提供改进建议（如果需要）。
    
    Args:
        state: 当前工作流状态，包含要评估的笑话
        
    Returns:
        更新后的状态，包含评估结果和反馈
    """
    try:
        logger.info("开始评估笑话...")
        grade = evaluator.invoke(f"Grade the joke {state['joke']}")
        logger.info(f"笑话评估结果: {grade.grade}")
        if grade.grade == "not funny":
            logger.info(f"改进建议: {grade.feedback}")
        return {"funny_or_not": grade.grade, "feedback": grade.feedback}
    except Exception as e:
        logger.error(f"评估笑话失败: {e}")
        raise


# 条件路由函数：根据评估结果决定工作流走向
def route_joke(state: State):
    """
    根据笑话评估结果决定工作流走向
    
    如果笑话被评估为"有趣"，则结束工作流；
    如果笑话被评估为"不有趣"，则返回笑话生成节点重新生成。
    
    Args:
        state: 当前工作流状态，包含笑话评估结果
        
    Returns:
        路由决策结果，决定下一个访问的节点
    """
    logger.info(f"根据评估结果路由工作流: {state['funny_or_not']}")
    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"
    else:
        # 处理意外的评估结果
        logger.warning(f"收到意外的评估结果: {state['funny_or_not']}，默认接受")
        return "Accepted"


# 构建工作流
logger.info("开始构建笑话优化工作流...")
# 创建状态图构建器
optimizer_builder = StateGraph(State)

# 添加工作流节点
optimizer_builder.add_node("llm_call_generator", llm_call_generator)  # 笑话生成节点
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)  # 笑话评估节点

# 添加工作流边
optimizer_builder.add_edge(START, "llm_call_generator")  # 从开始节点到笑话生成节点
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")  # 从笑话生成节点到笑话评估节点

# 添加条件边：根据评估结果决定工作流走向
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",  # 源节点：笑话评估节点
    route_joke,  # 条件函数：路由决策
    {  # 路由映射：决策结果 -> 下一个节点
        "Accepted": END,  # 如果笑话有趣，结束工作流
        "Rejected + Feedback": "llm_call_generator",  # 如果笑话不有趣，返回重新生成
    },
)

# 编译工作流
optimizer_workflow = optimizer_builder.compile()
logger.info("成功构建笑话优化工作流")

# 尝试可视化工作流图
logger.info("生成工作流可视化图...")
try:
    # 生成工作流图的PNG数据
    graph_png = optimizer_workflow.get_graph().draw_mermaid_png()
    
    # 保存图片到文件
    with open("./joke_optimizer_workflow.png", "wb") as f:
        f.write(graph_png)
    logger.info("工作流图已保存到: joke_optimizer_workflow.png")
    
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
try:
    logger.info("开始执行笑话优化工作流...")
    # 输入参数：笑话主题
    input_topic = "Cats"
    logger.info(f"笑话主题: {input_topic}")
    state = optimizer_workflow.invoke({"topic": input_topic})
    logger.info("工作流执行完成")
    
    # 输出结果
    print("\n" + "="*50)
    print("最终生成的笑话：")
    print("="*50)
    print(state["joke"])
    print("\n" + "="*50)
except Exception as e:
    logger.error(f"执行工作流失败: {e}")
    print(f"执行失败: {str(e)}")

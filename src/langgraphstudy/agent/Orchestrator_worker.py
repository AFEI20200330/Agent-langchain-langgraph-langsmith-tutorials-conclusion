# -*- coding: utf-8 -*-
"""
报告生成工作流模块

此模块实现了一个基于LangGraph的报告生成工作流，包含以下主要功能：
1. 报告结构规划：生成报告的章节结构
2. 并行章节撰写：为每个章节分配独立的工作节点并行撰写
3. 报告合成：将所有章节合成为最终报告

使用LangGraph的状态图和并行处理能力，提高报告生成效率。
"""

# 导入类型注解相关模块
from typing import Annotated, List, TypedDict
# 导入操作符模块，用于列表合并操作
import operator
# 导入Pydantic的BaseModel和Field，用于定义结构化数据模型
from pydantic import BaseModel, Field
# 导入LangChain的消息类型，用于与LLM交互
from langchain.messages import SystemMessage, HumanMessage
# 导入LangChain的ChatOpenAI，用于与OpenAI兼容的API交互
from langchain_openai import ChatOpenAI
# 导入LangGraph的核心组件，用于构建状态图工作流
from langgraph.graph import START, StateGraph, END
# 导入LangGraph的Send类型，用于并行任务分配
from langgraph.types import Send
# 导入操作系统模块，用于获取环境变量
import os
# 导入日志模块，用于记录运行信息和错误
import logging

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


# 报告章节数据模型
class Section(BaseModel):
    """
    报告章节数据模型
    
    定义了报告中单个章节的结构，包括章节名称和章节描述。
    用于指导LLM生成符合预期结构的报告章节。
    """
    name: str = Field(
        description="报告章节的名称。",
    )
    description: str = Field(
        description="该章节将涵盖的主要主题和概念的简要概述。",
    )


# 报告章节集合数据模型
class Sections(BaseModel):
    """
    报告章节集合数据模型
    
    定义了报告的完整章节结构，包含多个Section对象。
    用于接收LLM生成的报告规划结果。
    """
    sections: List[Section] = Field(
        description="报告的所有章节列表。",
    )


# 增强LLM，使其能够生成结构化输出
# 使用Sections模型作为输出格式约束，确保LLM生成符合预期结构的报告章节
planner = llm.with_structured_output(Sections)

from langgraph.types import Send


# 工作流全局状态定义
class State(TypedDict):
    """
    工作流全局状态字典
    
    用于在工作流节点间传递数据，包含报告生成的所有关键信息。
    """
    topic: str  # 报告主题
    sections: list[Section]  # 报告章节列表
    completed_sections: Annotated[
        list, operator.add
    ]  # 所有工作节点并行写入的已完成章节列表
    final_report: str  # 最终生成的完整报告


# 工作节点状态定义
class WorkerState(TypedDict):
    """
    工作节点状态字典
    
    用于在工作节点间传递数据，包含单个章节的撰写信息。
    """
    section: Section  # 当前要撰写的章节信息
    completed_sections: Annotated[list, operator.add]  # 已完成章节列表（用于合并结果）


# 定义工作流节点函数

def orchestrator(state: State):
    """报告规划节点函数
    
    根据报告主题生成报告的章节结构规划。
    
    Args:
        state: 当前工作流状态，包含报告主题
        
    Returns:
        更新后的状态，包含生成的报告章节列表
    """
    try:
        logger.info(f"开始为主题'{state['topic']}'生成报告章节规划...")
        # 调用结构化输出的LLM生成报告章节规划
        report_sections = planner.invoke(
            [
                SystemMessage(content="Generate a plan for the report."),
                HumanMessage(content=f"Here is the report topic: {state['topic']}"),
            ]
        )
        logger.info(f"成功生成{len(report_sections.sections)}个报告章节")
        return {"sections": report_sections.sections}
    except Exception as e:
        logger.error(f"生成报告章节规划失败: {e}")
        raise


def llm_call(state: WorkerState):
    """章节撰写工作节点函数
    
    根据章节名称和描述生成具体的报告章节内容。
    
    Args:
        state: 当前工作节点状态，包含要撰写的章节信息
        
    Returns:
        更新后的状态，包含生成的章节内容
    """
    try:
        logger.info(f"开始撰写章节: {state['section'].name}...")
        # 调用LLM生成章节内容
        section = llm.invoke(
            [
                SystemMessage(
                    content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
                ),
                HumanMessage(
                    content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
                ),
            ]
        )
        logger.info(f"成功撰写章节: {state['section'].name}")
        # 将生成的章节内容添加到已完成章节列表
        return {"completed_sections": [section.content]}
    except Exception as e:
        logger.error(f"撰写章节'{state['section'].name}'失败: {e}")
        # 如果生成失败，返回错误信息作为章节内容
        return {"completed_sections": [f"生成章节'{state['section'].name}'失败: {str(e)}"]}


def synthesizer(state: State):
    """报告合成节点函数
    
    将所有已完成的章节合成为最终的完整报告。
    
    Args:
        state: 当前工作流状态，包含所有已完成的章节
        
    Returns:
        更新后的状态，包含最终生成的完整报告
    """
    try:
        logger.info("开始合成最终报告...")
        # 获取所有已完成的章节
        completed_sections = state["completed_sections"]
        # 使用分隔线连接所有章节，形成完整报告
        completed_report_sections = "\n\n---\n\n".join(completed_sections)
        logger.info("成功合成最终报告")
        return {"final_report": completed_report_sections}
    except Exception as e:
        logger.error(f"合成最终报告失败: {e}")
        raise


# 条件边函数：为每个章节分配工作节点
def assign_workers(state: State):
    """
    根据报告章节列表为每个章节分配独立的工作节点
    
    使用LangGraph的Send API实现并行处理，为每个章节创建一个独立的工作任务。
    
    Args:
        state: 当前工作流状态，包含报告章节列表
        
    Returns:
        Send对象列表，用于启动并行工作节点
    """
    logger.info("开始为每个章节分配工作节点...")
    # 为每个章节创建一个Send对象，启动并行工作节点
    return [Send("llm_call", {"section": s}) for s in state["sections"]]


# 构建工作流
logger.info("开始构建报告生成工作流...")
# 创建状态图构建器
orchestrator_worker_builder = StateGraph(State)

# 添加工作流节点
orchestrator_worker_builder.add_node("orchestrator", orchestrator)  # 报告规划节点
orchestrator_worker_builder.add_node("llm_call", llm_call)  # 章节撰写工作节点
orchestrator_worker_builder.add_node("synthesizer", synthesizer)  # 报告合成节点

# 添加工作流边
orchestrator_worker_builder.add_edge(START, "orchestrator")  # 从开始节点到报告规划节点

# 添加条件边：根据规划结果分配工作节点
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator",  # 源节点：报告规划节点
    assign_workers,  # 条件函数：分配工作节点
    ["llm_call"]  # 目标节点：章节撰写工作节点
)

orchestrator_worker_builder.add_edge("llm_call", "synthesizer")  # 从章节撰写节点到报告合成节点
orchestrator_worker_builder.add_edge("synthesizer", END)  # 从报告合成节点到结束节点

# 编译工作流
orchestrator_worker = orchestrator_worker_builder.compile()
logger.info("成功构建报告生成工作流")

# 尝试可视化工作流图（仅在IPython环境中可用）
try:
    # 尝试可视化工作流图
    logger.info("生成工作流可视化图...")
    try:
        # 生成工作流图的PNG数据
        graph_png = orchestrator_worker.get_graph().draw_mermaid_png()
        
        # 保存图片到文件
        with open("./orchestrator_worker_workflow.png", "wb") as f:
            f.write(graph_png)
        logger.info("工作流图已保存到: orchestrator_worker_workflow.png")
        
        # 尝试在IPython环境中显示图片
        try:
            from IPython.display import Image, display, Markdown
            display(Image(graph_png))
            logger.info("工作流图已在IPython环境中显示")
        except ImportError:
            logger.info("未安装IPython，图片已保存到文件，可在当前目录查看")
    except Exception as e:
        logger.error(f"生成工作流可视化图失败: {e}")
    
    # 执行工作流
    logger.info("开始执行报告生成工作流...")
    state = orchestrator_worker.invoke({"topic": "Create a report on LLM scaling laws"})
    logger.info("工作流执行完成")
    
    # 输出结果
    print("\n" + "="*50)
    print("最终报告结果：")
    print("="*50)
    try:
        from IPython.display import Markdown, display
        display(Markdown(state["final_report"]))
    except ImportError:
        # 在非IPython环境中直接打印报告
        print(state["final_report"])
    
    # 同时打印文本格式的报告，便于在非IPython环境中查看
    print("\n" + "="*50)
    print("文本格式报告：")
    print("="*50)
    print(state["final_report"])
except Exception as e:
    logger.error(f"执行工作流失败: {e}")
    print(f"执行失败: {str(e)}")
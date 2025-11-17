from typing import TypedDict, List, Any, Dict, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 状态类型定义 ==========

class ApprovalState(TypedDict):
    """审批工作流状态"""
    input: str
    analysis: Optional[str] = None
    approved: Optional[bool] = None
    action_taken: Optional[str] = None
    history: List[str] = []

class ReviewState(TypedDict):
    """内容审查工作流状态"""
    document: str
    generated_content: Optional[str] = None
    reviewed_content: Optional[str] = None
    approved: Optional[bool] = None
    final_content: Optional[str] = None

class ToolCallState(TypedDict):
    """工具调用工作流状态"""
    query: str
    tool_call: Optional[Dict[str, Any]] = None
    tool_response: Optional[str] = None
    executed: Optional[bool] = None

# ========== 工具函数 ==========

def mock_llm_analysis(text: str) -> str:
    """模拟LLM分析功能"""
    logger.info(f"正在分析文本: {text}")
    time.sleep(0.5)  # 模拟处理延迟
    return f"LLM分析结果: {text} 包含需要审批的内容"


def mock_tool_call(tool_call: Dict[str, Any]) -> str:
    """模拟工具调用"""
    logger.info(f"执行工具调用: {tool_call}")
    time.sleep(0.5)  # 模拟工具执行延迟
    return f"工具调用 '{tool_call['name']}' 已执行，参数: {tool_call['args']}"

# ========== 节点函数 ==========

# 1. 审批工作流节点
def analyze_input(state: ApprovalState) -> ApprovalState:
    """分析输入内容"""
    logger.info(f"Analyze input node processing: {state}")
    analysis = mock_llm_analysis(state["input"])
    history = state.get("history", []) + [f"分析完成: {analysis}"]
    return {"analysis": analysis, "history": history}

def request_approval(state: ApprovalState) -> ApprovalState:
    """请求审批"""
    logger.info(f"Request approval node processing: {state}")
    # 中断执行，等待审批
    approved = interrupt({"message": "是否批准此操作？", "analysis": state["analysis"]})
    history = state.get("history", []) + [f"审批结果: {'批准' if approved else '拒绝'}"]
    return {"approved": approved, "history": history}

def execute_action(state: ApprovalState) -> ApprovalState:
    """执行操作"""
    logger.info(f"Execute action node processing: {state}")
    if state["approved"]:
        action_taken = f"已执行操作: {state['input']}"
        history = state.get("history", []) + [action_taken]
        return {"action_taken": action_taken, "history": history}
    else:
        action_taken = f"操作被拒绝: {state['input']}"
        history = state.get("history", []) + [action_taken]
        return {"action_taken": action_taken, "history": history}

# 2. 内容审查工作流节点
def generate_content(state: ReviewState) -> ReviewState:
    """生成内容"""
    logger.info(f"Generate content node processing: {state}")
    generated_content = f"基于文档生成的内容: {state['document']}"
    return {"generated_content": generated_content}

def review_content(state: ReviewState) -> ReviewState:
    """审查内容"""
    logger.info(f"Review content node processing: {state}")
    # 中断执行，等待内容审查
    reviewed_content = interrupt({"message": "请审查并修改生成的内容", "generated_content": state["generated_content"]})
    return {"reviewed_content": reviewed_content}

def finalize_content(state: ReviewState) -> ReviewState:
    """最终确认内容"""
    logger.info(f"Finalize content node processing: {state}")
    # 中断执行，等待最终确认
    approved = interrupt({"message": "是否确认最终内容？", "content": state["reviewed_content"]})
    if approved:
        return {"approved": True, "final_content": state["reviewed_content"]}
    else:
        return {"approved": False}

# 3. 工具调用工作流节点
def plan_tool_call(state: ToolCallState) -> ToolCallState:
    """计划工具调用"""
    logger.info(f"Plan tool call node processing: {state}")
    tool_call = {
        "name": "search",
        "args": {
            "query": state["query"]
        }
    }
    return {"tool_call": tool_call}

def confirm_tool_call(state: ToolCallState) -> ToolCallState:
    """确认工具调用"""
    logger.info(f"Confirm tool call node processing: {state}")
    # 中断执行，等待工具调用确认
    executed = interrupt({"message": "是否执行此工具调用？", "tool_call": state["tool_call"]})
    return {"executed": executed}

def execute_tool_call(state: ToolCallState) -> ToolCallState:
    """执行工具调用"""
    logger.info(f"Execute tool call node processing: {state}")
    if state["executed"]:
        tool_response = mock_tool_call(state["tool_call"])
        return {"tool_response": tool_response}
    else:
        return {"tool_response": "工具调用被取消"}

# ========== 工作流构建函数 ==========

def create_approval_workflow() -> Any:
    """创建审批工作流"""
    # 创建检查点存储
    checkpointer = InMemorySaver()
    
    # 创建工作流
    workflow = StateGraph(ApprovalState)
    
    # 添加节点
    workflow.add_node("analyze", analyze_input)
    workflow.add_node("approve", request_approval)
    workflow.add_node("execute", execute_action)
    
    # 添加边
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "approve")
    workflow.add_edge("approve", "execute")
    workflow.add_edge("execute", END)
    
    # 编译工作流，启用检查点
    return workflow.compile(checkpointer=checkpointer)

def create_review_workflow() -> Any:
    """创建内容审查工作流"""
    # 创建检查点存储
    checkpointer = InMemorySaver()
    
    # 创建工作流
    workflow = StateGraph(ReviewState)
    
    # 添加节点
    workflow.add_node("generate", generate_content)
    workflow.add_node("review", review_content)
    workflow.add_node("finalize", finalize_content)
    
    # 添加边
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "review")
    workflow.add_edge("review", "finalize")
    workflow.add_edge("finalize", END)
    
    # 编译工作流，启用检查点
    return workflow.compile(checkpointer=checkpointer)

def create_tool_call_workflow() -> Any:
    """创建工具调用工作流"""
    # 创建检查点存储
    checkpointer = InMemorySaver()
    
    # 创建工作流
    workflow = StateGraph(ToolCallState)
    
    # 添加节点
    workflow.add_node("plan", plan_tool_call)
    workflow.add_node("confirm", confirm_tool_call)
    workflow.add_node("execute", execute_tool_call)
    
    # 添加边
    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "confirm")
    workflow.add_edge("confirm", "execute")
    workflow.add_edge("execute", END)
    
    # 编译工作流，启用检查点
    return workflow.compile(checkpointer=checkpointer)

# ========== 演示函数 ==========

def demo_approval_workflow():
    """演示审批工作流"""
    logger.info("\n===== 演示审批工作流 =====")
    
    # 创建工作流
    workflow = create_approval_workflow()
    
    # 配置（使用相同的thread_id来跟踪状态）
    config = {"configurable": {"thread_id": "approval-thread-1"}}
    
    # 初始运行 - 会触发中断
    logger.info("\n1. 初始运行（触发审批中断）:")
    result = workflow.invoke({"input": "执行重要操作"}, config=config)
    logger.info(f"   中断信息: {result['__interrupt__'][0].value}")
    
    # 恢复运行 - 批准
    logger.info("\n2. 恢复运行（批准操作）:")
    result = workflow.invoke(Command(resume=True), config=config)
    logger.info(f"   最终状态: {result}")

def demo_review_workflow():
    """演示内容审查工作流"""
    logger.info("\n===== 演示内容审查工作流 =====")
    
    # 创建工作流
    workflow = create_review_workflow()
    
    # 配置
    config = {"configurable": {"thread_id": "review-thread-1"}}
    
    # 初始运行 - 生成内容
    logger.info("\n1. 初始运行（生成内容）:")
    result = workflow.invoke({"document": "关于LangGraph的文档"}, config=config)
    logger.info(f"   中断信息: {result['__interrupt__'][0].value}")
    
    # 恢复运行 - 审查内容
    logger.info("\n2. 恢复运行（审查修改内容）:")
    reviewed_content = "修改后的内容：LangGraph是一个强大的工作流框架"
    result = workflow.invoke(Command(resume=reviewed_content), config=config)
    logger.info(f"   中断信息: {result['__interrupt__'][0].value}")
    
    # 恢复运行 - 最终确认
    logger.info("\n3. 恢复运行（最终确认）:")
    result = workflow.invoke(Command(resume=True), config=config)
    logger.info(f"   最终状态: {result}")

def demo_tool_call_workflow():
    """演示工具调用工作流"""
    logger.info("\n===== 演示工具调用工作流 =====")
    
    # 创建工作流
    workflow = create_tool_call_workflow()
    
    # 配置
    config = {"configurable": {"thread_id": "tool-thread-1"}}
    
    # 初始运行 - 计划工具调用
    logger.info("\n1. 初始运行（计划工具调用）:")
    result = workflow.invoke({"query": "LangChain最新版本"}, config=config)
    logger.info(f"   中断信息: {result['__interrupt__'][0].value}")
    
    # 恢复运行 - 确认并执行工具调用
    logger.info("\n2. 恢复运行（确认工具调用）:")
    result = workflow.invoke(Command(resume=True), config=config)
    logger.info(f"   最终状态: {result}")

def demo_reject_scenario():
    """演示拒绝场景"""
    logger.info("\n===== 演示拒绝场景 =====")
    
    # 创建工作流
    workflow = create_approval_workflow()
    
    # 配置
    config = {"configurable": {"thread_id": "reject-thread-1"}}
    
    # 初始运行
    logger.info("\n1. 初始运行:")
    result = workflow.invoke({"input": "执行有风险的操作"}, config=config)
    logger.info(f"   中断信息: {result['__interrupt__'][0].value}")
    
    # 恢复运行 - 拒绝
    logger.info("\n2. 恢复运行（拒绝操作）:")
    result = workflow.invoke(Command(resume=False), config=config)
    logger.info(f"   最终状态: {result}")

# ========== 主函数 ==========

if __name__ == "__main__":
    # 运行所有演示
    demo_approval_workflow()
    demo_review_workflow()
    demo_tool_call_workflow()
    demo_reject_scenario()
    
    logger.info("\n所有演示完成!")
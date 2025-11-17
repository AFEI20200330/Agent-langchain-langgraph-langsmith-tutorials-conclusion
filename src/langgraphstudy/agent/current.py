# 导入操作系统模块，用于获取环境变量
import os
# 导入TypedDict用于定义类型化的字典结构
from typing import TypedDict

# 导入IPython的显示组件，用于可视化工作流图
from IPython.display import display, Image
# 导入LangGraph的核心组件，用于构建状态图工作流
from langgraph.graph import StateGraph, START, END
# 导入ChatOpenAI用于与OpenAI兼容的API进行交互
from langchain_openai import ChatOpenAI
# 从环境变量中获取OpenRouter API密钥
api_key = os.getenv("OPENROUTER_API_KEY")

# 初始化ChatOpenAI实例，配置模型参数
llm = ChatOpenAI(
    model = "meituan/longcat-flash-chat:free",  # 使用美团的免费模型
    base_url = "https://openrouter.ai/api/v1" if api_key else None,  # 如果有API密钥则使用OpenRouter的API地址
    api_key = api_key,  # 设置API密钥
    temperature = 0.7,  # 设置生成文本的创造性程度，0.7为中等创造性
    timeout = 30,  # 设置请求超时时间为30秒
)

# 定义工作流的状态结构，使用TypedDict确保类型安全
class State(TypedDict):
    topic: str  # 生成内容的主题
    joke: str  # 生成的笑话
    story: str  # 生成的故事
    poem: str  # 生成的诗歌
    combined_output: str  # 组合后的最终输出


# 定义工作流的第一个节点函数：生成笑话
# 输入：包含主题的状态
# 输出：更新后的状态，添加生成的笑话
# 功能：使用LLM根据给定主题生成一个笑话
def call_llm_1(state: State):
    """第一次LLM调用，生成初始笑话"""
    msg = llm.invoke(f"Generate a joke about {state['topic']}")
    return {"joke": msg.content}

# 定义工作流的第二个节点函数：生成故事
# 输入：包含主题的状态
# 输出：更新后的状态，添加生成的故事
# 功能：使用LLM根据给定主题生成一个短故事
def call_llm_2(state: State):
    """第二次LLM调用，生成初始故事"""
    msg = llm.invoke(f"Write a short story about {state['topic']}")
    return {"story": msg.content}

# 定义工作流的第三个节点函数：生成诗歌
# 输入：包含主题的状态
# 输出：更新后的状态，添加生成的诗歌
# 功能：使用LLM根据给定主题生成一首短诗
def call_llm_3(state: State):
    """第三次LLM调用，生成初始诗歌"""
    msg = llm.invoke(f"Write a short poem about {state['topic']}")
    return {"poem": msg.content}

# 定义工作流的聚合节点函数：组合所有输出
# 输入：包含笑话、故事和诗歌的状态
# 输出：更新后的状态，添加组合后的输出
# 功能：将生成的笑话、故事和诗歌组合成一个字符串
def aggregator(state: State):
    """将所有生成的内容组合成一个字符串"""
    return {
        "combined_output": f"Joke: {state['joke']}\nStory: {state['story']}\nPoem: {state['poem']}"}


# 创建StateGraph实例，指定状态类型为我们定义的State
parallel_builder = StateGraph(State)

# 向工作流中添加节点
parallel_builder.add_node("llm_1", call_llm_1)  # 生成笑话的节点
parallel_builder.add_node("llm_2", call_llm_2)  # 生成故事的节点
parallel_builder.add_node("llm_3", call_llm_3)  # 生成诗歌的节点
parallel_builder.add_node("aggregator", aggregator)  # 聚合结果的节点

# 定义工作流的边
# 从START节点并行执行三个生成任务
parallel_builder.add_edge(START, "llm_1")
parallel_builder.add_edge(START, "llm_2")
parallel_builder.add_edge(START, "llm_3")

# 三个生成任务完成后都指向聚合节点
parallel_builder.add_edge("llm_1", "aggregator")
parallel_builder.add_edge("llm_2", "aggregator")
parallel_builder.add_edge("llm_3", "aggregator")

# 聚合完成后结束工作流
parallel_builder.add_edge("aggregator", END)

# 编译工作流，生成可执行的parallel_workflow
parallel_workflow = parallel_builder.compile()

# 使用Mermaid可视化工作流图并在IPython中显示
# 这有助于直观地了解工作流的结构和节点之间的连接关系
display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

# 调用工作流，传入初始主题为"man"
# 这将并行执行三个生成任务，然后聚合结果
state = parallel_workflow.invoke({"topic":"man"})

# 打印组合后的最终输出
print(state["combined_output"])

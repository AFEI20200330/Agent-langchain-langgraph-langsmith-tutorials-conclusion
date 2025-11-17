

# 导入操作系统模块，用于获取环境变量
import os
# 导入日志模块，用于记录运行信息和错误
import logging

# 导入TypedDict用于定义类型化的字典结构
from typing import TypedDict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# 导入ChatOpenAI用于与OpenAI兼容的API进行交互
from langchain_openai import ChatOpenAI
# 导入LangGraph的核心组件，用于构建状态图工作流
from langgraph.graph import START, StateGraph, END
# 导入Pydantic的BaseModel和Field用于定义结构化输出模型
from pydantic import BaseModel, Field

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

# 定义一个Pydantic模型，用于结构化LLM的输出
class SearchQuery(BaseModel):
    search_query: str = Field(None,description="优化后的网络搜索查询")
    justification: str = Field(
        None,description="该查询与用户问题相关性的说明"
    )

# 创建一个支持结构化输出的LLM实例
structured_llm = llm.with_structured_output(SearchQuery)

# 调用结构化LLM并输出结果
output = structured_llm.invoke("为什么巧克力和高卡路里有关系")
print(output)


# 定义一个简单的乘法工具函数
def multiply(a: int, b: int) -> int:
    """将两个数字相乘并返回结果"""
    return a * b

# 创建一个绑定了乘法工具的LLM实例
llm_with_tools = llm.bind_tools([multiply])

# 调用绑定工具的LLM并输出工具调用结果
msg = llm_with_tools.invoke("2乘以3等于多少")
print(msg.tool_calls)


# 定义工作流的状态结构，使用TypedDict确保类型安全
class State(TypedDict):
    topic:str  # 笑话的主题
    joke:str  # 初始生成的笑话
    improved_joke:str  # 经过改进的笑话
    final_joke:str  # 最终润色后的笑话


# 定义工作流的第一个节点函数：生成初始笑话
# 输入：包含主题的状态
# 输出：更新后的状态，添加生成的笑话
# 功能：使用LLM根据给定主题生成一个简短笑话
def generate_joke(state:State):
    """第一次LLM调用，根据主题生成初始笑话"""
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke":msg.content}

# 定义条件判断函数：检查笑话是否包含笑点
# 输入：包含笑话内容的状态
# 输出："Pass"或"Fail"，用于决定工作流的下一个节点
# 功能：简单通过检查笑话中是否包含问号或感叹号来判断是否有笑点
def check_punchline(state:State):
    """条件判断函数，检查笑话是否包含笑点"""
    # 如果笑话中包含问号或感叹号，则认为有笑点，返回Pass
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    # 否则返回Fail
    return "Fail"


# 定义工作流的第二个节点函数：改进笑话
# 输入：包含初始笑话的状态
# 输出：更新后的状态，添加改进后的笑话
# 功能：使用LLM对初始笑话进行改进，添加文字游戏使其更有趣
def improve_joke(state:State):
    """第二次LLM调用，改进笑话的趣味性"""
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke":msg.content}


# 定义工作流的第三个节点函数：润色笑话
# 输入：包含改进后笑话的状态
# 输出：更新后的状态，添加最终润色的笑话
# 功能：使用LLM为改进后的笑话添加一个令人惊讶的转折
def polish_joke(state:State):
    """第三次LLM调用，为笑话添加令人惊讶的转折"""
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke":msg.content}

# 创建StateGraph实例，指定状态类型为我们定义的State
workflow = StateGraph(State)

# 向工作流中添加节点
# "generate_joke"节点使用generate_joke函数
workflow.add_node("generate_joke",generate_joke)
# "improve_joke"节点使用improve_joke函数
workflow.add_node("improve_joke",improve_joke)
# "polish_joke"节点使用polish_joke函数
workflow.add_node("polish_joke",polish_joke)

# 定义工作流的起始边：从START节点指向generate_joke节点
workflow.add_edge(START,"generate_joke")

# 定义条件边：从generate_joke节点出发，根据check_punchline函数的返回值决定下一个节点
# 如果返回"Fail"，则进入improve_joke节点
# 如果返回"Pass"，则直接结束工作流
workflow.add_conditional_edges(
    "generate_joke",check_punchline,{"Fail":"improve_joke","Pass":END}
)

# 定义普通边：从improve_joke节点指向polish_joke节点
workflow.add_edge("improve_joke","polish_joke")
# 定义普通边：从polish_joke节点指向END节点，结束工作流
workflow.add_edge("polish_joke",END)

# 编译工作流，生成可执行的chain
chain = workflow.compile()

# 尝试可视化工作流图
logger.info("生成工作流可视化图...")
try:
    # 生成工作流图的PNG数据
    graph_png = chain.get_graph().draw_mermaid_png()
    
    # 保存图片到文件
    with open("./flow_t_workflow.png", "wb") as f:
        f.write(graph_png)
    logger.info("工作流图已保存到: flow_t_workflow.png")
    
    # 尝试在IPython环境中显示图片
    try:
        from IPython.display import Image, display
        display(Image(graph_png))
        logger.info("工作流图已在IPython环境中显示")
    except ImportError:
        logger.info("未安装IPython，图片已保存到文件，可在当前目录查看")
except Exception as e:
    logger.error(f"生成工作流可视化图失败: {e}")

# 调用工作流，传入初始主题为"man"
# 这将执行整个工作流并返回最终状态
state = chain.invoke({"topic":"man"})

# 打印初始生成的笑话
print("Initial Joke:",state["joke"])
print("==========================\n")

# 检查状态中是否包含improved_joke字段，以确定笑话是否经过了改进和润色
if "improved_joke" in state:
    # 如果笑话经过了改进，打印改进后的笑话
    print("Improved Joke:",state["improved_joke"])
    print("==========================\n")
    # 打印最终润色后的笑话
    print("Final Joke:",state["final_joke"])
    print("==========================\n")
else:
    # 如果笑话直接通过了笑点检查，打印该信息
    print("Joke passed the check:",state["joke"])
    print("==========================\n")


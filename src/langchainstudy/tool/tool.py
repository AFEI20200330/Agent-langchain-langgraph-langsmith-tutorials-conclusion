


from langchain.tools import ToolRuntime
from langchain_core.messages import RemoveMessage, tool
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command


@tool
def search_database(query: str,limit: int = 10)->str:
    # 这里的文字描述了函数的作用，以及参数的含义，这个部分是必须的，不能省略，
    # 否则在调用函数时，会提示参数缺失
    """
    search the database for records matching the query.

    Args:
        query: the search query string.
        limit: the maximum number of records to return. Defaults to 10.
    """
    return f"Found {limit} records for query: {query}"


@tool("web_search")
def search(query:str)->str:
    """ Search the web for the query.
    Args:
        query: the search query string.
    """
    return f"Results for: {query}"

print(search.__name__)


@tool("calculator",description="useful for when you want to perform mathematical calculations")
def calc(expression: str) -> str:
    """
    Calculate the result of the expression.
    Args:
        expression: the mathematical expression to evaluate.
    """
    return str(eval(expression))

@tool
def summary_conversation(runtime:ToolRuntime)-> str:
    """
    Summarize the conversation so far.
    Args:
        runtime: the tool runtime object.
    """
    # 从运行时对象中获取所有消息
    messages = runtime.messages
    # 提取用户和助手的消息
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
    # 格式化摘要
    summary = "User: " + " | ".join([msg["content"] for msg in user_messages]) + "\n"
    summary += "Assistant: " + " | ".join([msg["content"] for msg in assistant_messages])
    # 打印摘要
    print(summary)
    return summary

def  clear_conversation()->Command:
    """
    Clear the conversation history.
    """
    return Command(
    update={
        "messages": [RemoveMessage(id = REMOVE_ALL_MESSAGES)],
    }
   )
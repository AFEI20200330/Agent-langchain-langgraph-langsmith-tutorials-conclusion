from langchain_core.messages.tool import ToolCall
from langchain_openai import ChatOpenAI
from langchain.tools import tool

import os 
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

model = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
    timeout=60,
)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers together."""
    return a / b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers together."""
    return a - b

tools = [multiply, add, divide, subtract]

tools_by_name = {tool.name: tool for tool in tools}

model_with_tools = model.bind_tools(tools)

from langgraph.graph import add_messages
from langchain.messages import(
    SystemMessage,
    HumanMessage,
    AIMessage,
)


from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint,task

@task
def call_llm(messages:list[BaseMessage]):
    """ LLM decides whether to use a tool or not. """
    response = model_with_tools.invoke(messages)
    return model_with_tools.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant that can use tools to answer questions."
            )
        ]+messages
    )

@task
def call_tool(tool_call: ToolCall):
    """Performs the tool call and returns the result."""
    # Get the tool from the tools_by_name dictionary
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)


# The agent function that runs the loop,
# until the LLM no longer requests to use a tool.
# It then returns the final response from the LLM.
@entrypoint()
def agent(messages:list[BaseMessage]):
    model_response = call_llm(messages).result()
    while True:
        if not model_response.tool_calls:
            break

        # Call the tools in parallel
        tool_result_futures =[
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        # Wait for all tool calls to complete
        tool_results = [future.result() for future in tool_result_futures]
        # Add the model response and tool results to the messages
        messages = add_messages(messages,[model_response,*tool_results])
        # Call the LLM again to get the final response
        model_response = call_llm(messages).result
    # Add the final model response to the messages
    messages = add_messages(messages,[model_response])
    return messages

messages = [HumanMessage(content="Multiply 2 and 3")]
for chunk in agent.stream(messages,stream_mode = "updates"):
    print(chunk)
    print("="*20)
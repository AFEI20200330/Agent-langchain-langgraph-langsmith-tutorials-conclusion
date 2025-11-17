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

response = model_with_tools.invoke("Multiply 2 and 3")
print(response)

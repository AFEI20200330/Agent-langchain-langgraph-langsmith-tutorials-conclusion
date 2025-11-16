
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

@dataclass
class Context:
    user_name:str

import os
# 从环境变量读取API密钥
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

model =ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

agent = create_agent(
    model=model,
    context_schema=Context,
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "你好,我叫什么名字?"}]},
    context=Context(user_name="张三"),
)


print(response)

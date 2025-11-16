
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

@dataclass
class Context:
    user_name:str

OPENROUTER_API_KEY = "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"

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



import os
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

# 从环境变量读取API密钥
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

model = ChatOpenAI(
    model = "openai/gpt-oss-20b:free",
    base_url = "https://openrouter.ai/api/v1",
    api_key = OPENROUTER_API_KEY,
    temperature = 0.7,
    max_tokens = 1024,
)

agen = create_agent(
    model = model,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True, # 所有写文件操作都需要人类确认
                "execute_sql":{"allowed_decisions":["approve", "reject"]}, # 执行SQL操作需要人类确认
                "read_data":False, # 读取数据操作不需要人类确认
            },
            # 自定义确认提示
            description_prefix="请确认以下操作是否正确：",
        ),
    ],
    checkpointer= InMemorySaver()
)
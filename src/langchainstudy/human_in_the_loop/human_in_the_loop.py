

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

OPENROUTER_API_KEY = "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"

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
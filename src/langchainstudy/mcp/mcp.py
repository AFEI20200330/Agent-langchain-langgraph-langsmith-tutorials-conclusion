from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from src.langchainstudy.chat.example import agent
client = MutilServerClient(
    {
        # 数学计算服务
        "math":{
            # transport: 通信方式，这里使用标准输入输出流
            "transport":"stdio",
            # command: 执行的命令，这里使用Python解释器
            "command":"python",
            # args: 执行命令的参数，这里是执行Python代码计算圆周率
            "args":[
                "-c",
                "import math; print(math.pi)"
            ]
        },
        # 天气查询服务
        "weather":{
            # transport: 通信方式，这里使用标准输入输出流
            "transport":"stdio",
            "url":"https://api.openweathermap.org/data/2.5/weather?q=Beijing&appid=YOUR_API_KEY"
        }
    }
)

model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
)

def test():
    # 获取所有工具
    tools = client.get_tools()
    agent = create_agent(
        model,
        tools,
        verbose=True,
    )
    math_response = agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the square root of 16?"}]}
    )
    print(math_response)
    weather_response = agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the weather like in Beijing?"}]}
    )
    print(weather_response)
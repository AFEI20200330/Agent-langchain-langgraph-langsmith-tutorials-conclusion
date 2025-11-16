from langchain.tools import tool
from langchain.agents import create_agent


subagent1 = create_agent(model="...", tools=[...])

@tool(
    "subagent1_name",
    description="subagent1_description"
)
def call_subagent1(query: str):
    result = subagent1.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

agent = create_agent(model="...", tools=[call_subagent1])

#当主代理call_subagent1认为任务与子代理的描述相符时，就会调用该子代理。
#子代理独立运行并返回其结果。
#主控人员收到结果后继续进行协调。
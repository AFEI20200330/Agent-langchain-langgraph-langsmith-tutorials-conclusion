from langchain_core.messages import AIMessage
from langgraph.graph import START, MessagesState, StateGraph, END

def mock_llm(state: MessagesState):
    return {"messages":[{
        "role": "assistant",
        "content": "Hello, I am a mock LLM."
    }]}

g = StateGraph(MessagesState)
g.add_node(mock_llm)
g.add_edge(START, "mock_llm")
g.add_edge("mock_llm", END)
g = g.compile()

response = g.invoke({"messages":[{"role": "user", "content": "Hello"}]})
print(response["messages"][-1].content)

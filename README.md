# Agent应用开发学习路线-Tutorials

langchain     LangChain学习实践案例，囊括基本model对话、agent创建于使用、中间件使用、消息类型、结构化输出、Stream、长短期记忆、guardrails、MCP、运行时等等
langgraph
langsmith
- [x] langchain
- [x] langgraph
- [x] langsmith



# 模型配置
MODEL_CONFIGS = {
    "primary": {
        "model": "openai/gpt-oss-20b:free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_API_KEY,
        "temperature": 0.7,
        "timeout": 30
    },
}

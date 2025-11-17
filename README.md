# Agent应用开发学习路线-Tutorials

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain Streaming 示例

本文件演示了LangChain中Streaming的使用方法，包括：
1. 流式传输代理进度
2. 流式传输LLM tokens
3. 流式传输自定义更新
4. 多种模式同时流式传输

所有示例均使用OpenRouter API实现
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.config import get_stream_writer

# 静态API密钥配置
# 注意：在生产环境中，不建议将API密钥直接硬编码在代码中
import os
# 从环境变量读取API密钥
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# 配置使用OpenRouter的ChatOpenAI实例
try:
    llm = ChatOpenAI(
        model="openai/gpt-oss-20b:free",  # 可以替换为其他OpenRouter支持的模型
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0.7,
        timeout=30
    )
    LLM_AVAILABLE = True
except Exception as e:
    llm = None
    LLM_AVAILABLE = False
    print(f"LLM初始化失败: {e}")
    print("请检查您的API密钥和网络连接是否正确。")

# 创建一个简单的天气工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"{city}的天气晴朗，温度25°C"

# 创建带自定义更新的天气工具
@tool
def get_weather_with_custom_stream(city: str) -> str:
    """获取指定城市的天气信息，并流式传输自定义更新"""
    writer = get_stream_writer()
    writer(f"正在查找城市 {city} 的天气数据...")
    writer(f"已获取城市 {city} 的天气数据")
    return f"{city}的天气晴朗，温度25°C"

def stream_agent_progress():
    """
    示例1: 流式传输代理进度
    使用stream_mode="updates"来获取每个代理步骤后的状态更新
    """
    print("\n=== 1. 流式传输代理进度 ===")
    
    if not LLM_AVAILABLE:
        print("LLM不可用，跳过此示例")
        return
    
    try:
        # 创建代理
        agent = create_agent(
            model=llm,
            tools=[get_weather],
        )
        
        # 流式传输代理进度
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": "上海的天气怎么样？"}]},
            stream_mode="updates",
        ):
            for step, data in chunk.items():
                print(f"步骤: {step}")
                print(f"内容: {data['messages'][-1].content}")
                print()
                
    except Exception as e:
        print(f"示例1执行失败: {e}")
        if "429" in str(e):
            print("速率限制错误，请稍后重试")
        elif "502" in str(e):
            print("服务器错误，暂时不可用")

def stream_llm_tokens():
    """
    示例2: 流式传输LLM tokens
    使用stream_mode="messages"来实时获取LLM生成的tokens
    """
    print("\n=== 2. 流式传输LLM tokens ===")
    
    if not LLM_AVAILABLE:
        print("LLM不可用，跳过此示例")
        return
    
    try:
        # 创建代理
        agent = create_agent(
            model=llm,
            tools=[get_weather],
        )
        
        # 流式传输LLM tokens
        print("AI回复:")
        for token, metadata in agent.stream(
            {"messages": [{"role": "user", "content": "北京的天气怎么样？"}]},
            stream_mode="messages",
        ):
            # 打印生成的文本内容
            if token.content:
                print(token.content, end="", flush=True)
        
        print()  # 换行
                
    except Exception as e:
        print(f"示例2执行失败: {e}")
        if "429" in str(e):
            print("速率限制错误，请稍后重试")
        elif "502" in str(e):
            print("服务器错误，暂时不可用")

def stream_custom_updates():
    """
    示例3: 流式传输自定义更新
    在工具中使用get_stream_writer来发送自定义更新
    """
    print("\n=== 3. 流式传输自定义更新 ===")
    
    if not LLM_AVAILABLE:
        print("LLM不可用，跳过此示例")
        return
    
    try:
        # 创建代理
        agent = create_agent(
            model=llm,
            tools=[get_weather_with_custom_stream],
        )
        
        # 流式传输自定义更新
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": "广州的天气怎么样？"}]},
            stream_mode="custom"
        ):
            print(f"自定义更新: {chunk}")
                
    except Exception as e:
        print(f"示例3执行失败: {e}")
        if "429" in str(e):
            print("速率限制错误，请稍后重试")
        elif "502" in str(e):
            print("服务器错误，暂时不可用")

def stream_multiple_modes():
    """
    示例4: 多种模式同时流式传输
    使用stream_mode=["updates", "custom"]来同时获取多种类型的更新
    """
    print("\n=== 4. 多种模式同时流式传输 ===")
    
    if not LLM_AVAILABLE:
        print("LLM不可用，跳过此示例")
        return
    
    try:
        # 创建代理
        agent = create_agent(
            model=llm,
            tools=[get_weather_with_custom_stream],
        )
        
        # 同时流式传输多种模式
        for stream_mode, chunk in agent.stream(
            {"messages": [{"role": "user", "content": "深圳的天气怎么样？"}]},
            stream_mode=["updates", "custom"]
        ):
            print(f"流模式: {stream_mode}")
            print(f"内容: {chunk}")
            print()
                
    except Exception as e:
        print(f"示例4执行失败: {e}")
        if "429" in str(e):
            print("速率限制错误，请稍后重试")
        elif "502" in str(e):
            print("服务器错误，暂时不可用")

if __name__ == "__main__":
    print("LangChain Streaming with OpenRouter API 示例")
    print("=" * 60)
    
    stream_agent_progress()
    stream_llm_tokens()
    stream_custom_updates()
    stream_multiple_modes()
    
    print("\n所有示例执行完毕！")
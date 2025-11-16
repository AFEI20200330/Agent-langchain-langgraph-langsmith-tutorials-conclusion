#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型回退功能

该脚本演示了模型提供器的全局错误捕捉和处理机制。
"""

import logging
from model_provider import with_model_fallback, ModelContext, call_model, get_all_model_names

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 测试使用装饰器的方式
@with_model_fallback
def test_with_decorator(prompt, model):
    """使用装饰器的方式调用模型"""
    messages = [
        ("system", "你是一个智能助手，用简洁的语言回答问题。"),
        ("human", prompt)
    ]
    return model.invoke(messages)

# 测试使用上下文管理器的方式
def test_with_context_manager(prompt):
    """使用上下文管理器的方式调用模型"""
    messages = [
        ("system", "你是一个智能助手，用简洁的语言回答问题。"),
        ("human", prompt)
    ]
    with ModelContext() as context:
        return context.invoke(messages)

# 测试使用便捷函数的方式
def test_with_convenience_function(prompt):
    """使用便捷函数的方式调用模型"""
    messages = [
        ("system", "你是一个智能助手，用简洁的语言回答问题。"),
        ("human", prompt)
    ]
    return call_model(messages)

if __name__ == "__main__":
    print("=== 模型回退功能测试 ===")
    print(f"可用模型列表: {get_all_model_names()}")
    print()
    
    prompt = "你好，请介绍一下你自己。"
    
    print("1. 测试使用装饰器的方式:")
    try:
        response = test_with_decorator(prompt)
        print(f"响应: {response.content}")
    except Exception as e:
        print(f"错误: {e}")
    print()
    
    print("2. 测试使用上下文管理器的方式:")
    try:
        response = test_with_context_manager(prompt)
        print(f"响应: {response.content}")
    except Exception as e:
        print(f"错误: {e}")
    print()
    
    print("3. 测试使用便捷函数的方式:")
    try:
        response = test_with_convenience_function(prompt)
        print(f"响应: {response.content}")
    except Exception as e:
        print(f"错误: {e}")
    print()
    
    print("=== 测试完成 ===")
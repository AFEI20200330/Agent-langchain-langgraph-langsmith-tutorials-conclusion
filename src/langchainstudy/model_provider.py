#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型提供器(Model Provider)

该模块提供了模型选择和创建功能，支持多个备选模型，当主要模型不可用时自动切换到其他模型。
实现了全局错误捕捉和处理机制，在模型调用出错时自动尝试下一个模型。
"""

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from typing import Optional, List, Dict, Callable, Any
import time
import logging
import functools

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API密钥配置
import os
# 从环境变量读取API密钥
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# 模型配置
MODEL_CONFIGS = {
    "primary": {
        "model": "openai/gpt-oss-20b:free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_API_KEY,
        "temperature": 0.7,
        "timeout": 30
    },
    "alternatives": [
        {
            "model": "meituan/longcat-flash-chat:free",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": OPENROUTER_API_KEY,
            "temperature": 0.7,
            "timeout": 30
        },
        {
            "model": "qwen/qwen3-coder:free",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": OPENROUTER_API_KEY,
            "temperature": 0.7,
            "timeout": 30
        }
    ]
}

class ModelProvider:
    """模型提供器类，用于创建和管理模型实例"""
    
    def __init__(self):
        """初始化模型提供器"""
        self.models = {}
        self.current_model_index = 0
        self.model_configs = [MODEL_CONFIGS["primary"]] + MODEL_CONFIGS["alternatives"]
    
    def _create_model(self, config: Dict[str, Any]) -> ChatOpenAI:
        """创建模型实例"""
        return ChatOpenAI(
            model=config["model"],
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=config["temperature"],
            timeout=config["timeout"]
        )
    
    def get_model(self, index: int = 0) -> Optional[ChatOpenAI]:
        """获取指定索引的模型实例，如果不存在则创建"""
        if index < 0 or index >= len(self.model_configs):
            logger.error(f"模型索引超出范围: {index}")
            return None
            
        model_key = f"model_{index}"
        if model_key not in self.models:
            config = self.model_configs[index]
            self.models[model_key] = self._create_model(config)
            logger.info(f"创建模型实例: {config['model']}")
        
        return self.models[model_key]
    
    def get_current_model(self) -> Optional[ChatOpenAI]:
        """获取当前使用的模型实例"""
        return self.get_model(self.current_model_index)
    
    def get_next_model(self) -> Optional[ChatOpenAI]:
        """获取下一个模型实例，如果当前是最后一个则返回None"""
        self.current_model_index += 1
        if self.current_model_index < len(self.model_configs):
            return self.get_model(self.current_model_index)
        else:
            logger.error("已经尝试了所有模型，没有更多备选模型！")
            return None
    
    def reset_model_index(self):
        """重置当前模型索引为0"""
        self.current_model_index = 0
    
    def get_all_model_names(self) -> List[str]:
        """获取所有模型的名称列表"""
        return [config["model"] for config in self.model_configs]
    
    def get_current_model_name(self) -> str:
        """获取当前使用的模型名称"""
        if self.current_model_index < len(self.model_configs):
            return self.model_configs[self.current_model_index]["model"]
        return ""

# 单例模式的模型提供器实例
model_provider = None

def get_model_provider() -> ModelProvider:
    """获取模型提供器的单例实例"""
    global model_provider
    if model_provider is None:
        model_provider = ModelProvider()
    return model_provider

# 全局模型调用装饰器
def with_model_fallback(func: Callable) -> Callable:
    """
    模型调用装饰器，自动处理模型调用错误并切换到下一个模型
    
    Args:
        func: 需要包装的函数，该函数应接受一个model参数
        
    Returns:
        包装后的函数，自动处理模型切换逻辑
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        provider = get_model_provider()
        provider.reset_model_index()  # 每次调用都从第一个模型开始尝试
        
        # 尝试所有模型
        for _ in range(len(provider.get_all_model_names())):
            model = provider.get_current_model()
            if not model:
                continue
                
            try:
                logger.info(f"使用模型 {provider.get_current_model_name()} 执行请求")
                result = func(*args, model=model, **kwargs)
                logger.info(f"模型 {provider.get_current_model_name()} 请求成功")
                return result
            except Exception as e:
                logger.warning(f"模型 {provider.get_current_model_name()} 请求失败: {e}")
                # 尝试下一个模型
                next_model = provider.get_next_model()
                if not next_model:
                    logger.error("所有模型都请求失败！")
                    raise
        
        logger.error("没有可用的模型！")
        raise Exception("没有可用的模型")
    
    return wrapper

# 全局模型调用上下文管理器
class ModelContext:
    """
    模型调用上下文管理器，自动处理模型调用错误并切换到下一个模型
    """
    
    def __init__(self):
        self.provider = get_model_provider()
        self.model = self.provider.get_current_model()  # 初始化当前模型
        
    def __enter__(self):
        self.provider.reset_model_index()
        self.model = self.provider.get_current_model()  # 更新当前模型
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    @property
    def current_model(self):
        """获取当前使用的模型"""
        return self.model
    
    def invoke(self, messages: Any, **kwargs) -> Any:
        """
        调用模型，自动处理错误并切换模型
        
        Args:
            messages: 发送给模型的消息
            **kwargs: 其他参数
            
        Returns:
            模型的响应，如果stream=True则返回生成器
        """
        is_streaming = kwargs.get("stream", False)
        
        def generate_chunks():
            # 重置模型索引
            self.provider.reset_model_index()
            # 尝试所有模型
            for _ in range(len(self.provider.get_all_model_names())):
                model = self.provider.get_current_model()
                self.model = model  # 更新当前模型引用
                if not model:
                    continue
                    
                try:
                    logger.info(f"使用模型 {self.provider.get_current_model_name()} 执行请求")
                    # 流式输出情况 - 使用stream方法
                    chunks = model.stream(messages, **kwargs)
                    logger.info(f"模型 {self.provider.get_current_model_name()} 请求成功")
                    # 返回生成器，直接迭代模型的流式输出
                    for chunk in chunks:
                        yield chunk
                    return  # 流式输出结束
                except Exception as e:
                    logger.warning(f"模型 {self.provider.get_current_model_name()} 请求失败: {e}")
                    # 尝试下一个模型
                    next_model = self.provider.get_next_model()
                    self.model = next_model  # 更新当前模型引用
                    if not next_model:
                        logger.error("所有模型都请求失败！")
                        raise
            
            logger.error("没有可用的模型！")
            raise Exception("没有可用的模型")
        
        if is_streaming:
            # 返回生成器
            return generate_chunks()
        else:
            # 非流式输出情况 - 尝试所有模型
            self.provider.reset_model_index()
            for _ in range(len(self.provider.get_all_model_names())):
                model = self.provider.get_current_model()
                self.model = model  # 更新当前模型引用
                if not model:
                    continue
                    
                try:
                    logger.info(f"使用模型 {self.provider.get_current_model_name()} 执行请求")
                    result = model.invoke(messages, **kwargs)
                    logger.info(f"模型 {self.provider.get_current_model_name()} 请求成功")
                    return result
                except Exception as e:
                    logger.warning(f"模型 {self.provider.get_current_model_name()} 请求失败: {e}")
                    # 尝试下一个模型
                    next_model = self.provider.get_next_model()
                    self.model = next_model  # 更新当前模型引用
                    if not next_model:
                        logger.error("所有模型都请求失败！")
                        raise
            
            logger.error("没有可用的模型！")
            raise Exception("没有可用的模型")

def get_chat_model() -> Optional[ChatOpenAI]:
    """获取当前使用的聊天模型实例"""
    provider = get_model_provider()
    return provider.get_current_model()

def get_all_model_names() -> List[str]:
    """获取所有模型的名称列表"""
    provider = get_model_provider()
    return provider.get_all_model_names()

def init_model_with_fallback(model_name: str = "openai/gpt-oss-20b:free", **kwargs) -> Optional[ChatOpenAI]:
    """
    使用指定模型名称初始化模型
    
    Args:
        model_name: 要使用的模型名称
        **kwargs: 模型配置参数
        
    Returns:
        模型实例
    """
    # 构建模型配置
    config = {
        "model": model_name,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_API_KEY,
        "temperature": kwargs.get("temperature", 0.7),
        "timeout": kwargs.get("timeout", 30)
    }
    
    model = ChatOpenAI(**config)
    logger.info(f"初始化模型: {model_name}")
    return model

def init_openrouter_chat_model_with_fallback(model_name: str = "openai/gpt-oss-20b:free", **kwargs) -> Optional[ChatOpenAI]:
    """
    使用init_chat_model函数初始化OpenRouter模型
    
    Args:
        model_name: 要使用的模型名称
        **kwargs: 模型配置参数
        
    Returns:
        模型实例
    """
    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=kwargs.get("temperature", 0.7),
        timeout=kwargs.get("timeout", 30)
    )
    logger.info(f"使用init_chat_model初始化模型: {model_name}")
    return model

# 便捷函数：直接调用模型，自动处理错误和切换
def call_model(messages: Any, **kwargs) -> Any:
    """
    直接调用模型，自动处理错误和切换到备选模型
    
    Args:
        messages: 发送给模型的消息
        **kwargs: 其他参数
        
    Returns:
        模型的响应
    """
    with ModelContext() as context:
        return context.invoke(messages, **kwargs)

# 单例模式的模型提供器实例
model_provider = None

def get_model_provider() -> ModelProvider:
    """获取模型提供器的单例实例"""
    global model_provider
    if model_provider is None:
        model_provider = ModelProvider()
    return model_provider
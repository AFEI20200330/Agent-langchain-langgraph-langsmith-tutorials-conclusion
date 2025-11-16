#!/usr/bin/env python3
"""
LangChainStudy - 主程序入口

这是LangChainStudy项目的主程序入口文件，用于演示LangChain的基本用法。
"""

import sys
import os

from src.langchainstudy import init_chat_model

# 将src目录添加到Python模块搜索路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from langchainstudy import example


def main():
    # 调用示例模块
    # example.greet()
    # # 测试LLM连接
    # example.test_llm_connection()
    init_chat_model.test()


if __name__ == "__main__":
    main()
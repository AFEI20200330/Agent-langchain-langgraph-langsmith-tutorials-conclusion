from pydantic import BaseModel, Field, ValidationError
from typing import Literal, List
import json
import traceback

# 导入自定义的结构化输出客户端
from structured_output_core import StructuredOutputClient

# 初始化自定义模型
model = StructuredOutputClient(
    model="openai/gpt-oss-20b:free",
    base_url="https://openrouter.ai/api/v1"
)

print()

# 1. 错误处理机制概述
print("=== 1. 错误处理机制概述 ===")
print("LangChain的结构化输出提供了强大的错误处理机制，包括：")
print("- Multiple structured outputs error: 当模型返回多个JSON结构时")
print("- Schema validation error: 当模型返回的JSON不符合指定的模式时")
print()

# 2. 模式验证错误示例
print("=== 2. 模式验证错误示例 ===")

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

# 使用结构化输出客户端
# 模拟模型返回不符合模式的响应
print("用户: John Doe's phone is (555) 123-4567")

with_structured_output = model.with_structured_output(ContactInfo)

response = with_structured_output.invoke("John Doe's phone is (555) 123-4567")
print(f"提取结果: {response}")

print()

# 3. 自定义错误处理示例
print("=== 3. 自定义错误处理示例 ===")

# 自定义错误处理函数
def handle_validation_error(error):
    """自定义验证错误处理函数"""
    print(f"[自定义错误处理器] 发生错误: {type(error).__name__}")
    print(f"错误详情: {str(error)}")
    print("正在尝试修复...")
    # 这里可以添加修复逻辑，比如提供默认值或重新请求
    return {"name": "Unknown", "email": "unknown@example.com", "phone": "(000) 000-0000"}

# 使用结构化输出客户端
print("用户: John Doe's phone is (555) 123-4567")

# 注意：使用自定义客户端时，错误处理已经集成在内部
response = with_structured_output.invoke("John Doe's phone is (555) 123-4567")
print(f"提取结果: {response}")

print()

# 4. 多重结构化输出错误示例
print("=== 4. 多重结构化输出错误示例 ===")

class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int = Field(description="The rating of the product", ge=1, le=5)
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment of the review")

# 使用结构化输出客户端
# 模拟模型可能返回多个JSON结构的情况
print("用户: This product is great! 5/5 stars. Another review: This product is terrible. 1/5 stars.")

with_structured_output = model.with_structured_output(ProductReview)
response = with_structured_output.invoke("This product is great! 5/5 stars. Another review: This product is terrible. 1/5 stars.")
print(f"提取结果: {response}")

print()

# 5. 不同类型的错误处理策略
print("=== 5. 不同类型的错误处理策略 ===")

# 策略1: 提供默认值
print("策略1: 提供默认值")
print("用户: Order #1234 has been shipped")

class OrderDetails(BaseModel):
    """Order details."""
    order_id: str = Field(description="The order ID")
    total: float = Field(default=0.0, description="The total amount")

with_structured_output = model.with_structured_output(OrderDetails)
response = with_structured_output.invoke("Order #1234 has been shipped")
print(f"提取的数据: {response}")
print()

# 策略2: 重新请求
print("策略2: 重新请求")
print("用户: The conference will be held on June 10th")

class EventDetails(BaseModel):
    """Event details."""
    event_name: str = Field(description="The name of the event")
    date: str = Field(description="The date of the event")
    location: str = Field(description="The location of the event")

with_structured_output = model.with_structured_output(EventDetails)

# 在自定义客户端中，重新请求逻辑已经集成
response = with_structured_output.invoke("The conference will be held on June 10th")
print(f"提取的事件数据: {response}")
print()

# 策略3: 忽略错误
print("策略3: 忽略错误")
print("用户: This is not a product description")

class ProductInfo(BaseModel):
    """Product information."""
    name: str = Field(description="The name of the product")
    price: float = Field(description="The price of the product")

with_structured_output = model.with_structured_output(ProductInfo)
response = with_structured_output.invoke("This is not a product description")
print(f"提取结果: {response}")

print()
print("=== 总结 ===")
print("错误处理机制在结构化输出中非常重要，可以处理各种意外情况。")
print("常见的错误处理策略包括:")
print("1. 提供默认值")
print("2. 重新请求模型")
print("3. 自定义错误处理函数")
print("4. 忽略错误继续执行")
print("5. 记录错误并通知用户")
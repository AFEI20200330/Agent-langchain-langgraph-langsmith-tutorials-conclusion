from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing_extensions import TypedDict
from structured_output_core import StructuredOutputClient
import json

OPENROUTER_API_KEY = "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"

# Initialize the model using our custom client
model = StructuredOutputClient(
    model="openai/gpt-oss-20b:free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

print("=== Provider Strategy Examples ===")
print("使用自定义实现的结构化输出API")
print()

# 1. Pydantic 模型示例
print("=== 1. Pydantic 模型示例 ===")

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

# Create model with structured output capability
structured_model = model.with_structured_output(ContactInfo)

# Invoke the model
response = structured_model.invoke("Extract contact info from: John Doe, john@example.com, (555) 123-4567")

print("结构化响应:")
print(response)
print(f"姓名: {response.name}")
print(f"邮箱: {response.email}")
print(f"电话: {response.phone}")

print()

# 2. 数据类示例
print("\n=== 2. 数据类示例 ===")

@dataclass
class ProductReview:
    """Analysis of a product review."""
    rating: int | None  # The rating of the product (1-5)
    sentiment: str  # The sentiment of the review (positive/negative)
    key_points: list[str]  # The key points of the review

# Create model with structured output capability
structured_model = model.with_structured_output(ProductReview)

# Invoke the model
response = structured_model.invoke("Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'")

print("结构化响应:")
print(response)
print(f"评分: {response.rating}")
print(f"情感: {response.sentiment}")
print(f"关键点: {response.key_points}")

print()

# 3. TypedDict 示例
print("\n=== 3. TypedDict 示例 ===")

class EventDetails(TypedDict):
    """Details of an event."""
    event_name: str  # Name of the event
    date: str  # Event date
    location: str  # Event location

# Create model with structured output capability
structured_model = model.with_structured_output(EventDetails)

# Invoke the model
response = structured_model.invoke("The Tech Conference will be held on March 15th at the Convention Center")

print("结构化响应:")
print(response)
print(f"事件名称: {response['event_name']}")
print(f"日期: {response['date']}")
print(f"地点: {response['location']}")

print()

# 4. JSON Schema 示例
print("\n=== 4. JSON Schema 示例 ===")

product_schema = {
    "type": "object",
    "description": "Product information.",
    "properties": {
        "name": {"type": "string", "description": "The name of the product"},
        "price": {"type": "number", "description": "The price of the product"},
        "category": {"type": "string", "description": "The category of the product"}
    },
    "required": ["name", "price", "category"]
}

# Create model with structured output capability using JSON Schema
structured_model = model.with_structured_output(product_schema)

# Invoke the model
response = structured_model.invoke("I bought a new laptop: MacBook Pro for $1999, which falls under the electronics category")

print("结构化响应:")
print(response)
print(f"产品名称: {response['name']}")
print(f"价格: ${response['price']}")
print(f"类别: {response['category']}")
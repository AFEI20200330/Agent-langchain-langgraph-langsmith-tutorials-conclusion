#!/usr/bin/env python3
"""
完整的结构化输出示例
模拟一个客户服务系统，使用结构化输出处理客户请求
"""

from pydantic import BaseModel, Field, ValidationError
from typing import Literal, List, Optional, Union, TypeAlias
from dataclasses import dataclass
import json
import re

# 导入自定义的结构化输出客户端
from structured_output_core import StructuredOutputClient

OPENROUTER_API_KEY = "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" 

# 初始化模型
model = StructuredOutputClient(
    model="openai/gpt-oss-20b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

print("完整的结构化输出示例")
print("=" * 70)

# ------------------------------
# 1. 定义数据模型
# ------------------------------

# 客户查询类型
QueryType: TypeAlias = Literal["order_status", "product_info", "complaint", "return_request", "other"]

# 客户信息模型
class CustomerInfo(BaseModel):
    """从客户消息中提取的客户信息"""
    name: Optional[str] = Field(None, description="客户姓名")
    email: Optional[str] = Field(None, description="客户邮箱")
    phone: Optional[str] = Field(None, description="客户电话")
    order_id: Optional[str] = Field(None, description="订单ID")

# 查询分类模型
class QueryClassification(BaseModel):
    """客户查询的分类"""
    query_type: QueryType = Field(description="查询类型")
    urgency: Literal["low", "medium", "high"] = Field(description="紧急程度")
    keywords: List[str] = Field(description="关键关键词")

# 产品信息请求模型
class ProductInfoRequest(BaseModel):
    """产品信息请求"""
    product_name: str = Field(description="产品名称")
    requested_info: List[str] = Field(description="请求的信息类型")

# 订单状态请求模型
class OrderStatusRequest(BaseModel):
    """订单状态请求"""
    order_id: str = Field(description="订单ID")
    requested_updates: List[str] = Field(description="请求的更新信息")

# 投诉请求模型
class ComplaintRequest(BaseModel):
    """投诉请求"""
    issue_type: str = Field(description="问题类型")
    description: str = Field(description="问题描述")
    desired_resolution: str = Field(description="期望的解决方案")

# 返回请求模型
class ReturnRequest(BaseModel):
    """返回请求"""
    order_id: str = Field(description="订单ID")
    product_name: str = Field(description="产品名称")
    reason: str = Field(description="返回原因")
    return_method: Literal["pickup", "dropoff", "shipping"] = Field(description="返回方式")

# 综合响应模型
class CustomerResponse(BaseModel):
    """客户请求的综合响应"""
    customer_info: CustomerInfo = Field(description="客户信息")
    query_classification: QueryClassification = Field(description="查询分类")
    request_details: Union[ProductInfoRequest, OrderStatusRequest, ComplaintRequest, ReturnRequest] = Field(description="请求详情")

# ------------------------------
# 2. 辅助函数
# ------------------------------

def extract_json_from_response(response_content: str) -> str:
    """从模型响应中提取JSON部分"""
    # 尝试使用正则表达式提取JSON
    json_pattern = r'\{.*?\}'
    json_matches = re.findall(json_pattern, response_content, re.DOTALL)
    
    if json_matches:
        # 尝试找最大的JSON结构
        largest_json = max(json_matches, key=len)
        try:
            json.loads(largest_json)
            return largest_json
        except json.JSONDecodeError:
            # 如果最大的JSON无效，尝试其他
            for json_str in json_matches:
                try:
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    continue
    
    # 如果没有找到有效的JSON，返回原始内容
    return response_content

def handle_validation_error(error: ValidationError, context) -> dict:
    """处理验证错误并尝试修复"""
    print(f"验证错误: {error}")
    print("正在尝试修复...")
    
    # 获取错误信息
    errors = error.errors()
    
    # 创建修复后的数据
    fixed_data = {}
    
    # 根据错误类型进行修复
    for err in errors:
        field_name = err["loc"][0]
        err_type = err["type"]
        
        # 处理必填字段缺失的情况
        if err_type == "missing":
            # 根据字段类型提供默认值
            if field_name in ["name", "email", "phone", "order_id", "product_name", "description", "reason"]:
                fixed_data[field_name] = "unknown"
            elif field_name in ["requested_info", "keywords"]:
                fixed_data[field_name] = []
            elif field_name in ["query_type"]:
                fixed_data[field_name] = "other"
            elif field_name in ["urgency"]:
                fixed_data[field_name] = "medium"
            elif field_name in ["return_method"]:
                fixed_data[field_name] = "shipping"
        
        # 处理类型错误
        elif err_type == "type_error.string":
            fixed_data[field_name] = str(err["input"])
        
        # 处理枚举类型错误
        elif err_type == "type_error.enum":
            # 使用第一个有效值
            if field_name == "query_type":
                fixed_data[field_name] = "other"
            elif field_name == "urgency":
                fixed_data[field_name] = "medium"
            elif field_name == "return_method":
                fixed_data[field_name] = "shipping"
    
    return fixed_data

# ------------------------------
# 3. 主处理函数
# ------------------------------

def process_customer_message(message: str) -> Optional[CustomerResponse]:
    """处理客户消息并返回结构化响应"""
    try:
        print(f"\n客户消息: {message}")
        print("-" * 70)
        
        # 创建带有结构化输出的模型
        with_structured_output = model.with_structured_output(CustomerResponse)
        
        # 调用模型获取结构化输出
        response = with_structured_output.invoke(message)
        
        print("处理成功!")
        print(f"客户信息: {response.customer_info}")
        print(f"查询类型: {response.query_classification.query_type}, 紧急程度: {response.query_classification.urgency}")
        print(f"关键词: {response.query_classification.keywords}")
        print(f"请求详情: {response.request_details}")
        
        return response
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return None

# ------------------------------
# 4. 生成响应
# ------------------------------

def generate_response(customer_response: CustomerResponse) -> str:
    """根据结构化响应生成客户回复"""
    
    # 基于不同的查询类型生成不同的响应
    query_type = customer_response.query_classification.query_type
    customer_info = customer_response.customer_info
    request_details = customer_response.request_details
    
    # 基础回复
    response = f"亲爱的{customer_info.name or '客户'}，"
    
    if query_type == "order_status":
        # 订单状态查询
        response += f"您的订单 #{request_details.order_id} 的状态是：正在配送中。"
        if hasattr(request_details, 'requested_updates') and request_details.requested_updates:
            response += f"\n您请求的更新信息：{', '.join(request_details.requested_updates)}"
    
    elif query_type == "product_info":
        # 产品信息查询
        response += f"关于{request_details.product_name}的信息："
        if hasattr(request_details, 'requested_info') and request_details.requested_info:
            response += f"\n您查询的信息：{', '.join(request_details.requested_info)}"
        response += "\n我们的产品质量保证，欢迎您的购买！"
    
    elif query_type == "complaint":
        # 投诉处理
        response += f"我们已经收到您关于{request_details.issue_type}的投诉。"
        response += f"\n问题描述：{request_details.description}"
        response += f"\n您期望的解决方案：{request_details.desired_resolution}"
        response += "\n我们会尽快处理您的投诉，并在24小时内与您联系。"
    
    elif query_type == "return_request":
        # 返回请求
        response += f"我们已经收到您关于订单 #{request_details.order_id} 中{request_details.product_name}的退货请求。"
        response += f"\n退货原因：{request_details.reason}"
        response += f"\n退货方式：{request_details.return_method}"
        response += "\n我们会尽快处理您的退货请求，并发送退货标签到您的邮箱。"
    
    else:
        # 其他类型查询
        response += "感谢您的咨询，我们会尽快处理并回复您。"
    
    # 根据紧急程度添加不同的结束语
    urgency = customer_response.query_classification.urgency
    if urgency == "high":
        response += "\n\n*** 紧急处理 ***"
    
    return response

# ------------------------------
# 5. 主函数
# ------------------------------

def main():
    """主函数，演示完整的客户服务流程"""
    
    # 测试用例
    test_messages = [
        "我叫张三，我的订单#12345一直没有收到，麻烦帮我查一下状态",
        "请问你们的最新款智能手机有什么特点？电池续航怎么样？",
        "我对昨天收到的商品非常不满意，质量太差了，我要退款！",
        "我想退货，订单号是#67890，商品是无线耳机，因为音质不好，我希望上门取件",
        "你们的客服电话是多少？"
    ]
    
    print("开始处理客户请求...")
    print("=" * 70)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n\n--- 测试用例 {i} ---")
        
        # 处理客户消息
        response = process_customer_message(message)
        
        if response:
            # 生成回复
            reply = generate_response(response)
            print("\n" + "=" * 70)
            print("生成的回复：")
            print(reply)
            print("=" * 70)
        else:
            print("处理失败，无法生成响应")

if __name__ == "__main__":
    main()
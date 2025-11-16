import json
from typing import Any, Dict, Type, Union, Optional, List, TypeVar
from pydantic import BaseModel, ValidationError
from dataclasses import is_dataclass, asdict
from typing_extensions import TypedDict
import requests

T = TypeVar('T')

class StructuredOutputClient:
    """自定义的结构化输出客户端，不依赖langchain的modelprovider"""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", model: str = "openai/gpt-oss-20b:free", temperature: float = 0.7):
        """初始化客户端"""
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
    
    def _make_api_request(self, messages: List[Dict[str, str]], response_format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """发送API请求到OpenAI兼容的端点"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")
        
        return response.json()
    
    def with_structured_output(self, schema: Union[Type[BaseModel], Type[TypedDict], Type, Dict[str, Any]]):
        """创建支持结构化输出的模型"""
        return StructuredOutputModel(self, schema)
    
    def invoke(self, messages: List[Dict[str, str]]) -> Any:
        """直接调用模型，获取原始响应"""
        response = self._make_api_request(messages)
        return response["choices"][0]["message"]["content"]


class StructuredOutputModel:
    """支持结构化输出的模型包装器"""
    
    def __init__(self, client: StructuredOutputClient, schema: Union[Type[BaseModel], Type[TypedDict], Type, Dict[str, Any]]):
        """初始化结构化输出模型"""
        self.client = client
        self.schema = schema
        self._schema_type = self._determine_schema_type()
    
    def _determine_schema_type(self) -> str:
        """确定schema的类型"""
        if isinstance(self.schema, dict):
            return "json_schema"
        try:
            if issubclass(self.schema, BaseModel):
                return "pydantic"
        except TypeError:
            pass
        if is_dataclass(self.schema):
            return "dataclass"
        elif hasattr(self.schema, "__annotations__") and hasattr(self.schema, "__total__"):
            return "typeddict"
        else:
            raise ValueError(f"不支持的schema类型: {type(self.schema)}")
    
    def _convert_schema_to_json(self) -> Dict[str, Any]:
        """将各种类型的schema转换为JSON Schema"""
        if self._schema_type == "pydantic":
            return self.schema.model_json_schema()
        elif self._schema_type == "dataclass":
            # 数据类转换为JSON Schema（简化实现）
            schema = {"type": "object", "properties": {}, "required": []}
            for field_name, field_type in self.schema.__annotations__.items():
                # 简化类型处理
                field_type_str = str(field_type)
                if "str" in field_type_str:
                    json_type = "string"
                elif "int" in field_type_str:
                    json_type = "integer"
                elif "float" in field_type_str:
                    json_type = "number"
                elif "bool" in field_type_str:
                    json_type = "boolean"
                elif "list" in field_type_str:
                    json_type = "array"
                elif "dict" in field_type_str:
                    json_type = "object"
                else:
                    json_type = "string"
                
                schema["properties"][field_name] = {"type": json_type}
                schema["required"].append(field_name)
            return schema
        elif self._schema_type == "typeddict":
            # TypedDict转换为JSON Schema（简化实现）
            schema = {"type": "object", "properties": {}, "required": []}
            for field_name, field_type in self.schema.__annotations__.items():
                field_type_str = str(field_type)
                if "str" in field_type_str:
                    json_type = "string"
                elif "int" in field_type_str:
                    json_type = "integer"
                elif "float" in field_type_str:
                    json_type = "number"
                elif "bool" in field_type_str:
                    json_type = "boolean"
                elif "list" in field_type_str:
                    json_type = "array"
                elif "dict" in field_type_str:
                    json_type = "object"
                else:
                    json_type = "string"
                
                schema["properties"][field_name] = {"type": json_type}
            return schema
        elif self._schema_type == "json_schema":
            return self.schema
        
        raise ValueError(f"无法处理schema类型: {self._schema_type}")
    
    def _parse_response(self, response_content: str) -> Any:
        """解析API响应内容"""
        try:
            # 提取JSON内容（处理可能的格式问题）
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_content.strip()
            
            # 解析JSON
            data = json.loads(json_str)
            
            # 根据schema类型进行转换
            if self._schema_type == "pydantic":
                return self.schema.model_validate(data)
            elif self._schema_type == "dataclass":
                return self.schema(**data)
            elif self._schema_type == "typeddict":
                return data  # TypedDict返回字典
            elif self._schema_type == "json_schema":
                return data  # JSON Schema返回字典
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析JSON响应: {e}")
        except ValidationError as e:
            raise ValueError(f"响应不符合Pydantic模型: {e}")
        except Exception as e:
            raise ValueError(f"解析响应失败: {e}")
    
    def invoke(self, user_message: str) -> Any:
        """调用模型获取结构化输出"""
        # 获取JSON Schema
        json_schema = self._convert_schema_to_json()
        
        # 构建系统提示词
        system_prompt = f"You are an assistant that returns structured data.\n"
        system_prompt += f"Return JSON that matches this schema:\n"
        system_prompt += f"{json.dumps(json_schema, indent=2)}\n"
        system_prompt += "Only return the JSON, no other text."
        
        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # 调用API
        response_content = self.client.invoke(messages)
        
        # 解析响应
        return self._parse_response(response_content)
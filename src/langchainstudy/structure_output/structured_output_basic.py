import os
from structured_output_core import StructuredOutputClient
from pydantic import BaseModel, Field

# Define structured output schema
class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

# 从环境变量读取API密钥
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
print(OPENROUTER_API_KEY)
if __name__ == "__main__":
    # 检查API密钥是否存在
    if not OPENROUTER_API_KEY:
        print("错误: 未设置OPENROUTER_API_KEY环境变量")
        print("请使用以下命令设置环境变量:")
        print("Windows (PowerShell): $env:OPENROUTER_API_KEY = \"your-api-key-here\"")
        print("macOS/Linux (终端): export OPENROUTER_API_KEY=\"your-api-key-here\"")
        exit(1)
    
    try:
        # Initialize the model using our custom client
        model = StructuredOutputClient(
            model="openai/gpt-oss-20b:free",
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
        )
        
        # Create model with structured output capability
        structured_model = model.with_structured_output(ContactInfo)
        
        # Invoke the model
        response = structured_model.invoke("Extract contact info from: John Doe, john@example.com, (555) 123-4567")
        
        print("Structured response:")
        print(response)
        print(f"Name: {response.name}")
        print(f"Email: {response.email}")
        print(f"Phone: {response.phone}")
    except Exception as e:
        print(f"执行错误: {str(e)}")
        print("请检查:")
        print("1. API密钥是否正确")
        print("2. 网络连接是否正常")
        print("3. OpenRouter API服务是否可用")
        exit(1)
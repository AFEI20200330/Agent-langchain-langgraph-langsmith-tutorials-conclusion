from structured_output_core import StructuredOutputClient
from pydantic import BaseModel, Field

# Define structured output schema
class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

print()

import os
# 从环境变量读取API密钥
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

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
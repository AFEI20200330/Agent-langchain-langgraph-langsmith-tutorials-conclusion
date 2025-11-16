
from langchain.chat_models import init_chat_model
from langchain.messages import(
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from pydantic_core.core_schema import none_schema
from model_provider import ModelContext, get_all_model_names

print(f"可用模型列表: {get_all_model_names()}")
print()

def basic_message_usage():
    with ModelContext() as context:
        sys_msg = SystemMessage(content="你是一个专业的助手，能够回答用户的问题。")
        human_msg = HumanMessage("what'is your role and target?")

        messages = [sys_msg, human_msg]

        response = context.invoke(messages)
        print("the type of response is ",type(response).__name__)
        print("the content of response is ",response.content)


def text_promot_usage():
    with ModelContext() as context:
        response = context.invoke("please write a poem about the sunset.")
        print("the type of response is ",type(response).__name__)
        print("the content of response is ",response.content)


def dict_format_usage():
    with ModelContext() as context:
        messages =[
            {"role": "system", "content": "你是一个专业的助手，能够回答用户的问题。"},
            {"role": "user", "content": "请写一首关于日落的诗。"},
            {"role": "assistant", "content": "日落是一个美丽的时间，它的景色是如此的壮观。"}
        ]
        response = context.invoke(messages)
        print("the type of response is ",type(response).__name__)
        print("the content of response is ",response.content)


def message_metadata_usage():
    with ModelContext() as context:
        human_msg = HumanMessage(
            content="what'is your role and target?",
            name="ivan",
            id="123456",
        )

        messages = [human_msg]
        response = context.invoke(messages)
        print("the type of response is ",type(response).__name__)
        print("the content of response is ",response.content)


def ai_msg_attribute():
    with ModelContext() as context:
        response = context.invoke("what'is your role and target?")
        print("the type of response is ",type(response).__name__)
        print("the content of response is ",response.content)
        print("the id of response is ",response.id)
        print("the type of response is ",response.type)

def stream_msg_demo():
    with ModelContext() as context:
        chunks = []
        full_msg = None
        for chunk in context.invoke("please explain what is the LangChain?", stream=True):
            chunks.append(chunk)
            print(chunk.content,end="",flush=True)
            full_msg = chunk if full_msg is None else full_msg + chunk
    
    print("\n完整消息:", full_msg.content)

def tool_calling_demo():
    @tool
    def get_weather(city: str) -> str:
        """
        查询指定城市的天气情况。
        """
        weather_data ={
            "广州": "天气晴朗, 25摄氏度",
            "深圳": "天气多云, 22摄氏度",
            "北京": "天气 rainy, 20摄氏度"
        }
        return weather_data.get(city, "城市未找到")
    
    with ModelContext() as context:
        model_with_tools = context.model.bind_tools([get_weather])
        response = model_with_tools.invoke("广州的天气怎么样？")
        print("the type of response is ",type(response).__name__)
        print("the content of response is ",response.content)

        if hasattr(response,"tool_calls") and response.tool_calls is not None:
            for tool_call in response.tool_calls:
                print(tool_call)

def content_block_demo():
    with ModelContext() as context:
        msg = HumanMessage(
            content_blocks=[
                {"type": "text", "text": "请查询广州的天气"},
            ]
        )
        response = context.invoke(msg)
        print("the type of response is ",type(response).__name__)
        print("the content of response is ",response.content)

        if hasattr(response,"content_blocks") and response.content_blocks is not None:
            for content_block in response.content_blocks:
                print("the content of content block is ",content_block)

def test():
    basic_message_usage()
    print("\n")
    text_promot_usage()
    print("\n")
    dict_format_usage()
    print("\n")
    message_metadata_usage()
    print("\n")
    ai_msg_attribute()
    print("\n")
    stream_msg_demo()
    print("\n")
    tool_calling_demo()
    print("\n")
    content_block_demo()


    
test()
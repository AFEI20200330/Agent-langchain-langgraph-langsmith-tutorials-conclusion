from langchain.agents import AgentState
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware, DockerExecutionPolicy, HostExecutionPolicy, LLMToolEmulator, LLMToolSelectorMiddleware, ModelCallLimitMiddleware, ModelFallbackMiddleware, PIIMiddleware, ShellToolMiddleware, TodoListMiddleware, ToolCallLimitMiddleware, ToolRetryMiddleware
from langchain.agents.middleware.file_search import FilesystemFileSearchMiddleware
from langchain_openai import ChatOpenAI
from langchain.agent.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware
from langgraph.runtime import Runtime
import os
# 从环境变量读取API密钥
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

model = ChatOpenAI(
    model_name="openai/gpt-oss-20b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.7,
    max_tokens=1024,
)

def LoggginMiddleware(AgentMiddleware):
    def before_model(self,state: AgentState,runtime:Runtime) -> dict[str,Any] | None:
        print(f"About to call model with state: {state}")
        return None
    def after_model(self,state: AgentState,runtime:Runtime) -> None:
        print(f"After calling model with state: {state}")
        return None

agent = create_agent(
    model = model,
    tools = [],
    middleware =[
        # 摘要中间件
        SummarizationMiddleware(
            model = model,
            trigger ={"fraction":0.8},
            keep = {"fraction":0.3},
        ),
        # 人类在循环中间件
        HumanInTheLoopMiddleware(
            interrupt_on = {
                # 中断发送邮件工具, 允许审批、编辑、拒绝
                "send_email_tool":{
                    "allowed_decisions":["approve","edit","reject"],
                },
                # 不中断读取邮件工具, 允许继续执行
                "read_emailtool": False
            }
        ),
        # 模型调用限制中间件
        ModelCallLimitMiddleware(
            # 线程限制, 每个线程最多调用模型10次，默认无限制
            thread_limit=10,
            # 运行限制, 每个运行最多调用模型5次,默认无限制
            run_limit=5,
            # 超出限制时的退出行为, 这里设置为结束运行
            exit_behavior="end",
        ),
        # 工具调用限制中间件
        ToolCallLimitMiddleware(
            # Global limit, 全局工具调用限制, 每个线程最多调用工具20次，默认无限制
            thread_limit=20,
            # 运行限制, 每个运行最多调用工具5次,默认无限制
            run_limit=10,
        ),
        # 工具调用限制中间件, 对发送邮件工具进行单独限制
        ToolCallLimitMiddleware(
            # 工具名称, 对发送邮件工具进行限制
            tool_name="send_email_tool",
            thread_limit=10,
            run_limit=5,
        ),
        # 模型回退中间件, 当主模型调用失败时, 尝试使用备用模型
        ModelFallbackMiddleware(
            # 主模型, 这里使用OpenAI的gpt-oss-20b模型
            "openai/gpt-oss-20b:free",
            # 备用模型, 这里使用OpenAI的gpt-oss-35b模型
            "openai/gpt-oss-35b:free",
        ),
        # PII中间件, 对输入中的个人身份信息进行处理
        PIIMiddleware(
            # 字段名称, 这里对邮箱进行处理，redact策略会将邮箱替换为[REDACTED]
            "email",strategy="redact",apply_to_input=True
        ),
        # PII中间件, 对输入中的个人身份信息进行处理,除开redact、mask策略, 还支持replace策略
        # replace策略会将个人身份信息替换为指定的字符串, 这里替换为[REDACTED], 对信用卡号也生效
        PIIMiddleware(
            # 字段名称, 这里对信用卡号进行处理，mask策略会将信用卡号替换为**** **** **** ****
            "credit_card",strategy="mask",apply_to_input=True
        ),
        # PII中间件, 对输入中的个人身份信息进行处理
        PIIMiddleware(
            # 字段名称, 这里对api_key进行处理, 正则表达式匹配sk-开头的32位字符串
            "api_key",
            # 检测器, 这里使用正则表达式匹配sk-开头的32位字符串
            detector=r"sk-[a-zA-Z0-9-]{32,}",
            strategy="block",
            apply_to_input=True
        ),

        # 待办事项列表中间件, 对输入中的待办事项列表进行处理
        TodoListMiddleware(),
        # LLM工具选择中间件, 对输入中的工具调用进行处理
        LLMToolSelectorMiddleware(
            model = model,
            # 最大工具数, 这里设置为3, 即最多选择3个工具
            max_tools=3,
            # 系统提示, 这里指定了工具列表, 智能助手只能使用这些工具
            system_prompt="你是一个智能助手，只能使用以下工具：{tools}",
            # 总是包含的工具, 这里指定了发送邮件工具
            always_include=["send_email_tool"],
        ),
        # 工具重试中间件, 对工具调用进行重试
        ToolRetryMiddleware(
            # 工具名称, 这里对发送邮件工具进行重试
            tools = ["send_email_tool"],
            # 最大重试次数, 这里设置为3, 即最多重试3次
            max_retries=3,
            # 重试退避因子, 这里设置为2, 即每次重试延迟时间翻倍
            backoff_factor=2,
            # 初始延迟时间, 这里设置为1秒, 即第一次重试延迟1秒
            initial_delay=1,
            # 重试时的回调函数, 这里打印重试信息
            on_failure=lambda tool_name, error: print(f"[重试中间件] 工具 {tool_name} 调用失败: {error}"),
            # 最大重试延迟时间, 这里设置为30秒, 即最大重试延迟30秒
            max_delay=30,
            # 重试时的抖动时间, 这里设置为0.1秒, 即每次重试延迟时间随机增加0.1秒
            jitter=0.1,
            # 重试时的异常类型, 这里指定了连接错误和超时错误
            retry_on=(ConnectionError, TimeoutError),
        ),
        # LLM工具模拟中间件, 对工具调用进行模拟,所有的工具调用都返回成功
        LLMToolEmulator(),
        # LLM工具模拟中间件, 对发送邮件工具进行模拟
        LLMToolEmulator(
            tools =["send_email_tool"]
        ),
        # LLM工具模拟中间件, 对openai/gpt-oss-35b:free模型进行模拟
        LLMToolEmulator(
            model = "openai/gpt-oss-35b:free"
        ),

        # 上下文编辑中间件, 对模型调用上下文进行编辑
        ContextEditingMiddleware(
            edits=[
                # 清除工具调用编辑, 当模型调用上下文超过2000个token时, 清除最近3次调用的工具, 但不清除工具输入
                ClearToolUsesEdit(
                    # 触发条件, 当模型调用上下文超过2000个token时触发
                    trigger= 2000,
                    # 保留次数, 这里保留最近3次调用的工具
                    keep =3,
                    # 不清除工具输入, 即保留工具调用的参数
                    clear_tool_inputs= False,
                    # 排除工具, 这里排除发送邮件工具, 即不清除发送邮件工具的调用
                    exclude_tools=["send_email_tool"],
                    # 占位符, 这里使用[cleared]作为占位符, 表示清除的工具调用
                    placeholder="[cleared]"
                ),
            ]
        ),


        # 壳工具中间件, 对壳工具调用进行处理
        ShellToolMiddleware(
            # shell工具工作目录, 这里指定为./workspace
            workspace_root="./workspace",
            # 执行策略, 这里使用主机执行策略, 即只允许在主机上执行shell命令
            execution_policy= HostExecutionPolicy(),
        ),

        ShellToolMiddleware(
            # shell工具工作目录, 这里指定为./workspace
            workspace_root="./workspace",
            # 启动命令, 这里安装requests库
            startup_commands = ["pip install requests"],
            # 执行策略, 这里使用Docker执行策略, 即只允许在Docker容器中执行shell命令
            execution_policy= DockerExecutionPolicy(
                image = "python:3.11-alpine",
                command_timeout = 60,
            )
        ),

        # 文件搜索中间件, 对输入中的文件路径进行处理
        FilesystemFileSearchMiddleware(
            # 文件搜索工作目录, 这里指定为./workspace
            root_path="./workspace",
            # 是否使用ripgrep, 这里设置为True, 即使用ripgrep进行文件搜索
            use_ripgrep=True,
            # 最大文件大小, 这里设置为10MB, 即最大搜索10MB的文件
            max_file_size_mb=10,
        ),

        # 自定义日志中间件
        LoggginMiddleware(),

    ]
)
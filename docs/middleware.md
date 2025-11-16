# Overview

> Control and customize agent execution at every step

Middleware provides a way to more tightly control what happens inside the agent. Middleware is useful for the following:

* Tracking agent behavior with logging, analytics, and debugging.
* Transforming prompts, [tool selection](/oss/python/langchain/middleware/built-in#llm-tool-selector), and output formatting.
* Adding [retries](/oss/python/langchain/middleware/built-in#tool-retry), [fallbacks](/oss/python/langchain/middleware/built-in#model-fallback), and early termination logic.
* Applying [rate limits](/oss/python/langchain/middleware/built-in#model-call-limit), guardrails, and [PII detection](/oss/python/langchain/middleware/built-in#pii-detection).

Add middleware by passing them to [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent):

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(...),
        HumanInTheLoopMiddleware(...)
    ],
)
```

## The agent loop

The core agent loop involves calling a model, letting it choose tools to execute, and then finishing when it calls no more tools:

<img src="https://mintcdn.com/langchain-5e9cc07a/Tazq8zGc0yYUYrDl/oss/images/core_agent_loop.png?fit=max&auto=format&n=Tazq8zGc0yYUYrDl&q=85&s=ac72e48317a9ced68fd1be64e89ec063" alt="Core agent loop diagram" style={{height: "200px", width: "auto", justifyContent: "center"}} className="rounded-lg block mx-auto" data-og-width="300" width="300" data-og-height="268" height="268" data-path="oss/images/core_agent_loop.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/langchain-5e9cc07a/Tazq8zGc0yYUYrDl/oss/images/core_agent_loop.png?w=280&fit=max&auto=format&n=Tazq8zGc0yYUYrDl&q=85&s=a4c4b766b6678ef52a6ed556b1a0b032 280w, https://mintcdn.com/langchain-5e9cc07a/Tazq8zGc0yYUYrDl/oss/images/core_agent_loop.png?w=560&fit=max&auto=format&n=Tazq8zGc0yYUYrDl&q=85&s=111869e6e99a52c0eff60a1ef7ddc49c 560w, https://mintcdn.com/langchain-5e9cc07a/Tazq8zGc0yYUYrDl/oss/images/core_agent_loop.png?w=840&fit=max&auto=format&n=Tazq8zGc0yYUYrDl&q=85&s=6c1e21de7b53bd0a29683aca09c6f86e 840w, https://mintcdn.com/langchain-5e9cc07a/Tazq8zGc0yYUYrDl/oss/images/core_agent_loop.png?w=1100&fit=max&auto=format&n=Tazq8zGc0yYUYrDl&q=85&s=88bef556edba9869b759551c610c60f4 1100w, https://mintcdn.com/langchain-5e9cc07a/Tazq8zGc0yYUYrDl/oss/images/core_agent_loop.png?w=1650&fit=max&auto=format&n=Tazq8zGc0yYUYrDl&q=85&s=9b0bdd138e9548eeb5056dc0ed2d4a4b 1650w, https://mintcdn.com/langchain-5e9cc07a/Tazq8zGc0yYUYrDl/oss/images/core_agent_loop.png?w=2500&fit=max&auto=format&n=Tazq8zGc0yYUYrDl&q=85&s=41eb4f053ed5e6b0ba5bad2badf6d755 2500w" />

Middleware exposes hooks before and after each of those steps:

<img src="https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=eb4404b137edec6f6f0c8ccb8323eaf1" alt="Middleware flow diagram" style={{height: "300px", width: "auto", justifyContent: "center"}} className="rounded-lg mx-auto" data-og-width="500" width="500" data-og-height="560" height="560" data-path="oss/images/middleware_final.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?w=280&fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=483413aa87cf93323b0f47c0dd5528e8 280w, https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?w=560&fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=41b7dd647447978ff776edafe5f42499 560w, https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?w=840&fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=e9b14e264f68345de08ae76f032c52d4 840w, https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?w=1100&fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=ec45e1932d1279b1beee4a4b016b473f 1100w, https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?w=1650&fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=3bca5ebf8aa56632b8a9826f7f112e57 1650w, https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?w=2500&fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=437f141d1266f08a95f030c2804691d9 2500w" />

## Additional resources

<CardGroup cols={2}>
  <Card title="Built-in middleware" icon="box" href="/oss/python/langchain/middleware/built-in">
    Explore built-in middleware for common use cases.
  </Card>

  <Card title="Custom middleware" icon="code" href="/oss/python/langchain/middleware/custom">
    Build your own middleware with hooks and decorators.
  </Card>

  <Card title="Middleware API reference" icon="book" href="https://reference.langchain.com/python/langchain/middleware/">
    Complete API reference for middleware.
  </Card>

  <Card title="Testing agents" icon="scale-unbalanced" href="/oss/python/langchain/test">
    Test your agents with LangSmith.
  </Card>
</CardGroup>

***

<Callout icon="pen-to-square" iconType="regular">
  [Edit the source of this page on GitHub.](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/middleware/overview.mdx)
</Callout>

<Tip icon="terminal" iconType="regular">
  [Connect these docs programmatically](/use-these-docs) to Claude, VSCode, and more via MCP for    real-time answers.
</Tip>



# Built-in middleware

> Prebuilt middleware for common agent use cases

LangChain provides prebuilt middleware for common use cases. Each middleware is production-ready and configurable for your specific needs.

## Provider-agnostic middleware

The following middleware work with any LLM provider:

| Middleware                              | Description                                                                 |
| --------------------------------------- | --------------------------------------------------------------------------- |
| [Summarization](#summarization)         | Automatically summarize conversation history when approaching token limits. |
| [Human-in-the-loop](#human-in-the-loop) | Pause execution for human approval of tool calls.                           |
| [Model call limit](#model-call-limit)   | Limit the number of model calls to prevent excessive costs.                 |
| [Tool call limit](#tool-call-limit)     | Control tool execution by limiting call counts.                             |
| [Model fallback](#model-fallback)       | Automatically fallback to alternative models when primary fails.            |
| [PII detection](#pii-detection)         | Detect and handle Personally Identifiable Information (PII).                |
| [To-do list](#to-do-list)               | Equip agents with task planning and tracking capabilities.                  |
| [LLM tool selector](#llm-tool-selector) | Use an LLM to select relevant tools before calling main model.              |
| [Tool retry](#tool-retry)               | Automatically retry failed tool calls with exponential backoff.             |
| [LLM tool emulator](#llm-tool-emulator) | Emulate tool execution using an LLM for testing purposes.                   |
| [Context editing](#context-editing)     | Manage conversation context by trimming or clearing tool uses.              |
| [Shell tool](#shell-tool)               | Expose a persistent shell session to agents for command execution.          |
| [File search](#file-search)             | Provide Glob and Grep search tools over filesystem files.                   |

### Summarization

Automatically summarize conversation history when approaching token limits, preserving recent messages while compressing older context. Summarization is useful for the following:

* Long-running conversations that exceed context windows.
* Multi-turn dialogues with extensive history.
* Applications where preserving full conversation context matters.

**API reference:** [`SummarizationMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.SummarizationMiddleware)

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[weather_tool, calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger={"tokens": 4000},
            keep={"messages": 20},
        ),
    ],
)
```

<Accordion title="Configuration options">
  <ParamField body="model" type="string | BaseChatModel" required>
    Model for generating summaries. Can be a model identifier string (e.g., `'openai:gpt-4o-mini'`) or a `BaseChatModel` instance. See [`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model\(model\)) for more information.
  </ParamField>

  <ParamField body="trigger" type="dict | list[dict]">
    Conditions for triggering summarization. Can be:

    * A single condition dict (all properties must be met - AND logic)
    * A list of condition dicts (any condition must be met - OR logic)

    Each condition can include:

    * `fraction` (float): Fraction of model's context size (0-1)
    * `tokens` (int): Absolute token count
    * `messages` (int): Message count

    At least one property must be specified per condition. If not provided, summarization will not trigger automatically.
  </ParamField>

  <ParamField body="keep" type="dict" default="{messages: 20}">
    How much context to preserve after summarization. Specify exactly one of:

    * `fraction` (float): Fraction of model's context size to keep (0-1)
    * `tokens` (int): Absolute token count to keep
    * `messages` (int): Number of recent messages to keep
  </ParamField>

  <ParamField body="token_counter" type="function">
    Custom token counting function. Defaults to character-based counting.
  </ParamField>

  <ParamField body="summary_prompt" type="string">
    Custom prompt template for summarization. Uses built-in template if not specified. The template should include `{messages}` placeholder where conversation history will be inserted.
  </ParamField>

  <ParamField body="trim_tokens_to_summarize" type="number" default="4000">
    Maximum number of tokens to include when generating the summary. Messages will be trimmed to fit this limit before summarization.
  </ParamField>

  <ParamField body="summary_prefix" type="string">
    Prefix to add to the summary message. If not provided, a default prefix is used.
  </ParamField>

  <ParamField body="max_tokens_before_summary" type="number" deprecated>
    **Deprecated:** Use `trigger: {"tokens": value}` instead. Token threshold for triggering summarization.
  </ParamField>

  <ParamField body="messages_to_keep" type="number" deprecated>
    **Deprecated:** Use `keep: {"messages": value}` instead. Recent messages to preserve.
  </ParamField>
</Accordion>

<Accordion title="Full example">
  The summarization middleware monitors message token counts and automatically summarizes older messages when thresholds are reached.

  **Trigger conditions** control when summarization runs:

  * Single condition object (all properties must be met - AND logic)
  * Array of conditions (any condition must be met - OR logic)
  * Each condition can use `fraction` (of model's context size), `tokens` (absolute count), or `messages` (message count)

  **Keep conditions** control how much context to preserve (specify exactly one):

  * `fraction` - Fraction of model's context size to keep
  * `tokens` - Absolute token count to keep
  * `messages` - Number of recent messages to keep

  ```python  theme={null}
  from langchain.agents import create_agent
  from langchain.agents.middleware import SummarizationMiddleware


  # Single condition: trigger if tokens >= 4000 AND messages >= 10
  agent = create_agent(
      model="gpt-4o",
      tools=[weather_tool, calculator_tool],
      middleware=[
          SummarizationMiddleware(
              model="gpt-4o-mini",
              trigger={"tokens": 4000, "messages": 10},
              keep={"messages": 20},
          ),
      ],
  )

  # Multiple conditions
  agent2 = create_agent(
      model="gpt-4o",
      tools=[weather_tool, calculator_tool],
      middleware=[
          SummarizationMiddleware(
              model="gpt-4o-mini",
              trigger=[
                  {"tokens": 5000, "messages": 3},
                  {"tokens": 3000, "messages": 6},
              ],
              keep={"messages": 20},
          ),
      ],
  )

  # Using fractional limits
  agent3 = create_agent(
      model="gpt-4o",
      tools=[weather_tool, calculator_tool],
      middleware=[
          SummarizationMiddleware(
              model="gpt-4o-mini",
              trigger={"fraction": 0.8},
              keep={"fraction": 0.3},
          ),
      ],
  )
  ```
</Accordion>

### Human-in-the-loop

Pause agent execution for human approval, editing, or rejection of tool calls before they execute. [Human-in-the-loop](/oss/python/langchain/human-in-the-loop) is useful for the following:

* High-stakes operations requiring human approval (e.g. database writes, financial transactions).
* Compliance workflows where human oversight is mandatory.
* Long-running conversations where human feedback guides the agent.

**API reference:** [`HumanInTheLoopMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.HumanInTheLoopMiddleware)

<Warning>
  Human-in-the-loop middleware requires a [checkpointer](/oss/python/langgraph/persistence#checkpoints) to maintain state across interruptions.
</Warning>

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "read_email_tool": False,
            }
        ),
    ],
)
```

<Tip>
  For complete examples, configuration options, and integration patterns, see the [Human-in-the-loop documentation](/oss/python/langchain/human-in-the-loop).
</Tip>

### Model call limit

Limit the number of model calls to prevent infinite loops or excessive costs. Model call limit is useful for the following:

* Preventing runaway agents from making too many API calls.
* Enforcing cost controls on production deployments.
* Testing agent behavior within specific call budgets.

**API reference:** [`ModelCallLimitMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelCallLimitMiddleware)

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end",
        ),
    ],
)
```

<Accordion title="Configuration options">
  <ParamField body="thread_limit" type="number">
    Maximum model calls across all runs in a thread. Defaults to no limit.
  </ParamField>

  <ParamField body="run_limit" type="number">
    Maximum model calls per single invocation. Defaults to no limit.
  </ParamField>

  <ParamField body="exit_behavior" type="string" default="end">
    Behavior when limit is reached. Options: `'end'` (graceful termination) or `'error'` (raise exception)
  </ParamField>
</Accordion>

### Tool call limit

Control agent execution by limiting the number of tool calls, either globally across all tools or for specific tools. Tool call limits are useful for the following:

* Preventing excessive calls to expensive external APIs.
* Limiting web searches or database queries.
* Enforcing rate limits on specific tool usage.
* Protecting against runaway agent loops.

**API reference:** [`ToolCallLimitMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ToolCallLimitMiddleware)

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool],
    middleware=[
        # Global limit
        ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
        # Tool-specific limit
        ToolCallLimitMiddleware(
            tool_name="search",
            thread_limit=5,
            run_limit=3,
        ),
    ],
)
```

<Accordion title="Configuration options">
  <ParamField body="tool_name" type="string">
    Name of specific tool to limit. If not provided, limits apply to **all tools globally**.
  </ParamField>

  <ParamField body="thread_limit" type="number">
    Maximum tool calls across all runs in a thread (conversation). Persists across multiple invocations with the same thread ID. Requires a checkpointer to maintain state. `None` means no thread limit.
  </ParamField>

  <ParamField body="run_limit" type="number">
    Maximum tool calls per single invocation (one user message → response cycle). Resets with each new user message. `None` means no run limit.

    **Note:** At least one of `thread_limit` or `run_limit` must be specified.
  </ParamField>

  <ParamField body="exit_behavior" type="string" default="continue">
    Behavior when limit is reached:

    * `'continue'` (default) - Block exceeded tool calls with error messages, let other tools and the model continue. The model decides when to end based on the error messages.
    * `'error'` - Raise a `ToolCallLimitExceededError` exception, stopping execution immediately
    * `'end'` - Stop execution immediately with a `ToolMessage` and AI message for the exceeded tool call. Only works when limiting a single tool; raises `NotImplementedError` if other tools have pending calls.
  </ParamField>
</Accordion>

<Accordion title="Full example">
  Specify limits with:

  * **Thread limit** - Max calls across all runs in a conversation (requires checkpointer)
  * **Run limit** - Max calls per single invocation (resets each turn)

  Exit behaviors:

  * `'continue'` (default) - Block exceeded calls with error messages, agent continues
  * `'error'` - Raise exception immediately
  * `'end'` - Stop with ToolMessage + AI message (single-tool scenarios only)

  ```python  theme={null}
  from langchain.agents import create_agent
  from langchain.agents.middleware import ToolCallLimitMiddleware


  global_limiter = ToolCallLimitMiddleware(thread_limit=20, run_limit=10)
  search_limiter = ToolCallLimitMiddleware(tool_name="search", thread_limit=5, run_limit=3)
  database_limiter = ToolCallLimitMiddleware(tool_name="query_database", thread_limit=10)
  strict_limiter = ToolCallLimitMiddleware(tool_name="scrape_webpage", run_limit=2, exit_behavior="error")

  agent = create_agent(
      model="gpt-4o",
      tools=[search_tool, database_tool, scraper_tool],
      middleware=[global_limiter, search_limiter, database_limiter, strict_limiter],
  )
  ```
</Accordion>

### Model fallback

Automatically fallback to alternative models when the primary model fails. Model fallback is useful for the following:

* Building resilient agents that handle model outages.
* Cost optimization by falling back to cheaper models.
* Provider redundancy across OpenAI, Anthropic, etc.

**API reference:** [`ModelFallbackMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelFallbackMiddleware)

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        ModelFallbackMiddleware(
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022",
        ),
    ],
)
```

<Accordion title="Configuration options">
  <ParamField body="first_model" type="string | BaseChatModel" required>
    First fallback model to try when the primary model fails. Can be a model identifier string (e.g., `'openai:gpt-4o-mini'`) or a `BaseChatModel` instance.
  </ParamField>

  <ParamField body="*additional_models" type="string | BaseChatModel">
    Additional fallback models to try in order if previous models fail
  </ParamField>
</Accordion>

### PII detection

Detect and handle Personally Identifiable Information (PII) in conversations using configurable strategies. PII detection is useful for the following:

* Healthcare and financial applications with compliance requirements.
* Customer service agents that need to sanitize logs.
* Any application handling sensitive user data.

**API reference:** [`PIIMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.PIIMiddleware)

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
    ],
)
```

#### Custom PII types

You can create custom PII types by providing a `detector` parameter. This allows you to detect patterns specific to your use case beyond the built-in types.

**Three ways to create custom detectors:**

1. **Regex pattern string** - Simple pattern matching

2. **Custom function** - Complex detection logic with validation

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
import re


# Method 1: Regex pattern string
agent1 = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
        ),
    ],
)

# Method 2: Compiled regex pattern
agent2 = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        PIIMiddleware(
            "phone_number",
            detector=re.compile(r"\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{4}"),
            strategy="mask",
        ),
    ],
)

# Method 3: Custom detector function
def detect_ssn(content: str) -> list[dict[str, str | int]]:
    """Detect SSN with validation.

    Returns a list of dictionaries with 'text', 'start', and 'end' keys.
    """
    import re
    matches = []
    pattern = r"\d{3}-\d{2}-\d{4}"
    for match in re.finditer(pattern, content):
        ssn = match.group(0)
        # Validate: first 3 digits shouldn't be 000, 666, or 900-999
        first_three = int(ssn[:3])
        if first_three not in [0, 666] and not (900 <= first_three <= 999):
            matches.append({
                "text": ssn,
                "start": match.start(),
                "end": match.end(),
            })
    return matches

agent3 = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        PIIMiddleware(
            "ssn",
            detector=detect_ssn,
            strategy="hash",
        ),
    ],
)
```

**Custom detector function signature:**

The detector function must accept a string (content) and return matches:

Returns a list of dictionaries with `text`, `start`, and `end` keys:

```python  theme={null}
def detector(content: str) -> list[dict[str, str | int]]:
    return [
        {"text": "matched_text", "start": 0, "end": 12},
        # ... more matches
    ]
```

<Tip>
  For custom detectors:

  * Use regex strings for simple patterns
  * Use RegExp objects when you need flags (e.g., case-insensitive matching)
  * Use custom functions when you need validation logic beyond pattern matching
  * Custom functions give you full control over detection logic and can implement complex validation rules
</Tip>

<Accordion title="Configuration options">
  <ParamField body="pii_type" type="string" required>
    Type of PII to detect. Can be a built-in type (`email`, `credit_card`, `ip`, `mac_address`, `url`) or a custom type name.
  </ParamField>

  <ParamField body="strategy" type="string" default="redact">
    How to handle detected PII. Options:

    * `'block'` - Raise exception when detected
    * `'redact'` - Replace with `[REDACTED_TYPE]`
    * `'mask'` - Partially mask (e.g., `****-****-****-1234`)
    * `'hash'` - Replace with deterministic hash
  </ParamField>

  <ParamField body="detector" type="function | regex">
    Custom detector function or regex pattern. If not provided, uses built-in detector for the PII type.
  </ParamField>

  <ParamField body="apply_to_input" type="boolean" default="True">
    Check user messages before model call
  </ParamField>

  <ParamField body="apply_to_output" type="boolean" default="False">
    Check AI messages after model call
  </ParamField>

  <ParamField body="apply_to_tool_results" type="boolean" default="False">
    Check tool result messages after execution
  </ParamField>
</Accordion>

### To-do list

Equip agents with task planning and tracking capabilities for complex multi-step tasks. To-do lists are useful for the following:

* Complex multi-step tasks requiring coordination across multiple tools.
* Long-running operations where progress visibility is important.

<Note>
  This middleware automatically provides agents with a `write_todos` tool and system prompts to guide effective task planning.
</Note>

**API reference:** [`TodoListMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.TodoListMiddleware)

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[read_file, write_file, run_tests],
    middleware=[TodoListMiddleware()],
)
```

<Accordion title="Configuration options">
  <ParamField body="system_prompt" type="string">
    Custom system prompt for guiding todo usage. Uses built-in prompt if not specified.
  </ParamField>

  <ParamField body="tool_description" type="string">
    Custom description for the `write_todos` tool. Uses built-in description if not specified.
  </ParamField>
</Accordion>

### LLM tool selector

Use an LLM to intelligently select relevant tools before calling the main model. LLM tool selectors are useful for the following:

* Agents with many tools (10+) where most aren't relevant per query.
* Reducing token usage by filtering irrelevant tools.
* Improving model focus and accuracy.

This middleware uses structured output to ask an LLM which tools are most relevant for the current query. The structured output schema defines the available tool names and descriptions. Model providers often add this structured output information to the system prompt behind the scenes.

**API reference:** [`LLMToolSelectorMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.LLMToolSelectorMiddleware)

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[tool1, tool2, tool3, tool4, tool5, ...],
    middleware=[
        LLMToolSelectorMiddleware(
            model="gpt-4o-mini",
            max_tools=3,
            always_include=["search"],
        ),
    ],
)
```

<Accordion title="Configuration options">
  <ParamField body="model" type="string | BaseChatModel">
    Model for tool selection. Can be a model identifier string (e.g., `'openai:gpt-4o-mini'`) or a `BaseChatModel` instance. See [`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model\(model\)) for more information.

    Defaults to the agent's main model.
  </ParamField>

  <ParamField body="system_prompt" type="string">
    Instructions for the selection model. Uses built-in prompt if not specified.
  </ParamField>

  <ParamField body="max_tools" type="number">
    Maximum number of tools to select. If the model selects more, only the first max\_tools will be used. No limit if not specified.
  </ParamField>

  <ParamField body="always_include" type="list[string]">
    Tool names to always include regardless of selection. These do not count against the max\_tools limit.
  </ParamField>
</Accordion>

### Tool retry

Automatically retry failed tool calls with configurable exponential backoff. Tool retry is useful for the following:

* Handling transient failures in external API calls.
* Improving reliability of network-dependent tools.
* Building resilient agents that gracefully handle temporary errors.

**API reference:** [`ToolRetryMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ToolRetryMiddleware)

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
    ],
)
```

<Accordion title="Configuration options">
  <ParamField body="max_retries" type="number" default="2">
    Maximum number of retry attempts after the initial call (3 total attempts with default)
  </ParamField>

  <ParamField body="tools" type="list[BaseTool | str]">
    Optional list of tools or tool names to apply retry logic to. If `None`, applies to all tools.
  </ParamField>

  <ParamField body="retry_on" type="tuple[type[Exception], ...] | callable" default="(Exception,)">
    Either a tuple of exception types to retry on, or a callable that takes an exception and returns `True` if it should be retried.
  </ParamField>

  <ParamField body="on_failure" type="string | callable" default="return_message">
    Behavior when all retries are exhausted. Options:

    * `'return_message'` - Return a `ToolMessage` with error details (allows LLM to handle failure)
    * `'raise'` - Re-raise the exception (stops agent execution)
    * Custom callable - Function that takes the exception and returns a string for the `ToolMessage` content
  </ParamField>

  <ParamField body="backoff_factor" type="number" default="2.0">
    Multiplier for exponential backoff. Each retry waits `initial_delay * (backoff_factor ** retry_number)` seconds. Set to `0.0` for constant delay.
  </ParamField>

  <ParamField body="initial_delay" type="number" default="1.0">
    Initial delay in seconds before first retry
  </ParamField>

  <ParamField body="max_delay" type="number" default="60.0">
    Maximum delay in seconds between retries (caps exponential backoff growth)
  </ParamField>

  <ParamField body="jitter" type="boolean" default="true">
    Whether to add random jitter (`±25%`) to delay to avoid thundering herd
  </ParamField>
</Accordion>

<Accordion title="Full example">
  The middleware automatically retries failed tool calls with exponential backoff.

  **Key configuration:**

  * `max_retries` - Number of retry attempts (default: 2)
  * `backoff_factor` - Multiplier for exponential backoff (default: 2.0)
  * `initial_delay` - Starting delay in seconds (default: 1.0)
  * `max_delay` - Cap on delay growth (default: 60.0)
  * `jitter` - Add random variation (default: True)

  **Failure handling:**

  * `on_failure='return_message'` - Return error message
  * `on_failure='raise'` - Re-raise exception
  * Custom function - Function returning error message

  ```python  theme={null}
  from langchain.agents import create_agent
  from langchain.agents.middleware import ToolRetryMiddleware


  agent = create_agent(
      model="gpt-4o",
      tools=[search_tool, database_tool, api_tool],
      middleware=[
          ToolRetryMiddleware(
              max_retries=3,
              backoff_factor=2.0,
              initial_delay=1.0,
              max_delay=60.0,
              jitter=True,
              tools=["api_tool"],
              retry_on=(ConnectionError, TimeoutError),
              on_failure="return_message",
          ),
      ],
  )
  ```
</Accordion>

### LLM tool emulator

Emulate tool execution using an LLM for testing purposes, replacing actual tool calls with AI-generated responses. LLM tool emulators are useful for the following:

* Testing agent behavior without executing real tools.
* Developing agents when external tools are unavailable or expensive.
* Prototyping agent workflows before implementing actual tools.

**API reference:** [`LLMToolEmulator`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.LLMToolEmulator)

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather, search_database, send_email],
    middleware=[
        LLMToolEmulator(),  # Emulate all tools
    ],
)
```

<Accordion title="Configuration options">
  <ParamField body="tools" type="list[str | BaseTool]">
    List of tool names (str) or BaseTool instances to emulate. If `None` (default), ALL tools will be emulated. If empty list `[]`, no tools will be emulated. If array with tool names/instances, only those tools will be emulated.
  </ParamField>

  <ParamField body="model" type="string | BaseChatModel">
    Model to use for generating emulated tool responses. Can be a model identifier string (e.g., `'anthropic:claude-sonnet-4-5-20250929'`) or a `BaseChatModel` instance. Defaults to the agent's model if not specified. See [`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model\(model\)) for more information.
  </ParamField>
</Accordion>

<Accordion title="Full example">
  The middleware uses an LLM to generate plausible responses for tool calls instead of executing the actual tools.

  ```python  theme={null}
  from langchain.agents import create_agent
  from langchain.agents.middleware import LLMToolEmulator
  from langchain_core.tools import tool


  @tool
  def get_weather(location: str) -> str:
      """Get the current weather for a location."""
      return f"Weather in {location}"

  @tool
  def send_email(to: str, subject: str, body: str) -> str:
      """Send an email."""
      return "Email sent"


  # Emulate all tools (default behavior)
  agent = create_agent(
      model="gpt-4o",
      tools=[get_weather, send_email],
      middleware=[LLMToolEmulator()],
  )

  # Emulate specific tools only
  agent2 = create_agent(
      model="gpt-4o",
      tools=[get_weather, send_email],
      middleware=[LLMToolEmulator(tools=["get_weather"])],
  )

  # Use custom model for emulation
  agent4 = create_agent(
      model="gpt-4o",
      tools=[get_weather, send_email],
      middleware=[LLMToolEmulator(model="anthropic:claude-sonnet-4-5-20250929")],
  )
  ```
</Accordion>

### Context editing

Manage conversation context by clearing older tool call outputs when token limits are reached, while preserving recent results. This helps keep context windows manageable in long conversations with many tool calls. Context editing is useful for the following:

* Long conversations with many tool calls that exceed token limits
* Reducing token costs by removing older tool outputs that are no longer relevant
* Maintaining only the most recent N tool results in context

**API reference:** [`ContextEditingMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ContextEditingMiddleware), [`ClearToolUsesEdit`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ClearToolUsesEdit)

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=100000,
                    keep=3,
                ),
            ],
        ),
    ],
)
```

<Accordion title="Configuration options">
  <ParamField body="edits" type="list[ContextEdit]" default="[ClearToolUsesEdit()]">
    List of [`ContextEdit`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ContextEdit) strategies to apply
  </ParamField>

  <ParamField body="token_count_method" type="string" default="approximate">
    Token counting method. Options: `'approximate'` or `'model'`
  </ParamField>

  **[`ClearToolUsesEdit`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ClearToolUsesEdit) options:**

  <ParamField body="trigger" type="number" default="100000">
    Token count that triggers the edit. When the conversation exceeds this token count, older tool outputs will be cleared.
  </ParamField>

  <ParamField body="clear_at_least" type="number" default="0">
    Minimum number of tokens to reclaim when the edit runs. If set to 0, clears as much as needed.
  </ParamField>

  <ParamField body="keep" type="number" default="3">
    Number of most recent tool results that must be preserved. These will never be cleared.
  </ParamField>

  <ParamField body="clear_tool_inputs" type="boolean" default="False">
    Whether to clear the originating tool call parameters on the AI message. When `True`, tool call arguments are replaced with empty objects.
  </ParamField>

  <ParamField body="exclude_tools" type="list[string]" default="()">
    List of tool names to exclude from clearing. These tools will never have their outputs cleared.
  </ParamField>

  <ParamField body="placeholder" type="string" default="[cleared]">
    Placeholder text inserted for cleared tool outputs. This replaces the original tool message content.
  </ParamField>
</Accordion>

<Accordion title="Full example">
  The middleware applies context editing strategies when token limits are reached. The most common strategy is `ClearToolUsesEdit`, which clears older tool results while preserving recent ones.

  **How it works:**

  1. Monitor token count in conversation
  2. When threshold is reached, clear older tool outputs
  3. Keep most recent N tool results
  4. Optionally preserve tool call arguments for context

  ```python  theme={null}
  from langchain.agents import create_agent
  from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit


  agent = create_agent(
      model="gpt-4o",
      tools=[search_tool, calculator_tool, database_tool],
      middleware=[
          ContextEditingMiddleware(
              edits=[
                  ClearToolUsesEdit(
                      trigger=2000,
                      keep=3,
                      clear_tool_inputs=False,
                      exclude_tools=[],
                      placeholder="[cleared]",
                  ),
              ],
          ),
      ],
  )
  ```
</Accordion>

### Shell tool

Expose a persistent shell session to agents for command execution. Shell tool middleware is useful for the following:

* Agents that need to execute system commands
* Development and deployment automation tasks
* Testing and validation workflows
* File system operations and script execution

<Warning>
  **Security consideration**: Use appropriate execution policies (`HostExecutionPolicy`, `DockerExecutionPolicy`, or `CodexSandboxExecutionPolicy`) to match your deployment's security requirements.
</Warning>

<Note>
  **Limitation**: Persistent shell sessions do not currently work with interrupts (human-in-the-loop). We anticipate adding support for this in the future.
</Note>

**API reference:** @\[`ShellToolMiddleware`]

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ShellToolMiddleware,
    HostExecutionPolicy,
)

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/workspace",
            execution_policy=HostExecutionPolicy(),
        ),
    ],
)
```

<Accordion title="Configuration options">
  <ParamField body="workspace_root" type="str | Path | None">
    Base directory for the shell session. If omitted, a temporary directory is created when the agent starts and removed when it ends.
  </ParamField>

  <ParamField body="startup_commands" type="tuple[str, ...] | list[str] | str | None">
    Optional commands executed sequentially after the session starts
  </ParamField>

  <ParamField body="shutdown_commands" type="tuple[str, ...] | list[str] | str | None">
    Optional commands executed before the session shuts down
  </ParamField>

  <ParamField body="execution_policy" type="BaseExecutionPolicy | None">
    Execution policy controlling timeouts, output limits, and resource configuration. Options:

    * `HostExecutionPolicy` - Full host access (default); best for trusted environments where the agent already runs inside a container or VM
    * `DockerExecutionPolicy` - Launches a separate Docker container for each agent run, providing harder isolation
    * `CodexSandboxExecutionPolicy` - Reuses the Codex CLI sandbox for additional syscall/filesystem restrictions
  </ParamField>

  <ParamField body="redaction_rules" type="tuple[RedactionRule, ...] | list[RedactionRule] | None">
    Optional redaction rules to sanitize command output before returning it to the model
  </ParamField>

  <ParamField body="tool_description" type="str | None">
    Optional override for the registered shell tool description
  </ParamField>

  <ParamField body="shell_command" type="Sequence[str] | str | None">
    Optional shell executable (string) or argument sequence used to launch the persistent session. Defaults to `/bin/bash`.
  </ParamField>

  <ParamField body="env" type="Mapping[str, Any] | None">
    Optional environment variables to supply to the shell session. Values are coerced to strings before command execution.
  </ParamField>
</Accordion>

<Accordion title="Full example">
  The middleware provides a single persistent shell session that agents can use to execute commands sequentially.

  **Execution policies:**

  * `HostExecutionPolicy` (default) - Native execution with full host access
  * `DockerExecutionPolicy` - Isolated Docker container execution
  * `CodexSandboxExecutionPolicy` - Sandboxed execution via Codex CLI

  ```python  theme={null}
  from langchain.agents import create_agent
  from langchain.agents.middleware import (
      ShellToolMiddleware,
      HostExecutionPolicy,
      DockerExecutionPolicy,
      RedactionRule,
  )


  # Basic shell tool with host execution
  agent = create_agent(
      model="gpt-4o",
      tools=[search_tool],
      middleware=[
          ShellToolMiddleware(
              workspace_root="/workspace",
              execution_policy=HostExecutionPolicy(),
          ),
      ],
  )

  # Docker isolation with startup commands
  agent_docker = create_agent(
      model="gpt-4o",
      tools=[],
      middleware=[
          ShellToolMiddleware(
              workspace_root="/workspace",
              startup_commands=["pip install requests", "export PYTHONPATH=/workspace"],
              execution_policy=DockerExecutionPolicy(
                  image="python:3.11-slim",
                  command_timeout=60.0,
              ),
          ),
      ],
  )

  # With output redaction
  agent_redacted = create_agent(
      model="gpt-4o",
      tools=[],
      middleware=[
          ShellToolMiddleware(
              workspace_root="/workspace",
              redaction_rules=[
                  RedactionRule(pii_type="api_key", detector=r"sk-[a-zA-Z0-9]{32}"),
              ],
          ),
      ],
  )
  ```
</Accordion>

### File search

Provide Glob and Grep search tools over filesystem files. File search middleware is useful for the following:

* Code exploration and analysis
* Finding files by name patterns
* Searching code content with regex
* Large codebases where file discovery is needed

**API reference:** @\[`FilesystemFileSearchMiddleware`]

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        FilesystemFileSearchMiddleware(
            root_path="/workspace",
            use_ripgrep=True,
        ),
    ],
)
```

<Accordion title="Configuration options">
  <ParamField body="root_path" type="str" required>
    Root directory to search. All file operations are relative to this path.
  </ParamField>

  <ParamField body="use_ripgrep" type="bool" default="True">
    Whether to use ripgrep for search. Falls back to Python regex if ripgrep is unavailable.
  </ParamField>

  <ParamField body="max_file_size_mb" type="int" default="10">
    Maximum file size to search in MB. Files larger than this are skipped.
  </ParamField>
</Accordion>

<Accordion title="Full example">
  The middleware adds two search tools to agents:

  **Glob tool** - Fast file pattern matching:

  * Supports patterns like `**/*.py`, `src/**/*.ts`
  * Returns matching file paths sorted by modification time

  **Grep tool** - Content search with regex:

  * Full regex syntax support
  * Filter by file patterns with `include` parameter
  * Three output modes: `files_with_matches`, `content`, `count`

  ```python  theme={null}
  from langchain.agents import create_agent
  from langchain.agents.middleware import FilesystemFileSearchMiddleware
  from langchain_core.messages import HumanMessage


  agent = create_agent(
      model="gpt-4o",
      tools=[],
      middleware=[
          FilesystemFileSearchMiddleware(
              root_path="/workspace",
              use_ripgrep=True,
              max_file_size_mb=10,
          ),
      ],
  )

  # Agent can now use glob_search and grep_search tools
  result = agent.invoke({
      "messages": [HumanMessage("Find all Python files containing 'async def'")]
  })

  # The agent will use:
  # 1. glob_search(pattern="**/*.py") to find Python files
  # 2. grep_search(pattern="async def", include="*.py") to find async functions
  ```
</Accordion>

## Provider-specific middleware

These middleware are optimized for specific LLM providers.

### Anthropic

Middleware specifically designed for Anthropic's Claude models.

| Middleware                        | Description                                                    |
| --------------------------------- | -------------------------------------------------------------- |
| [Prompt caching](#prompt-caching) | Reduce costs by caching repetitive prompt prefixes             |
| [Bash tool](#bash-tool)           | Execute Claude's native bash tool with local command execution |
| [Text editor](#text-editor)       | Provide Claude's text editor tool for file editing             |
| [Memory](#memory)                 | Provide Claude's memory tool for persistent agent memory       |
| [File search](#file-search-1)     | Search tools for state-based file systems                      |

#### Prompt caching

Reduce costs and latency by caching static or repetitive prompt content (like system prompts, tool definitions, and conversation history) on Anthropic's servers. This middleware implements a **conversational caching strategy** that places cache breakpoints after the most recent message, allowing the entire conversation history (including the latest user message) to be cached and reused in subsequent API calls. Prompt caching is useful for the following:

* Applications with long, static system prompts that don't change between requests
* Agents with many tool definitions that remain constant across invocations
* Conversations where early message history is reused across multiple turns
* High-volume deployments where reducing API costs and latency is critical

<Info>
  Learn more about [Anthropic prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching#cache-limitations) strategies and limitations.
</Info>

**API reference:** [`AnthropicPromptCachingMiddleware`](https://reference.langchain.com/python/integrations/langchain_anthropic/middleware/#langchain_anthropic.middleware.AnthropicPromptCachingMiddleware)

```python  theme={null}
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain.agents import create_agent

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
    system_prompt="<Your long system prompt here>",
    middleware=[AnthropicPromptCachingMiddleware(ttl="5m")],
)
```

<Accordion title="Configuration options">
  <ParamField body="type" type="string" default="ephemeral">
    Cache type. Only `'ephemeral'` is currently supported.
  </ParamField>

  <ParamField body="ttl" type="string" default="5m">
    Time to live for cached content. Valid values: `'5m'` or `'1h'`
  </ParamField>

  <ParamField body="min_messages_to_cache" type="number" default="0">
    Minimum number of messages before caching starts
  </ParamField>

  <ParamField body="unsupported_model_behavior" type="string" default="warn">
    Behavior when using non-Anthropic models. Options: `'ignore'`, `'warn'`, or `'raise'`
  </ParamField>
</Accordion>

<Accordion title="Full example">
  The middleware caches content up to and including the latest message in each request. On subsequent requests within the TTL window (5 minutes or 1 hour), previously seen content is retrieved from cache rather than reprocessed, significantly reducing costs and latency.

  **How it works:**

  1. First request: System prompt, tools, and the user message "Hi, my name is Bob" are sent to the API and cached
  2. Second request: The cached content (system prompt, tools, and first message) is retrieved from cache. Only the new message "What's my name?" needs to be processed, plus the model's response from the first request
  3. This pattern continues for each turn, with each request reusing the cached conversation history

  ```python  theme={null}
  from langchain_anthropic import ChatAnthropic
  from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
  from langchain.agents import create_agent
  from langchain_core.messages import HumanMessage


  LONG_PROMPT = """
  Please be a helpful assistant.

  <Lots more context ...>
  """

  agent = create_agent(
      model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
      system_prompt=LONG_PROMPT,
      middleware=[AnthropicPromptCachingMiddleware(ttl="5m")],
  )

  # First invocation: Creates cache with system prompt, tools, and "Hi, my name is Bob"
  agent.invoke({"messages": [HumanMessage("Hi, my name is Bob")]})

  # Second invocation: Reuses cached system prompt, tools, and previous messages
  # Only processes the new message "What's my name?" and the previous AI response
  agent.invoke({"messages": [HumanMessage("What's my name?")]})
  ```
</Accordion>

#### Bash tool

Execute Claude's native `bash_20250124` tool with local command execution. The bash tool middleware is useful for the following:

* Using Claude's built-in bash tool with local execution
* Leveraging Claude's optimized bash tool interface
* Agents that need persistent shell sessions with Anthropic models

<Info>
  This middleware wraps `ShellToolMiddleware` and exposes it as Claude's native bash tool.
</Info>

**API reference:** @\[`ClaudeBashToolMiddleware`]

```python  theme={null}
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import ClaudeBashToolMiddleware
from langchain.agents import create_agent

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
    tools=[],
    middleware=[
        ClaudeBashToolMiddleware(
            workspace_root="/workspace",
        ),
    ],
)
```

<Accordion title="Configuration options">
  `ClaudeBashToolMiddleware` accepts all parameters from @\[`ShellToolMiddleware`], including:

  <ParamField body="workspace_root" type="str | Path | None">
    Base directory for the shell session
  </ParamField>

  <ParamField body="startup_commands" type="tuple[str, ...] | list[str] | str | None">
    Commands to run when the session starts
  </ParamField>

  <ParamField body="execution_policy" type="BaseExecutionPolicy | None">
    Execution policy (`HostExecutionPolicy`, `DockerExecutionPolicy`, or `CodexSandboxExecutionPolicy`)
  </ParamField>

  <ParamField body="redaction_rules" type="tuple[RedactionRule, ...] | list[RedactionRule] | None">
    Rules for sanitizing command output
  </ParamField>

  See [Shell tool](#shell-tool) for full configuration details.
</Accordion>

<Accordion title="Full example">
  ```python  theme={null}
  from langchain_anthropic import ChatAnthropic
  from langchain_anthropic.middleware import ClaudeBashToolMiddleware
  from langchain.agents import create_agent
  from langchain.agents.middleware import DockerExecutionPolicy


  agent = create_agent(
      model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
      tools=[],
      middleware=[
          ClaudeBashToolMiddleware(
              workspace_root="/workspace",
              startup_commands=["pip install requests"],
              execution_policy=DockerExecutionPolicy(
                  image="python:3.11-slim",
              ),
          ),
      ],
  )

  # Claude can now use its native bash tool
  result = agent.invoke({
      "messages": [{"role": "user", "content": "List files in the workspace"}]
  })
  ```
</Accordion>

#### Text editor

Provide Claude's text editor tool (`text_editor_20250728`) for file creation and editing. The text editor middleware is useful for the following:

* File-based agent workflows
* Code editing and refactoring tasks
* Multi-file project work
* Agents that need persistent file storage

<Note>
  Available in two variants: **State-based** (files in LangGraph state) and **Filesystem-based** (files on disk).
</Note>

**API reference:** @\[`StateClaudeTextEditorMiddleware`], @\[`FilesystemClaudeTextEditorMiddleware`]

```python  theme={null}
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import StateClaudeTextEditorMiddleware
from langchain.agents import create_agent

# State-based (files in LangGraph state)
agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
    tools=[],
    middleware=[
        StateClaudeTextEditorMiddleware(),
    ],
)
```

<Accordion title="Configuration options">
  **@\[`StateClaudeTextEditorMiddleware`] (state-based)**

  <ParamField body="allowed_path_prefixes" type="Sequence[str] | None">
    Optional list of allowed path prefixes. If specified, only paths starting with these prefixes are allowed.
  </ParamField>

  **@\[`FilesystemClaudeTextEditorMiddleware`] (filesystem-based)**

  <ParamField body="root_path" type="str" required>
    Root directory for file operations
  </ParamField>

  <ParamField body="allowed_prefixes" type="list[str] | None">
    Optional list of allowed virtual path prefixes (default: `["/"]`)
  </ParamField>

  <ParamField body="max_file_size_mb" type="int" default="10">
    Maximum file size in MB
  </ParamField>
</Accordion>

<Accordion title="Full example">
  Claude's text editor tool supports the following commands:

  * `view` - View file contents or list directory
  * `create` - Create a new file
  * `str_replace` - Replace string in file
  * `insert` - Insert text at line number
  * `delete` - Delete a file
  * `rename` - Rename/move a file

  ```python  theme={null}
  from langchain_anthropic import ChatAnthropic
  from langchain_anthropic.middleware import (
      StateClaudeTextEditorMiddleware,
      FilesystemClaudeTextEditorMiddleware,
  )
  from langchain.agents import create_agent


  # State-based: Files persist in LangGraph state
  agent_state = create_agent(
      model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
      tools=[],
      middleware=[
          StateClaudeTextEditorMiddleware(
              allowed_path_prefixes=["/project"],
          ),
      ],
  )

  # Filesystem-based: Files persist on disk
  agent_fs = create_agent(
      model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
      tools=[],
      middleware=[
          FilesystemClaudeTextEditorMiddleware(
              root_path="/workspace",
              allowed_prefixes=["/src"],
              max_file_size_mb=10,
          ),
      ],
  )
  ```
</Accordion>

#### Memory

Provide Claude's memory tool (`memory_20250818`) for persistent agent memory across conversation turns. The memory middleware is useful for the following:

* Long-running agent conversations
* Maintaining context across interruptions
* Task progress tracking
* Persistent agent state management

<Info>
  Claude's memory tool uses a `/memories` directory and automatically injects a system prompt encouraging the agent to check and update memory.
</Info>

**API reference:** @\[`StateClaudeMemoryMiddleware`], @\[`FilesystemClaudeMemoryMiddleware`]

```python  theme={null}
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import StateClaudeMemoryMiddleware
from langchain.agents import create_agent

# State-based memory
agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
    tools=[],
    middleware=[
        StateClaudeMemoryMiddleware(),
    ],
)
```

<Accordion title="Configuration options">
  **@\[`StateClaudeMemoryMiddleware`] (state-based)**

  <ParamField body="allowed_path_prefixes" type="Sequence[str] | None">
    Optional list of allowed path prefixes. Defaults to `["/memories"]`.
  </ParamField>

  <ParamField body="system_prompt" type="str">
    System prompt to inject. Defaults to Anthropic's recommended memory prompt that encourages the agent to check and update memory.
  </ParamField>

  **@\[`FilesystemClaudeMemoryMiddleware`] (filesystem-based)**

  <ParamField body="root_path" type="str" required>
    Root directory for file operations
  </ParamField>

  <ParamField body="allowed_prefixes" type="list[str] | None">
    Optional list of allowed virtual path prefixes. Defaults to `["/memories"]`.
  </ParamField>

  <ParamField body="max_file_size_mb" type="int" default="10">
    Maximum file size in MB
  </ParamField>

  <ParamField body="system_prompt" type="str">
    System prompt to inject
  </ParamField>
</Accordion>

<Accordion title="Full example">
  ```python  theme={null}
  from langchain_anthropic import ChatAnthropic
  from langchain_anthropic.middleware import (
      StateClaudeMemoryMiddleware,
      FilesystemClaudeMemoryMiddleware,
  )
  from langchain.agents import create_agent


  # State-based: Memory persists in LangGraph state
  agent_state = create_agent(
      model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
      tools=[],
      middleware=[
          StateClaudeMemoryMiddleware(),
      ],
  )

  # Filesystem-based: Memory persists on disk
  agent_fs = create_agent(
      model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
      tools=[],
      middleware=[
          FilesystemClaudeMemoryMiddleware(
              root_path="/workspace",
          ),
      ],
  )

  # The agent will automatically:
  # 1. Check /memories directory at start
  # 2. Record progress and thoughts during execution
  # 3. Update memory files as work progresses
  ```
</Accordion>

#### File search

Provide Glob and Grep search tools for files stored in LangGraph state. File search middleware is useful for the following:

* Searching through state-based virtual file systems
* Works with text editor and memory tools
* Finding files by patterns
* Content search with regex

**API reference:** @\[`StateFileSearchMiddleware`]

```python  theme={null}
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import (
    StateClaudeTextEditorMiddleware,
    StateFileSearchMiddleware,
)
from langchain.agents import create_agent

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
    tools=[],
    middleware=[
        StateClaudeTextEditorMiddleware(),
        StateFileSearchMiddleware(),  # Search text editor files
    ],
)
```

<Accordion title="Configuration options">
  <ParamField body="state_key" type="str" default="text_editor_files">
    State key containing files to search. Use `"text_editor_files"` for text editor files or `"memory_files"` for memory files.
  </ParamField>
</Accordion>

<Accordion title="Full example">
  The middleware adds Glob and Grep search tools that work with state-based files.

  ```python  theme={null}
  from langchain_anthropic import ChatAnthropic
  from langchain_anthropic.middleware import (
      StateClaudeTextEditorMiddleware,
      StateClaudeMemoryMiddleware,
      StateFileSearchMiddleware,
  )
  from langchain.agents import create_agent


  # Search text editor files
  agent = create_agent(
      model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
      tools=[],
      middleware=[
          StateClaudeTextEditorMiddleware(),
          StateFileSearchMiddleware(state_key="text_editor_files"),
      ],
  )

  # Search memory files
  agent_memory = create_agent(
      model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
      tools=[],
      middleware=[
          StateClaudeMemoryMiddleware(),
          StateFileSearchMiddleware(state_key="memory_files"),
      ],
  )
  ```
</Accordion>

### OpenAI

Middleware specifically designed for OpenAI models.

| Middleware                                | Description                                               |
| ----------------------------------------- | --------------------------------------------------------- |
| [Content moderation](#content-moderation) | Moderate agent traffic using OpenAI's moderation endpoint |

#### Content moderation

Moderate agent traffic (user input, model output, and tool results) using OpenAI's moderation endpoint to detect and handle unsafe content. Content moderation is useful for the following:

* Applications requiring content safety and compliance
* Filtering harmful, hateful, or inappropriate content
* Customer-facing agents that need safety guardrails
* Meeting platform moderation requirements

<Info>
  Learn more about [OpenAI's moderation models](https://platform.openai.com/docs/guides/moderation) and categories.
</Info>

**API reference:** @\[`OpenAIModerationMiddleware`]

```python  theme={null}
from langchain_openai import ChatOpenAI
from langchain_openai.middleware import OpenAIModerationMiddleware
from langchain.agents import create_agent

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[search_tool, database_tool],
    middleware=[
        OpenAIModerationMiddleware(
            model="omni-moderation-latest",
            check_input=True,
            check_output=True,
            exit_behavior="end",
        ),
    ],
)
```

<Accordion title="Configuration options">
  <ParamField body="model" type="ModerationModel" default="omni-moderation-latest">
    OpenAI moderation model to use. Options: `'omni-moderation-latest'`, `'omni-moderation-2024-09-26'`, `'text-moderation-latest'`, `'text-moderation-stable'`
  </ParamField>

  <ParamField body="check_input" type="bool" default="True">
    Whether to check user input messages before the model is called
  </ParamField>

  <ParamField body="check_output" type="bool" default="True">
    Whether to check model output messages after the model is called
  </ParamField>

  <ParamField body="check_tool_results" type="bool" default="False">
    Whether to check tool result messages before the model is called
  </ParamField>

  <ParamField body="exit_behavior" type="string" default="end">
    How to handle violations when content is flagged. Options:

    * `'end'` - End agent execution immediately with a violation message
    * `'error'` - Raise `OpenAIModerationError` exception
    * `'replace'` - Replace the flagged content with the violation message and continue
  </ParamField>

  <ParamField body="violation_message" type="str | None">
    Custom template for violation messages. Supports template variables:

    * `{categories}` - Comma-separated list of flagged categories
    * `{category_scores}` - JSON string of category scores
    * `{original_content}` - The original flagged content

    Default: `"I'm sorry, but I can't comply with that request. It was flagged for {categories}."`
  </ParamField>

  <ParamField body="client" type="OpenAI | None">
    Optional pre-configured OpenAI client to reuse. If not provided, a new client will be created.
  </ParamField>

  <ParamField body="async_client" type="AsyncOpenAI | None">
    Optional pre-configured AsyncOpenAI client to reuse. If not provided, a new async client will be created.
  </ParamField>
</Accordion>

<Accordion title="Full example">
  The middleware integrates OpenAI's moderation endpoint to check content at different stages:

  **Moderation stages:**

  * `check_input` - User messages before model call
  * `check_output` - AI messages after model call
  * `check_tool_results` - Tool outputs before model call

  **Exit behaviors:**

  * `'end'` (default) - Stop execution with violation message
  * `'error'` - Raise exception for application handling
  * `'replace'` - Replace flagged content and continue

  ```python  theme={null}
  from langchain_openai import ChatOpenAI
  from langchain_openai.middleware import OpenAIModerationMiddleware
  from langchain.agents import create_agent


  # Basic moderation
  agent = create_agent(
      model=ChatOpenAI(model="gpt-4o"),
      tools=[search_tool, customer_data_tool],
      middleware=[
          OpenAIModerationMiddleware(
              model="omni-moderation-latest",
              check_input=True,
              check_output=True,
          ),
      ],
  )

  # Strict moderation with custom message
  agent_strict = create_agent(
      model=ChatOpenAI(model="gpt-4o"),
      tools=[search_tool, customer_data_tool],
      middleware=[
          OpenAIModerationMiddleware(
              model="omni-moderation-latest",
              check_input=True,
              check_output=True,
              check_tool_results=True,
              exit_behavior="error",
              violation_message=(
                  "Content policy violation detected: {categories}. "
                  "Please rephrase your request."
              ),
          ),
      ],
  )

  # Moderation with replacement behavior
  agent_replace = create_agent(
      model=ChatOpenAI(model="gpt-4o"),
      tools=[search_tool],
      middleware=[
          OpenAIModerationMiddleware(
              check_input=True,
              exit_behavior="replace",
              violation_message="[Content removed due to safety policies]",
          ),
      ],
  )
  ```
</Accordion>

***

<Callout icon="pen-to-square" iconType="regular">
  [Edit the source of this page on GitHub.](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/middleware/built-in.mdx)
</Callout>

<Tip icon="terminal" iconType="regular">
  [Connect these docs programmatically](/use-these-docs) to Claude, VSCode, and more via MCP for    real-time answers.
</Tip>




# Custom middleware

Build custom middleware by implementing hooks that run at specific points in the agent execution flow.

## Hooks

Middleware provides two styles of hooks to intercept agent execution:

<CardGroup cols={2}>
  <Card title="Node-style hooks" icon="share-nodes" href="#node-style-hooks">
    Run sequentially at specific execution points.
  </Card>

  <Card title="Wrap-style hooks" icon="container-storage" href="#wrap-style-hooks">
    Run around each model or tool call.
  </Card>
</CardGroup>

### Node-style hooks

Run sequentially at specific execution points. Use for logging, validation, and state updates.

**Available hooks:**

* `before_agent` - Before agent starts (once per invocation)
* `before_model` - Before each model call
* `after_model` - After each model response
* `after_agent` - After agent completes (once per invocation)

**Example:**

<Tabs>
  <Tab title="Decorator">
    ```python  theme={null}
    from langchain.agents.middleware import before_model, after_model, AgentState
    from langchain.messages import AIMessage
    from langgraph.runtime import Runtime
    from typing import Any


    @before_model(can_jump_to=["end"])
    def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if len(state["messages"]) >= 50:
            return {
                "messages": [AIMessage("Conversation limit reached.")],
                "jump_to": "end"
            }
        return None

    @after_model
    def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None
    ```
  </Tab>

  <Tab title="Class">
    ```python  theme={null}
    from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
    from langchain.messages import AIMessage
    from langgraph.runtime import Runtime
    from typing import Any

    class MessageLimitMiddleware(AgentMiddleware):
        def __init__(self, max_messages: int = 50):
            super().__init__()
            self.max_messages = max_messages

        @hook_config(can_jump_to=["end"])
        def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            if len(state["messages"]) == self.max_messages:
                return {
                    "messages": [AIMessage("Conversation limit reached.")],
                    "jump_to": "end"
                }
            return None

        def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            print(f"Model returned: {state['messages'][-1].content}")
            return None
    ```
  </Tab>
</Tabs>

### Wrap-style hooks

Intercept execution and control when the handler is called. Use for retries, caching, and transformation.

You decide if the handler is called zero times (short-circuit), once (normal flow), or multiple times (retry logic).

**Available hooks:**

* `wrap_model_call` - Around each model call
* `wrap_tool_call` - Around each tool call

**Example:**

<Tabs>
  <Tab title="Decorator">
    ```python  theme={null}
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    from typing import Callable


    @wrap_model_call
    def retry_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        for attempt in range(3):
            try:
                return handler(request)
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"Retry {attempt + 1}/3 after error: {e}")
    ```
  </Tab>

  <Tab title="Class">
    ```python  theme={null}
    from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
    from typing import Callable

    class RetryMiddleware(AgentMiddleware):
        def __init__(self, max_retries: int = 3):
            super().__init__()
            self.max_retries = max_retries

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            for attempt in range(self.max_retries):
                try:
                    return handler(request)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    print(f"Retry {attempt + 1}/{self.max_retries} after error: {e}")
    ```
  </Tab>
</Tabs>

## Create middleware

You can create middleware in two ways:

<CardGroup cols={2}>
  <Card title="Decorator-based middleware" icon="at" href="#decorator-based-middleware">
    Quick and simple for single-hook middleware. Use decorators to wrap individual functions.
  </Card>

  <Card title="Class-based middleware" icon="brackets-curly" href="#class-based-middleware">
    More powerful for complex middleware with multiple hooks or configuration.
  </Card>
</CardGroup>

### Decorator-based middleware

Quick and simple for single-hook middleware. Use decorators to wrap individual functions.

**Available decorators:**

**Node-style:**

* [`@before_agent`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.before_agent) - Runs before agent starts (once per invocation)
* [`@before_model`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.before_model) - Runs before each model call
* [`@after_model`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.after_model) - Runs after each model response
* [`@after_agent`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.after_agent) - Runs after agent completes (once per invocation)

**Wrap-style:**

* [`@wrap_model_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_model_call) - Wraps each model call with custom logic
* [`@wrap_tool_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_tool_call) - Wraps each tool call with custom logic

**Convenience:**

* [`@dynamic_prompt`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.dynamic_prompt) - Generates dynamic system prompts

**Example:**

```python  theme={null}
from langchain.agents.middleware import before_model, wrap_model_call
from langchain.agents.middleware import AgentState, ModelRequest, ModelResponse
from langchain.agents import create_agent
from langgraph.runtime import Runtime
from typing import Any, Callable


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"About to call model with {len(state['messages'])} messages")
    return None

@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")

agent = create_agent(
    model="gpt-4o",
    middleware=[log_before_model, retry_model],
    tools=[...],
)
```

**When to use decorators:**

* Single hook needed
* No complex configuration
* Quick prototyping

### Class-based middleware

More powerful for complex middleware with multiple hooks or configuration. Use classes when you need to define both sync and async implementations for the same hook, or when you want to combine multiple hooks in a single middleware.

**Example:**

```python  theme={null}
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langgraph.runtime import Runtime
from typing import Any, Callable

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None

agent = create_agent(
    model="gpt-4o",
    middleware=[LoggingMiddleware()],
    tools=[...],
)
```

**When to use classes:**

* Defining both sync and async implementations for the same hook
* Multiple hooks needed in a single middleware
* Complex configuration required (e.g., configurable thresholds, custom models)
* Reuse across projects with init-time configuration

## Custom state schema

Middleware can extend the agent's state with custom properties.

<Tabs>
  <Tab title="Decorator">
    ```python  theme={null}
    from langchain.agents.middleware import AgentState, before_model, after_model
    from typing_extensions import NotRequired
    from typing import Any
    from langgraph.runtime import Runtime

    class CustomState(AgentState):
        model_call_count: NotRequired[int]
        user_id: NotRequired[str]

    @before_model(state_schema=CustomState, can_jump_to=["end"])
    def check_call_limit(state: CustomState, runtime: Runtime) -> dict[str, Any] | None:
        count = state.get("model_call_count", 0)
        if count > 10:
            return {"jump_to": "end"}
        return None

    @after_model(state_schema=CustomState)
    def increment_counter(state: CustomState, runtime: Runtime) -> dict[str, Any] | None:
        return {"model_call_count": state.get("model_call_count", 0) + 1}

    agent = create_agent(
        model="gpt-4o",
        middleware=[check_call_limit, increment_counter],
        tools=[...],
    )

    # Invoke with custom state
    result = agent.invoke({
        "messages": [HumanMessage("Hello")],
        "model_call_count": 0,
        "user_id": "user-123",
    })
    ```
  </Tab>

  <Tab title="Class">
    ```python  theme={null}
    from langchain.agents.middleware import AgentState, AgentMiddleware
    from typing_extensions import NotRequired
    from typing import Any

    class CustomState(AgentState):
        model_call_count: NotRequired[int]
        user_id: NotRequired[str]

    class CallCounterMiddleware(AgentMiddleware[CustomState]):
        state_schema = CustomState

        def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
            count = state.get("model_call_count", 0)
            if count > 10:
                return {"jump_to": "end"}
            return None

        def after_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
            return {"model_call_count": state.get("model_call_count", 0) + 1}

    agent = create_agent(
        model="gpt-4o",
        middleware=[CallCounterMiddleware()],
        tools=[...],
    )

    # Invoke with custom state
    result = agent.invoke({
        "messages": [HumanMessage("Hello")],
        "model_call_count": 0,
        "user_id": "user-123",
    })
    ```
  </Tab>
</Tabs>

## Execution order

When using multiple middleware, understand how they execute:

```python  theme={null}
agent = create_agent(
    model="gpt-4o",
    middleware=[middleware1, middleware2, middleware3],
    tools=[...],
)
```

<Accordion title="Execution flow">
  **Before hooks run in order:**

  1. `middleware1.before_agent()`
  2. `middleware2.before_agent()`
  3. `middleware3.before_agent()`

  **Agent loop starts**

  4. `middleware1.before_model()`
  5. `middleware2.before_model()`
  6. `middleware3.before_model()`

  **Wrap hooks nest like function calls:**

  7. `middleware1.wrap_model_call()` → `middleware2.wrap_model_call()` → `middleware3.wrap_model_call()` → model

  **After hooks run in reverse order:**

  8. `middleware3.after_model()`
  9. `middleware2.after_model()`
  10. `middleware1.after_model()`

  **Agent loop ends**

  11. `middleware3.after_agent()`
  12. `middleware2.after_agent()`
  13. `middleware1.after_agent()`
</Accordion>

**Key rules:**

* `before_*` hooks: First to last
* `after_*` hooks: Last to first (reverse)
* `wrap_*` hooks: Nested (first middleware wraps all others)

## Agent jumps

To exit early from middleware, return a dictionary with `jump_to`:

**Available jump targets:**

* `'end'`: Jump to the end of the agent execution (or the first `after_agent` hook)
* `'tools'`: Jump to the tools node
* `'model'`: Jump to the model node (or the first `before_model` hook)

<Tabs>
  <Tab title="Decorator">
    ```python  theme={null}
    from langchain.agents.middleware import after_model, hook_config, AgentState
    from langchain.messages import AIMessage
    from langgraph.runtime import Runtime
    from typing import Any


    @after_model
    @hook_config(can_jump_to=["end"])
    def check_for_blocked(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        if "BLOCKED" in last_message.content:
            return {
                "messages": [AIMessage("I cannot respond to that request.")],
                "jump_to": "end"
            }
        return None
    ```
  </Tab>

  <Tab title="Class">
    ```python  theme={null}
    from langchain.agents.middleware import AgentMiddleware, hook_config, AgentState
    from langchain.messages import AIMessage
    from langgraph.runtime import Runtime
    from typing import Any

    class BlockedContentMiddleware(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            last_message = state["messages"][-1]
            if "BLOCKED" in last_message.content:
                return {
                    "messages": [AIMessage("I cannot respond to that request.")],
                    "jump_to": "end"
                }
            return None
    ```
  </Tab>
</Tabs>

## Best practices

1. Keep middleware focused - each should do one thing well
2. Handle errors gracefully - don't let middleware errors crash the agent
3. **Use appropriate hook types**:
   * Node-style for sequential logic (logging, validation)
   * Wrap-style for control flow (retry, fallback, caching)
4. Clearly document any custom state properties
5. Unit test middleware independently before integrating
6. Consider execution order - place critical middleware first in the list
7. Use built-in middleware when possible

## Examples

### Dynamic model selection

<Tabs>
  <Tab title="Decorator">
    ```python  theme={null}
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    from langchain.chat_models import init_chat_model
    from typing import Callable


    complex_model = init_chat_model("gpt-4o")
    simple_model = init_chat_model("gpt-4o-mini")

    @wrap_model_call
    def dynamic_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # Use different model based on conversation length
       if len(request.messages) > 10:
           model = complex_model
        else:
           model = simple_model
        return handler(request.override(model=model))
    ```
  </Tab>

  <Tab title="Class">
    ```python  theme={null}
    from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
    from langchain.chat_models import init_chat_model
    from typing import Callable

    complex_model = init_chat_model("gpt-4o")
    simple_model = init_chat_model("gpt-4o-mini")

    class DynamicModelMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            # Use different model based on conversation length
            if len(request.messages) > 10:
               model = complex_model
            else:
               model = simple_model
            return handler(request.override(model=model))
    ```
  </Tab>
</Tabs>

### Tool call monitoring

<Tabs>
  <Tab title="Decorator">
    ```python  theme={null}
    from langchain.agents.middleware import wrap_tool_call
    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command
    from typing import Callable


    @wrap_tool_call
    def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        print(f"Executing tool: {request.tool_call['name']}")
        print(f"Arguments: {request.tool_call['args']}")
        try:
            result = handler(request)
            print(f"Tool completed successfully")
            return result
        except Exception as e:
            print(f"Tool failed: {e}")
            raise
    ```
  </Tab>

  <Tab title="Class">
    ```python  theme={null}
    from langchain.tools.tool_node import ToolCallRequest
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command
    from typing import Callable

    class ToolMonitoringMiddleware(AgentMiddleware):
        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command],
        ) -> ToolMessage | Command:
            print(f"Executing tool: {request.tool_call['name']}")
            print(f"Arguments: {request.tool_call['args']}")
            try:
                result = handler(request)
                print(f"Tool completed successfully")
                return result
            except Exception as e:
                print(f"Tool failed: {e}")
                raise
    ```
  </Tab>
</Tabs>

### Dynamically selecting tools

Select relevant tools at runtime to improve performance and accuracy.

**Benefits:**

* **Shorter prompts** - Reduce complexity by exposing only relevant tools
* **Better accuracy** - Models choose correctly from fewer options
* **Permission control** - Dynamically filter tools based on user access

<Tabs>
  <Tab title="Decorator">
    ```python  theme={null}
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    from typing import Callable


    @wrap_model_call
    def select_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Middleware to select relevant tools based on state/context."""
        # Select a small, relevant subset of tools based on state/context
        relevant_tools = select_relevant_tools(request.state, request.runtime)
        return handler(request.override(tools=relevant_tools))

    agent = create_agent(
        model="gpt-4o",
        tools=all_tools,  # All available tools need to be registered upfront
        middleware=[select_tools],
    )
    ```
  </Tab>

  <Tab title="Class">
    ```python  theme={null}
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
    from typing import Callable


    class ToolSelectorMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            """Middleware to select relevant tools based on state/context."""
            # Select a small, relevant subset of tools based on state/context
            relevant_tools = select_relevant_tools(request.state, request.runtime)
            return handler(request.override(tools=relevant_tools))

    agent = create_agent(
        model="gpt-4o",
        tools=all_tools,  # All available tools need to be registered upfront
        middleware=[ToolSelectorMiddleware()],
    )
    ```
  </Tab>
</Tabs>

## Additional resources

* [Middleware API reference](https://reference.langchain.com/python/langchain/middleware/)
* [Built-in middleware](/oss/python/langchain/middleware/built-in)
* [Testing agents](/oss/python/langchain/test)

***

<Callout icon="pen-to-square" iconType="regular">
  [Edit the source of this page on GitHub.](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/middleware/custom.mdx)
</Callout>

<Tip icon="terminal" iconType="regular">
  [Connect these docs programmatically](/use-these-docs) to Claude, VSCode, and more via MCP for    real-time answers.
</Tip>

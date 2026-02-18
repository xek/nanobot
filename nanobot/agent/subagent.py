"""Subagent manager for background task execution."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool


class SubagentManager:
    """
    Manages background subagent execution.
    
    Subagents are lightweight agent instances that run in the background
    to handle specific tasks. They share the same LLM provider but have
    isolated context and a focused system prompt.
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._shared_tools: list[Any] = []
        self._mlflow_client: Any = None
        self._mlflow_experiment_id: str | None = None

    def set_mlflow(self, mlflow_module: Any) -> None:
        """Set the mlflow module for tracing subagent calls via MlflowClient.

        Uses the imperative client API (start_trace/start_span/end_span/end_trace)
        which doesn't depend on async context propagation.
        """
        try:
            self._mlflow_client = mlflow_module.MlflowClient()
            exp = mlflow_module.get_experiment_by_name("nanobot")
            self._mlflow_experiment_id = exp.experiment_id if exp else None
        except Exception as e:
            logger.warning(f"Failed to init MLflow client for subagent: {e}")
            self._mlflow_client = None

    def set_shared_tools(self, parent_registry: "ToolRegistry") -> None:
        """Copy MCP and other shared tools from the parent agent's registry.

        Called after MCP servers are connected so the subagent can use
        Confluence, Jira, etc.  Tools that should stay private to the
        main agent (message, think, cron) are excluded.
        """
        from nanobot.agent.tools.mcp import MCPToolWrapper
        exclude = {"message", "think", "cron"}
        self._shared_tools = [
            tool for name, tool in parent_registry._tools.items()
            if name not in exclude and isinstance(tool, MCPToolWrapper)
        ]

    def _build_tools(self) -> ToolRegistry:
        """Build the tool registry for a subagent run."""
        tools = ToolRegistry()
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        tools.register(ReadFileTool(allowed_dir=allowed_dir))
        tools.register(WriteFileTool(allowed_dir=allowed_dir))
        tools.register(EditFileTool(allowed_dir=allowed_dir))
        tools.register(ListDirTool(allowed_dir=allowed_dir))
        tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        tools.register(WebSearchTool(api_key=self.brave_api_key))
        tools.register(WebFetchTool())
        for shared_tool in self._shared_tools:
            tools.register(shared_tool)
        return tools

    async def _run_task_loop(
        self,
        task: str,
        tag: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_iterations: int = 15,
        trace_id: str | None = None,
        root_span_id: str | None = None,
    ) -> str:
        """Core agent loop: chat with tools until the LLM produces a final answer.

        Used by both ``spawn`` (background) and ``run_inline`` (synchronous).
        ``trace_id`` / ``root_span_id`` are for nesting MLflow spans.
        """
        tools = self._build_tools()
        system_prompt = self._build_subagent_prompt(task)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        use_model = model or self.model
        use_temp = temperature if temperature is not None else self.temperature
        use_max = max_tokens if max_tokens is not None else self.max_tokens

        for iteration in range(1, max_iterations + 1):
            chat_span = self._span_start(
                "subagent.chat", trace_id, root_span_id,
                attributes={"tag": tag, "iteration": iteration, "model": use_model},
            )
            try:
                response = await self.provider.chat(
                    messages=messages, tools=tools.get_definitions(),
                    model=use_model, temperature=use_temp, max_tokens=use_max,
                )
            finally:
                self._span_end(trace_id, chat_span)

            if not response.has_tool_calls:
                return response.content or "Task completed but no final response was generated."

            tool_call_dicts = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
            messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": tool_call_dicts,
            })

            for tool_call in response.tool_calls:
                logger.debug(f"[{tag}] executing: {tool_call.name}")
                tool_span = self._span_start(
                    f"subagent.tool.{tool_call.name}", trace_id, root_span_id,
                    attributes={"tag": tag, "tool": tool_call.name,
                                "args": json.dumps(tool_call.arguments)[:500]},
                )
                result = await tools.execute(tool_call.name, tool_call.arguments)
                self._span_end(trace_id, tool_span, outputs={"preview": result[:300]})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": result,
                })

        return "Task completed but no final response was generated."

    # -- MLflow client-based tracing (no context dependency) -------------------

    def _trace_start(self, name: str, **attrs: Any) -> tuple[str | None, str | None]:
        """Start a new MLflow trace and root span.  Returns (trace_id, span_id)."""
        if not self._mlflow_client:
            return None, None
        try:
            root = self._mlflow_client.start_trace(
                name=name,
                experiment_id=self._mlflow_experiment_id,
                attributes=attrs,
            )
            return root.trace_id, root.span_id
        except Exception as e:
            logger.debug(f"MLflow start_trace failed: {e}")
            return None, None

    def _trace_end(self, trace_id: str | None, **outputs: Any) -> None:
        """End an MLflow trace."""
        if not trace_id or not self._mlflow_client:
            return
        try:
            self._mlflow_client.end_trace(trace_id, outputs=outputs or None)
        except Exception as e:
            logger.debug(f"MLflow end_trace failed: {e}")

    def _span_start(
        self, name: str, trace_id: str | None, parent_id: str | None, **kwargs: Any
    ) -> str | None:
        """Create a child span.  Returns span_id or None."""
        if not trace_id or not parent_id or not self._mlflow_client:
            return None
        try:
            span = self._mlflow_client.start_span(
                name=name, trace_id=trace_id, parent_id=parent_id, **kwargs,
            )
            return span.span_id
        except Exception as e:
            logger.debug(f"MLflow start_span failed: {e}")
            return None

    def _span_end(
        self, trace_id: str | None, span_id: str | None, outputs: dict | None = None,
    ) -> None:
        """End a child span."""
        if not trace_id or not span_id or not self._mlflow_client:
            return
        try:
            self._mlflow_client.end_span(trace_id, span_id, outputs=outputs)
        except Exception as e:
            logger.debug(f"MLflow end_span failed: {e}")

    # -- public entry points -------------------------------------------------

    async def run_inline(
        self,
        task: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run a task synchronously (awaited) and return the result directly.

        Used by the think tool for quick/deep inline modes.
        """
        tag = f"inline-{str(uuid.uuid4())[:6]}"
        logger.info(f"[{tag}] starting inline task: {task[:60]}...")
        trace_id, root_span_id = self._trace_start(
            "think.inline", tag=tag, model=model or self.model,
            task_preview=task[:200],
        )
        try:
            result = await self._run_task_loop(
                task, tag, model=model, temperature=temperature,
                max_tokens=max_tokens, max_iterations=10,
                trace_id=trace_id, root_span_id=root_span_id,
            )
            self._trace_end(trace_id, result_preview=result[:500])
            return result
        except Exception as e:
            logger.error(f"[{tag}] failed: {e}")
            self._trace_end(trace_id, error=str(e))
            return f"Error: {e}"

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        bg_task = asyncio.create_task(
            self._run_background(task_id, task, display_label, origin)
        )
        self._running_tasks[task_id] = bg_task
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        return f"Background task [{display_label}] started (id: {task_id}). I'll report back when it's done."

    # -- internal helpers ----------------------------------------------------

    async def _run_background(
        self, task_id: str, task: str, label: str, origin: dict[str, str],
    ) -> None:
        """Background wrapper: runs the loop and announces the result."""
        logger.info(f"Subagent [{task_id}] starting task: {label}")
        tag = f"bg-{task_id}"
        trace_id, root_span_id = self._trace_start(
            "think.background", tag=tag, label=label, task_preview=task[:200],
        )
        try:
            result = await self._run_task_loop(
                task, tag=tag, trace_id=trace_id, root_span_id=root_span_id,
            )
            logger.info(f"Subagent [{task_id}] completed successfully")
            self._trace_end(trace_id, result_preview=result[:500])
            await self._announce_result(task_id, label, task, result, origin, "ok")
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Subagent [{task_id}] failed: {e}")
            self._trace_end(trace_id, error=str(e))
            await self._announce_result(task_id, label, task, error_msg, origin, "error")

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )
        await self.bus.publish_inbound(msg)
        logger.debug(f"Subagent [{task_id}] announced result to {origin['channel']}:{origin['chat_id']}")
    
    def _build_subagent_prompt(self, task: str) -> str:
        """Build a focused system prompt for the subagent."""
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"

        return f"""# Subagent

## Current Time
{now} ({tz})

You are a subagent spawned by the main agent to complete a specific task.

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages
- Use Jira and Confluence tools (if available)
- Complete the task thoroughly

## What You Cannot Do
- Send messages directly to users (no message tool available)
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}
Skills are available at: {self.workspace}/skills/ (read SKILL.md files as needed)

When you have completed the task, provide a clear summary of your findings or actions."""
    
    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)

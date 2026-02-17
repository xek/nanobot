"""Agent loop: the core processing engine."""

import asyncio
from contextlib import AsyncExitStack
import json
import json_repair
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import Session, SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        agent_defaults: "AgentDefaults | None" = None,
    ):
        from nanobot.config.schema import ExecToolConfig, AgentDefaults
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.agent_defaults = agent_defaults or AgentDefaults()

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()

        # Subagents default to the quick tier
        sub_model, sub_temp, sub_max_tokens = self.agent_defaults.resolve_tier("quick")
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=sub_model if sub_model != self.agent_defaults.model else self.model,
            temperature=sub_temp,
            max_tokens=sub_max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        # DSPy LM tiers (quick/normal/deep) for use with dspy modules.
        # Falls back gracefully if dspy is not installed.
        self.lm_quick, self.lm_normal, self.lm_deep = self._init_dspy_tiers()

        # DSPy observability callback (Phase 2): logs tier, model, latency,
        # token counts for every dspy.LM call.
        self._dspy_callback = self._init_dspy_callback()

        # Enable MLflow tracing if MLFLOW_TRACKING_URI is set
        self._init_mlflow_tracing()

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._react_agent: Any = None  # Lazy-initialised after MCP connect
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
    
    def _init_dspy_tiers(self) -> tuple[Any, Any, Any]:
        """Create dspy.LM instances for the quick / normal / deep tiers.

        Returns (lm_quick, lm_normal, lm_deep). If dspy is not installed
        or tier models are not configured, returns (None, None, None).
        The normal tier is set as the global default via dspy.configure().
        """
        try:
            import dspy
        except ImportError:
            logger.debug("dspy not installed, skipping LM tier init")
            return None, None, None

        # Collect connection kwargs from the underlying provider
        lm_kwargs: dict[str, Any] = {}
        if getattr(self.provider, "api_key", None):
            lm_kwargs["api_key"] = self.provider.api_key
        if getattr(self.provider, "api_base", None):
            lm_kwargs["api_base"] = self.provider.api_base

        q_model, q_temp, q_max = self.agent_defaults.resolve_tier("quick")
        n_model, n_temp, n_max = self.agent_defaults.resolve_tier("normal")
        d_model, d_temp, d_max = self.agent_defaults.resolve_tier("deep")

        # Only create LMs when at least one tier has a distinct model
        has_tiers = any(
            t.model for t in [
                self.agent_defaults.tiers.quick,
                self.agent_defaults.tiers.normal,
                self.agent_defaults.tiers.deep,
            ]
        )
        if not has_tiers:
            logger.debug("No tier models configured, skipping dspy.LM init")
            return None, None, None

        # Resolve model names through the provider's gateway logic so that
        # e.g. "gemini/gemini-3-flash" becomes "openai/gemini-3-flash" when
        # routed through a LiteLLM proxy.
        resolve = getattr(self.provider, "_resolve_model", None)
        if resolve:
            q_model = resolve(q_model)
            n_model = resolve(n_model)
            d_model = resolve(d_model)

        try:
            lm_quick = dspy.LM(
                q_model, temperature=q_temp, max_tokens=q_max,
                cache=False, **lm_kwargs,
            )
            lm_normal = dspy.LM(
                n_model, temperature=n_temp, max_tokens=n_max,
                cache=False, **lm_kwargs,
            )
            lm_deep = dspy.LM(
                d_model, temperature=d_temp, max_tokens=d_max,
                cache=False, **lm_kwargs,
            )
            dspy.configure(lm=lm_normal)
            logger.info(
                f"DSPy LM tiers initialised: "
                f"quick={q_model}, normal={n_model}, deep={d_model}"
            )
            return lm_quick, lm_normal, lm_deep
        except Exception as e:
            logger.warning(f"Failed to initialise dspy.LM tiers: {e}")
            return None, None, None

    def _init_dspy_callback(self) -> Any:
        """Create and register the NanobotCallback for per-tier logging.

        Maps each dspy.LM tier instance to its name so the callback can
        log which tier is being used.  Also registers the provider's
        underlying LM (the ``normal`` default) if it exposes one.
        """
        try:
            import dspy
            from nanobot.providers.dspy_callbacks import NanobotCallback
        except ImportError:
            return None

        cb = NanobotCallback()

        # Map tier LMs
        if self.lm_quick:
            cb.register_tier(self.lm_quick, "quick")
        if self.lm_normal:
            cb.register_tier(self.lm_normal, "normal")
        if self.lm_deep:
            cb.register_tier(self.lm_deep, "deep")

        # Map the provider's own LM (used by _run_agent_loop)
        provider_lm = getattr(self.provider, "lm", None)
        if provider_lm:
            cb.register_tier(provider_lm, "normal")
        # Also register cached LMs from DSPyProvider
        lm_cache = getattr(self.provider, "_lm_cache", {})
        for lm in lm_cache.values():
            if id(lm) not in cb._tier_map:
                cb.register_tier(lm, "normal")

        # Register globally so all dspy.LM calls are observed
        existing = dspy.settings.get("callbacks", []) or []
        dspy.configure(callbacks=existing + [cb])
        logger.debug("DSPy NanobotCallback registered")
        return cb

    def _init_mlflow_tracing(self) -> None:
        """Enable MLflow tracing if MLFLOW_TRACKING_URI is set.

        Calls mlflow.dspy.autolog() so every dspy.LM call is automatically
        traced.  Each conversation turn is wrapped in its own
        ``mlflow.start_span()`` context so that traces are scoped per
        turn rather than per process lifetime.
        """
        import os
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            self._mlflow = None
            return

        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("nanobot")
            mlflow.dspy.autolog()
            self._mlflow = mlflow

            logger.info(f"MLflow tracing enabled → {tracking_uri}")
        except ImportError:
            self._mlflow = None
            logger.debug("mlflow not installed, skipping tracing setup")
        except Exception as e:
            self._mlflow = None
            logger.warning(f"Failed to initialise MLflow tracing: {e}")

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or not self._mcp_servers:
            return
        self._mcp_connected = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        self._mcp_stack = AsyncExitStack()
        await self._mcp_stack.__aenter__()
        await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    def _init_react_agent(self) -> Any:
        """Create the dspy.ReAct module (lazy, called after MCP connect).

        Returns the module, or None if dspy is unavailable.
        """
        try:
            import dspy
            from nanobot.agent.dspy_agent import AgentTurn, wrap_registry
        except ImportError:
            logger.debug("dspy not installed, falling back to manual agent loop")
            return None

        dspy_tools = wrap_registry(self.tools)
        instructions = self.context.build_system_prompt()
        sig = AgentTurn
        if instructions:
            sig = sig.with_instructions(f"{AgentTurn.__doc__}\n\n{instructions}")
        agent = dspy.ReAct(sig, tools=dspy_tools, max_iters=self.max_iterations)
        logger.info(f"dspy.ReAct initialised with {len(dspy_tools)} tools")
        return agent

    async def _traced_agent_loop(
        self, initial_messages: list[dict], msg: InboundMessage,
    ) -> tuple[str | None, list[str], dict[str, dict]]:
        """Run the agent loop with optional DSPy usage tracking and MLflow span."""
        usage_totals: dict[str, dict] = {}

        try:
            import dspy
        except ImportError:
            content, tools = await self._run_agent_loop(initial_messages)
            return content, tools, usage_totals

        with dspy.track_usage() as tracker:
            if self._mlflow:
                with self._mlflow.start_span(
                    name="agent_turn",
                    attributes={
                        "channel": msg.channel,
                        "chat_id": msg.chat_id,
                        "sender": msg.sender_id,
                        "message_preview": msg.content[:120],
                    },
                ):
                    content, tools = await self._run_agent_loop(initial_messages)
            else:
                content, tools = await self._run_agent_loop(initial_messages)
        usage_totals = tracker.get_total_tokens()
        return content, tools, usage_totals

    async def _run_agent_loop(self, initial_messages: list[dict]) -> tuple[str | None, list[str]]:
        """Dispatch to dspy.ReAct or the manual tool-calling loop."""
        if self._react_agent is not None:
            return await self._run_react_loop(initial_messages)
        return await self._run_manual_loop(initial_messages)

    async def _run_react_loop(self, initial_messages: list[dict]) -> tuple[str | None, list[str]]:
        """Run the agent turn via dspy.ReAct.acall()."""
        from nanobot.agent.dspy_agent import build_history

        # Extract current user message (last user turn)
        current_message = ""
        for m in reversed(initial_messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, list):
                    c = " ".join(p.get("text", "") for p in c if p.get("type") == "text")
                current_message = c
                break

        # Build history from prior user/assistant pairs (exclude current turn)
        history_msgs = [m for m in initial_messages if m.get("role") in ("user", "assistant")]
        if history_msgs and history_msgs[-1].get("role") == "user":
            history_msgs = history_msgs[:-1]
        history = build_history(history_msgs) if history_msgs else None

        kwargs: dict[str, Any] = {"message": current_message}
        if history is not None:
            kwargs["history"] = history

        try:
            prediction = await self._react_agent.acall(**kwargs)
        except Exception as e:
            logger.error(f"ReAct agent failed: {e}")
            return f"I encountered an error processing your request: {e}", []

        # Extract tool names from the trajectory
        trajectory = getattr(prediction, "trajectory", {}) or {}
        tools_used: list[str] = []
        for key, val in sorted(trajectory.items()):
            if key.startswith("tool_name_") and val and val != "finish":
                tools_used.append(val)
                idx = key.split("_")[-1]
                args = trajectory.get(f"tool_args_{idx}", {})
                args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                logger.info(f"Tool call: {val}({args_str[:200]})")

        return getattr(prediction, "response", None) or "", tools_used

    async def _run_manual_loop(self, initial_messages: list[dict]) -> tuple[str | None, list[str]]:
        """Fallback: manual tool-calling loop via provider.chat()."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                messages.append({"role": "user", "content": "Reflect on the results and decide next steps."})
            else:
                final_content = response.content
                break

        return final_content, tools_used

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        # Initialise ReAct after MCP so all tools (including MCP) are available
        if self._react_agent is None:
            self._react_agent = self._init_react_agent()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(self, msg: InboundMessage, session_key: str | None = None) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
        
        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        
        # Handle slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # Capture messages before clearing (avoid race condition with background task)
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            async def _consolidate_and_cleanup():
                temp_session = Session(key=session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_consolidate_and_cleanup())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started. Memory consolidation in progress.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")
        
        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        self._set_tool_context(msg.channel, msg.chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        # Run the agent loop with optional MLflow tracing and token tracking
        final_content, tools_used, usage_totals = await self._traced_agent_loop(
            initial_messages, msg,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        if usage_totals:
            parts = []
            for model, u in usage_totals.items():
                short = model.rsplit("/", 1)[-1]
                parts.append(f"{short}=[in={u.get('prompt_tokens', 0)} out={u.get('completion_tokens', 0)}]")
            logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview} | usage: {' '.join(parts)}")
        else:
            logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(origin_channel, origin_chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _ = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "Background task completed."
        
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        After successful summarisation the session is trimmed to
        ``keep_count`` recent messages and saved.  The LLM prefix
        cache resets at this point; between compactions messages
        remain append-only for cache efficiency.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, summarise old messages then trim the session.
        """
        memory = MemoryStore(self.workspace)

        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(f"Memory consolidation (archive_all): {len(session.messages)} total messages archived")
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})")
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})")
                return

            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep")

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if not text:
                logger.warning("Memory consolidation: LLM returned empty response, skipping")
                return
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}")
                return

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            if archive_all:
                session.last_consolidated = 0
            else:
                removed = session.trim(keep_count)
                self.sessions.save(session)
                logger.info(
                    f"Memory consolidation done: trimmed {removed} messages, "
                    f"{len(session.messages)} remaining"
                )
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        tier: str | None = None,
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
            tier: LLM tier override ("quick", "normal", "deep"). None uses default.
        
        Returns:
            The agent's response.
        """
        # Temporarily override model/temperature/max_tokens for this call
        orig_model, orig_temp, orig_max = self.model, self.temperature, self.max_tokens
        if tier:
            t_model, t_temp, t_max = self.agent_defaults.resolve_tier(tier)
            self.model = t_model
            self.temperature = t_temp
            self.max_tokens = t_max
            logger.info(f"process_direct: tier={tier} -> model={t_model} for session={session_key}")

        try:
            await self._connect_mcp()
            if self._react_agent is None:
                self._react_agent = self._init_react_agent()
            msg = InboundMessage(
                channel=channel,
                sender_id="user",
                chat_id=chat_id,
                content=content
            )
            response = await self._process_message(msg, session_key=session_key)
            return response.content if response else ""
        finally:
            self.model, self.temperature, self.max_tokens = orig_model, orig_temp, orig_max

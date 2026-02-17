"""DSPy ReAct agent module (Phase 4).

Wraps nanobot's tool-calling loop as a ``dspy.ReAct`` module so that:
- Every conversation turn is a single MLflow trace
- The agent loop becomes optimisable via SIMBA/GEPA
- ``dspy.History`` carries multi-turn context natively
"""

from __future__ import annotations

from typing import Any

import dspy
from loguru import logger

from nanobot.agent.tools.base import Tool as NanobotTool
from nanobot.agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

class AgentTurn(dspy.Signature):
    """You are a helpful AI assistant with access to tools.
    Respond concisely and accurately. Use tools when you need
    external information or to take action.
    Focus on the current user message. History is provided only for
    short-term continuity — do NOT blend unrelated earlier topics
    into your answer unless the user explicitly refers to them."""

    message: str = dspy.InputField(desc="Current user message")
    history: dspy.History = dspy.InputField(
        desc="Recent conversation turns for short-term continuity", default=None,
    )
    response: str = dspy.OutputField(desc="Response to the user")


# ---------------------------------------------------------------------------
# Tool bridge: nanobot Tool -> dspy.Tool
# ---------------------------------------------------------------------------

def wrap_nanobot_tool(tool: NanobotTool) -> dspy.Tool:
    """Convert a nanobot ``Tool`` instance into a ``dspy.Tool``.

    We build a thin async wrapper whose signature is derived from the
    nanobot tool's JSON-schema ``parameters`` property.  ``dspy.Tool``
    picks up the wrapper's name and docstring automatically, and we
    override ``args`` with the original JSON schema so the LLM sees
    the right parameter descriptions.
    """
    # Build the JSON-schema dict that dspy.Tool expects for `args`
    schema = tool.parameters or {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    # dspy.Tool.args format: {name: {type: ..., ...}}
    dspy_args: dict[str, Any] = {}
    for param_name, param_schema in properties.items():
        entry: dict[str, Any] = dict(param_schema)
        if param_name not in required:
            entry.setdefault("default", None)
        dspy_args[param_name] = entry

    # Async function that delegates to tool.execute()
    async def _call(**kwargs: Any) -> str:
        errors = tool.validate_params(kwargs)
        if errors:
            return f"Error: Invalid parameters: {'; '.join(errors)}"
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            return f"Error executing {tool.name}: {e}"

    return dspy.Tool(
        func=_call,
        name=tool.name,
        desc=tool.description,
        args=dspy_args,
    )


def wrap_registry(registry: ToolRegistry) -> list[dspy.Tool]:
    """Wrap every tool in a ``ToolRegistry`` as a ``dspy.Tool``."""
    wrapped: list[dspy.Tool] = []
    for name in registry.tool_names:
        nanobot_tool = registry.get(name)
        if nanobot_tool is None:
            continue
        try:
            wrapped.append(wrap_nanobot_tool(nanobot_tool))
        except Exception as e:
            logger.warning(f"Failed to wrap tool '{name}' as dspy.Tool: {e}")
    return wrapped


# ---------------------------------------------------------------------------
# NanobotReAct module
# ---------------------------------------------------------------------------

class NanobotReAct(dspy.Module):
    """Wraps ``dspy.ReAct`` for use as nanobot's agent loop.

    Parameters
    ----------
    tools : list[dspy.Tool]
        Tools available to the agent.
    max_iters : int
        Maximum thought-action-observation cycles.
    instructions : str | None
        Extra instructions injected into the signature (system prompt
        fragments, persona, etc.).
    """

    def __init__(
        self,
        tools: list[dspy.Tool],
        max_iters: int = 20,
        instructions: str | None = None,
    ):
        super().__init__()
        sig = AgentTurn
        if instructions:
            sig = sig.with_instructions(
                f"{AgentTurn.__doc__}\n\n{instructions}"
            )
        self.react = dspy.ReAct(sig, tools=tools, max_iters=max_iters)

    def forward(self, message: str, history: dspy.History | None = None) -> dspy.Prediction:
        kwargs: dict[str, Any] = {"message": message}
        if history is not None:
            kwargs["history"] = history
        return self.react(**kwargs)

    async def aforward(self, message: str, history: dspy.History | None = None) -> dspy.Prediction:
        kwargs: dict[str, Any] = {"message": message}
        if history is not None:
            kwargs["history"] = history
        return await self.react.acall(**kwargs)


# ---------------------------------------------------------------------------
# History bridge: nanobot session -> dspy.History
# ---------------------------------------------------------------------------

def build_history(session_messages: list[dict[str, Any]]) -> dspy.History | None:
    """Convert nanobot session messages into a ``dspy.History``.

    Each entry in the returned ``History`` has ``message`` (user) and
    ``response`` (assistant) keys, matching the ``AgentTurn`` signature.
    Messages without a paired assistant response are skipped.
    """
    pairs: list[dict[str, str]] = []
    i = 0
    while i < len(session_messages):
        msg = session_messages[i]
        if msg.get("role") == "user":
            content_user = msg.get("content", "")
            # Look ahead for the paired assistant response
            if i + 1 < len(session_messages) and session_messages[i + 1].get("role") == "assistant":
                content_asst = session_messages[i + 1].get("content", "")
                pairs.append({"message": content_user, "response": content_asst})
                i += 2
                continue
        i += 1

    if not pairs:
        return None
    return dspy.History(messages=pairs)

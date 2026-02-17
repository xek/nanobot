"""DSPy agent helpers.

Bridges nanobot's tool registry and session history to DSPy's
``dspy.ReAct`` module and ``dspy.History`` type.
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
        desc="Recent conversation turns for short-term continuity",
        default=None,
    )
    response: str = dspy.OutputField(desc="Response to the user")


# ---------------------------------------------------------------------------
# Tool bridge: nanobot Tool -> dspy.Tool
# ---------------------------------------------------------------------------

def wrap_nanobot_tool(tool: NanobotTool) -> dspy.Tool:
    """Convert a nanobot ``Tool`` into a ``dspy.Tool``."""
    schema = tool.parameters or {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    dspy_args: dict[str, Any] = {}
    for name, prop in properties.items():
        entry: dict[str, Any] = dict(prop)
        if name not in required:
            entry.setdefault("default", None)
        dspy_args[name] = entry

    async def _call(**kwargs: Any) -> str:
        errors = tool.validate_params(kwargs)
        if errors:
            return f"Error: Invalid parameters: {'; '.join(errors)}"
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            return f"Error executing {tool.name}: {e}"

    return dspy.Tool(func=_call, name=tool.name, desc=tool.description, args=dspy_args)


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
            logger.warning(f"Failed to wrap tool '{name}': {e}")
    return wrapped


# ---------------------------------------------------------------------------
# History bridge: nanobot session -> dspy.History
# ---------------------------------------------------------------------------

def build_history(session_messages: list[dict[str, Any]]) -> dspy.History | None:
    """Convert nanobot session messages into ``dspy.History``.

    Pairs consecutive user/assistant messages into turns matching the
    ``AgentTurn`` signature fields (``message``, ``response``).
    """
    pairs: list[dict[str, str]] = []
    i = 0
    while i < len(session_messages):
        msg = session_messages[i]
        if msg.get("role") == "user":
            if (
                i + 1 < len(session_messages)
                and session_messages[i + 1].get("role") == "assistant"
            ):
                pairs.append({
                    "message": msg.get("content", ""),
                    "response": session_messages[i + 1].get("content", ""),
                })
                i += 2
                continue
        i += 1

    return dspy.History(messages=pairs) if pairs else None

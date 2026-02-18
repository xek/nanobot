"""Think tool: inline or background reasoning with tier selection."""

from typing import Any, TYPE_CHECKING

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class ThinkTool(Tool):
    """Delegate a subtask to a specific thinking mode.

    All modes run a full agent loop with tools (file, shell, web,
    Confluence, Jira, etc.).  The difference is the model tier and
    whether the task blocks or runs in the background.

    Modes:
      - **quick** — synchronous, cheap/fast model.
      - **deep** — synchronous, powerful model with extended reasoning.
      - **background** — asynchronous, reports back when done.
    """

    def __init__(
        self,
        subagent_manager: "SubagentManager | None" = None,
        quick_config: tuple[str, float, int] | None = None,
        deep_config: tuple[str, float, int] | None = None,
    ):
        self._subagent_manager = subagent_manager
        self._quick_config = quick_config  # (model, temperature, max_tokens)
        self._deep_config = deep_config
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for background task announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id

    @property
    def name(self) -> str:
        return "think"

    @property
    def description(self) -> str:
        return (
            "Delegate a subtask to a different thinking mode. "
            "All modes have full tool access (files, shell, web, Jira, Confluence). "
            "'quick' = fast/cheap model, synchronous — use for simple subtasks. "
            "'deep' = powerful model, synchronous — use for hard problems or when stuck. "
            "'background' = runs asynchronously and reports back — use for "
            "time-consuming research or multi-step investigations."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task to accomplish. Be specific and self-contained.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["quick", "deep", "background"],
                    "description": (
                        "quick = fast/cheap model, blocks until done. "
                        "deep = powerful model, blocks until done. "
                        "background = runs async, reports back later."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "Short label for the task (used for display/logging)",
                },
            },
            "required": ["prompt", "mode"],
        }

    async def execute(
        self,
        prompt: str,
        mode: str = "quick",
        label: str | None = None,
        **kwargs: Any,
    ) -> str:
        if self._subagent_manager is None:
            return "Error: think tool is not available (no subagent manager)."

        if mode == "background":
            return await self._subagent_manager.spawn(
                task=prompt,
                label=label,
                origin_channel=self._origin_channel,
                origin_chat_id=self._origin_chat_id,
            )

        config = self._quick_config if mode == "quick" else self._deep_config
        if config is None:
            return f"Error: '{mode}' tier is not configured."

        model, temperature, max_tokens = config
        return await self._subagent_manager.run_inline(
            task=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

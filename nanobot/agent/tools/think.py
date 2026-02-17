"""Think tool: inline or background reasoning with tier selection."""

from typing import Any, TYPE_CHECKING

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class ThinkTool(Tool):
    """Delegate a subtask to a specific thinking mode, inline or in the background.

    Modes:
      - **quick** — fast/cheap single LLM call, inline. Good for
        classification, reformatting, yes/no questions.
      - **deep** — powerful single LLM call, inline. Good for hard
        reasoning, self-reflection, evaluation.
      - **background** — spawns a full subagent with tools that runs
        asynchronously and reports back when done. Good for complex,
        time-consuming tasks that can run independently.
    """

    def __init__(
        self,
        lm_quick: Any = None,
        lm_deep: Any = None,
        subagent_manager: "SubagentManager | None" = None,
    ):
        self._lm_quick = lm_quick
        self._lm_deep = lm_deep
        self._subagent_manager = subagent_manager
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
            "'quick' = fast/cheap inline call for simple subtasks (classification, reformatting, yes/no). "
            "'deep' = powerful inline call for hard problems or self-reflection. "
            "'background' = spawn a full subagent with tools that runs asynchronously and reports back."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task or question to think about",
                },
                "mode": {
                    "type": "string",
                    "enum": ["quick", "deep", "background"],
                    "description": (
                        "quick = fast/cheap inline call. "
                        "deep = slow/powerful inline call. "
                        "background = full subagent with tools, runs async."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "Short label for the task (used in background mode for display)",
                },
            },
            "required": ["prompt", "mode"],
        }

    async def execute(
        self, prompt: str, mode: str = "quick", label: str | None = None, **kwargs: Any,
    ) -> str:
        if mode == "background":
            return await self._run_background(prompt, label)
        return await self._run_inline(prompt, mode)

    async def _run_inline(self, prompt: str, mode: str) -> str:
        """Single dspy.Predict call on the chosen tier."""
        import dspy

        lm = self._lm_quick if mode == "quick" else self._lm_deep
        if lm is None:
            return f"Error: '{mode}' tier is not configured."

        try:
            with dspy.context(lm=lm):
                result = dspy.Predict("prompt -> result")(prompt=prompt)
            return result.result
        except Exception as e:
            return f"Error in think({mode}): {e}"

    async def _run_background(self, task: str, label: str | None) -> str:
        """Spawn a full subagent to handle the task asynchronously."""
        if self._subagent_manager is None:
            return "Error: background mode is not available (no subagent manager)."

        return await self._subagent_manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
        )

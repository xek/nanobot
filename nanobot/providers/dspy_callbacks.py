"""DSPy callback for per-tier observability."""

import time
from typing import Any

from loguru import logger

from dspy.utils.callback import BaseCallback


class NanobotCallback(BaseCallback):
    """Log tier, model, latency, and token counts for every dspy.LM call.

    Inherits from ``dspy.utils.callback.BaseCallback`` so all hook
    signatures match and no "missing method" warnings are emitted.
    """

    def __init__(self, tier_map: dict[int, str] | None = None):
        self._tier_map: dict[int, str] = tier_map or {}
        self._starts: dict[str, float] = {}

    def register_tier(self, lm: Any, tier: str) -> None:
        """Associate a dspy.LM instance with a tier name."""
        self._tier_map[id(lm)] = tier

    def _tier_for(self, instance: Any) -> str:
        return self._tier_map.get(id(instance), "unknown")

    # -- LM hooks ----------------------------------------------------------

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._starts[call_id] = time.monotonic()
        model = getattr(instance, "model", "?")
        tier = self._tier_for(instance)
        n_messages = len(inputs.get("messages", []))
        has_tools = bool(inputs.get("tools"))
        logger.debug(
            f"LM call start  | tier={tier} model={model} "
            f"messages={n_messages} tools={has_tools}"
        )

    def on_lm_end(
        self, call_id: str, outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        elapsed = time.monotonic() - self._starts.pop(call_id, time.monotonic())
        if exception:
            logger.warning(f"LM call failed  | {elapsed:.1f}s | {exception}")
            return

        usage: dict[str, int] = {}
        if outputs:
            response = outputs.get("response") if isinstance(outputs, dict) else None
            if response and hasattr(response, "usage") and response.usage:
                u = response.usage
                usage = {
                    "prompt": getattr(u, "prompt_tokens", 0) or 0,
                    "completion": getattr(u, "completion_tokens", 0) or 0,
                    "total": getattr(u, "total_tokens", 0) or 0,
                }

        tok_str = ""
        if usage:
            tok_str = (
                f" tokens=[in={usage['prompt']} out={usage['completion']} "
                f"total={usage['total']}]"
            )
        logger.info(f"LM call done   | {elapsed:.1f}s{tok_str}")

    # -- Module hooks ------------------------------------------------------

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._starts[call_id] = time.monotonic()
        logger.debug(f"Module start   | {type(instance).__name__}")

    def on_module_end(
        self, call_id: str, outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        elapsed = time.monotonic() - self._starts.pop(call_id, time.monotonic())
        if exception:
            logger.warning(f"Module failed  | {elapsed:.1f}s | {exception}")
        else:
            logger.debug(f"Module done    | {elapsed:.1f}s")

    # -- Tool hooks --------------------------------------------------------

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._starts[call_id] = time.monotonic()
        name = getattr(instance, "name", type(instance).__name__)
        logger.debug(f"Tool start     | {name}")

    def on_tool_end(
        self, call_id: str, outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        elapsed = time.monotonic() - self._starts.pop(call_id, time.monotonic())
        if exception:
            logger.warning(f"Tool failed    | {elapsed:.1f}s | {exception}")
        else:
            logger.debug(f"Tool done      | {elapsed:.1f}s")

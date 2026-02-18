"""DSPy callback for per-tier observability and OpenTelemetry tracing."""

import time
from typing import Any

from loguru import logger


try:
    from dspy.utils.callback import BaseCallback as _Base
except ImportError:
    _Base = object


class NanobotCallback(_Base):
    """Log and trace every dspy.LM / Module / Tool call.

    Inherits from ``dspy.utils.callback.BaseCallback`` so DSPy
    recognises the callback and calls all hook methods.

    When an OpenTelemetry ``Tracer`` is supplied via ``set_tracer()``,
    each on_*_start / on_*_end pair is wrapped in an OTel span so all
    DSPy activity shows up in Jaeger automatically — no manual
    instrumentation needed in the agent loops.
    """

    def __init__(self, tier_map: dict[int, str] | None = None):
        self._tier_map: dict[int, str] = tier_map or {}
        self._starts: dict[str, float] = {}
        self._tracer: Any = None
        self._spans: dict[str, Any] = {}

    def set_tracer(self, tracer: Any) -> None:
        self._tracer = tracer

    def register_tier(self, lm: Any, tier: str) -> None:
        """Associate a dspy.LM instance with a tier name."""
        self._tier_map[id(lm)] = tier

    def _tier_for(self, instance: Any) -> str:
        return self._tier_map.get(id(instance), "unknown")

    def _start_span(self, call_id: str, name: str, attributes: dict | None = None) -> None:
        if not self._tracer:
            return
        span = self._tracer.start_span(name, attributes=attributes)
        self._spans[call_id] = span

    def _end_span(self, call_id: str, attributes: dict | None = None,
                  exception: Exception | None = None) -> None:
        span = self._spans.pop(call_id, None)
        if not span:
            return
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        if exception:
            span.set_status(
                _otel_status_error(str(exception))
            )
            span.record_exception(exception)
        span.end()

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
        self._start_span(call_id, f"lm.{tier}", {
            "tier": tier, "model": model,
            "message_count": n_messages, "has_tools": has_tools,
        })

    def on_lm_end(
        self, call_id: str, outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        elapsed = time.monotonic() - self._starts.pop(call_id, time.monotonic())
        if exception:
            logger.warning(f"LM call failed  | {elapsed:.1f}s | {exception}")
            self._end_span(call_id, exception=exception)
            return

        usage: dict[str, int] = {}
        response_preview = ""
        if outputs:
            response = outputs.get("response") if isinstance(outputs, dict) else None
            if response and hasattr(response, "usage") and response.usage:
                u = response.usage
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(u, "total_tokens", 0) or 0,
                }
            if response and hasattr(response, "choices") and response.choices:
                msg = response.choices[0].message
                response_preview = (getattr(msg, "content", None) or "")[:500]

        tok_str = ""
        if usage:
            tok_str = (
                f" tokens=[in={usage['prompt_tokens']} out={usage['completion_tokens']} "
                f"total={usage['total_tokens']}]"
            )
        logger.info(f"LM call done   | {elapsed:.1f}s{tok_str}")

        attrs = {"elapsed_s": round(elapsed, 2)}
        attrs.update(usage)
        if response_preview:
            attrs["response_preview"] = response_preview
        self._end_span(call_id, attributes=attrs)

    # -- Module hooks ------------------------------------------------------

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._starts[call_id] = time.monotonic()
        name = type(instance).__name__
        logger.debug(f"Module start   | {name}")
        input_preview = ""
        if isinstance(inputs, dict):
            for v in inputs.values():
                if isinstance(v, str) and v:
                    input_preview = v[:200]
                    break
        self._start_span(call_id, f"module.{name}", {
            "module": name,
            "input_preview": input_preview,
        })

    def on_module_end(
        self, call_id: str, outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        elapsed = time.monotonic() - self._starts.pop(call_id, time.monotonic())
        attrs = {"elapsed_s": round(elapsed, 2)}
        if exception:
            logger.warning(f"Module failed  | {elapsed:.1f}s | {exception}")
        else:
            logger.debug(f"Module done    | {elapsed:.1f}s")
            if outputs and hasattr(outputs, "answer"):
                attrs["answer_preview"] = str(outputs.answer)[:500]
        self._end_span(call_id, attributes=attrs, exception=exception)

    # -- Tool hooks --------------------------------------------------------

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._starts[call_id] = time.monotonic()
        name = getattr(instance, "name", type(instance).__name__)
        logger.debug(f"Tool start     | {name}")
        self._start_span(call_id, f"tool.{name}", {
            "tool.name": name,
            "tool.args_preview": str(inputs)[:500],
        })

    def on_tool_end(
        self, call_id: str, outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        elapsed = time.monotonic() - self._starts.pop(call_id, time.monotonic())
        attrs = {"elapsed_s": round(elapsed, 2)}
        if exception:
            logger.warning(f"Tool failed    | {elapsed:.1f}s | {exception}")
        else:
            logger.debug(f"Tool done      | {elapsed:.1f}s")
            if outputs:
                attrs["result_preview"] = str(outputs)[:500]
        self._end_span(call_id, attributes=attrs, exception=exception)


def _otel_status_error(description: str):
    """Create an OTel ERROR status without importing at module level."""
    from opentelemetry.trace import StatusCode, Status
    return Status(StatusCode.ERROR, description)

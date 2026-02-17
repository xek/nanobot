"""DSPy-based LLM provider with caching, retry, and observability hooks."""

from typing import Any

import json_repair

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class DSPyProvider(LLMProvider):
    """
    LLM provider that delegates to dspy.LM.

    Provides the same interface as LiteLLMProvider but routes calls through
    DSPy's LM abstraction, gaining:
    - Automatic response caching (deduplicates identical requests)
    - Retry with exponential backoff
    - Call history for observability (Phase 2 callbacks)
    - Foundation for dspy.Predict / dspy.ReAct integration (Phase 4)

    Under the hood dspy.LM still uses litellm, so model naming, API keys,
    and fallback chains work identically.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        cache: bool = False,
        num_retries: int = 3,
        **extra_kwargs: Any,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model

        import dspy

        lm_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cache": cache,
            "num_retries": num_retries,
        }
        if api_key:
            lm_kwargs["api_key"] = api_key
        if api_base:
            lm_kwargs["api_base"] = api_base
        lm_kwargs.update(extra_kwargs)

        self._lm = dspy.LM(model=default_model, **lm_kwargs)

    @property
    def lm(self) -> Any:
        """Expose the underlying dspy.LM for direct use in DSPy modules."""
        return self._lm

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via dspy.LM.acall().

        Uses dspy's native async path (alitellm_completion) so the event
        loop is never blocked.
        """
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max(1, max_tokens),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            await self._lm.acall(messages=messages, **kwargs)

            # dspy.LM stores the full litellm response in history
            if self._lm.history:
                entry = self._lm.history[-1]
                response = entry.get("response") or entry.get("outputs")
                if response and hasattr(response, "choices"):
                    return self._parse_response(response)
                # Fallback: entry["outputs"] is a list of content strings
                outputs = entry.get("outputs", [])
                content = outputs[0] if outputs else ""
                return LLMResponse(content=content)

            return LLMResponse(content="")
        except Exception as e:
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
            )

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse a litellm ModelResponse into LLMResponse.

        The response object is the same format as litellm.completion() returns,
        so this mirrors LiteLLMProvider._parse_response.
        """
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCallRequest] = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json_repair.loads(args)
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        usage: dict[str, int] = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        reasoning_content = getattr(message, "reasoning_content", None)

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model

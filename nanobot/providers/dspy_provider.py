"""DSPy-based LLM provider with caching, retry, and observability hooks."""

from typing import Any

import json_repair
import litellm

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.registry import find_by_model, find_gateway


class DSPyProvider(LLMProvider):
    """
    LLM provider that delegates to dspy.LM.

    Provides the same interface as LiteLLMProvider but routes calls through
    DSPy's LM abstraction, gaining:
    - Automatic MLflow tracing via mlflow.dspy.autolog()
    - Retry with exponential backoff
    - Call history for observability
    - Foundation for dspy.Predict / dspy.ReAct integration

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
        provider_name: str | None = None,
        **extra_kwargs: Any,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self._cache = cache
        self._num_retries = num_retries

        # Detect gateway (LiteLLM proxy, OpenRouter, etc.)
        self._gateway = find_gateway(provider_name, api_key, api_base)

        # Configure litellm globals (same as LiteLLMProvider)
        if api_base:
            litellm.api_base = api_base
        litellm.suppress_debug_info = True
        litellm.drop_params = True

        import dspy

        resolved = self._resolve_model(default_model)
        self._lm = self._make_lm(resolved, temperature, max_tokens)
        # Cache dspy.LM instances by *original* model name for tier switching
        self._lm_cache: dict[str, Any] = {default_model: self._lm}

    def _resolve_model(self, model: str) -> str:
        """Resolve model name for dspy.LM (which uses litellm internally).

        For gateway/proxy setups the model name must keep its original
        provider prefix (e.g. ``gemini/``) so the proxy can route it,
        but also needs ``openai/`` so litellm uses the OpenAI protocol.

        Unlike LiteLLMProvider._resolve_model we never strip the model
        prefix — the proxy needs it to select the correct upstream.
        """
        if self._gateway:
            prefix = self._gateway.litellm_prefix
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model

        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"
        return model

    def _make_lm(self, model: str, temperature: float, max_tokens: int) -> Any:
        """Create a dspy.LM instance with an already-resolved model name."""
        import dspy

        lm_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cache": self._cache,
            "num_retries": self._num_retries,
        }
        if self.api_key:
            lm_kwargs["api_key"] = self.api_key
        if self.api_base:
            lm_kwargs["api_base"] = self.api_base
        return dspy.LM(model=model, **lm_kwargs)

    def _get_lm(self, model: str | None, temperature: float, max_tokens: int) -> Any:
        """Get or create a dspy.LM for the given model."""
        model = model or self.default_model
        if model not in self._lm_cache:
            resolved = self._resolve_model(model)
            self._lm_cache[model] = self._make_lm(resolved, temperature, max_tokens)
        return self._lm_cache[model]

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
        lm = self._get_lm(model, temperature, max_tokens)

        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max(1, max_tokens),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            await lm.acall(messages=messages, **kwargs)

            # dspy.LM stores the full litellm response in history
            if lm.history:
                entry = lm.history[-1]
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

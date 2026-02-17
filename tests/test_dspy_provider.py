"""Tests for DSPyProvider and dspy.LM tier initialisation in AgentLoop."""

from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass, field
from typing import Any
import asyncio

import pytest

from nanobot.providers.base import LLMResponse, ToolCallRequest


# ---------------------------------------------------------------------------
# DSPyProvider unit tests
# ---------------------------------------------------------------------------


class TestDSPyProviderInit:
    """Test DSPyProvider construction."""

    @patch("dspy.LM")
    def test_creates_dspy_lm(self, mock_lm_cls):
        from nanobot.providers.dspy_provider import DSPyProvider

        provider = DSPyProvider(
            api_key="sk-test",
            api_base="http://localhost:4000",
            default_model="gemini/gemini-3-flash",
            temperature=0.5,
            max_tokens=2048,
        )
        mock_lm_cls.assert_called_once()
        call_kwargs = mock_lm_cls.call_args
        assert call_kwargs.kwargs["model"] == "gemini/gemini-3-flash"
        assert call_kwargs.kwargs["api_key"] == "sk-test"
        assert call_kwargs.kwargs["api_base"] == "http://localhost:4000"
        assert call_kwargs.kwargs["temperature"] == 0.5
        assert call_kwargs.kwargs["max_tokens"] == 2048

    @patch("dspy.LM")
    def test_default_model(self, mock_lm_cls):
        from nanobot.providers.dspy_provider import DSPyProvider

        provider = DSPyProvider(default_model="openai/gpt-4")
        assert provider.get_default_model() == "openai/gpt-4"

    @patch("dspy.LM")
    def test_lm_property_exposes_instance(self, mock_lm_cls):
        from nanobot.providers.dspy_provider import DSPyProvider

        provider = DSPyProvider(default_model="test/model")
        assert provider.lm is mock_lm_cls.return_value


class TestDSPyProviderChat:
    """Test DSPyProvider.chat() method."""

    @pytest.fixture
    def provider(self):
        """Create a DSPyProvider with a mocked dspy.LM."""
        with patch("dspy.LM") as mock_lm_cls:
            mock_lm = MagicMock()
            mock_lm.acall = AsyncMock()
            mock_lm_cls.return_value = mock_lm
            from nanobot.providers.dspy_provider import DSPyProvider
            p = DSPyProvider(default_model="test/model")
            return p, mock_lm

    @pytest.mark.asyncio
    async def test_chat_returns_content(self, provider):
        p, mock_lm = provider

        # Simulate dspy.LM history entry with a litellm-like response
        mock_response = _make_litellm_response(content="Hello there!")
        mock_lm.history = [{"response": mock_response, "outputs": ["Hello there!"]}]

        result = await p.chat(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            max_tokens=1024,
        )
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello there!"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_parses_tool_calls(self, provider):
        p, mock_lm = provider

        mock_response = _make_litellm_response(
            content=None,
            tool_calls=[
                _make_tool_call("call_1", "read_file", '{"path": "/tmp/test.txt"}'),
            ],
        )
        mock_lm.history = [{"response": mock_response, "outputs": [""]}]

        result = await p.chat(
            messages=[{"role": "user", "content": "read the file"}],
            tools=[{"type": "function", "function": {"name": "read_file"}}],
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "/tmp/test.txt"}

    @pytest.mark.asyncio
    async def test_chat_parses_usage(self, provider):
        p, mock_lm = provider

        mock_response = _make_litellm_response(
            content="Done",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        mock_lm.history = [{"response": mock_response, "outputs": ["Done"]}]

        result = await p.chat(
            messages=[{"role": "user", "content": "test"}],
        )
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_chat_handles_error(self, provider):
        p, mock_lm = provider

        mock_lm.acall = AsyncMock(side_effect=RuntimeError("API timeout"))
        mock_lm.history = []

        result = await p.chat(
            messages=[{"role": "user", "content": "test"}],
        )
        assert "Error calling LLM" in result.content
        assert result.finish_reason == "error"

    @pytest.mark.asyncio
    async def test_chat_fallback_to_outputs(self, provider):
        """When history entry has no response object, fall back to outputs."""
        p, mock_lm = provider

        mock_lm.history = [{"outputs": ["fallback content"]}]

        result = await p.chat(
            messages=[{"role": "user", "content": "test"}],
        )
        assert result.content == "fallback content"

    @pytest.mark.asyncio
    async def test_chat_clamps_max_tokens(self, provider):
        """max_tokens=0 should be clamped to 1."""
        p, mock_lm = provider

        mock_lm.history = [{"outputs": ["ok"]}]

        await p.chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=0,
        )
        call_kwargs = mock_lm.acall.call_args
        assert call_kwargs.kwargs["max_tokens"] == 1


# ---------------------------------------------------------------------------
# AgentLoop._init_dspy_tiers tests
# ---------------------------------------------------------------------------


class TestAgentLoopDspyTiers:
    """Test dspy.LM tier initialisation in AgentLoop."""

    def _make_agent_loop(self, tiers_dict=None, tmp_path=None):
        """Create a minimal AgentLoop with mocked dependencies."""
        from nanobot.config.schema import AgentDefaults, TierConfig, TiersConfig

        tiers_cfg = {}
        if tiers_dict:
            for name, cfg in tiers_dict.items():
                tiers_cfg[name] = TierConfig(**cfg)

        defaults = AgentDefaults(
            model="base/default-model",
            temperature=0.7,
            max_tokens=4096,
            tiers=TiersConfig(**tiers_cfg),
        )

        mock_provider = MagicMock()
        mock_provider.api_key = "sk-test"
        mock_provider.api_base = "http://localhost:4000"
        mock_provider.get_default_model.return_value = "base/default-model"
        mock_provider._resolve_model = lambda m: m  # no gateway in tests

        mock_bus = MagicMock()
        mock_bus.publish_outbound = AsyncMock()

        workspace = tmp_path or Path("/tmp/test-nanobot-workspace")
        workspace.mkdir(parents=True, exist_ok=True)

        from nanobot.agent.loop import AgentLoop
        with patch("nanobot.agent.loop.SubagentManager"):
            agent = AgentLoop(
                bus=mock_bus,
                provider=mock_provider,
                workspace=workspace,
                agent_defaults=defaults,
            )
        return agent

    @patch("dspy.configure")
    @patch("dspy.LM")
    def test_tiers_initialised_when_configured(self, mock_lm_cls, mock_configure):
        agent = self._make_agent_loop(tiers_dict={
            "quick": {"model": "gemini/flash-lite", "temperature": 0.3},
            "normal": {"model": "gemini/pro"},
            "deep": {"model": "openai/o3", "temperature": 1.0, "max_tokens": 16384},
        })
        assert agent.lm_quick is not None
        assert agent.lm_normal is not None
        assert agent.lm_deep is not None
        assert mock_lm_cls.call_count == 3
        mock_configure.assert_called_once()

    @patch("dspy.configure")
    @patch("dspy.LM")
    def test_tiers_none_when_not_configured(self, mock_lm_cls, mock_configure):
        agent = self._make_agent_loop(tiers_dict={})
        assert agent.lm_quick is None
        assert agent.lm_normal is None
        assert agent.lm_deep is None
        mock_lm_cls.assert_not_called()

    @patch("dspy.configure")
    @patch("dspy.LM")
    def test_tier_models_passed_correctly(self, mock_lm_cls, mock_configure):
        agent = self._make_agent_loop(tiers_dict={
            "quick": {"model": "fast/model", "temperature": 0.2},
            "normal": {"model": "mid/model"},
            "deep": {"model": "big/model", "max_tokens": 32000},
        })
        calls = mock_lm_cls.call_args_list
        # quick
        assert calls[0].args[0] == "fast/model"
        assert calls[0].kwargs["temperature"] == 0.2
        # normal
        assert calls[1].args[0] == "mid/model"
        assert calls[1].kwargs["temperature"] == 0.7  # inherited
        # deep
        assert calls[2].args[0] == "big/model"
        assert calls[2].kwargs["max_tokens"] == 32000

    @patch("dspy.configure")
    @patch("dspy.LM")
    def test_provider_credentials_passed_to_lm(self, mock_lm_cls, mock_configure):
        agent = self._make_agent_loop(tiers_dict={
            "quick": {"model": "fast/model"},
        })
        for call in mock_lm_cls.call_args_list:
            assert call.kwargs["api_key"] == "sk-test"
            assert call.kwargs["api_base"] == "http://localhost:4000"

    @patch.dict("sys.modules", {"dspy": None})
    def test_graceful_without_dspy(self):
        """AgentLoop still works when dspy is not installed."""
        agent = self._make_agent_loop(tiers_dict={
            "quick": {"model": "fast/model"},
        })
        assert agent.lm_quick is None
        assert agent.lm_normal is None
        assert agent.lm_deep is None


# ---------------------------------------------------------------------------
# Helpers to build mock litellm responses
# ---------------------------------------------------------------------------


def _make_litellm_response(
    content: str | None = "",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    usage: dict | None = None,
):
    """Build a mock litellm ModelResponse."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []
    message.reasoning_content = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    response = MagicMock()
    response.choices = [choice]

    if usage:
        response.usage = MagicMock()
        response.usage.prompt_tokens = usage["prompt_tokens"]
        response.usage.completion_tokens = usage["completion_tokens"]
        response.usage.total_tokens = usage["total_tokens"]
    else:
        response.usage = None

    return response


def _make_tool_call(call_id: str, name: str, arguments: str):
    """Build a mock tool call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc

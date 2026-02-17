"""Tests for NanobotReAct module, tool wrapping, and history bridge."""

import asyncio
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from nanobot.agent.tools.base import Tool as NanobotTool


# ---------------------------------------------------------------------------
# Fixtures: minimal nanobot tools
# ---------------------------------------------------------------------------


class DummyTool(NanobotTool):
    """A simple async tool for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "A dummy tool that echoes its input."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo"},
            },
            "required": ["text"],
        }

    async def execute(self, text: str, **kwargs: Any) -> str:
        return f"echo: {text}"


class FailingTool(NanobotTool):
    """A tool that always raises."""

    @property
    def name(self) -> str:
        return "fail"

    @property
    def description(self) -> str:
        return "Always fails."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        raise RuntimeError("intentional failure")


# ---------------------------------------------------------------------------
# Tool wrapping tests
# ---------------------------------------------------------------------------


class TestWrapNanobotTool:
    """Test conversion from nanobot Tool to dspy.Tool."""

    def test_basic_wrapping(self):
        from nanobot.agent.dspy_agent import wrap_nanobot_tool

        tool = wrap_nanobot_tool(DummyTool())
        assert tool.name == "dummy"
        assert "echo" in tool.desc.lower()
        assert "text" in tool.args

    @pytest.mark.asyncio
    async def test_async_execution(self):
        from nanobot.agent.dspy_agent import wrap_nanobot_tool

        tool = wrap_nanobot_tool(DummyTool())
        result = await tool.acall(text="hello")
        assert result == "echo: hello"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        from nanobot.agent.dspy_agent import wrap_nanobot_tool

        tool = wrap_nanobot_tool(FailingTool())
        result = await tool.acall()
        assert "Error executing fail" in result

    @pytest.mark.asyncio
    async def test_validation_error(self):
        from nanobot.agent.dspy_agent import wrap_nanobot_tool

        tool = wrap_nanobot_tool(DummyTool())
        # Missing required "text" parameter
        result = await tool.acall()
        assert "Error" in result


class TestWrapRegistry:
    """Test bulk wrapping of a ToolRegistry."""

    def test_wraps_all_tools(self):
        from nanobot.agent.tools.registry import ToolRegistry
        from nanobot.agent.dspy_agent import wrap_registry

        reg = ToolRegistry()
        reg.register(DummyTool())
        reg.register(FailingTool())

        wrapped = wrap_registry(reg)
        assert len(wrapped) == 2
        names = {t.name for t in wrapped}
        assert names == {"dummy", "fail"}


# ---------------------------------------------------------------------------
# History bridge tests
# ---------------------------------------------------------------------------


class TestBuildHistory:
    """Test conversion from session messages to dspy.History."""

    def test_paired_messages(self):
        from nanobot.agent.dspy_agent import build_history

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good!"},
        ]
        history = build_history(messages)
        assert history is not None
        assert len(history.messages) == 2
        assert history.messages[0]["message"] == "Hello"
        assert history.messages[0]["response"] == "Hi there!"
        assert history.messages[1]["message"] == "How are you?"

    def test_empty_messages(self):
        from nanobot.agent.dspy_agent import build_history

        assert build_history([]) is None

    def test_unpaired_user_message(self):
        from nanobot.agent.dspy_agent import build_history

        messages = [
            {"role": "user", "content": "Hello"},
        ]
        # Single user message without assistant response -> no pairs
        assert build_history(messages) is None

    def test_skips_system_messages(self):
        from nanobot.agent.dspy_agent import build_history

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        history = build_history(messages)
        assert history is not None
        assert len(history.messages) == 1


# ---------------------------------------------------------------------------
# NanobotReAct module tests
# ---------------------------------------------------------------------------


class TestNanobotReAct:
    """Test the NanobotReAct dspy.Module."""

    def test_module_creation(self):
        import dspy
        from nanobot.agent.dspy_agent import NanobotReAct, wrap_nanobot_tool

        tool = wrap_nanobot_tool(DummyTool())
        agent = NanobotReAct(tools=[tool], max_iters=5)
        assert agent.react is not None

    def test_module_with_instructions(self):
        import dspy
        from nanobot.agent.dspy_agent import NanobotReAct, wrap_nanobot_tool

        tool = wrap_nanobot_tool(DummyTool())
        agent = NanobotReAct(
            tools=[tool],
            max_iters=5,
            instructions="You are Jirard, a sharp AI assistant.",
        )
        assert agent.react is not None


# ---------------------------------------------------------------------------
# AgentLoop integration: _run_react_loop
# ---------------------------------------------------------------------------


class TestAgentLoopReactIntegration:
    """Test that AgentLoop correctly initialises and delegates to ReAct."""

    @patch("dspy.configure")
    @patch("dspy.LM")
    def test_react_agent_lazy_init(self, mock_lm_cls, mock_configure):
        """ReAct agent is None until run() or process_direct() is called."""
        from nanobot.config.schema import AgentDefaults, TierConfig, TiersConfig
        from nanobot.agent.loop import AgentLoop
        from pathlib import Path
        import tempfile

        defaults = AgentDefaults(
            model="test/model",
            tiers=TiersConfig(
                quick=TierConfig(model="fast/m"),
                normal=TierConfig(model="mid/m"),
                deep=TierConfig(model="big/m"),
            ),
        )
        mock_provider = MagicMock()
        mock_provider.api_key = "sk-test"
        mock_provider.api_base = "http://localhost:4000"
        mock_provider.get_default_model.return_value = "test/model"
        mock_provider._resolve_model = lambda m: m

        mock_bus = MagicMock()
        mock_bus.publish_outbound = AsyncMock()

        with tempfile.TemporaryDirectory() as td:
            with patch("nanobot.agent.loop.SubagentManager"):
                agent = AgentLoop(
                    bus=mock_bus,
                    provider=mock_provider,
                    workspace=Path(td),
                    agent_defaults=defaults,
                )
            # Should be None before run()
            assert agent._react_agent is None

    @patch("dspy.configure")
    @patch("dspy.LM")
    def test_init_react_agent_wraps_tools(self, mock_lm_cls, mock_configure):
        """_init_react_agent creates a NanobotReAct with all registered tools."""
        from nanobot.config.schema import AgentDefaults, TierConfig, TiersConfig
        from nanobot.agent.loop import AgentLoop
        from nanobot.agent.dspy_agent import NanobotReAct
        from pathlib import Path
        import tempfile

        defaults = AgentDefaults(
            model="test/model",
            tiers=TiersConfig(
                quick=TierConfig(model="fast/m"),
                normal=TierConfig(model="mid/m"),
                deep=TierConfig(model="big/m"),
            ),
        )
        mock_provider = MagicMock()
        mock_provider.api_key = "sk-test"
        mock_provider.api_base = "http://localhost:4000"
        mock_provider.get_default_model.return_value = "test/model"
        mock_provider._resolve_model = lambda m: m

        mock_bus = MagicMock()
        mock_bus.publish_outbound = AsyncMock()

        with tempfile.TemporaryDirectory() as td:
            with patch("nanobot.agent.loop.SubagentManager"):
                agent = AgentLoop(
                    bus=mock_bus,
                    provider=mock_provider,
                    workspace=Path(td),
                    agent_defaults=defaults,
                )
            react_agent = agent._init_react_agent()
            assert isinstance(react_agent, NanobotReAct)

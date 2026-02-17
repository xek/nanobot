"""Tests for the unified ThinkTool (inline quick/deep + background)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.tools.think import ThinkTool


# -- fixtures --

@pytest.fixture
def lm_quick():
    return MagicMock(name="lm_quick")


@pytest.fixture
def lm_deep():
    return MagicMock(name="lm_deep")


@pytest.fixture
def subagent_manager():
    mgr = AsyncMock()
    mgr.spawn = AsyncMock(return_value="Background task started (id: abc123).")
    return mgr


@pytest.fixture
def full_tool(lm_quick, lm_deep, subagent_manager):
    return ThinkTool(
        lm_quick=lm_quick,
        lm_deep=lm_deep,
        subagent_manager=subagent_manager,
    )


# -- metadata --

def test_name_is_think(full_tool):
    assert full_tool.name == "think"


def test_description_mentions_all_modes(full_tool):
    desc = full_tool.description
    assert "quick" in desc
    assert "deep" in desc
    assert "background" in desc


def test_parameters_schema(full_tool):
    params = full_tool.parameters
    props = params["properties"]
    assert "prompt" in props
    assert "mode" in props
    assert "label" in props
    assert params["required"] == ["prompt", "mode"]
    assert set(props["mode"]["enum"]) == {"quick", "deep", "background"}


# -- inline modes --

@pytest.mark.asyncio
async def test_inline_quick(full_tool, lm_quick):
    mock_result = MagicMock()
    mock_result.result = "42"

    with patch("dspy.context") as ctx_mock, \
         patch("dspy.Predict") as predict_cls:
        ctx_mock.return_value.__enter__ = MagicMock()
        ctx_mock.return_value.__exit__ = MagicMock(return_value=False)
        predict_cls.return_value.return_value = mock_result

        result = await full_tool.execute(prompt="What is 6*7?", mode="quick")

    assert result == "42"
    ctx_mock.assert_called_once_with(lm=lm_quick)


@pytest.mark.asyncio
async def test_inline_deep(full_tool, lm_deep):
    mock_result = MagicMock()
    mock_result.result = "deep answer"

    with patch("dspy.context") as ctx_mock, \
         patch("dspy.Predict") as predict_cls:
        ctx_mock.return_value.__enter__ = MagicMock()
        ctx_mock.return_value.__exit__ = MagicMock(return_value=False)
        predict_cls.return_value.return_value = mock_result

        result = await full_tool.execute(prompt="Hard question", mode="deep")

    assert result == "deep answer"
    ctx_mock.assert_called_once_with(lm=lm_deep)


@pytest.mark.asyncio
async def test_inline_missing_tier():
    tool = ThinkTool(lm_quick=None, lm_deep=None)
    result = await tool.execute(prompt="test", mode="quick")
    assert "Error" in result
    assert "'quick' tier is not configured" in result


@pytest.mark.asyncio
async def test_inline_error_handling(full_tool):
    with patch("dspy.context") as ctx_mock, \
         patch("dspy.Predict") as predict_cls:
        ctx_mock.return_value.__enter__ = MagicMock()
        ctx_mock.return_value.__exit__ = MagicMock(return_value=False)
        predict_cls.return_value.side_effect = RuntimeError("LLM down")

        result = await full_tool.execute(prompt="test", mode="quick")

    assert "Error in think(quick)" in result
    assert "LLM down" in result


# -- background mode --

@pytest.mark.asyncio
async def test_background_delegates_to_subagent_manager(full_tool, subagent_manager):
    result = await full_tool.execute(
        prompt="Research topic X",
        mode="background",
        label="research-x",
    )
    assert "Background task started" in result
    subagent_manager.spawn.assert_awaited_once_with(
        task="Research topic X",
        label="research-x",
        origin_channel="cli",
        origin_chat_id="direct",
    )


@pytest.mark.asyncio
async def test_background_uses_set_context(full_tool, subagent_manager):
    full_tool.set_context("slack", "C123:thread_456")
    await full_tool.execute(prompt="Do a thing", mode="background")
    subagent_manager.spawn.assert_awaited_once_with(
        task="Do a thing",
        label=None,
        origin_channel="slack",
        origin_chat_id="C123:thread_456",
    )


@pytest.mark.asyncio
async def test_background_no_manager():
    tool = ThinkTool(lm_quick=MagicMock(), lm_deep=MagicMock(), subagent_manager=None)
    result = await tool.execute(prompt="test", mode="background")
    assert "Error" in result
    assert "not available" in result


# -- registration in AgentLoop --

def test_think_tool_registered_in_agent_loop(tmp_path):
    """ThinkTool is registered and replaces the old SpawnTool."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.config.schema import AgentDefaults

    class FakeProvider:
        def get_default_model(self):
            return "test/model"

    provider = FakeProvider()
    bus = MessageBus()
    defaults = AgentDefaults(model="test/model", temperature=0.7, max_tokens=4096)

    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test/model",
        agent_defaults=defaults,
    )

    assert agent.tools.get("think") is not None
    assert isinstance(agent.tools.get("think"), ThinkTool)
    # spawn tool should no longer be registered
    assert agent.tools.get("spawn") is None

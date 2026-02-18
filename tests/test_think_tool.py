"""Tests for the unified ThinkTool (inline quick/deep + background)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.think import ThinkTool


# -- fixtures --

@pytest.fixture
def subagent_manager():
    mgr = AsyncMock()
    mgr.spawn = AsyncMock(return_value="Background task started (id: abc123).")
    mgr.run_inline = AsyncMock(return_value="Inline result from subagent.")
    return mgr


@pytest.fixture
def full_tool(subagent_manager):
    return ThinkTool(
        subagent_manager=subagent_manager,
        quick_config=("fast/model", 0.2, 2048),
        deep_config=("powerful/model", 1.0, 16384),
    )


# -- metadata --

def test_name_is_think(full_tool):
    assert full_tool.name == "think"


def test_description_mentions_all_modes(full_tool):
    desc = full_tool.description
    assert "quick" in desc
    assert "deep" in desc
    assert "background" in desc
    assert "tool" in desc.lower()


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
async def test_inline_quick(full_tool, subagent_manager):
    result = await full_tool.execute(prompt="Classify this text", mode="quick")

    assert result == "Inline result from subagent."
    subagent_manager.run_inline.assert_awaited_once_with(
        task="Classify this text",
        model="fast/model",
        temperature=0.2,
        max_tokens=2048,
    )


@pytest.mark.asyncio
async def test_inline_deep(full_tool, subagent_manager):
    result = await full_tool.execute(prompt="Hard question", mode="deep")

    assert result == "Inline result from subagent."
    subagent_manager.run_inline.assert_awaited_once_with(
        task="Hard question",
        model="powerful/model",
        temperature=1.0,
        max_tokens=16384,
    )


@pytest.mark.asyncio
async def test_inline_missing_tier():
    mgr = AsyncMock()
    tool = ThinkTool(subagent_manager=mgr, quick_config=None, deep_config=None)
    result = await tool.execute(prompt="test", mode="quick")
    assert "Error" in result
    assert "'quick' tier is not configured" in result


@pytest.mark.asyncio
async def test_no_subagent_manager():
    tool = ThinkTool(subagent_manager=None)
    result = await tool.execute(prompt="test", mode="quick")
    assert "Error" in result
    assert "not available" in result


# -- background mode --

@pytest.mark.asyncio
async def test_background_delegates_to_spawn(full_tool, subagent_manager):
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
    assert agent.tools.get("spawn") is None

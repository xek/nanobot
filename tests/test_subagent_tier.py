"""Tests for subagent and process_direct tier defaults."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from nanobot.bus.queue import MessageBus
from nanobot.config.schema import AgentDefaults, TierConfig, TiersConfig
from nanobot.providers.base import LLMProvider, LLMResponse


class FakeProvider(LLMProvider):
    """Minimal LLM provider for testing."""

    def __init__(self):
        super().__init__()
        self.last_model: str | None = None
        self.last_temperature: float | None = None
        self.last_max_tokens: int | None = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.last_model = model
        self.last_temperature = temperature
        self.last_max_tokens = max_tokens
        return LLMResponse(content="ok", finish_reason="stop")

    def get_default_model(self) -> str:
        return "default/model"


def _make_defaults(**tier_kwargs) -> AgentDefaults:
    tiers = {}
    for name in ("quick", "normal", "deep"):
        if name in tier_kwargs:
            tiers[name] = TierConfig(**tier_kwargs[name])
    return AgentDefaults(
        model="base/model",
        temperature=0.7,
        max_tokens=4096,
        tiers=TiersConfig(**tiers),
    )


# === SubagentManager uses quick tier ===

def test_subagent_manager_uses_quick_tier(tmp_path):
    from nanobot.agent.loop import AgentLoop

    defaults = _make_defaults(
        quick={"model": "fast/lite", "temperature": 0.2, "max_tokens": 2048},
    )
    provider = FakeProvider()
    bus = MessageBus()

    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="base/model",
        agent_defaults=defaults,
    )

    assert agent.subagents.model == "fast/lite"
    assert agent.subagents.temperature == 0.2
    assert agent.subagents.max_tokens == 2048


def test_subagent_manager_falls_back_when_no_quick_tier(tmp_path):
    from nanobot.agent.loop import AgentLoop

    defaults = _make_defaults()
    provider = FakeProvider()
    bus = MessageBus()

    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="base/model",
        agent_defaults=defaults,
    )

    # No quick tier configured -> falls back to base model
    assert agent.subagents.model == "base/model"
    assert agent.subagents.temperature == 0.7
    assert agent.subagents.max_tokens == 4096


# === process_direct tier switching ===

async def test_process_direct_applies_tier(tmp_path):
    from nanobot.agent.loop import AgentLoop

    defaults = _make_defaults(
        quick={"model": "fast/lite", "temperature": 0.2, "max_tokens": 2048},
        deep={"model": "openai/o3", "temperature": 1.0, "max_tokens": 16384},
    )
    provider = FakeProvider()
    bus = MessageBus()

    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="base/model",
        agent_defaults=defaults,
    )

    # Tier switching happens through model/temperature/max_tokens override
    # during process_direct. We verify the agent's state is restored after.
    assert agent.model == "base/model"
    assert agent.temperature == 0.7

    # process_direct with tier="quick" should temporarily switch
    await agent.process_direct("test", tier="quick")
    # The provider should have received the quick tier's model
    assert provider.last_model == "fast/lite"
    assert provider.last_temperature == 0.2
    assert provider.last_max_tokens == 2048

    # After the call, the agent's state should be restored
    assert agent.model == "base/model"
    assert agent.temperature == 0.7
    assert agent.max_tokens == 4096


async def test_process_direct_restores_state_on_error(tmp_path):
    from nanobot.agent.loop import AgentLoop

    defaults = _make_defaults(
        deep={"model": "openai/o3", "temperature": 1.0},
    )

    class FailingProvider(FakeProvider):
        async def chat(self, **kwargs):
            raise RuntimeError("boom")

    provider = FailingProvider()
    bus = MessageBus()

    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="base/model",
        agent_defaults=defaults,
    )

    # Even if process_direct fails, state should be restored
    try:
        await agent.process_direct("test", tier="deep")
    except Exception:
        pass

    assert agent.model == "base/model"
    assert agent.temperature == 0.7


async def test_process_direct_no_tier_uses_default(tmp_path):
    from nanobot.agent.loop import AgentLoop

    defaults = _make_defaults(
        quick={"model": "fast/lite", "temperature": 0.2},
    )
    provider = FakeProvider()
    bus = MessageBus()

    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="base/model",
        agent_defaults=defaults,
    )

    await agent.process_direct("test")
    assert provider.last_model == "base/model"
    assert provider.last_temperature == 0.7

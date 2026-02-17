"""Tests for TierConfig / TiersConfig and AgentDefaults.resolve_tier()."""

from nanobot.config.schema import AgentDefaults, TierConfig, TiersConfig


def _defaults_with_tiers(**tier_kwargs) -> AgentDefaults:
    """Build AgentDefaults with custom tier overrides."""
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


# === resolve_tier: no tiers configured ===

def test_resolve_tier_none_returns_defaults():
    d = _defaults_with_tiers()
    model, temp, max_tok = d.resolve_tier(None)
    assert model == "base/model"
    assert temp == 0.7
    assert max_tok == 4096


def test_resolve_tier_unknown_returns_defaults():
    d = _defaults_with_tiers()
    model, temp, max_tok = d.resolve_tier("unknown")
    assert model == "base/model"


def test_resolve_tier_empty_model_returns_defaults():
    """A tier with no model set falls back to top-level defaults."""
    d = _defaults_with_tiers(quick={"model": ""})
    model, temp, max_tok = d.resolve_tier("quick")
    assert model == "base/model"
    assert temp == 0.7
    assert max_tok == 4096


# === resolve_tier: tiers configured ===

def test_resolve_tier_quick_overrides_model():
    d = _defaults_with_tiers(quick={"model": "fast/lite", "temperature": 0.3})
    model, temp, max_tok = d.resolve_tier("quick")
    assert model == "fast/lite"
    assert temp == 0.3
    assert max_tok == 4096  # inherited


def test_resolve_tier_deep_overrides_all():
    d = _defaults_with_tiers(deep={
        "model": "openai/o3",
        "temperature": 1.0,
        "max_tokens": 16384,
    })
    model, temp, max_tok = d.resolve_tier("deep")
    assert model == "openai/o3"
    assert temp == 1.0
    assert max_tok == 16384


def test_resolve_tier_normal_with_partial_override():
    d = _defaults_with_tiers(normal={"model": "gemini/pro", "temperature": None})
    model, temp, max_tok = d.resolve_tier("normal")
    assert model == "gemini/pro"
    assert temp == 0.7  # inherited from top-level
    assert max_tok == 4096  # inherited


def test_resolve_tier_inherits_max_tokens_when_not_set():
    d = _defaults_with_tiers(quick={"model": "fast/lite"})
    _, _, max_tok = d.resolve_tier("quick")
    assert max_tok == 4096


# === TiersConfig defaults ===

def test_tiers_config_defaults_are_empty():
    t = TiersConfig()
    assert t.quick.model == ""
    assert t.normal.model == ""
    assert t.deep.model == ""


def test_tier_config_defaults():
    tc = TierConfig()
    assert tc.model == ""
    assert tc.temperature is None
    assert tc.max_tokens is None


# === Config loading from dict (simulates JSON config) ===

def test_agent_defaults_from_dict_with_tiers():
    data = {
        "model": "anthropic/claude-opus-4-5",
        "temperature": 0.7,
        "max_tokens": 8192,
        "tiers": {
            "quick": {"model": "gemini/flash-lite", "temperature": 0.3},
            "normal": {"model": "gemini/pro"},
            "deep": {"model": "openai/o3", "temperature": 1.0, "max_tokens": 16384},
        },
    }
    d = AgentDefaults(**data)
    assert d.tiers.quick.model == "gemini/flash-lite"
    assert d.tiers.quick.temperature == 0.3
    assert d.tiers.normal.model == "gemini/pro"
    assert d.tiers.normal.temperature is None
    assert d.tiers.deep.model == "openai/o3"
    assert d.tiers.deep.max_tokens == 16384


def test_agent_defaults_from_dict_without_tiers():
    data = {
        "model": "anthropic/claude-opus-4-5",
        "temperature": 0.7,
        "max_tokens": 8192,
    }
    d = AgentDefaults(**data)
    assert d.tiers.quick.model == ""
    assert d.tiers.normal.model == ""
    assert d.tiers.deep.model == ""
    # resolve_tier still works, falls back
    model, _, _ = d.resolve_tier("quick")
    assert model == "anthropic/claude-opus-4-5"

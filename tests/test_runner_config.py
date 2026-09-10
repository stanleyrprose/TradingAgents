from copy import deepcopy

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.runner_config import build_codex_oauth_config

CODEX_DEFAULTS = {
    "quick_think_llm_provider": "codex_cli",
    "quick_think_llm": "gpt-5.6-luna",
    "codex_quick_reasoning_effort": "low",
    "trader_think_llm_provider": "codex_cli",
    "trader_think_llm": "gpt-5.6-sol",
    "codex_trader_reasoning_effort": "medium",
    "deep_think_llm_provider": "codex_cli",
    "deep_think_llm": "gpt-5.6-sol",
    "codex_deep_reasoning_effort": "high",
    "checkpoint_enabled": True,
}


def test_build_returns_fresh_dict_with_stable_defaults(monkeypatch):
    for env in (
        "TRADINGAGENTS_QUICK_THINK_LLM_PROVIDER",
        "TRADINGAGENTS_QUICK_THINK_LLM",
        "TRADINGAGENTS_CODEX_QUICK_REASONING_EFFORT",
        "TRADINGAGENTS_TRADER_THINK_LLM_PROVIDER",
        "TRADINGAGENTS_TRADER_THINK_LLM",
        "TRADINGAGENTS_CODEX_TRADER_REASONING_EFFORT",
        "TRADINGAGENTS_DEEP_THINK_LLM_PROVIDER",
        "TRADINGAGENTS_DEEP_THINK_LLM",
        "TRADINGAGENTS_CODEX_DEEP_REASONING_EFFORT",
        "TRADINGAGENTS_CHECKPOINT_ENABLED",
    ):
        monkeypatch.delenv(env, raising=False)

    first = build_codex_oauth_config()
    second = build_codex_oauth_config()

    assert first is not second
    assert {key: first[key] for key in CODEX_DEFAULTS} == CODEX_DEFAULTS


def test_present_env_preserves_env_derived_default_config(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_DEEP_THINK_LLM", "configured-in-environment")
    monkeypatch.setenv("TRADINGAGENTS_CHECKPOINT_ENABLED", "false")
    monkeypatch.setitem(DEFAULT_CONFIG, "deep_think_llm", "configured-in-environment")
    monkeypatch.setitem(DEFAULT_CONFIG, "checkpoint_enabled", False)

    config = build_codex_oauth_config()

    assert config["deep_think_llm"] == "configured-in-environment"
    assert config["checkpoint_enabled"] is False


def test_mutation_does_not_leak_to_default_config():
    original = deepcopy(DEFAULT_CONFIG)
    config = build_codex_oauth_config()

    config["quick_think_llm"] = "changed"
    config["new_key"] = "new"

    assert original == DEFAULT_CONFIG

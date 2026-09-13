"""Small shared configuration helpers for programmatic graph runners."""

from __future__ import annotations

import os

from tradingagents.default_config import DEFAULT_CONFIG

_CODEX_OAUTH_DEFAULTS = {
    "TRADINGAGENTS_QUICK_THINK_LLM_PROVIDER": ("quick_think_llm_provider", "codex_cli"),
    "TRADINGAGENTS_QUICK_THINK_LLM": ("quick_think_llm", "gpt-5.6-luna"),
    "TRADINGAGENTS_CODEX_QUICK_REASONING_EFFORT": ("codex_quick_reasoning_effort", "low"),
    "TRADINGAGENTS_TRADER_THINK_LLM_PROVIDER": ("trader_think_llm_provider", "codex_cli"),
    "TRADINGAGENTS_TRADER_THINK_LLM": ("trader_think_llm", "gpt-5.6-sol"),
    "TRADINGAGENTS_CODEX_TRADER_REASONING_EFFORT": ("codex_trader_reasoning_effort", "medium"),
    "TRADINGAGENTS_DEEP_THINK_LLM_PROVIDER": ("deep_think_llm_provider", "codex_cli"),
    "TRADINGAGENTS_DEEP_THINK_LLM": ("deep_think_llm", "gpt-5.6-sol"),
    "TRADINGAGENTS_CODEX_DEEP_REASONING_EFFORT": ("codex_deep_reasoning_effort", "high"),
    "TRADINGAGENTS_CHECKPOINT_ENABLED": ("checkpoint_enabled", True),
}


def build_codex_oauth_config() -> dict:
    """Return runner config with the stable Codex OAuth defaults unless overridden."""
    config = DEFAULT_CONFIG.copy()
    for env, (key, value) in _CODEX_OAUTH_DEFAULTS.items():
        if env not in os.environ:
            config[key] = value
    return config

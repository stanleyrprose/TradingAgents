"""Unit tests for the deterministic option scenario engine."""

from datetime import date
from unittest import mock

import pytest

from tradingagents.dataflows import option_scenarios as scenarios


def test_black_scholes_known_call_and_put_values():
    call = scenarios.black_scholes_price(100, 100, 1, 0.05, 0.20, "C")
    put = scenarios.black_scholes_price(100, 100, 1, 0.05, 0.20, "P")
    assert call == pytest.approx(10.45, abs=0.01)
    assert put == pytest.approx(5.57, abs=0.01)


def test_black_scholes_expiry_intrinsic():
    assert scenarios.black_scholes_price(110, 100, 0, 0.05, 0.20, "C") == 10
    assert scenarios.black_scholes_price(90, 100, 0, 0.05, 0.20, "P") == 10


def test_expiry_breakeven_for_both_rights():
    assert scenarios.expiry_breakeven(100, 4.5, "C") == 104.5
    assert scenarios.expiry_breakeven(100, 4.5, "P") == 95.5


def test_resolver_falls_back_without_key_and_does_not_request(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with mock.patch.object(scenarios.fred, "_request") as request:
        assert scenarios.resolve_scenario_risk_free_rate("2026-01-15") == (
            0.04,
            "fallback 4.00% (FRED FEDFUNDS unavailable)",
        )
    request.assert_not_called()


def test_resolver_parses_latest_valid_fedfunds(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "configured")
    observations = {
        "observations": [
            {"date": "2026-01-14", "value": "."},
            {"date": "2026-01-13", "value": "4.33"},
        ]
    }
    with (
        mock.patch.object(scenarios.fred, "_fred_today", return_value="2026-01-20"),
        mock.patch.object(scenarios.fred, "_request", return_value=observations) as request,
    ):
        rate, provenance = scenarios.resolve_scenario_risk_free_rate("2026-01-15")
    assert rate == pytest.approx(0.0433)
    assert "FEDFUNDS 4.33%" in provenance
    assert request.call_args.args[0] == "series/observations"
    assert request.call_args.args[1]["sort_order"] == "desc"


def test_required_spot_moves_with_time_decay_for_call_and_put():
    call_premium = scenarios.black_scholes_price(100, 100, 30 / 365, 0.04, 0.25, "C")
    put_premium = scenarios.black_scholes_price(100, 100, 30 / 365, 0.04, 0.25, "P")
    call_spot = scenarios.required_spot_to_preserve_premium(
        100, 100, call_premium, 20 / 365, 0.04, 0.25, "C"
    )
    put_spot = scenarios.required_spot_to_preserve_premium(
        100, 100, put_premium, 20 / 365, 0.04, 0.25, "P"
    )
    assert call_spot is not None and call_spot > 100
    assert put_spot is not None and put_spot < 100


def test_short_dte_report_dedupes_and_skips_invalid_horizons():
    report = scenarios.build_option_scenario_report(
        "XYZ", "C", 100, date(2026, 1, 2), date(2026, 1, 1), 100, 2, 0.25,
        risk_free_rate=0.04,
    )
    theta_section = report.split("## Theta burn", 1)[1].split("## Required", 1)[0]
    assert theta_section.count("| +0 |") == 1
    assert theta_section.count("| +1 |") == 1
    assert "| +3 |" not in theta_section
    assert "| +5 |" not in theta_section


def test_expiry_date_report_uses_day_zero_intrinsic_matrix():
    report = scenarios.build_option_scenario_report(
        "XYZ",
        "C",
        100,
        date(2026, 1, 1),
        date(2026, 1, 1),
        100,
        2,
        0.25,
        risk_free_rate=0.04,
    )
    matrix_section = report.split("## Spot × IV matrix", 1)[1].split("## Expiry payoff", 1)[0]
    assert "Spot × IV matrix (day +0)" in report
    assert "Spot × IV matrix (day +1)" not in report
    assert "| $105.00 | $5.00 / +300.00 | $5.00 / +300.00 | $5.00 / +300.00 |" in matrix_section


def test_report_has_required_sections_and_only_finite_output():
    report = scenarios.build_option_scenario_report(
        "XYZ",
        "P",
        100,
        date(2026, 2, 15),
        date(2026, 1, 15),
        100,
        4.25,
        0.25,
        vendor_theo=4.10,
        vendor_delta=-0.45,
        vendor_gamma=0.03,
        vendor_theta=-0.08,
        risk_free_rate=0.04,
        risk_free_provenance="test rate",
    )
    for label in (
        "Deterministic option scenario engine",
        "Assumptions",
        "Current premium and expiry breakeven",
        "Base Black-Scholes model cross-check",
        "Theta burn",
        "Required underlying to preserve current premium",
        "Spot × IV matrix",
        "Expiry payoff",
        "Delta/gamma local stress",
        "Caveats",
    ):
        assert label in report
    assert "model cross-check, not market fair value or target" in report
    assert "Risk-free-rate proxy: 4.00% (test rate)." in report
    assert "FEDFUNDS/fallback is a proxy, not a maturity-matched Treasury/OIS curve." in report
    assert "vendor theta" in report.lower()
    assert "nan" not in report.lower()
    assert "inf" not in report.lower()

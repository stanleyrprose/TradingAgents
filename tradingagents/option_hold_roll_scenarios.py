"""Scenario-normalized deterministic comparison of HOLD versus long-option rolls."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from tradingagents.dataflows.equity_options import EquityOptionSnapshot
from tradingagents.dataflows.option_scenarios import (
    black_scholes_price,
    resolve_scenario_risk_free_rate,
)
from tradingagents.option_capital_allocator import CONTRACT_MULTIPLIER
from tradingagents.option_roll_planner import OptionRollPlanResult

_SPOT_SHOCKS = (-0.05, 0.0, 0.05)
_IV_SHOCKS = (-0.05, 0.0, 0.05)
_MIN_IV = 0.01


@dataclass(frozen=True)
class RollScenarioAlternative:
    """One roll candidate under one normalized market scenario."""

    symbol: str
    model_value_per_share: float
    lifecycle_pnl_dollars: float
    advantage_vs_hold_dollars: float


@dataclass(frozen=True)
class HoldRollScenarioPoint:
    """HOLD and all roll candidates under one shared spot/time/IV-shock scenario."""

    horizon_days: int
    spot_change_pct: float
    iv_shock_pp: float
    scenario_spot: float
    hold_model_value_per_share: float
    hold_lifecycle_pnl_dollars: float
    hold_forward_pnl_dollars: float
    alternatives: tuple[RollScenarioAlternative, ...]


@dataclass(frozen=True)
class RollScenarioSummary:
    """Descriptive grid dominance statistics for one roll candidate."""

    symbol: str
    scenario_count: int
    outperform_count: int
    tie_count: int
    underperform_count: int
    min_advantage_dollars: float
    median_advantage_dollars: float
    max_advantage_dollars: float


@dataclass(frozen=True)
class HoldRollScenarioResult:
    """Normalized scenario grid; descriptive only, never an execution decision."""

    status: str
    reason: str
    risk_free_rate: float | None
    risk_free_provenance: str | None
    horizons: tuple[int, ...]
    points: tuple[HoldRollScenarioPoint, ...]
    summaries: tuple[RollScenarioSummary, ...]
    report: str


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive(value: object, name: str) -> float:
    number = _finite(value)
    if number is None or number <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return number


def _nonnegative(value: object, name: str) -> float:
    number = _finite(value)
    if number is None or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


def _horizons(dte: int) -> tuple[int, ...]:
    if isinstance(dte, bool) or not isinstance(dte, int) or dte < 0:
        raise ValueError("snapshot dte must be a nonnegative whole number")
    return tuple(sorted({day for day in (3, 7, dte) if 0 <= day <= dte}))


def _scenario_iv(base_iv: float, shock: float) -> float:
    return max(base_iv + shock, _MIN_IV)


def _render_unavailable(reason: str) -> str:
    return f"<scenario-normalized hold-vs-roll unavailable: {reason}>"


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _signed_money(value: float) -> str:
    return f"{value:+,.2f}"


def _render(
    snapshot: EquityOptionSnapshot,
    *,
    contracts: int,
    entry_premium: float,
    rate: float,
    provenance: str,
    horizons: tuple[int, ...],
    points: tuple[HoldRollScenarioPoint, ...],
    summaries: tuple[RollScenarioSummary, ...],
) -> str:
    symbols = tuple(summary.symbol for summary in summaries)
    lines = [
        "# Scenario-normalized hold vs roll comparison",
        "",
        "Status: **COMPARE**",
        f"Current contract: {snapshot.symbol} | Contracts: {contracts} | Original entry premium: ${entry_premium:.2f}",
        f"Underlying spot baseline: ${snapshot.underlying_spot:.2f} | Old DTE: {snapshot.dte}",
        f"Risk-free-rate proxy: {rate:.2%} ({provenance}); q=0.00%.",
        f"Horizons: {', '.join(f'+{day}d' for day in horizons)} | Spot shocks: -5%, 0%, +5% | IV shocks: -5pp, 0pp, +5pp.",
        "Each contract keeps its own current IV baseline; the same IV shock is applied to each baseline.",
        "All values use the same European Black-Scholes scenario model so HOLD and ROLL alternatives are compared on one normalized model basis.",
        "",
        "## Lifecycle P/L semantics",
        "",
        "HOLD lifecycle P/L = future old-option model value - original entry premium.",
        "ROLL lifecycle P/L = (old delayed bid - original entry premium) + (future replacement model value - replacement delayed ask).",
        "ROLL advantage vs HOLD is the difference between those two lifecycle P/L values under the exact same scenario.",
        "",
        "## Grid summary",
        "",
        "The counts below are descriptive grid counts only. They are not probabilities, expected returns, or equally likely market forecasts.",
        "",
        "| Roll candidate | Better than HOLD | Tie | Worse than HOLD | Worst advantage | Median advantage | Best advantage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.symbol} | {summary.outperform_count}/{summary.scenario_count} | "
            f"{summary.tie_count} | {summary.underperform_count} | "
            f"{_signed_money(summary.min_advantage_dollars)} | "
            f"{_signed_money(summary.median_advantage_dollars)} | "
            f"{_signed_money(summary.max_advantage_dollars)} |"
        )

    for horizon in horizons:
        for iv_shock in _IV_SHOCKS:
            lines.extend(
                [
                    "",
                    f"## Horizon +{horizon}d | IV shock {iv_shock * 100:+.0f}pp",
                    "",
                    "| Spot shock | Scenario spot | HOLD lifecycle P/L | HOLD forward P/L | "
                    + " | ".join(f"{symbol} advantage vs HOLD" for symbol in symbols)
                    + " |",
                    "|---:|---:|---:|---:|" + "---:|" * len(symbols),
                ]
            )
            matching = [
                point
                for point in points
                if point.horizon_days == horizon and point.iv_shock_pp == iv_shock
            ]
            for point in matching:
                advantages = {item.symbol: item.advantage_vs_hold_dollars for item in point.alternatives}
                lines.append(
                    f"| {point.spot_change_pct:+.0%} | {_money(point.scenario_spot)} | "
                    f"{_signed_money(point.hold_lifecycle_pnl_dollars)} | "
                    f"{_signed_money(point.hold_forward_pnl_dollars)} | "
                    + " | ".join(_signed_money(advantages[symbol]) for symbol in symbols)
                    + " |"
                )

    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            "A positive roll advantage means that replacement has higher lifecycle P/L than HOLD in that specific modeled scenario; a negative value means HOLD is better in that scenario.",
            "No win count or median advantage is a probability-weighted forecast. The grid deliberately does not assign scenario probabilities or collapse the trade-offs into a single roll score.",
            "",
            "## Caveats",
            "",
            "This reuses the v1.1 European Black-Scholes scenario engine with q=0 and fixed per-contract IV plus the stated shock. It does not model American early exercise, dividends, volatility smile dynamics, earnings/event jumps, path dependence, transaction costs, taxes, slippage, or changing rates.",
            "Cboe inputs are delayed. Old-position close economics use the delayed bid already frozen in the roll plan; replacement entry economics use each candidate delayed ask. Model scenario values are not executable quotes, fair values, or guaranteed P/L.",
        ]
    )
    return "\n".join(lines)


def compare_hold_vs_roll_scenarios(
    snapshot: EquityOptionSnapshot,
    roll_plan: OptionRollPlanResult,
    *,
    entry_premium: object,
    contracts: object,
    risk_free_rate: object | None = None,
    risk_free_provenance: str | None = None,
) -> HoldRollScenarioResult:
    """Compare HOLD and roll candidates over the same deterministic scenario grid."""

    premium = _positive(entry_premium, "entry_premium")
    if isinstance(contracts, bool) or not isinstance(contracts, int) or contracts <= 0:
        raise ValueError("contracts must be a positive whole number")
    if roll_plan.status != "COMPARE" or not roll_plan.candidates:
        reason = "scenario comparison requires a COMPARE roll plan with candidates"
        return HoldRollScenarioResult(
            "NO_COMPARE", reason, None, None, (), (), (), _render_unavailable(reason)
        )

    spot = _finite(snapshot.underlying_spot)
    hold_iv = _finite(snapshot.iv)
    bid = _finite(snapshot.bid)
    if spot is None or spot <= 0:
        reason = "current underlying spot is unavailable"
        return HoldRollScenarioResult(
            "NO_COMPARE", reason, None, None, (), (), (), _render_unavailable(reason)
        )
    if hold_iv is None or hold_iv <= 0:
        reason = "current option IV is unavailable"
        return HoldRollScenarioResult(
            "NO_COMPARE", reason, None, None, (), (), (), _render_unavailable(reason)
        )
    if bid is None or bid < 0:
        reason = "current delayed bid is unavailable"
        return HoldRollScenarioResult(
            "NO_COMPARE", reason, None, None, (), (), (), _render_unavailable(reason)
        )
    horizons = _horizons(snapshot.dte)
    if not horizons:
        reason = "no valid scenario horizon is available"
        return HoldRollScenarioResult(
            "NO_COMPARE", reason, None, None, (), (), (), _render_unavailable(reason)
        )

    if risk_free_rate is None:
        rate, provenance = resolve_scenario_risk_free_rate(snapshot.as_of)
    else:
        parsed_rate = _finite(risk_free_rate)
        if parsed_rate is None:
            raise ValueError("risk_free_rate must be a finite number")
        rate = parsed_rate
        provenance = risk_free_provenance or "caller supplied"

    scale = CONTRACT_MULTIPLIER * contracts
    for candidate in roll_plan.candidates:
        _positive(candidate.iv, f"{candidate.symbol} iv")
        _positive(candidate.open_debit_per_share, f"{candidate.symbol} open_debit_per_share")
        close_credit = _nonnegative(
            candidate.close_credit_per_share, f"{candidate.symbol} close_credit_per_share"
        )
        if not math.isclose(close_credit, bid, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"{candidate.symbol} roll close credit does not match current snapshot bid"
            )

    points: list[HoldRollScenarioPoint] = []
    advantages: dict[str, list[float]] = {
        candidate.symbol: [] for candidate in roll_plan.candidates
    }
    for horizon in horizons:
        hold_time = max(snapshot.dte - horizon, 0) / 365.0
        for iv_shock in _IV_SHOCKS:
            shocked_hold_iv = _scenario_iv(hold_iv, iv_shock)
            for spot_shock in _SPOT_SHOCKS:
                scenario_spot = spot * (1 + spot_shock)
                hold_value = black_scholes_price(
                    scenario_spot,
                    snapshot.strike,
                    hold_time,
                    rate,
                    shocked_hold_iv,
                    snapshot.right,
                    0.0,
                )
                hold_lifecycle_pnl = (hold_value - premium) * scale
                hold_forward_pnl = (hold_value - bid) * scale
                roll_outcomes: list[RollScenarioAlternative] = []
                for candidate in roll_plan.candidates:
                    candidate_time = max(candidate.dte - horizon, 0) / 365.0
                    candidate_iv = _scenario_iv(candidate.iv, iv_shock)
                    new_value = black_scholes_price(
                        scenario_spot,
                        candidate.strike,
                        candidate_time,
                        rate,
                        candidate_iv,
                        snapshot.right,
                        0.0,
                    )
                    lifecycle_pnl = (
                        (candidate.close_credit_per_share - premium)
                        + (new_value - candidate.open_debit_per_share)
                    ) * scale
                    advantage = lifecycle_pnl - hold_lifecycle_pnl
                    advantages[candidate.symbol].append(advantage)
                    roll_outcomes.append(
                        RollScenarioAlternative(
                            candidate.symbol,
                            new_value,
                            lifecycle_pnl,
                            advantage,
                        )
                    )
                points.append(
                    HoldRollScenarioPoint(
                        horizon,
                        spot_shock,
                        iv_shock,
                        scenario_spot,
                        hold_value,
                        hold_lifecycle_pnl,
                        hold_forward_pnl,
                        tuple(roll_outcomes),
                    )
                )

    summaries: list[RollScenarioSummary] = []
    tolerance = 0.01
    for candidate in roll_plan.candidates:
        values = advantages[candidate.symbol]
        outperform = sum(value > tolerance for value in values)
        underperform = sum(value < -tolerance for value in values)
        ties = len(values) - outperform - underperform
        summaries.append(
            RollScenarioSummary(
                candidate.symbol,
                len(values),
                outperform,
                ties,
                underperform,
                min(values),
                median(values),
                max(values),
            )
        )

    point_tuple = tuple(points)
    summary_tuple = tuple(summaries)
    return HoldRollScenarioResult(
        "COMPARE",
        "HOLD and roll candidates were evaluated under identical spot/time shocks and per-contract IV shocks",
        rate,
        provenance,
        horizons,
        point_tuple,
        summary_tuple,
        _render(
            snapshot,
            contracts=contracts,
            entry_premium=premium,
            rate=rate,
            provenance=provenance,
            horizons=horizons,
            points=point_tuple,
            summaries=summary_tuple,
        ),
    )

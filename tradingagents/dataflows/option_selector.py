"""Deterministic current-chain selector for long US equity options."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import date
from statistics import median

import requests

from .equity_options import _UA, _URL, _finite, _parse_occ, _row_number, _today

logger = logging.getLogger(__name__)

_UNDERLYING_RE = re.compile(r"[A-Z]{1,6}")
_WEIGHTS = (
    ("Delta fit", 25.0),
    ("Bid-ask spread", 20.0),
    ("Open interest", 15.0),
    ("Volume", 10.0),
    ("DTE fit", 10.0),
    ("Theta burden", 10.0),
    ("Breakeven", 5.0),
    ("IV-relative", 5.0),
)


@dataclass(frozen=True)
class OptionCandidate:
    """One eligible option contract with its deterministic ranking details."""

    symbol: str
    expiry: date
    right: str
    strike: float
    dte: int
    bid: float
    ask: float
    midpoint: float
    spread_pct: float
    delta: float
    abs_delta: float
    iv: float
    oi: float
    volume: float
    theta: float | None
    theta_burden: float | None
    breakeven: float
    required_move: float
    components: tuple[tuple[str, float, float], ...]
    score: float


@dataclass(frozen=True)
class OptionSelectionResult:
    """Structured current-chain selection plus its backward-compatible report."""

    candidates: tuple[OptionCandidate, ...]
    report: str
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.candidates)


def _unavailable(reason: str) -> str:
    return f"<equity option selection unavailable: {reason}>"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _nonnegative(row: dict, field: str) -> float | None:
    value = row.get(field)
    if value is None:
        return 0.0
    number = _finite(value)
    return number if number is not None and number >= 0 else None


def _points(
    candidate: dict[str, object],
    *,
    target_delta: float,
    target_dte: int,
    min_dte: int,
    max_dte: int,
    median_iv: float,
) -> tuple[tuple[str, float, float], ...]:
    abs_delta = float(candidate["abs_delta"])
    spread_pct = float(candidate["spread_pct"])
    oi = float(candidate["oi"])
    volume = float(candidate["volume"])
    dte = int(candidate["dte"])
    theta_burden = candidate["theta_burden"]
    required_move = float(candidate["required_move"])
    iv = float(candidate["iv"])

    delta_score = 25 * _clamp(1 - abs(abs_delta - target_delta) / 0.30, 0, 1)
    spread_score = 20 * _clamp(1 - spread_pct / 25, 0, 1)
    oi_score = 15 * min(math.log1p(oi) / math.log1p(5000), 1)
    volume_score = 10 * min(math.log1p(volume) / math.log1p(1000), 1)
    if min_dte == max_dte:
        dte_score = 10.0
    else:
        farther_boundary = max(target_dte - min_dte, max_dte - target_dte)
        dte_score = 10 * _clamp(1 - abs(dte - target_dte) / farther_boundary, 0, 1)
    theta_score = (
        5.0
        if theta_burden is None
        else 10 * _clamp(1 - float(theta_burden) / 0.10, 0, 1)
    )
    breakeven_score = 5 * _clamp(1 - required_move / 0.15, 0, 1)
    iv_ratio = iv / median_iv
    iv_score = 5 * _clamp(1.5 - iv_ratio, 0, 1)
    return (
        ("Delta fit", delta_score, 25.0),
        ("Bid-ask spread", spread_score, 20.0),
        ("Open interest", oi_score, 15.0),
        ("Volume", volume_score, 10.0),
        ("DTE fit", dte_score, 10.0),
        ("Theta burden", theta_score, 10.0),
        ("Breakeven", breakeven_score, 5.0),
        ("IV-relative", iv_score, 5.0),
    )


def _eligible_candidates(
    options: list,
    *,
    underlying: str,
    right: str,
    spot: float,
    today: date,
    min_dte: int,
    max_dte: int,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for row in options:
        if not isinstance(row, dict):
            continue
        contract = _parse_occ(row.get("option"))
        if contract is None or contract.root != underlying or contract.right != right:
            continue
        dte = (contract.expiry - today).days
        if not min_dte <= dte <= max_dte:
            continue
        bid = _row_number(row, "bid", positive=True)
        ask = _row_number(row, "ask", positive=True)
        iv = _row_number(row, "iv", positive=True)
        delta = _row_number(row, "delta")
        oi = _nonnegative(row, "open_interest")
        volume = _nonnegative(row, "volume")
        theta = _row_number(row, "theta")
        if None in (bid, ask, iv, delta, oi, volume) or ask < bid:
            continue
        if (right == "C" and delta <= 0) or (right == "P" and delta >= 0):
            continue
        abs_delta = abs(delta)
        if not 0.10 <= abs_delta <= 0.90:
            continue
        midpoint = (bid + ask) / 2
        if midpoint <= 0:
            continue
        spread_pct = (ask - bid) / midpoint * 100
        if spread_pct > 25:
            continue
        breakeven = (
            contract.strike + midpoint if right == "C" else contract.strike - midpoint
        )
        required_move = (
            max(breakeven / spot - 1, 0)
            if right == "C"
            else max(1 - breakeven / spot, 0)
        )
        candidates.append(
            {
                "symbol": contract.symbol,
                "expiry": contract.expiry,
                "right": contract.right,
                "strike": contract.strike,
                "dte": dte,
                "bid": bid,
                "ask": ask,
                "midpoint": midpoint,
                "spread_pct": spread_pct,
                "delta": delta,
                "abs_delta": abs_delta,
                "iv": iv,
                "oi": oi,
                "volume": volume,
                "theta": theta,
                "theta_burden": abs(theta) / midpoint if theta is not None else None,
                "breakeven": breakeven,
                "required_move": required_move,
            }
        )
    return candidates


def _rank_candidates(
    raw_candidates: list[dict[str, object]],
    *,
    target_delta: float,
    target_dte: int,
    min_dte: int,
    max_dte: int,
) -> list[OptionCandidate]:
    median_iv = median(float(candidate["iv"]) for candidate in raw_candidates)
    scored: list[OptionCandidate] = []
    for candidate in raw_candidates:
        components = _points(
            candidate,
            target_delta=target_delta,
            target_dte=target_dte,
            min_dte=min_dte,
            max_dte=max_dte,
            median_iv=median_iv,
        )
        scored.append(
            OptionCandidate(
                **candidate,
                components=components,
                score=round(sum(points for _, points, _ in components), 2),
            )
        )
    call = scored[0].right == "C"
    return sorted(
        scored,
        key=lambda item: (
            -item.score,
            abs(item.abs_delta - target_delta),
            item.spread_pct,
            -item.oi,
            -item.volume,
            abs(item.dte - target_dte),
            item.expiry,
            item.strike if call else -item.strike,
            item.symbol,
        ),
    )


def _number(value: float, decimals: int = 2) -> str:
    rendered = f"{value:.{decimals}f}"
    return rendered.rstrip("0").rstrip(".") if decimals else rendered


def _candidate_row(rank: int | str, item: OptionCandidate) -> str:
    theta = "N/A" if item.theta is None else _number(item.theta, 4)
    burden = "N/A" if item.theta_burden is None else f"{item.theta_burden * 100:.2f}%"
    return (
        f"| {rank} | {item.symbol} | {item.expiry.isoformat()} / {item.dte} | "
        f"{_number(item.strike, 3)} | {_number(item.delta, 3)} | "
        f"{item.bid:.2f} / {item.ask:.2f} / {item.midpoint:.2f} | "
        f"{item.spread_pct:.2f}% | {item.iv * 100:.2f}% | {_number(item.oi, 0)} | "
        f"{_number(item.volume, 0)} | {theta} | {burden} | {item.breakeven:.2f} | "
        f"{item.required_move * 100:.2f}% | {item.score:.2f} |"
    )


def _render_report(
    ranked: list[OptionCandidate],
    *,
    underlying: str,
    spot: float,
    timestamp: object,
    direction: str,
    min_dte: int,
    max_dte: int,
    target_delta: float,
    top_n: int,
) -> str:
    best = ranked[0]
    timestamp_text = str(timestamp).strip() if timestamp is not None else ""
    timestamp_text = timestamp_text or "DATA_UNAVAILABLE"
    right_name = "call" if best.right == "C" else "put"
    theta_text = (
        "theta unavailable (neutral 5/10 theta points)"
        if best.theta_burden is None
        else f"theta burden {best.theta_burden * 100:.2f}% of midpoint per day"
    )
    weights = ", ".join(f"{name} {weight:g}" for name, weight in _WEIGHTS)
    lines = [
        "# Cboe delayed equity option contract selection",
        "",
        f"Underlying: {underlying} | Spot: {spot:.2f} | Source timestamp: {timestamp_text}",
        f"Direction: {direction} | Right: {right_name} ({best.right})",
        f"Requested DTE range: {min_dte}-{max_dte} inclusive | Target absolute delta: {target_delta:.2f}",
        f"Total eligible count: {len(ranked)}",
        f"Scoring weights (100 points): {weights}.",
        "This is a transparent heuristic ranking, not personalized advice, fair value, expected return, or profit probability.",
        "",
        "## Recommended contract:",
        "",
        f"**{best.symbol}** — score {best.score:.2f}/100; delta {best.delta:.3f}; DTE {best.dte}; "
        f"spread {best.spread_pct:.2f}%; OI {_number(best.oi, 0)}; volume {_number(best.volume, 0)}; "
        f"IV {best.iv * 100:.2f}%; expiry breakeven {best.breakeven:.2f}; "
        f"required directional move {best.required_move * 100:.2f}%; {theta_text}.",
        "",
        "## Top candidates",
        "",
        "| Rank | Symbol | Expiry / DTE | Strike | Delta | Bid / Ask / Mid | Spread | IV | OI | Volume | Theta | Theta burden | Breakeven | Required move | Total score |",
        "|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_candidate_row(rank, item) for rank, item in enumerate(ranked[:top_n], 1))
    lines.extend(
        [
            "",
            "## Score breakdown",
            "",
            f"Recommended contract: {best.symbol}",
            "",
            "| Component | Points | Maximum |",
            "|---|---:|---:|",
        ]
    )
    lines.extend(
        f"| {name} | {points:.2f} | {maximum:.2f} |"
        for name, points, maximum in best.components
    )
    lines.append(f"| **Total** | **{best.score:.2f}** | **100.00** |")

    by_expiry: dict[date, OptionCandidate] = {}
    for item in ranked:
        by_expiry.setdefault(item.expiry, item)
    lines.extend(
        [
            "",
            "## Best candidate by expiry",
            "",
            "| Expiry | Symbol | DTE | Delta | Spread | OI | Volume | Score |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for expiry in sorted(by_expiry)[:8]:
        item = by_expiry[expiry]
        lines.append(
            f"| {expiry.isoformat()} | {item.symbol} | {item.dte} | {item.delta:.3f} | "
            f"{item.spread_pct:.2f}% | {_number(item.oi, 0)} | {_number(item.volume, 0)} | {item.score:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "Cboe data is delayed. This uses the current snapshot only and is not a historical-selection backtest. Liquidity fields can be stale, and midpoint is not an executable price. IV and Greeks are vendor-calculated. The ranking does not model dividends, American early exercise, earnings or event gaps, slippage, commissions, or tax. Its score is a heuristic—not expected return, fair value, or probability. It supports single-leg long calls and puts only.",
            "The IV-relative component is only a small relative premium-volatility input; it does not assert that lower IV is cheap or fair value.",
        ]
    )
    return "\n".join(lines)


def rank_equity_option_contracts(
    underlying: str,
    direction: str,
    end_date: str,
    min_dte: int = 7,
    max_dte: int = 45,
    target_delta: float = 0.55,
    top_n: int = 5,
    timeout: float = 15.0,
) -> OptionSelectionResult:
    """Rank the current delayed chain and return structured candidates and report."""

    def unavailable(reason: str) -> OptionSelectionResult:
        return OptionSelectionResult((), _unavailable(reason), reason)

    if not isinstance(underlying, str):
        return unavailable("invalid underlying; expected 1-6 letters")
    normalized = underlying.upper()
    if _UNDERLYING_RE.fullmatch(normalized) is None:
        return unavailable("invalid underlying; expected 1-6 letters")
    if not isinstance(direction, str) or direction.lower() not in {"bullish", "bearish"}:
        return unavailable("unsupported direction; use bullish or bearish")
    normalized_direction = direction.lower()
    if not isinstance(end_date, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", end_date) is None:
        return unavailable("invalid end_date; expected YYYY-MM-DD")
    try:
        requested_date = date.fromisoformat(end_date)
    except ValueError:
        return unavailable("invalid end_date; expected YYYY-MM-DD")
    today = _today()
    if requested_date != today:
        return unavailable(
            "current Cboe delayed chain cannot be used for historical/future selection"
        )
    if (
        isinstance(min_dte, bool)
        or isinstance(max_dte, bool)
        or not isinstance(min_dte, int)
        or not isinstance(max_dte, int)
        or not 1 <= min_dte <= max_dte <= 365
    ):
        return unavailable("invalid DTE range; require 1 <= min_dte <= max_dte <= 365")
    if (
        isinstance(target_delta, bool)
        or not isinstance(target_delta, (int, float))
        or not math.isfinite(target_delta)
        or not 0.10 <= target_delta <= 0.90
    ):
        return unavailable("invalid target_delta; require 0.10 through 0.90")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 20:
        return unavailable("invalid top_n; require 1 through 20")

    right = "C" if normalized_direction == "bullish" else "P"
    try:
        response = requests.get(
            _URL.format(underlying=normalized),
            headers={"User-Agent": _UA},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError
        options = data.get("options")
        if not isinstance(options, list):
            raise ValueError
        spot = _finite(data.get("current_price"), positive=True)
        if spot is None:
            raise ValueError
        raw_candidates = _eligible_candidates(
            options,
            underlying=normalized,
            right=right,
            spot=spot,
            today=today,
            min_dte=min_dte,
            max_dte=max_dte,
        )
        if not raw_candidates:
            return unavailable("no eligible contracts in the current delayed chain")
        target_dte = max(min_dte, min(21, max_dte))
        ranked = _rank_candidates(
            raw_candidates,
            target_delta=float(target_delta),
            target_dte=target_dte,
            min_dte=min_dte,
            max_dte=max_dte,
        )
        return OptionSelectionResult(
            tuple(ranked),
            _render_report(
                ranked,
                underlying=normalized,
                spot=spot,
                timestamp=data.get("timestamp", payload.get("timestamp")),
                direction=normalized_direction,
                min_dte=min_dte,
                max_dte=max_dte,
                target_delta=float(target_delta),
                top_n=top_n,
            ),
        )
    except (requests.RequestException, TypeError, ValueError, OverflowError):
        logger.warning("Cboe option selection failed")
        return unavailable("Cboe delayed chain request or payload failed")


def select_equity_option_contract(
    underlying: str,
    direction: str,
    end_date: str,
    min_dte: int = 7,
    max_dte: int = 45,
    target_delta: float = 0.55,
    top_n: int = 5,
    timeout: float = 15.0,
) -> str:
    """Select and explain one current-chain long call or put contract."""
    return rank_equity_option_contracts(
        underlying,
        direction,
        end_date,
        min_dte=min_dte,
        max_dte=max_dte,
        target_delta=target_delta,
        top_n=top_n,
        timeout=timeout,
    ).report

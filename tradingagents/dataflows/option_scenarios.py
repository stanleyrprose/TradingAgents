"""Deterministic, dependency-free option scenario calculations and reporting."""

import math
import os
from datetime import date, timedelta

from . import fred

_FALLBACK_RATE = 0.04
_FALLBACK_PROVENANCE = "fallback 4.00% (FRED FEDFUNDS unavailable)"


def _finite_number(value: object, name: str) -> float:
    """Return *value* as a finite float or raise a stable validation error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _right(value: str) -> str:
    if not isinstance(value, str) or value.upper() not in {"C", "P"}:
        raise ValueError("right must be C or P")
    return value.upper()


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def black_scholes_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    right: str,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European call or put using the Black-Scholes-Merton model."""
    spot = _finite_number(spot, "spot")
    strike = _finite_number(strike, "strike")
    time_years = _finite_number(time_years, "time_years")
    rate = _finite_number(rate, "rate")
    volatility = _finite_number(volatility, "volatility")
    dividend_yield = _finite_number(dividend_yield, "dividend_yield")
    right = _right(right)
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if time_years < 0:
        raise ValueError("time_years must be nonnegative")
    if time_years == 0:
        return max(spot - strike, 0.0) if right == "C" else max(strike - spot, 0.0)
    if volatility <= 0:
        raise ValueError("volatility must be positive when time_years is positive")

    root_t = math.sqrt(time_years)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * time_years
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    discounted_spot = spot * math.exp(-dividend_yield * time_years)
    discounted_strike = strike * math.exp(-rate * time_years)
    if right == "C":
        return discounted_spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    return discounted_strike * _normal_cdf(-d2) - discounted_spot * _normal_cdf(-d1)


def expiry_breakeven(strike: float, premium: float, right: str) -> float:
    """Return the expiry underlying breakeven for a long option."""
    strike = _finite_number(strike, "strike")
    premium = _finite_number(premium, "premium")
    right = _right(right)
    if strike <= 0:
        raise ValueError("strike must be positive")
    if premium < 0:
        raise ValueError("premium must be nonnegative")
    return strike + premium if right == "C" else strike - premium


def resolve_scenario_risk_free_rate(curr_date: str | date) -> tuple[float, str]:
    """Resolve a point-in-time FEDFUNDS rate, safely falling back to 4%."""
    if not os.getenv("FRED_API_KEY"):
        return _FALLBACK_RATE, _FALLBACK_PROVENANCE
    try:
        end = curr_date if isinstance(curr_date, date) else date.fromisoformat(curr_date)
        end_text = end.isoformat()
        pit = min(end_text, fred._fred_today())
        payload = fred._request(
            "series/observations",
            {
                "series_id": "FEDFUNDS",
                "observation_start": (end - timedelta(days=90)).isoformat(),
                "observation_end": end_text,
                "sort_order": "desc",
                "realtime_start": pit,
                "realtime_end": pit,
            },
        )
        for observation in payload.get("observations", []):
            raw = observation.get("value")
            if raw in (None, "", "."):
                continue
            value = float(raw)
            if math.isfinite(value):
                observation_date = observation.get("date", "date unavailable")
                return value / 100.0, f"FRED FEDFUNDS {value:.2f}% ({observation_date})"
    except Exception:
        pass
    return _FALLBACK_RATE, _FALLBACK_PROVENANCE


def required_spot_to_preserve_premium(
    spot: float,
    strike: float,
    premium: float,
    future_time_years: float,
    rate: float,
    volatility: float,
    right: str,
    dividend_yield: float = 0.0,
) -> float | None:
    """Solve for the future spot whose model value equals ``premium``."""
    spot = _finite_number(spot, "spot")
    premium = _finite_number(premium, "premium")
    if spot <= 0:
        raise ValueError("spot must be positive")
    if premium < 0:
        raise ValueError("premium must be nonnegative")
    # Let the pricing function consistently validate all remaining model inputs.
    low, high = spot * 0.05, spot * 5.0

    def difference(candidate: float) -> float:
        return black_scholes_price(
            candidate,
            strike,
            future_time_years,
            rate,
            volatility,
            right,
            dividend_yield,
        ) - premium

    low_value, high_value = difference(low), difference(high)
    if abs(low_value) <= 1e-10:
        return low
    if abs(high_value) <= 1e-10:
        return high
    if low_value * high_value > 0:
        return None
    for _ in range(100):
        midpoint = (low + high) / 2.0
        mid_value = difference(midpoint)
        if abs(mid_value) <= 1e-10 or high - low <= max(1e-10, spot * 1e-12):
            return midpoint
        if low_value * mid_value <= 0:
            high = midpoint
        else:
            low, low_value = midpoint, mid_value
    return (low + high) / 2.0


def _optional_finite(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _signed_money(value: float) -> str:
    return f"{value:+,.2f}"


def build_option_scenario_report(
    symbol: str,
    right: str,
    strike: float,
    expiry: date,
    as_of: date,
    spot: float,
    market_premium: float,
    iv_decimal: float,
    vendor_theo: float | None = None,
    vendor_delta: float | None = None,
    vendor_gamma: float | None = None,
    vendor_theta: float | None = None,
    risk_free_rate: float | None = None,
    risk_free_provenance: str | None = None,
) -> str:
    """Build a deterministic markdown scenario report for one option contract."""
    right = _right(right)
    strike = _finite_number(strike, "strike")
    spot = _finite_number(spot, "spot")
    premium = _finite_number(market_premium, "market_premium")
    volatility = _finite_number(iv_decimal, "iv_decimal")
    if strike <= 0 or spot <= 0:
        raise ValueError("spot and strike must be positive")
    if premium < 0:
        raise ValueError("market_premium must be nonnegative")
    if volatility <= 0:
        raise ValueError("iv_decimal must be positive")
    if not isinstance(expiry, date) or not isinstance(as_of, date):
        raise ValueError("expiry and as_of must be date objects")
    dte = (expiry - as_of).days
    if dte < 0:
        raise ValueError("expiry must not precede as_of")

    if risk_free_rate is None:
        rate, provenance = resolve_scenario_risk_free_rate(as_of)
    else:
        rate = _finite_number(risk_free_rate, "risk_free_rate")
        provenance = risk_free_provenance or "caller supplied"
    q = 0.0
    base_value = black_scholes_price(spot, strike, dte / 365.0, rate, volatility, right, q)
    breakeven = expiry_breakeven(strike, premium, right)
    option_name = "Call" if right == "C" else "Put"

    lines = [
        "# Deterministic option scenario engine",
        "",
        f"**Contract:** {symbol} {expiry.isoformat()} {strike:g} {option_name}",
        "",
        "## Assumptions",
        f"- European Black-Scholes cross-check; q=0.00%; IV={volatility:.2%}.",
        "- 100 shares/contract; constant IV within each scenario.",
        f"- Risk-free-rate proxy: {rate:.2%} ({provenance}).",
        "- FEDFUNDS/fallback is a proxy, not a maturity-matched Treasury/OIS curve.",
        "- Model outputs are a model cross-check, not market fair value or target.",
        "",
        "## Current premium and expiry breakeven",
        f"- Current premium: {_money(premium)} per share ({_money(premium * 100)} per contract).",
        f"- Expiry breakeven: {_money(breakeven)} underlying spot.",
        "",
        "## Base Black-Scholes model cross-check",
        f"- Base BS model value: {_money(base_value)}.",
        f"- Difference vs premium: {_signed_money(base_value - premium)} per share.",
    ]
    theo = _optional_finite(vendor_theo)
    if theo is not None:
        lines.append(f"- Difference vs vendor theo ({_money(theo)}): {_signed_money(base_value - theo)} per share.")
    theta = _optional_finite(vendor_theta)
    if theta is not None:
        lines.append(f"- Vendor theta: {theta:+.4f} per share (vendor convention/time unit applies).")
    lines.extend(["- This is a model cross-check, not market fair value or target.", ""])

    days = sorted({0, *(day for day in (1, 3, 5) if day <= dte), dte})
    lines.extend(
        [
            "## Theta burn (unchanged spot and IV)",
            "| Future day | Remaining DTE | Model value | P/L/share vs premium | P/L/contract |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for day in days:
        value = black_scholes_price(spot, strike, (dte - day) / 365.0, rate, volatility, right, q)
        pnl = value - premium
        lines.append(
            f"| +{day} | {dte - day} | {_money(value)} | {_signed_money(pnl)} | {_signed_money(pnl * 100)} |"
        )

    lines.extend(
        [
            "",
            "## Required underlying to preserve current premium",
            "Unchanged IV/rate/q; each row solves for the spot that retains the current premium.",
            "",
            "| Future day | Remaining DTE | Required spot | Move |",
            "|---:|---:|---:|---:|",
        ]
    )
    preserve_days = [day for day in (1, 3, 5) if day < dte]
    if not preserve_days:
        lines.append("| — | — | unavailable | unavailable |")
    for day in preserve_days:
        required = required_spot_to_preserve_premium(
            spot, strike, premium, (dte - day) / 365.0, rate, volatility, right, q
        )
        if required is None:
            lines.append(f"| +{day} | {dte - day} | unavailable | unavailable |")
        else:
            lines.append(
                f"| +{day} | {dte - day} | {_money(required)} | {(required / spot - 1):+.2%} |"
            )

    horizon = 0 if dte == 0 else min(3, max(dte - 1, 1))
    matrix_t = max(dte - horizon, 0) / 365.0
    iv_levels = [max(volatility - 0.05, 0.01), volatility, volatility + 0.05]
    spot_levels = [spot * 0.95, spot, spot * 1.05]
    lines.extend(
        [
            "",
            f"## Spot × IV matrix (day +{horizon})",
            "Each cell is model value / P/L per contract versus current premium.",
            "",
            f"| Spot / IV | {iv_levels[0]:.2%} | {iv_levels[1]:.2%} | {iv_levels[2]:.2%} |",
            "|---:|---:|---:|---:|",
        ]
    )
    for scenario_spot in spot_levels:
        cells = []
        for scenario_iv in iv_levels:
            value = black_scholes_price(
                scenario_spot, strike, matrix_t, rate, scenario_iv, right, q
            )
            cells.append(f"{_money(value)} / {_signed_money((value - premium) * 100)}")
        lines.append(f"| {_money(scenario_spot)} | " + " | ".join(cells) + " |")

    expiry_spots = [spot * factor for factor in (0.90, 0.95, 1.0, 1.05, 1.10)]
    if all(abs(breakeven - value) > max(0.005, spot * 1e-9) for value in expiry_spots):
        expiry_spots.append(breakeven)
    expiry_spots.sort()
    lines.extend(
        [
            "",
            "## Expiry payoff",
            "| Expiry spot | Intrinsic payoff/share | P/L/contract vs premium |",
            "|---:|---:|---:|",
        ]
    )
    for expiry_spot in expiry_spots:
        intrinsic = black_scholes_price(expiry_spot, strike, 0, rate, volatility, right, q)
        lines.append(
            f"| {_money(expiry_spot)} | {_money(intrinsic)} | {_signed_money((intrinsic - premium) * 100)} |"
        )

    delta = _optional_finite(vendor_delta)
    gamma = _optional_finite(vendor_gamma)
    if delta is not None and gamma is not None:
        lines.extend(
            [
                "",
                "## Delta/gamma local stress",
                "Greeks change as spot and time change; this is a local approximation only.",
                "",
                "| Spot shock | Approx option change/share | Approx P/L/contract |",
                "|---:|---:|---:|",
            ]
        )
        for shock in (-0.03, -0.01, 0.01, 0.03):
            spot_change = spot * shock
            option_change = delta * spot_change + 0.5 * gamma * spot_change**2
            lines.append(
                f"| {shock:+.0%} | {_signed_money(option_change)} | {_signed_money(option_change * 100)} |"
            )

    lines.extend(
        [
            "",
            "## Caveats",
            "- Cboe inputs may be delayed.",
            "- European Black-Scholes differs from American-style equity options.",
            "- q=0 ignores dividends/carry; fixed IV ignores smile/skew dynamics.",
            "- Commissions, slippage, assignment, and early exercise are excluded.",
            "- Scenario estimates are not executable quotes or guaranteed P&L.",
        ]
    )
    return "\n".join(lines)

"""Pure net-Greek exposure reporting and policy guard for sized long options."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from tradingagents.dataflows.option_selector import OptionCandidate
from tradingagents.option_capital_allocator import (
    CONTRACT_MULTIPLIER,
    OptionCapitalAllocation,
)


@dataclass(frozen=True)
class OptionGreekExposure:
    """Greek exposure proxies for one sized long-option position."""

    symbol: str
    contracts: int
    delta_shares: float | None
    delta_notional_proxy: float | None
    gamma_delta_shares_per_dollar: float | None
    vega_dollars_per_vol_point: float | None
    theta_dollars_per_day: float | None
    up_1pct_delta_gamma_pl_proxy: float | None
    down_1pct_delta_gamma_pl_proxy: float | None


@dataclass(frozen=True)
class OptionExposureBreach:
    """One exceeded limit or one limit whose required data is unavailable."""

    limit_name: str
    actual: float | None
    cap: float
    reason: str


@dataclass(frozen=True)
class OptionExposureGuardResult:
    """Structured portfolio exposure report and explicit-policy decision."""

    status: str
    allowed: bool | None
    exposures: tuple[OptionGreekExposure, ...]
    net_delta_shares: float | None
    gross_abs_delta_shares: float | None
    net_delta_notional_proxy: float | None
    gross_abs_delta_notional_proxy: float | None
    net_gamma_delta_shares_per_dollar: float | None
    gross_abs_gamma_delta_shares_per_dollar: float | None
    net_vega_dollars_per_vol_point: float | None
    gross_abs_vega_dollars_per_vol_point: float | None
    net_theta_dollars_per_day: float | None
    gross_abs_theta_dollars_per_day: float | None
    up_1pct_delta_gamma_pl_proxy: float | None
    down_1pct_delta_gamma_pl_proxy: float | None
    underlying_spot: float | None
    breaches: tuple[OptionExposureBreach, ...]
    report: str
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        """Whether the supplied positions could be matched and evaluated."""

        return self.unavailable_reason is None


_LIMITS = (
    ("max_abs_delta_shares", "net delta shares", "net_delta_shares"),
    (
        "max_abs_gamma_delta_shares_per_dollar",
        "net gamma delta-shares/$",
        "net_gamma_delta_shares_per_dollar",
    ),
    (
        "max_abs_vega_dollars_per_vol_point",
        "net vega $/vol-point proxy",
        "net_vega_dollars_per_vol_point",
    ),
    (
        "max_abs_theta_dollars_per_day",
        "net theta $/day proxy",
        "net_theta_dollars_per_day",
    ),
)


def _optional_limit(value: object | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number greater than zero")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return number


def resolve_option_exposure_limits(
    *,
    max_abs_delta_shares: object | None = None,
    max_abs_gamma_delta_shares_per_dollar: object | None = None,
    max_abs_vega_dollars_per_vol_point: object | None = None,
    max_abs_theta_dollars_per_day: object | None = None,
) -> dict[str, float | None]:
    """Validate and normalize optional explicit option-exposure limits."""

    raw_limits = {
        "max_abs_delta_shares": max_abs_delta_shares,
        "max_abs_gamma_delta_shares_per_dollar": max_abs_gamma_delta_shares_per_dollar,
        "max_abs_vega_dollars_per_vol_point": max_abs_vega_dollars_per_vol_point,
        "max_abs_theta_dollars_per_day": max_abs_theta_dollars_per_day,
    }
    return {
        name: _optional_limit(value, name) for name, value in raw_limits.items()
    }


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _aggregate(
    exposures: tuple[OptionGreekExposure, ...], field: str
) -> tuple[float | None, float | None]:
    values = [getattr(exposure, field) for exposure in exposures]
    if any(value is None for value in values):
        return None, None
    numeric = [float(value) for value in values]
    return sum(numeric), sum(abs(value) for value in numeric)


def _fmt(value: float | None, decimals: int = 2) -> str:
    return "N/A" if value is None else f"{value:,.{decimals}f}"


def _render_report(
    result: OptionExposureGuardResult,
    limits: dict[str, float | None],
) -> str:
    lines = [
        "## Long option portfolio Greeks / exposure guard",
        "",
        f"Status: **{result.status}** | Allowed: "
        + ("N/A" if result.allowed is None else str(result.allowed)),
        f"Underlying spot: {_fmt(result.underlying_spot)}",
    ]
    if result.unavailable_reason:
        lines.extend(["", f"Unavailable: {result.unavailable_reason}"])
    lines.extend(
        [
            "",
            "| Symbol | Contracts | Delta shares | Gamma delta-shares/$ | Vega $/vol-point proxy | Theta $/day proxy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {item.symbol} | {item.contracts} | {_fmt(item.delta_shares)} | "
        f"{_fmt(item.gamma_delta_shares_per_dollar)} | "
        f"{_fmt(item.vega_dollars_per_vol_point)} | "
        f"{_fmt(item.theta_dollars_per_day)} |"
        for item in result.exposures
    )
    if not result.exposures:
        empty_value = "N/A" if result.unavailable_reason else "0.00"
        lines.append(
            f"| — | 0 | {empty_value} | {empty_value} | {empty_value} | {empty_value} |"
        )
    lines.extend(
        [
            "",
            f"Net / gross absolute delta shares: {_fmt(result.net_delta_shares)} / "
            f"{_fmt(result.gross_abs_delta_shares)}",
            f"Net / gross absolute delta notional proxy: "
            f"${_fmt(result.net_delta_notional_proxy)} / "
            f"${_fmt(result.gross_abs_delta_notional_proxy)}",
            f"Net / gross absolute gamma delta-shares/$: "
            f"{_fmt(result.net_gamma_delta_shares_per_dollar)} / "
            f"{_fmt(result.gross_abs_gamma_delta_shares_per_dollar)}",
            f"Net / gross absolute vega $/vol-point proxy: "
            f"{_fmt(result.net_vega_dollars_per_vol_point)} / "
            f"{_fmt(result.gross_abs_vega_dollars_per_vol_point)}",
            f"Net / gross absolute theta $/day proxy: "
            f"{_fmt(result.net_theta_dollars_per_day)} / "
            f"{_fmt(result.gross_abs_theta_dollars_per_day)}",
            f"Local +1% / -1% delta-gamma P/L proxy: "
            f"${_fmt(result.up_1pct_delta_gamma_pl_proxy)} / "
            f"${_fmt(result.down_1pct_delta_gamma_pl_proxy)}",
            "",
            "### Explicit limits (absolute net aggregate exposure)",
            "",
        ]
    )
    supplied = False
    result_values = {
        name: getattr(result, field) for name, _, field in _LIMITS
    }
    breach_names = {breach.limit_name for breach in result.breaches}
    for name, label, _ in _LIMITS:
        cap = limits[name]
        if cap is None:
            continue
        supplied = True
        actual = result_values[name]
        outcome = "BLOCK" if name in breach_names else "PASS"
        lines.append(
            f"- {label}: abs(actual) {_fmt(None if actual is None else abs(actual))} "
            f"vs cap {_fmt(cap)} — {outcome}"
        )
    if not supplied:
        if result.unavailable_reason:
            lines.append(
                "Exposure report unavailable; no policy limits were supplied."
            )
        else:
            lines.append("No limits supplied; this is REPORT_ONLY and makes no allow/block decision.")
    for breach in result.breaches:
        if breach.reason:
            lines.append(f"- Breach: {breach.reason}")
    lines.extend(
        [
            "",
            "### Caveats",
            "",
            "Cboe Greeks are vendor-calculated delayed snapshot values, not exact hedges or executable risk. Vega is reported as a Cboe/vendor-convention $/vol-point proxy; vendor units can differ. Theta is treated as a per-share daily proxy for this report, subject to the vendor's convention and time unit.",
            "The ±1% result is a local first/second-order delta-gamma approximation. It ignores higher-order Greeks, IV-surface changes, correlation, portfolio positions outside these options, assignment/exercise, and slippage.",
            "Guard limits use absolute net aggregate exposure, not the sum of absolute exposures. This is a lightweight net-exposure guard, not a full portfolio stress engine. Delta notional is a proxy, not capital at risk. Same-underlying Top3 is not diversification.",
        ]
    )
    return "\n".join(lines)


def _error_result(
    reason: str,
    spot: float | None,
    limits: dict[str, float | None],
) -> OptionExposureGuardResult:
    labels = {name: label for name, label, _ in _LIMITS}
    breaches = tuple(
        OptionExposureBreach(
            limit_name=name,
            actual=None,
            cap=cap,
            reason=f"{labels[name]} not evaluable because the exposure report is unavailable",
        )
        for name, cap in limits.items()
        if cap is not None
    )
    result = OptionExposureGuardResult(
        status="BLOCK",
        allowed=False,
        exposures=(),
        net_delta_shares=None,
        gross_abs_delta_shares=None,
        net_delta_notional_proxy=None,
        gross_abs_delta_notional_proxy=None,
        net_gamma_delta_shares_per_dollar=None,
        gross_abs_gamma_delta_shares_per_dollar=None,
        net_vega_dollars_per_vol_point=None,
        gross_abs_vega_dollars_per_vol_point=None,
        net_theta_dollars_per_day=None,
        gross_abs_theta_dollars_per_day=None,
        up_1pct_delta_gamma_pl_proxy=None,
        down_1pct_delta_gamma_pl_proxy=None,
        underlying_spot=spot,
        breaches=breaches,
        report="",
        unavailable_reason=reason,
    )
    return OptionExposureGuardResult(
        **{**result.__dict__, "report": _render_report(result, limits)}
    )


def evaluate_long_option_exposure(
    candidates: Iterable[OptionCandidate],
    allocation: OptionCapitalAllocation,
    *,
    underlying_spot: float | None,
    max_abs_delta_shares: float | None = None,
    max_abs_gamma_delta_shares_per_dollar: float | None = None,
    max_abs_vega_dollars_per_vol_point: float | None = None,
    max_abs_theta_dollars_per_day: float | None = None,
) -> OptionExposureGuardResult:
    """Report sized long-option exposures and enforce only explicit user limits.

    Limits apply to absolute *net* aggregate exposure. With no limits, the result
    is informational and ``allowed`` is ``None``.
    """

    limits = resolve_option_exposure_limits(
        max_abs_delta_shares=max_abs_delta_shares,
        max_abs_gamma_delta_shares_per_dollar=max_abs_gamma_delta_shares_per_dollar,
        max_abs_vega_dollars_per_vol_point=max_abs_vega_dollars_per_vol_point,
        max_abs_theta_dollars_per_day=max_abs_theta_dollars_per_day,
    )

    try:
        supplied_candidates = tuple(candidates)
    except TypeError:
        return _error_result("candidates must be an iterable", _finite(underlying_spot), limits)
    positions = getattr(allocation, "positions", None)
    if not isinstance(positions, tuple):
        try:
            positions = tuple(positions)
        except TypeError:
            return _error_result(
                "allocation positions must be iterable", _finite(underlying_spot), limits
            )

    candidate_map: dict[str, OptionCandidate] = {}
    for candidate in supplied_candidates:
        symbol = getattr(candidate, "symbol", None)
        if not isinstance(symbol, str) or not symbol:
            return _error_result(
                "candidate has an empty or invalid symbol", _finite(underlying_spot), limits
            )
        if symbol in candidate_map:
            return _error_result(
                f"duplicate candidate symbol: {symbol}", _finite(underlying_spot), limits
            )
        candidate_map[symbol] = candidate

    seen_positions: set[str] = set()
    positive: list[tuple[object, OptionCandidate]] = []
    for position in positions:
        symbol = getattr(position, "symbol", None)
        contracts = getattr(position, "contracts", None)
        if not isinstance(symbol, str) or not symbol:
            return _error_result(
                "allocation position has an empty or invalid symbol",
                _finite(underlying_spot),
                limits,
            )
        if symbol in seen_positions:
            return _error_result(
                f"duplicate allocation symbol: {symbol}",
                _finite(underlying_spot),
                limits,
            )
        seen_positions.add(symbol)
        if isinstance(contracts, bool) or not isinstance(contracts, int) or contracts < 0:
            return _error_result(
                f"allocation contracts for {symbol} must be a nonnegative whole number",
                _finite(underlying_spot),
                limits,
            )
        if contracts == 0:
            continue
        candidate = candidate_map.get(symbol)
        if candidate is None:
            return _error_result(
                f"positive-contract allocation symbol missing from candidates: {symbol}",
                _finite(underlying_spot),
                limits,
            )
        positive.append((position, candidate))

    spot = _finite(underlying_spot)
    if positive and (spot is None or spot <= 0):
        return _error_result(
            "underlying_spot must be finite and greater than zero for positive contracts",
            spot,
            limits,
        )

    exposures_list: list[OptionGreekExposure] = []
    for position, candidate in positive:
        contracts = position.contracts
        scale = contracts * CONTRACT_MULTIPLIER
        delta = _finite(getattr(candidate, "delta", None))
        gamma = _finite(getattr(candidate, "gamma", None))
        vega = _finite(getattr(candidate, "vega", None))
        theta = _finite(getattr(candidate, "theta", None))
        delta_shares = None if delta is None else scale * delta
        delta_notional = None if delta_shares is None else delta_shares * spot
        gamma_exposure = None if gamma is None else scale * gamma
        vega_exposure = None if vega is None else scale * vega
        theta_exposure = None if theta is None else scale * theta
        if delta is None or gamma is None:
            up_proxy = down_proxy = None
        else:
            move = spot * 0.01
            up_proxy = scale * (delta * move + 0.5 * gamma * move**2)
            down_proxy = scale * (-delta * move + 0.5 * gamma * move**2)
        exposures_list.append(
            OptionGreekExposure(
                symbol=position.symbol,
                contracts=contracts,
                delta_shares=delta_shares,
                delta_notional_proxy=delta_notional,
                gamma_delta_shares_per_dollar=gamma_exposure,
                vega_dollars_per_vol_point=vega_exposure,
                theta_dollars_per_day=theta_exposure,
                up_1pct_delta_gamma_pl_proxy=up_proxy,
                down_1pct_delta_gamma_pl_proxy=down_proxy,
            )
        )
    exposures = tuple(exposures_list)

    if exposures:
        net_delta, gross_delta = _aggregate(exposures, "delta_shares")
        net_notional, gross_notional = _aggregate(exposures, "delta_notional_proxy")
        net_gamma, gross_gamma = _aggregate(exposures, "gamma_delta_shares_per_dollar")
        net_vega, gross_vega = _aggregate(exposures, "vega_dollars_per_vol_point")
        net_theta, gross_theta = _aggregate(exposures, "theta_dollars_per_day")
        up_proxy, _ = _aggregate(exposures, "up_1pct_delta_gamma_pl_proxy")
        down_proxy, _ = _aggregate(exposures, "down_1pct_delta_gamma_pl_proxy")
    else:
        net_delta = gross_delta = net_notional = gross_notional = 0.0
        net_gamma = gross_gamma = net_vega = gross_vega = 0.0
        net_theta = gross_theta = up_proxy = down_proxy = 0.0

    actuals = {
        "max_abs_delta_shares": net_delta,
        "max_abs_gamma_delta_shares_per_dollar": net_gamma,
        "max_abs_vega_dollars_per_vol_point": net_vega,
        "max_abs_theta_dollars_per_day": net_theta,
    }
    labels = {name: label for name, label, _ in _LIMITS}
    breaches: list[OptionExposureBreach] = []
    for name, cap in limits.items():
        if cap is None:
            continue
        actual = actuals[name]
        if actual is None:
            breaches.append(
                OptionExposureBreach(
                    limit_name=name,
                    actual=None,
                    cap=cap,
                    reason=f"{labels[name]} unavailable for one or more positive-contract positions",
                )
            )
        elif abs(actual) > cap:
            breaches.append(
                OptionExposureBreach(
                    limit_name=name,
                    actual=actual,
                    cap=cap,
                    reason=f"absolute {labels[name]} {abs(actual):,.2f} exceeds cap {cap:,.2f}",
                )
            )

    has_limits = any(cap is not None for cap in limits.values())
    status = "BLOCK" if breaches else "PASS" if has_limits else "REPORT_ONLY"
    allowed = False if breaches else True if has_limits else None
    result = OptionExposureGuardResult(
        status=status,
        allowed=allowed,
        exposures=exposures,
        net_delta_shares=net_delta,
        gross_abs_delta_shares=gross_delta,
        net_delta_notional_proxy=net_notional,
        gross_abs_delta_notional_proxy=gross_notional,
        net_gamma_delta_shares_per_dollar=net_gamma,
        gross_abs_gamma_delta_shares_per_dollar=gross_gamma,
        net_vega_dollars_per_vol_point=net_vega,
        gross_abs_vega_dollars_per_vol_point=gross_vega,
        net_theta_dollars_per_day=net_theta,
        gross_abs_theta_dollars_per_day=gross_theta,
        up_1pct_delta_gamma_pl_proxy=up_proxy,
        down_1pct_delta_gamma_pl_proxy=down_proxy,
        underlying_spot=spot,
        breaches=tuple(breaches),
        report="",
    )
    return OptionExposureGuardResult(
        **{**result.__dict__, "report": _render_report(result, limits)}
    )

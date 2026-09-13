"""Point-in-time-safe positioning from official CFTC annual archives."""

from __future__ import annotations

import csv
import io
import logging
from datetime import date, timedelta
from functools import lru_cache
from urllib.request import Request, urlopen
from zipfile import ZipFile

from tradingagents.instrument_router import classify_instrument

logger = logging.getLogger(__name__)

_UA = "tradingagents/0.4 (+https://github.com/TauricResearch/TradingAgents)"
_ARCHIVE_URLS = {
    "tff": "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip",
    "disaggregated": (
        "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
    ),
}

_FOREX_CODES = {
    "EUR": "099741",
    "JPY": "097741",
    "GBP": "096742",
    "CHF": "092741",
    "CAD": "090741",
    "AUD": "232741",
    "NZD": "112741",
}

_FOREX_FUTURES = {
    "6E=F": "EUR",
    "6J=F": "JPY",
    "6B=F": "GBP",
    "6A=F": "AUD",
    "6C=F": "CAD",
    "6S=F": "CHF",
}

_COMMODITY_CODES = {
    "GC=F": "088691",
    "SI=F": "084691",
    "CL=F": "067651",
    "BZ=F": "06765T",
    "NG=F": "03565B",
    "HG=F": "085692",
    "ZC=F": "002602",
    "ZW=F": "001602",
    "ZS=F": "005602",
    "KC=F": "083731",
    "SB=F": "080732",
    "CT=F": "033661",
    "CC=F": "073732",
}

_DATE_FIELD = "Report_Date_as_YYYY-MM-DD"
_CODE_FIELD = "CFTC_Contract_Market_Code"
_TFF_FIELDS = (
    _DATE_FIELD,
    _CODE_FIELD,
    "Open_Interest_All",
    "Asset_Mgr_Positions_Long_All",
    "Asset_Mgr_Positions_Short_All",
    "Lev_Money_Positions_Long_All",
    "Lev_Money_Positions_Short_All",
    "Change_in_Asset_Mgr_Long_All",
    "Change_in_Asset_Mgr_Short_All",
    "Change_in_Lev_Money_Long_All",
    "Change_in_Lev_Money_Short_All",
)
_DISAGGREGATED_FIELDS = (
    _DATE_FIELD,
    _CODE_FIELD,
    "Open_Interest_All",
    "M_Money_Positions_Long_All",
    "M_Money_Positions_Short_All",
    "Change_in_M_Money_Long_All",
    "Change_in_M_Money_Short_All",
    "Prod_Merc_Positions_Long_All",
    "Prod_Merc_Positions_Short_All",
)


def _unavailable(reason: str) -> str:
    return f"<cftc positioning unavailable: {reason}>"


@lru_cache(maxsize=8)
def _load_annual_archive(
    report_type: str, year: int, timeout: float = 12.0
) -> tuple[dict[str, str], ...]:
    """Download and parse one official futures-only annual archive.

    Successful results are cached in-process so multiple currency legs and
    repeated agent calls do not redownload the multi-megabyte ZIP.
    """
    try:
        url = _ARCHIVE_URLS[report_type].format(year=year)
    except KeyError as exc:
        raise ValueError(f"unknown report type {report_type!r}") from exc

    request = Request(url, headers={"User-Agent": _UA, "Accept": "application/zip"})
    with urlopen(request, timeout=timeout) as response:
        archive_bytes = response.read()

    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".txt", ".csv")) and not name.endswith("/")
        ]
        if not members:
            raise ValueError("annual ZIP contains no text report")
        payload = archive.read(members[0])

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = payload.decode("latin-1")
    reader = csv.DictReader(io.StringIO(text))
    required = _TFF_FIELDS if report_type == "tff" else _DISAGGREGATED_FIELDS
    headers = set(reader.fieldnames or ())
    missing = [field for field in required if field not in headers]
    if missing:
        raise ValueError(f"annual CSV missing field {missing[0]}")
    return tuple(dict(row) for row in reader)


def _number(row: dict[str, str], field: str) -> float:
    value = row.get(field)
    if value is None or not value.strip():
        raise ValueError(f"missing {field}")
    try:
        return float(value.replace(",", "").strip())
    except ValueError as exc:
        raise ValueError(f"invalid {field}") from exc


def _latest_row(
    rows: tuple[dict[str, str], ...], contract_code: str, cutoff: date
) -> tuple[date, dict[str, str]] | None:
    eligible: list[tuple[date, dict[str, str]]] = []
    for row in rows:
        if (row.get(_CODE_FIELD) or "").strip() != contract_code:
            continue
        raw_date = (row.get(_DATE_FIELD) or "").strip()
        try:
            report_date = date.fromisoformat(raw_date[:10])
        except ValueError:
            continue
        if report_date <= cutoff:
            eligible.append((report_date, row))
    return max(eligible, key=lambda item: item[0]) if eligible else None


def _metrics(
    row: dict[str, str], long_field: str, short_field: str
) -> tuple[float, float]:
    open_interest = _number(row, "Open_Interest_All")
    if open_interest <= 0:
        raise ValueError("Open_Interest_All must be positive")
    net = _number(row, long_field) - _number(row, short_field)
    return net, net / open_interest * 100.0


def _net_change(row: dict[str, str], long_field: str, short_field: str) -> float:
    return _number(row, long_field) - _number(row, short_field)


def _contracts(value: float, *, signed: bool = False) -> str:
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value:,.0f}"


def _release_lag(cutoff: date) -> str:
    return (
        f"Release-lag rule: eligible report dates are on or before {cutoff.isoformat()} "
        "(end date minus 3 calendar days), so Tuesday positions first become "
        "eligible at Friday end-of-day."
    )


def _forex_report(
    canonical_symbol: str, end: date, timeout: float
) -> str:
    futures_currency = _FOREX_FUTURES.get(canonical_symbol)
    if futures_currency is not None:
        base, quote = futures_currency, "USD"
    else:
        pair = canonical_symbol.removesuffix("=X")
        if len(pair) != 6:
            return _unavailable(f"unsupported forex symbol {canonical_symbol}")
        base, quote = pair[:3], pair[3:]
    unsupported = [
        currency
        for currency in (base, quote)
        if currency != "USD" and currency not in _FOREX_CODES
    ]
    if unsupported:
        currencies = ", ".join(dict.fromkeys(unsupported))
        return _unavailable(
            f"unsupported CFTC forex mapping for {currencies}; no contract code guessed"
        )

    legs = [currency for currency in (base, quote) if currency != "USD"]
    if not legs:
        return _unavailable("USD has no standalone currency-futures leg")

    cutoff = end - timedelta(days=3)
    rows = _load_annual_archive("tff", end.year, timeout)
    results: dict[str, float] = {}
    lines = [
        f"CFTC TFF Futures Only positioning — {canonical_symbol}",
        _release_lag(cutoff),
    ]
    for currency in dict.fromkeys(legs):
        code = _FOREX_CODES[currency]
        selected = _latest_row(rows, code, cutoff)
        if selected is None:
            return _unavailable(
                f"no eligible {currency} TFF row in the {end.year} archive"
            )
        report_date, row = selected
        leveraged_net, leveraged_pct = _metrics(
            row, "Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All"
        )
        asset_net, asset_pct = _metrics(
            row,
            "Asset_Mgr_Positions_Long_All",
            "Asset_Mgr_Positions_Short_All",
        )
        weekly_change = _net_change(
            row, "Change_in_Lev_Money_Long_All", "Change_in_Lev_Money_Short_All"
        )
        # Validate the remaining requested change fields even though only the
        # leveraged-money weekly net change is part of the rendered metrics.
        _net_change(
            row,
            "Change_in_Asset_Mgr_Long_All",
            "Change_in_Asset_Mgr_Short_All",
        )
        results[currency] = leveraged_pct
        lines.extend(
            [
                f"### {currency} currency futures vs USD ({code})",
                f"Report date (positions as of): {report_date.isoformat()}",
                (
                    "Leveraged money: net "
                    f"{_contracts(leveraged_net)} contracts; {leveraged_pct:+.2f}% "
                    f"of OI; weekly net change {_contracts(weekly_change, signed=True)} "
                    "contracts"
                ),
                (
                    f"Asset managers: net {_contracts(asset_net)} contracts; "
                    f"{asset_pct:+.2f}% of OI"
                ),
            ]
        )

    if futures_currency is not None:
        proxy = results[futures_currency]
    elif quote == "USD":
        proxy = results[base]
    elif base == "USD":
        proxy = -results[quote]
    else:
        proxy = results[base] - results[quote]
    proxy_label = (
        "Contract leveraged-money positioning proxy"
        if futures_currency is not None
        else "Pair leveraged-money positioning differential proxy"
    )
    lines.append(f"{proxy_label}: {proxy:+.2f} percentage points of OI")
    if futures_currency is not None:
        lines.append(
            "This is the currency futures leg versus USD, not OTC spot positioning."
        )
    else:
        lines.append(
            "This is a differential proxy from currency futures versus USD, "
            "not direct OTC spot positioning or a direct cross-pair COT measure."
        )
    return "\n".join(lines)


def _commodity_report(symbol: str, end: date, timeout: float) -> str:
    code = _COMMODITY_CODES.get(symbol)
    if code is None:
        return _unavailable(
            f"unsupported CFTC commodity mapping for {symbol}; no contract code guessed"
        )

    cutoff = end - timedelta(days=3)
    rows = _load_annual_archive("disaggregated", end.year, timeout)
    selected = _latest_row(rows, code, cutoff)
    if selected is None:
        return _unavailable(
            f"no eligible {symbol} disaggregated row in the {end.year} archive"
        )
    report_date, row = selected
    managed_net, managed_pct = _metrics(
        row, "M_Money_Positions_Long_All", "M_Money_Positions_Short_All"
    )
    weekly_change = _net_change(
        row, "Change_in_M_Money_Long_All", "Change_in_M_Money_Short_All"
    )
    producer_net, producer_pct = _metrics(
        row, "Prod_Merc_Positions_Long_All", "Prod_Merc_Positions_Short_All"
    )
    return "\n".join(
        [
            f"CFTC Disaggregated Futures Only positioning — {symbol} ({code})",
            _release_lag(cutoff),
            f"Report date (positions as of): {report_date.isoformat()}",
            (
                f"Managed money: net {_contracts(managed_net)} contracts; "
                f"{managed_pct:+.2f}% of OI; weekly net change "
                f"{_contracts(weekly_change, signed=True)} contracts"
            ),
            (
                f"Producer/merchant: net {_contracts(producer_net)} contracts; "
                f"{producer_pct:+.2f}% of OI"
            ),
            (
                "Managed-money positioning is a proxy for speculative positioning, "
                "not a standalone forecast."
            ),
        ]
    )


def fetch_cftc_positioning(
    ticker: str, end_date: str, timeout: float = 12.0
) -> str:
    """Return official CFTC positioning without allowing failures to abort a graph."""
    try:
        end = date.fromisoformat(end_date)
    except (TypeError, ValueError):
        return _unavailable("invalid end_date; expected YYYY-MM-DD")

    try:
        profile = classify_instrument(ticker)
        if profile.asset_class == "forex":
            return _forex_report(profile.canonical_symbol, end, timeout)
        if profile.asset_class == "commodity":
            return _commodity_report(profile.analysis_symbol, end, timeout)
        return _unavailable(
            f"unsupported asset class for {profile.canonical_symbol}"
        )
    except Exception as exc:
        logger.warning("CFTC positioning fetch failed: %s", type(exc).__name__)
        detail = str(exc).strip()
        reason = type(exc).__name__ if not detail else f"{type(exc).__name__}: {detail}"
        return _unavailable(reason)

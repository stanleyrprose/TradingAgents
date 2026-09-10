"""Point-in-time-safe weekly energy inventory from official EIA DNav pages."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from functools import lru_cache
from html.parser import HTMLParser
from urllib.request import Request, urlopen

from tradingagents.instrument_router import classify_instrument

logger = logging.getLogger(__name__)

_UA = "tradingagents/0.4 (+https://github.com/TauricResearch/TradingAgents)"
_CRUDE_URL = (
    "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?"
    "f=W&n=PET&s=WCESTUS1"
)
_GAS_URL = "https://www.eia.gov/dnav/ng/hist/nw2_epg0_swo_r48_bcfw.htm"


class _DNavParser(HTMLParser):
    """Collect only the cells used by EIA's DNav historical tables."""

    _CLASSES = frozenset({"B6", "B5", "B3"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[tuple[str, str]] = []
        self._tag_stack: list[str] = []
        self._active_class: str | None = None
        self._active_depth: int | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._tag_stack.append(tag)
        if self._active_class is not None or tag not in {"td", "th"}:
            return
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        matched = self._CLASSES.intersection(classes)
        if matched:
            self._active_class = next(iter(matched))
            self._active_depth = len(self._tag_stack)
            self._parts = []

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._active_class is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if (
            self._active_class is not None
            and self._active_depth == len(self._tag_stack)
            and self._tag_stack
            and self._tag_stack[-1] == tag
        ):
            self.cells.append((self._active_class, "".join(self._parts)))
            self._active_class = None
            self._active_depth = None
            self._parts = []
        if self._tag_stack:
            self._tag_stack.pop()


def _clean(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def _parse_value(raw: str) -> float | None:
    value = _clean(raw)
    if not value or value.upper() in {"NA", "N/A", "--"}:
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _parse_eia_html(payload: str) -> tuple[tuple[date, float], ...]:
    """Parse dated observations from an EIA DNav historical HTML page."""
    parser = _DNavParser()
    parser.feed(payload)
    parser.close()

    current_month: tuple[int, int] | None = None
    pending_dates: list[tuple[tuple[int, int] | None, str]] = []
    observations: dict[date, float] = {}

    for cell_class, raw_text in parser.cells:
        text = _clean(raw_text)
        if cell_class == "B6":
            try:
                parsed_month = datetime.strptime(text, "%Y-%b")
            except ValueError:
                current_month = None
            else:
                current_month = (parsed_month.year, parsed_month.month)
        elif cell_class == "B5":
            # Keep blank slots so a following blank B3 cell remains aligned.
            pending_dates.append((current_month, text))
        elif cell_class == "B3" and pending_dates:
            month, raw_week = pending_dates.pop(0)
            value = _parse_value(text)
            if month is None or not raw_week or value is None:
                continue
            try:
                week_month, week_day = (int(part) for part in raw_week.split("/"))
                observation_date = date(month[0], week_month, week_day)
            except (TypeError, ValueError):
                continue
            if week_month != month[1]:
                continue
            observations[observation_date] = value

    return tuple(sorted(observations.items()))


@lru_cache(maxsize=4)
def _load_source_page(url: str, timeout: float) -> tuple[tuple[date, float], ...]:
    """Download and parse one official page, caching successful parsed results."""
    request = Request(
        url,
        headers={"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
    if isinstance(payload, bytes):
        try:
            html = payload.decode("utf-8")
        except UnicodeDecodeError:
            html = payload.decode("latin-1")
    else:
        html = str(payload)
    observations = _parse_eia_html(html)
    if not observations:
        raise ValueError("EIA page contained no usable weekly observations")
    return observations


def _unavailable(reason: str) -> str:
    return f"<eia inventory unavailable: {reason}>"


def _number(value: float, *, signed: bool = False) -> str:
    prefix = "+" if signed and value > 0 else ""
    if value.is_integer():
        return f"{prefix}{value:,.0f}"
    return f"{prefix}{value:,.2f}".rstrip("0").rstrip(".")


def _report(
    symbol: str,
    end: date,
    observations: tuple[tuple[date, float], ...],
) -> str:
    cutoff = end - timedelta(days=6)
    eligible = [(day, value) for day, value in observations if day <= cutoff]
    if not eligible:
        return _unavailable(
            f"no observation eligible by {cutoff.isoformat()} under the 6-day lag rule"
        )

    units = "Thousand Barrels" if symbol in {"CL=F", "BZ=F"} else "Billion Cubic Feet"

    if symbol == "CL=F":
        lines = [
            "EIA Weekly U.S. Ending Stocks excluding SPR of Crude Oil — CL=F",
            "This is a U.S. commercial crude stock measure (Thousand Barrels).",
        ]
    elif symbol == "BZ=F":
        lines = [
            "EIA U.S. crude inventory proxy for Brent/global crude — BZ=F",
            (
                "PROXY WARNING: WCESTUS1 is a U.S. commercial crude stock measure, "
                "not direct Brent/global inventory."
            ),
        ]
    else:
        lines = [
            "EIA Lower 48 States Natural Gas Working Underground Storage — NG=F",
            "Units: Billion Cubic Feet.",
        ]

    lines.append(
        "Publication-lag rule: only observation dates on or before "
        f"{cutoff.isoformat()} (analysis end date minus 6 calendar days) are eligible. "
        "This fixed lag is intentionally conservative for weekly EIA releases and "
        "does not infer historical availability from the page's Release Date field."
    )

    latest_date, latest_value = eligible[-1]
    lines.append(
        f"Latest eligible inventory: {_number(latest_value)} {units} "
        f"(observation date {latest_date.isoformat()})"
    )

    if len(eligible) >= 2:
        prior_value = eligible[-2][1]
        weekly_change = latest_value - prior_value
        if prior_value == 0:
            percent = "percent change unavailable because the prior value is zero"
        else:
            percent = f"{weekly_change / prior_value * 100:+.2f}%"
        lines.append(
            f"1-week change: {_number(weekly_change, signed=True)} {units} ({percent})"
        )
    else:
        lines.append("1-week change: unavailable (requires a prior eligible observation)")

    if len(eligible) >= 5:
        four_observation_change = latest_value - eligible[-5][1]
        lines.append(
            "4-observation change: "
            f"{_number(four_observation_change, signed=True)} {units}"
        )
    else:
        lines.append(
            "4-observation change: unavailable (requires five eligible observations)"
        )
    return "\n".join(lines)


def fetch_eia_inventory(
    ticker: str, end_date: str, timeout: float = 12.0
) -> str:
    """Return official EIA weekly energy inventory without aborting the graph."""
    try:
        end = date.fromisoformat(end_date)
    except (TypeError, ValueError):
        return _unavailable("invalid end_date; expected YYYY-MM-DD")

    try:
        profile = classify_instrument(ticker)
        symbol = profile.analysis_symbol
        if profile.asset_class != "commodity" or symbol not in {
            "CL=F",
            "BZ=F",
            "NG=F",
        }:
            return _unavailable(
                f"not available/applicable for {profile.canonical_symbol}; "
                "official weekly EIA integration covers crude oil and natural gas only"
            )
        url = _GAS_URL if symbol == "NG=F" else _CRUDE_URL
        return _report(symbol, end, _load_source_page(url, timeout))
    except Exception as exc:
        logger.warning("EIA inventory fetch failed: %s", type(exc).__name__)
        detail = str(exc).strip()
        reason = type(exc).__name__ if not detail else f"{type(exc).__name__}: {detail}"
        return _unavailable(reason)

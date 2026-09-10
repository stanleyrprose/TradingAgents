import pytest

from tradingagents.instrument_router import classify_instrument


@pytest.mark.parametrize(
    ("symbol", "canonical", "primary", "asset_class", "kind", "can_run"),
    [
        ("AAPL", "AAPL", "stock", "equity", "stock", True),
        ("BTCUSD", "BTC-USD", "crypto", "crypto", "spot", True),
        ("EURUSD", "EURUSD=X", "forex", "forex", "spot", True),
        ("EURUSD=X", "EURUSD=X", "forex", "forex", "spot", True),
        ("GC=F", "GC=F", "commodity", "commodity", "future", True),
        ("XAUUSD", "GC=F", "commodity", "commodity", "derivative", True),
        ("ES=F", "ES=F", "future", "index", "future", True),
        ("ZN=F", "ZN=F", "future", "fixed_income", "future", True),
        ("6E=F", "6E=F", "future", "forex", "future", True),
        ("SPX500", "^GSPC", "derivative", "index", "cfd", True),
        ("^GSPC", "^GSPC", "index", "index", "index", True),
        ("^TNX", "^TNX", "rate", "rate", "yield", True),
        ("DGS10", "DGS10", "rate", "rate", "yield", False),
        ("^VIX", "^VIX", "volatility", "volatility", "index", True),
        ("US0378331005", "US0378331005", "fixed_income", "fixed_income", "bond", False),
        ("SPY", "SPY", "fund", "equity", "etf", True),
        ("GLD", "GLD", "fund", "commodity", "etf", True),
        ("TLT", "TLT", "fund", "fixed_income", "etf", True),
        ("IBIT", "IBIT", "fund", "crypto", "etf", True),
    ],
)
def test_classification_matrix(symbol, canonical, primary, asset_class, kind, can_run):
    profile = classify_instrument(symbol)
    assert profile.canonical_symbol == canonical
    assert (profile.primary_type, profile.asset_class, profile.instrument_kind) == (
        primary,
        asset_class,
        kind,
    )
    assert profile.can_run is can_run
    assert profile.pipeline_asset_type == ("crypto" if primary == "crypto" else "stock")
    if primary not in {"stock", "crypto"}:
        assert profile.capability == "PARTIAL"


def test_occ_option_extracts_underlying_analysis_symbol():
    profile = classify_instrument("AAPL260918C00200000")
    assert profile.canonical_symbol == "AAPL260918C00200000"
    assert profile.analysis_symbol == "AAPL"
    assert (profile.primary_type, profile.asset_class, profile.instrument_kind) == (
        "option",
        "equity",
        "option",
    )
    assert "proxy" in profile.notes


def test_legacy_bond_override_maps_to_fixed_income():
    profile = classify_instrument("AAPL", "bond")
    assert (profile.primary_type, profile.asset_class, profile.instrument_kind) == (
        "fixed_income",
        "fixed_income",
        "bond",
    )


def test_fund_override_supports_unknown_network_free_funds():
    profile = classify_instrument("MYSTERY", "fund")
    assert (profile.primary_type, profile.asset_class, profile.instrument_kind) == (
        "fund",
        "other",
        "fund",
    )
    assert profile.analysts == ("market", "news")
    assert "fundamentals" not in profile.analysts


def test_future_override_on_ordinary_stock_has_unknown_asset_class():
    profile = classify_instrument("AAPL", "future")
    assert (profile.asset_class, profile.instrument_kind) == ("other", "future")


def test_option_override_keeps_crypto_exposure_and_uses_crypto_proxy():
    profile = classify_instrument("ETH-USD", "option")
    assert profile.analysis_symbol == "ETH-USD"
    assert (profile.primary_type, profile.asset_class, profile.instrument_kind) == (
        "option",
        "crypto",
        "option",
    )
    assert profile.pipeline_asset_type == "crypto"
    assert "crypto underlying" in profile.notes


def test_invalid_override_is_rejected():
    with pytest.raises(ValueError, match="override_type"):
        classify_instrument("AAPL", "swaption")


def test_full_capability_and_analyst_routing():
    stock = classify_instrument("AAPL")
    crypto = classify_instrument("BTCUSD")
    assert stock.capability == "FULL"
    assert stock.analysts == ("market", "social", "news", "fundamentals")
    assert crypto.capability == "FULL"
    assert crypto.analysts == ("market", "social", "news")


@pytest.mark.parametrize("symbol", ["SPY", "IBIT"])
def test_equity_and_crypto_funds_include_social_but_not_fundamentals(symbol):
    profile = classify_instrument(symbol)
    assert profile.analysts == ("market", "social", "news")
    assert profile.capability == "PARTIAL"


@pytest.mark.parametrize(
    ("symbol", "phrase"),
    [
        ("EURUSD", "carry"),
        ("SPY", "holdings"),
        ("^GSPC", "breadth"),
        ("^TNX", "term-premium"),
        ("^VIX", "skew"),
        ("TLT", "duration"),
        ("AAPL260918C00200000", "Greeks"),
        ("ES=F", "basis"),
        ("GC=F", "physical supply-demand"),
        ("SPX500", "contract semantics"),
    ],
)
def test_partial_notes_name_missing_domain_analytics(symbol, phrase):
    assert phrase in classify_instrument(symbol).notes


@pytest.mark.parametrize(
    ("symbol", "required_notes"),
    [
        ("EURUSD", ("forward points", "realized carry")),
        ("GC=F", ("inventory", "physical supply-demand", "cash basis")),
    ],
)
def test_cross_asset_profiles_remain_partial_with_explicit_data_gaps(
    symbol, required_notes
):
    profile = classify_instrument(symbol)

    assert profile.capability == "PARTIAL"
    assert all(note in profile.notes for note in required_notes)
    if symbol == "EURUSD":
        assert "robust positioning" not in profile.notes


def test_supported_metals_curve_notes_reflect_remaining_gaps():
    profile = classify_instrument("GC=F")

    assert profile.capability == "PARTIAL"
    assert "inventory" in profile.notes
    assert "physical supply-demand" in profile.notes
    assert "term-structure" not in profile.notes
    assert "futures curve" not in profile.notes


@pytest.mark.parametrize("symbol", ["CL=F", "NG=F"])
def test_supported_energy_curve_notes_reflect_eia_coverage(symbol):
    notes = classify_instrument(symbol).notes

    assert "inventory" not in notes
    assert "term-structure" not in notes
    assert "futures curve" not in notes
    assert "physical supply-demand" in notes
    assert "cash basis" in notes


def test_unsupported_brent_curve_notes_retain_inventory_and_curve_gaps():
    notes = classify_instrument("BZ=F").notes

    assert "inventory" in notes
    assert "term-structure" in notes


def test_gold_alias_inherits_curve_coverage_and_retains_derivative_semantics():
    notes = classify_instrument("XAUUSD").notes

    assert "inventory" in notes
    assert "term-structure" not in notes
    assert "futures curve" not in notes
    assert "derivative contract semantics" in notes

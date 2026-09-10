import pytest

from tradingagents.instrument_router import classify_instrument


@pytest.mark.parametrize(
    ("symbol", "primary", "asset_class", "kind", "can_run"),
    [
        ("AAPL", "stock", "stock", "cash", True),
        ("BTCUSD", "crypto", "crypto", "cash", True),
        ("ETH-USDT", "crypto", "crypto", "cash", True),
        ("AAPL240119C00100000", "option", "stock", "option", True),
        ("GC=F", "commodity", "commodity", "future", True),
        ("XAUUSD", "commodity", "commodity", "derivative", True),
        ("GOLD", "commodity", "commodity", "derivative", True),
        ("EURUSD", "derivative", "other", "derivative", True),
        ("ES=F", "future", "other", "future", True),
        ("SPX500", "derivative", "other", "derivative", True),
        ("^TNX", "bond", "bond", "bond", True),
        ("US0378331005", "bond", "bond", "bond", False),
    ],
)
def test_classification_matrix(symbol, primary, asset_class, kind, can_run):
    profile = classify_instrument(symbol)
    assert (profile.primary_type, profile.asset_class, profile.instrument_kind) == (
        primary,
        asset_class,
        kind,
    )
    assert profile.can_run is can_run
    assert profile.pipeline_asset_type == ("crypto" if primary == "crypto" else "stock")
    if primary not in {"stock", "crypto"}:
        assert profile.capability == "PARTIAL"


def test_override_produces_requested_coherent_metadata():
    profile = classify_instrument("AAPL", "future")
    assert profile.primary_type == "future"
    assert (profile.asset_class, profile.instrument_kind) == ("other", "future")
    assert profile.pipeline_asset_type == "stock"


def test_invalid_override_is_rejected():
    with pytest.raises(ValueError, match="override_type"):
        classify_instrument("AAPL", "forex")


@pytest.mark.parametrize(
    ("symbol", "phrase"),
    [
        ("AAPL240119C00100000", "option Greeks/chain"),
        ("ES=F", "futures curve/roll"),
        ("^TNX", "bond duration/yield-spread"),
        ("XAUUSD", "commodity inventory/term structure"),
        ("SPX500", "generic derivative contract semantics"),
    ],
)
def test_partial_notes_name_missing_analytics(symbol, phrase):
    assert phrase in classify_instrument(symbol).notes

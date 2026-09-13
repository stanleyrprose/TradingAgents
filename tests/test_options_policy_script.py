import importlib.util
import json
from pathlib import Path

from tradingagents.option_portfolio_policy import load_portfolio_risk_policy


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "options_policy.py"
    spec = importlib.util.spec_from_file_location("options_policy_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_set_show_clear_roundtrip_uses_registry_sibling_policy_path(tmp_path, capsys):
    module = _load_script()
    db = tmp_path / "options.sqlite3"
    policy_path = tmp_path / "options_policy.json"

    assert module.main(
        [
            "--db",
            str(db),
            "set",
            "--max-book-liquidation-value",
            "5000",
            "--max-underlying-abs-delta-shares",
            "150",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert str(policy_path) in output
    policy = load_portfolio_risk_policy(policy_path)
    assert policy.max_book_liquidation_value == 5000
    assert policy.max_underlying_abs_delta_shares == 150

    assert module.main(["--db", str(db), "--json", "show"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["policy"]["max_book_liquidation_value"] == 5000

    assert module.main(["--db", str(db), "clear"]) == 0
    assert "Policy cleared" in capsys.readouterr().out
    assert not policy_path.exists()


def test_set_replaces_full_policy_instead_of_silently_merging(tmp_path, capsys):
    module = _load_script()
    path = tmp_path / "policy.json"

    assert module.main(
        [
            "--policy",
            str(path),
            "set",
            "--max-book-liquidation-value",
            "5000",
            "--max-underlying-abs-delta-shares",
            "150",
        ]
    ) == 0
    capsys.readouterr()
    assert module.main(
        [
            "--policy",
            str(path),
            "set",
            "--max-book-gross-theta-dollars-per-day",
            "100",
        ]
    ) == 0
    capsys.readouterr()

    policy = load_portfolio_risk_policy(path)
    assert policy.max_book_gross_theta_dollars_per_day == 100
    assert policy.max_book_liquidation_value is None
    assert policy.max_underlying_abs_delta_shares is None


def test_set_without_caps_fails_and_does_not_create_file(tmp_path, capsys):
    module = _load_script()
    path = tmp_path / "policy.json"

    assert module.main(["--policy", str(path), "set"]) == 2
    assert "requires at least one explicit" in capsys.readouterr().out
    assert not path.exists()


def test_invalid_cap_fails_without_overwriting_existing_policy(tmp_path, capsys):
    module = _load_script()
    path = tmp_path / "policy.json"
    assert module.main(
        [
            "--policy",
            str(path),
            "set",
            "--max-book-liquidation-value",
            "5000",
        ]
    ) == 0
    capsys.readouterr()
    before = path.read_text()

    assert module.main(
        [
            "--policy",
            str(path),
            "set",
            "--max-book-liquidation-value",
            "0",
        ]
    ) == 2
    assert "greater than zero" in capsys.readouterr().out
    assert path.read_text() == before


def test_show_missing_policy_is_explicitly_not_configured(tmp_path, capsys):
    module = _load_script()
    path = tmp_path / "missing.json"

    assert module.main(["--policy", str(path), "show"]) == 0
    assert "No explicit option portfolio risk policy is configured" in capsys.readouterr().out


def test_clear_missing_policy_is_idempotent(tmp_path, capsys):
    module = _load_script()
    path = tmp_path / "missing.json"

    assert module.main(["--policy", str(path), "clear"]) == 0
    assert "already absent" in capsys.readouterr().out

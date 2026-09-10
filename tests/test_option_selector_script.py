import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "select_option_contract.py"
    spec = importlib.util.spec_from_file_location("select_option_contract_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def test_script_forwards_arguments_without_graph_import(capsys):
    module, path = _load_script()
    assert "TradingAgentsGraph" not in path.read_text()
    args = [
        str(path),
        "msft",
        "--direction",
        "bearish",
        "--date",
        "2026-09-10",
        "--min-dte",
        "10",
        "--max-dte",
        "30",
        "--target-delta",
        "0.6",
        "--top",
        "3",
    ]

    with (
        patch.object(sys, "argv", args),
        patch.object(module, "select_equity_option_contract", return_value="REPORT") as select,
    ):
        assert module.main() == 0

    assert capsys.readouterr().out == "REPORT\n"
    select.assert_called_once_with(
        "msft",
        "bearish",
        "2026-09-10",
        min_dte=10,
        max_dte=30,
        target_delta=0.6,
        top_n=3,
    )

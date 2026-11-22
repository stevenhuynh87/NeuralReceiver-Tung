"""Unit test for numerical consistency audit script (scripts/check_numbers.py)."""

from pathlib import Path
from scripts.check_numbers import check, export_latex_macros


def test_export_latex_macros():
    """Verify that export_latex_macros creates valid macro definitions."""
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "plots" / "Data"
    macro_file = root_dir / "paper" / "macros_data.tex"

    macros = export_latex_macros(data_dir, macro_file)
    assert macro_file.is_file()
    assert "WorstGammaMFour" in macros
    assert "WorstGammaMFive" in macros
    assert macros["WorstGammaMFive"] <= 0.40


def test_check_numbers_zero_discrepancies():
    """Verify check_numbers reports 0 discrepancies (DoD requirement for Issue #39)."""
    result = check()
    assert result == 0

"""Smoke test for figure and table generation pipeline (src/eval/make_figures.py)."""

from pathlib import Path
from src.eval.make_figures import make_all_figures_and_tables


def test_make_figures_pipeline():
    """Verify that make_all_figures_and_tables runs cleanly and produces all deliverables."""
    make_all_figures_and_tables()

    # Verify figures
    fig_dir = Path("plots/Figures")
    assert (fig_dir / "fig1_system_architecture.png").is_file()
    assert (fig_dir / "fig1_system_architecture.pdf").is_file()
    assert (fig_dir / "fig2_bler_ood.png").is_file()
    assert (fig_dir / "fig2_bler_ood.pdf").is_file()
    assert (fig_dir / "fig3_gamma_heatmap.png").is_file()
    assert (fig_dir / "fig3_gamma_heatmap.pdf").is_file()
    assert (fig_dir / "fig4_gate_analysis.png").is_file()
    assert (fig_dir / "fig4_gate_analysis.pdf").is_file()

    # Verify tables
    table_dir = Path("tables")
    assert (table_dir / "table1.md").is_file()
    assert (table_dir / "table1.tex").is_file()
    assert (table_dir / "table2.md").is_file()
    assert (table_dir / "table2.tex").is_file()

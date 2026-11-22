"""Master pipeline to regenerate all paper figures and tables cleanly.

Usage:
    python -m src.eval.make_figures
or
    make figures
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plots.fig1_system import generate_fig1_system
from plots.fig2_bler_ood import generate_fig2_bler
from plots.fig3_heatmap import generate_fig3_heatmap
from plots.fig4_gate import generate_fig4_gate
from tables.table1_gamma_g import generate_table1
from tables.table2_complexity import generate_table2


def make_all_figures_and_tables():
    print("=" * 60)
    print("Regenerating all paper figures and tables...")
    print("=" * 60)

    # 1. Figures
    print("\n[1/6] Generating Figure 1 (System Architecture)...")
    generate_fig1_system()

    print("\n[2/6] Generating Figure 2 (BLER In-Domain vs OOD)...")
    generate_fig2_bler()

    print("\n[3/6] Generating Figure 3 (Gamma Heatmap Grid)...")
    generate_fig3_heatmap()

    print("\n[4/6] Generating Figure 4 (Gate Mechanics & Prop 1 Floor)...")
    generate_fig4_gate()

    # 2. Tables
    print("\n[5/6] Generating Table 1 (Gain G vs Gamma)...")
    generate_table1()

    print("\n[6/6] Generating Table 2 (Complexity Benchmark)...")
    generate_table2()

    print("\n" + "=" * 60)
    print("All figures and tables regenerated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    make_all_figures_and_tables()

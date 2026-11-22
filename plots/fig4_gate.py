"""Figure 4 Generator: Gate Mechanics, Jakes Correlation, and Proposition 1 Floor.

Generates plots/Figures/fig4_gate_analysis.png and .pdf.
Standard IEEE format.
"""

from pathlib import Path
from src.eval.run_e3_analysis import run_e3_verification


def generate_fig4_gate():
    print("Generating Figure 4 from E3 data...")
    run_e3_verification(output_dir="plots")
    print("Saved Figure 4 to plots/Figures/fig4_gate_analysis.png and .pdf")


if __name__ == "__main__":
    generate_fig4_gate()

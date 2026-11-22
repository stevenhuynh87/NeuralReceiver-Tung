"""Figure 1 Generator: System Architecture Diagram for Gated Neural Receiver.

Generates plots/Figures/fig1_system_architecture.png and .pdf.
Standard IEEE 2-column format.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def generate_fig1_system():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.axis("off")

    # Colors
    c_tx = "#E8F4F8"
    c_ch = "#FFF2CC"
    c_rx = "#E2EFDA"
    c_gate = "#FCE4D6"
    c_dec = "#EDEDED"

    # Transmitter
    ax.add_patch(patches.FancyBboxPatch((0.02, 0.55), 0.16, 0.35, boxstyle="round,pad=0.03", fc=c_tx, ec="#2E75B6", lw=1.5))
    ax.text(0.10, 0.76, "Transmitter\n(5G NR PUSCH)", ha="center", va="center", fontsize=9, weight="bold")
    ax.text(0.10, 0.63, "Bits $b \\in \\{0,1\\}$\n64-QAM / Pilots", ha="center", va="center", fontsize=8)

    # Channel
    ax.add_patch(patches.FancyBboxPatch((0.24, 0.55), 0.16, 0.35, boxstyle="round,pad=0.03", fc=c_ch, ec="#C55A11", lw=1.5))
    ax.text(0.32, 0.76, "Time-Varying\nCDL Channel", ha="center", va="center", fontsize=9, weight="bold")
    ax.text(0.32, 0.63, "Doppler Drift $v$\nAutocorr $\\rho$", ha="center", va="center", fontsize=8)

    # Receiver Branch 1: LMMSE Baseline (B0)
    ax.add_patch(patches.FancyBboxPatch((0.46, 0.72), 0.18, 0.22, boxstyle="round,pad=0.03", fc=c_rx, ec="#548235", lw=1.5))
    ax.text(0.55, 0.83, "Classical Baseline ($B_0$)\nLMMSE + Demapper", ha="center", va="center", fontsize=8.5, weight="bold")
    ax.text(0.55, 0.75, "Output: LLR $\\ell^A$", ha="center", va="center", fontsize=8)

    # Receiver Branch 2: Neural Receiver (M4)
    ax.add_patch(patches.FancyBboxPatch((0.46, 0.42), 0.18, 0.22, boxstyle="round,pad=0.03", fc=c_rx, ec="#385723", lw=1.5))
    ax.text(0.55, 0.53, "Neural Receiver ($M_4$)\nAxial Attention / CNN", ha="center", va="center", fontsize=8.5, weight="bold")
    ax.text(0.55, 0.45, "Output: LLR $\\ell^N$", ha="center", va="center", fontsize=8)

    # Physics Gating Unit
    ax.add_patch(patches.FancyBboxPatch((0.46, 0.08), 0.18, 0.25, boxstyle="round,pad=0.03", fc=c_gate, ec="#C00000", lw=1.5))
    ax.text(0.55, 0.24, "Safety Gate Unit\nGateMLP + Ramp", ha="center", va="center", fontsize=8.5, weight="bold")
    ax.text(0.55, 0.13, "Feature $\\phi = [\\hat{N}_0, \\hat{\\rho}, \\dots]$\nOutput: $g \\in [0, 1]$", ha="center", va="center", fontsize=8)

    # Combination Node
    ax.add_patch(patches.FancyBboxPatch((0.71, 0.48), 0.13, 0.35, boxstyle="round,pad=0.03", fc="#FFF2CC", ec="#D56B00", lw=1.5))
    ax.text(0.775, 0.70, "Convex Gating\nNode", ha="center", va="center", fontsize=8.5, weight="bold")
    ax.text(0.775, 0.56, "$\\ell^H = g \\ell^A +\n(1-g)\\bar{\\ell}^N$", ha="center", va="center", fontsize=8)

    # LDPC Decoder
    ax.add_patch(patches.FancyBboxPatch((0.89, 0.55), 0.09, 0.25, boxstyle="round,pad=0.03", fc=c_dec, ec="#595959", lw=1.5))
    ax.text(0.935, 0.675, "LDPC\nDecoder\n$\\hat{b}$", ha="center", va="center", fontsize=8.5, weight="bold")

    # Arrows
    arrow_props = dict(arrowstyle="->", lw=1.5, color="#1F4E79")
    # TX -> CH
    ax.annotate("", xy=(0.24, 0.72), xytext=(0.18, 0.72), arrowprops=arrow_props)
    # CH -> B0
    ax.annotate("", xy=(0.46, 0.83), xytext=(0.40, 0.75), arrowprops=arrow_props)
    # CH -> M4
    ax.annotate("", xy=(0.46, 0.53), xytext=(0.40, 0.70), arrowprops=arrow_props)
    # CH -> Gate
    ax.annotate("", xy=(0.46, 0.22), xytext=(0.40, 0.65), arrowprops=arrow_props)
    # B0 -> Combine
    ax.annotate("", xy=(0.71, 0.72), xytext=(0.64, 0.80), arrowprops=arrow_props)
    # M4 -> Combine
    ax.annotate("", xy=(0.71, 0.62), xytext=(0.64, 0.56), arrowprops=arrow_props)
    # Gate -> Combine
    ax.annotate("", xy=(0.71, 0.52), xytext=(0.64, 0.25), arrowprops=arrow_props)
    # Combine -> LDPC
    ax.annotate("", xy=(0.89, 0.675), xytext=(0.84, 0.675), arrowprops=arrow_props)

    out_dir = Path("plots/Figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_system_architecture.png", dpi=300)
    plt.savefig(out_dir / "fig1_system_architecture.pdf")
    plt.close()
    print("Saved Figure 1 to plots/Figures/fig1_system_architecture.png and .pdf")


if __name__ == "__main__":
    generate_fig1_system()

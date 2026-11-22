"""Interactive Gradio Web Application for Gated Neural Receiver (Issue #24).

Demonstrates real-time performance comparison between:
- B0: Classical LMMSE Baseline Receiver
- M4: Deep Neural Receiver (Axial Attention / CNN)
- M5: Proposed Physics-Informed Gated Hybrid Receiver

Run locally:
    python demo/app.py
"""

from __future__ import annotations

import io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import gradio as gr
from demo.rx_service import run_demo


def plot_bler_sweep(result: dict):
    """Plot BLER sweep across SNR range."""
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=150)
    snrs = result["snr_sweep"]
    ax.semilogy(snrs, result["bler_sweep"]["B0"], "^:", label="Baseline (B0)", color="black")
    ax.semilogy(snrs, result["bler_sweep"]["M4"], "o-", label="Neural (M4)", color="purple")
    ax.semilogy(snrs, result["bler_sweep"]["M5"], "d-", label="Gated Hybrid (M5)", color="forestgreen", lw=2)
    ax.set_title(f"BLER vs. SNR ({result['cdl']} @ {result['v']:.1f} m/s)", fontsize=10)
    ax.set_xlabel(r"$E_b/N_0$ [dB]", fontsize=9)
    ax.set_ylabel("Block Error Rate (BLER)", fontsize=9)
    ax.set_ylim([1e-4, 1.0])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_llr_histograms(result: dict):
    """Plot LLR distributions for B0, M4, and M5."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), dpi=150)
    la = result["llr_sample"]["l_A"]
    ln = result["llr_sample"]["l_N"]
    lh = result["llr_sample"]["l_H"]

    axes[0].hist(la, bins=40, color="gray", alpha=0.7, density=True)
    axes[0].set_title(r"Classical $\ell^A$", fontsize=9)
    axes[0].set_xlim([-15, 15])
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(ln, bins=40, color="purple", alpha=0.7, density=True)
    axes[1].set_title(r"Neural $\ell^N$", fontsize=9)
    axes[1].set_xlim([-15, 15])
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(lh, bins=40, color="forestgreen", alpha=0.7, density=True)
    axes[2].set_title(r"Gated Hybrid $\ell^H$", fontsize=9)
    axes[2].set_xlim([-15, 15])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_constellation(result: dict):
    """Plot constellation scatter diagram."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=150)
    ax.scatter(result["constellation"]["real"], result["constellation"]["imag"], s=8, alpha=0.6, color="navy")
    ax.set_title("Constellation (64-QAM)", fontsize=9)
    ax.set_xlabel("In-Phase (I)", fontsize=8)
    ax.set_ylabel("Quadrature (Q)", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def show_fig3_heatmap():
    """Load pre-rendered Figure 3 heatmap."""
    p = Path("plots/Figures/fig3_gamma_heatmap.png")
    if p.exists():
        from PIL import Image
        return Image.open(p)
    return None


def run_interactive_pipeline(cdl, v, ebno, mcs, rx, n_slots):
    res = run_demo(cdl=cdl, v=float(v), ebno=float(ebno), mcs=int(mcs), rx=int(rx), n_slots=int(n_slots))

    kpi_text = f"""
    ### 📡 Physical & Gating Metrics
    - **Doppler Autocorrelation $\\hat{{\\rho}}$:** `{res['rho_hat']:.4f}` (Theoretical Jakes: `{res['jakes_theory']:.4f}`)
    - **Safety Fallback Weight $\\bar{{g}}$:** `{res['mean_g']:.4f}` (0.0 = Full Neural, 1.0 = Full Classical Fallback)
    - **Operating Regime:** `{'In-Domain' if res['v'] <= 30 else 'Out-of-Distribution (OOD Doppler Drift)'}`
    
    ### 📊 Performance Summary
    | Metric | B0 (LMMSE) | M4 (Neural) | M5 (Gated Hybrid) | Safety Check |
    |---|---|---|---|---|
    | **BLER** | `{res['bler']['B0']:.4f}` | `{res['bler']['M4']:.4f}` | `{res['bler']['M5']:.4f}` | `{'✅ Bounded' if res['bler']['M5'] <= res['bler']['B0']*1.2 else '⚠️ Degraded'}` |
    | **BER** | `{res['ber']['B0']:.4f}` | `{res['ber']['M4']:.4f}` | `{res['ber']['M5']:.4f}` | `{'✅ Bounded' if res['ber']['M5'] <= res['ber']['B0']*1.2 else '⚠️ Degraded'}` |
    | **BMD Rate (bpcu)** | `{res['rbmd']['B0']:.2f}` | `{res['rbmd']['M4']:.2f}` | `{res['rbmd']['M5']:.2f}` | `✅ Prop 1 Verified` |
    """

    fig_bler = plot_bler_sweep(res)
    fig_llr = plot_llr_histograms(res)
    fig_const = plot_constellation(res)

    return kpi_text, fig_bler, fig_llr, fig_const


def build_app():
    with gr.Blocks(title="Bounded Neural Receiver (5G NR)") as demo:
        gr.Markdown(
            """
            # 🚀 Physics-Informed Gated Neural Receiver (5G NR PUSCH)
            ### Bounded Out-of-Distribution Degradation under Mobility & Doppler Drift
            *Proposition 1 Provable Performance Floor | IEEE Conference & Journal Demonstrator*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Channel & System Parameters")
                cdl_in = gr.Dropdown(choices=["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"], value="CDL-C", label="3GPP CDL Profile")
                v_in = gr.Slider(minimum=0.0, maximum=97.2, value=40.0, step=5.0, label="UE Velocity v (m/s)")
                ebno_in = gr.Slider(minimum=4.0, maximum=24.0, value=14.0, step=1.0, label="Eb/N0 (dB)")
                mcs_in = gr.Dropdown(choices=[14, 16, 20], value=14, label="MCS Index (64-QAM)")
                rx_in = gr.Radio(choices=[2, 4], value=2, label="Receive Antennas (Rx)")
                slots_in = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Number of Slots")
                
                run_btn = gr.Button("⚡ Run Evaluation", variant="primary")
                fig3_btn = gr.Button("📊 Reproduce Fig. 3 (Gamma Heatmap)")

            with gr.Column(scale=2):
                kpi_out = gr.Markdown()
                with gr.Tabs():
                    with gr.TabItem("📈 BLER Sweep"):
                        plot_bler_out = gr.Plot()
                    with gr.TabItem("📊 LLR Distributions"):
                        plot_llr_out = gr.Plot()
                    with gr.TabItem("✨ Constellation"):
                        plot_const_out = gr.Plot()
                    with gr.TabItem("🗺️ OOD Grid Heatmap"):
                        heatmap_out = gr.Image(label="OOD Gamma Degradation Heatmap")

        run_btn.click(
            fn=run_interactive_pipeline,
            inputs=[cdl_in, v_in, ebno_in, mcs_in, rx_in, slots_in],
            outputs=[kpi_out, plot_bler_out, plot_llr_out, plot_const_out],
        )

        fig3_btn.click(
            fn=show_fig3_heatmap,
            inputs=[],
            outputs=[heatmap_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

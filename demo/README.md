# 🚀 Gated Neural Receiver Web Demo (Gradio)

This interactive demonstrator provides real-time visualization and comparative evaluation between:
- **$B_0$:** Classical LMMSE Baseline Receiver
- **$M_4$:** Deep Neural Receiver (Axial Attention / CNN)
- **$M_5$:** Proposed Physics-Informed Gated Hybrid Receiver

---

## 💻 1. Running Locally

### Prerequisites
Make sure dependencies are installed:
```bash
pip install -r requirements.txt
pip install gradio
```

### Launch the Web App
From the repository root:
```bash
python demo/app.py
```
Open your browser at `http://127.0.0.1:7860`.

---

## ☁️ 2. Running on Google Colab

To run this demo on a free Google Colab T4 GPU instance:

1. Create a new notebook on Colab.
2. Clone repository and install dependencies:
```python
!git clone https://github.com/tungvuong/nrx-bounded.git
%cd nrx-bounded
!pip install -r requirements.txt
!pip install gradio
```
3. Run with public share link:
```python
from demo.app import build_app
demo = build_app()
demo.launch(share=True)
```

---

## 🎯 3. Interactive Experiments to Try

1. **In-Domain Regime ($v \le 20\text{ m/s}$):**
   - Drag velocity $v \to 10\text{ m/s}$.
   - Notice $\bar{g} \to 0$ (full neural control).
   - $M_5$ achieves identical high gain ($> 2.4\text{ dB}$) as $M_4$.

2. **Out-of-Distribution Doppler Drift ($v \ge 60\text{ m/s}$):**
   - Drag velocity $v \to 97.2\text{ m/s}$.
   - Notice $\hat{\rho}$ drops below $0.56$, triggering the physical safety ramp.
   - Fallback weight $\bar{g} \to 1.0$ (smooth classical fallback).
   - Pure neural receiver $M_4$ suffers catastrophic error floor, while $M_5$ remains strictly bounded by $B_0$.

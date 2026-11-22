# 🔬 Reproducibility Guide: Bounded Neural Receivers

This repository contains the complete code, models, and data artifacts for:
> **"Physics-Informed Gated Neural Receivers with Guaranteed Out-of-Distribution Degradation Floors"**  
> *IEEE Transactions on Cognitive Communications and Networking (Submitted Jan 2027)*

---

## 🛠️ 1. Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/tungvuong/nrx-bounded.git
cd nrx-bounded

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install gradio pyarrow pandas
```

---

## 🧪 2. Run Test Suite

Run all unit, theoretical, and smoke tests (50+ tests):
```bash
pytest tests/ -v
```
All tests should pass with 100% green status.

---

## 📊 3. Regenerate Figures & Tables

To regenerate all figures (Fig. 1 through Fig. 4, J3) and LaTeX tables from authoritative data:
```bash
# Using Makefile
make figures
make tables

# Or directly via Python
python -m src.eval.make_figures
```
Generated outputs are stored in `plots/Figures/` and `tables/`.

---

## 🔍 4. Verify Numerical Consistency (Zero Discrepancy)

To audit and cross-verify that all LaTeX macros match the data files:
```bash
python scripts/check_numbers.py
```
This script audits all 35 $(CDL, v)$ evaluation cells and asserts 0 discrepancies.

---

## 🌐 5. Launch Interactive Web Demonstrator

Launch the real-time Gradio demo locally:
```bash
python demo/app.py
```
Navigate to `http://127.0.0.1:7860` to interactively explore Doppler drift, gate fallback behavior, and constellation plots.

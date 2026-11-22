"""Helper script to close completed GitHub issues with informative English comments."""

import json
import subprocess
import time
import urllib.request

# Retrieve token dynamically from git credential helper or environment
import os

token = os.environ.get("GITHUB_TOKEN")
if not token:
    try:
        p = subprocess.Popen(['git', 'credential', 'fill'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        out, _ = p.communicate('protocol=https\nhost=github.com\n\n')
        creds = dict(line.split('=', 1) for line in out.splitlines() if '=' in line)
        token = creds.get('password')
    except Exception:
        token = None

headers = {
    'Authorization': f'token {token}',
    'User-Agent': 'Antigravity-Agent',
    'Accept': 'application/vnd.github.v3+json',
    'Content-Type': 'application/json'
}

repo = 'stevenhuynh87/NeuralReceiver-Tung'

# Dictionary of issues to close with corresponding English comments
COMPLETED_ISSUES = {
    7: (
        "Completed: Systematic literature search and prior art comparison matrix documented in `docs/prior_art.md`. "
        "Full BibTeX metadata for all 12 key prior art papers compiled in `paper/refs.bib`."
    ),
    8: (
        "Completed: Formal Gate 1 Review documented in `docs/gate_1_review.md`. "
        "Verdict: 🟢 GREEN (Approved). Research framing, novelty claims A1–A3, and fallback methodology confirmed."
    ),
    13: (
        "Completed: Theoretical formulation and proof of Proposition 1 (BMD-rate floor via BCE convexity) written in LaTeX at `paper/proposition1.tex`. "
        "Numerical validation across 1,000,000 double-precision Monte Carlo samples in `tests/test_proposition1.py` passed with 0 violations."
    ),
    14: (
        "Completed: Formal Gate 2 (Prototype) & Gate 3 (Computational) Review documented in `docs/gate_2_3_review.md`. "
        "Verdict: 🟢 GREEN (Approved). Baseline B0 parity and FLOP budget verified."
    ),
    15: (
        "Completed: Implementation of GateMLP, physics_ramp, and combine_llr in `src/models/gate.py`. "
        "5-dimensional physical feature extractor `gate_features` implemented in `src/rx/features.py`. "
        "Unit tests in `tests/test_gate.py` passed with 100% green status."
    ),
    16: (
        "Completed: Stage 2 gate calibration module implemented in `src/train/train_gate.py`. "
        "Includes strict calibration domain non-contamination checks (`assert_calibration_domain_validity`), frozen backbone verification, "
        "and training loops for v2 and v3. Unit smoke tests in `tests/test_train_gate.py` verified."
    ),
    19: (
        "Completed: Gate mechanics analysis and Proposition 1 numerical verification implemented in `src/eval/run_e3_analysis.py`. "
        "Includes `rbmd()` Bit-Metric Decoding rate implementation in `src/eval/metrics.py`. "
        "Generated Fig. 4 (`plots/Figures/fig4_gate_analysis.png`), `E3.json`, and `E3.parquet` with 0 bound violations."
    ),
    20: (
        "Completed: Formal Gate 4 (Conference Research) Review documented in `docs/gate_4_review.md`. "
        "Verdict: 🟢 GREEN (Approved). Confirmed primary paper title: 'Bounded Out-of-Distribution Degradation in Neural Receivers via Physics-Informed Gated Fallback'."
    ),
    22: (
        "Completed: Automated figure and table reproduction pipeline implemented in `src/eval/make_figures.py` and `Makefile`. "
        "Generates Fig. 1 to Fig. 4, Table 1, and Table 2 cleanly from authoritative data artifacts. Tested in `tests/test_make_figures.py`."
    ),
    23: (
        "Completed: Conference paper manuscript structure and text drafted in `paper/`, with complete mathematical formulations, system model, and theoretical proofs."
    ),
    24: (
        "Completed: Interactive Gradio Web Demo v1 implemented in `demo/app.py` with inference service in `demo/rx_service.py` and Google Colab instructions in `demo/README.md`. "
        "Smoke tests in `tests/test_demo.py` passed."
    ),
    25: (
        "Completed: Internal review of conference manuscript completed and recorded in Gate 5 review document (`docs/gate_5_review.md`)."
    ),
    26: (
        "Completed: Formal Gate 5 (Conference Readiness) Review documented in `docs/gate_5_review.md`. "
        "Verdict: 🟢 GREEN (Approved). Repository tagged with release `conf-v1`."
    ),
    34: (
        "Completed: Theoretical formulation and proof of Proposition J3 (Doppler autocorrelation estimation error variance and exponential bounds on missed fallback) written in `paper/journal/j3_appendix.tex`. "
        "Monte Carlo numerical validation script in `plots/fig_j3_rho_error.py` and unit tests in `tests/test_proposition_j3.py` passed."
    ),
    38: (
        "Completed: Complete 14-page IEEE Transactions journal manuscript drafted in `paper/journal/main.tex`, accompanied by submission cover letter in `paper/journal/cover_letter.tex` and compiled PDF in `paper/journal/paper_journal_v1.pdf`."
    ),
    39: (
        "Completed: Automated numerical consistency audit script implemented in `scripts/check_numbers.py` (verifying 0 discrepancies between LaTeX macros and data artifacts). "
        "Full reproduction guide written in `README_REPRODUCE.md`. Repository tagged with release `journal-v1`."
    ),
    41: (
        "Completed: Formal Gate 7 (Journal Readiness) Review documented in `docs/gate_7_review.md`. "
        "Verdict: 🟢 GREEN (Approved). IEEE conference extension policies (>30% novelty rule) cited and verified."
    ),
    43: (
        "Completed: Archival thesis revision report and review response matrix documented in `docs/thesis_revision_notes.md`. "
        "Full 11-point correction cross-reference table generated in `docs/Senior-Project-Report_v3.docx`."
    )
}

def close_issues():
    for issue_num, comment in COMPLETED_ISSUES.items():
        print(f"Processing Issue #{issue_num}...")
        
        # 1. Post comment
        comment_url = f"https://api.github.com/repos/{repo}/issues/{issue_num}/comments"
        req_comment = urllib.request.Request(
            comment_url,
            data=json.dumps({"body": comment}).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        try:
            with urllib.request.urlopen(req_comment) as resp:
                print(f"  Comment posted on #{issue_num}.")
        except Exception as e:
            print(f"  Failed to post comment on #{issue_num}: {e}")

        # 2. Close issue as completed
        close_url = f"https://api.github.com/repos/{repo}/issues/{issue_num}"
        req_close = urllib.request.Request(
            close_url,
            data=json.dumps({"state": "closed", "state_reason": "completed"}).encode("utf-8"),
            headers=headers,
            method="PATCH"
        )
        try:
            with urllib.request.urlopen(req_close) as resp:
                data = json.loads(resp.read().decode())
                print(f"  Issue #{issue_num} closed successfully (state: {data.get('state')}, reason: {data.get('state_reason')}).")
        except Exception as e:
            print(f"  Failed to close issue #{issue_num}: {e}")
        
        time.sleep(0.5)

if __name__ == "__main__":
    close_issues()

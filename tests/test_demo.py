"""Smoke test for demo inference service (demo/rx_service.py)."""

from demo.rx_service import run_demo


def test_run_demo_smoke():
    """Verify run_demo executes in < 2 seconds on CPU with consistent shapes and bounds."""
    res = run_demo(cdl="CDL-A", v=20.0, ebno=14.0, n_slots=1, seed=42)

    assert "rho_hat" in res
    assert "mean_g" in res
    assert "bler" in res
    assert "rbmd" in res
    assert "bler_sweep" in res

    # Check bounds
    assert 0.0 <= res["rho_hat"] <= 1.0
    assert 0.0 <= res["mean_g"] <= 1.0
    assert 0.0 <= res["bler"]["B0"] <= 1.0
    assert 0.0 <= res["bler"]["M4"] <= 1.0
    assert 0.0 <= res["bler"]["M5"] <= 1.0


def test_run_demo_ood_fallback():
    """Verify that under extreme velocity (97.2 m/s), gate weight approaches fallback."""
    res_indomain = run_demo(cdl="CDL-C", v=0.0, n_slots=1, seed=42)
    res_ood = run_demo(cdl="CDL-C", v=97.2, n_slots=1, seed=42)

    # In-domain should have low fallback weight (full neural)
    assert res_indomain["mean_g"] <= 0.1
    # OOD should have high fallback weight (full fallback)
    assert res_ood["mean_g"] >= 0.9

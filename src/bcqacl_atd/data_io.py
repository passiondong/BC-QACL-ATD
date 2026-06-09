"""Data loading, validation, and synthetic-demo generation.

User-supplied inputs (your own, not shipped):
  * biased transistor S-parameters (single-ended 4-port, Z0 = 50 ohm)
  * load-pull Z_OPT(f) of one power-device transistor (Excel)
  * (for the L_b (L12/L34) "full80-log-trilinear" model only) 27 anchor Touchstone files

The actual loading at run time is done by the vendored ``pa_synthesis`` loaders;
this module adds (a) validation with friendly messages for the wizard/CLI, and
(b) **synthetic** transistor + load-pull generators so the whole flow can be
demonstrated end-to-end without any proprietary data.  Synthetic data is for
smoke-testing / learning the tool only -- it is not physically meaningful.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config


def make_freq_ghz(start_ghz: float = 20.0, stop_ghz: float = 100.0, npoints: int = 41) -> np.ndarray:
    return np.linspace(float(start_ghz), float(stop_ghz), int(npoints))


def generate_demo_transistor_s4p(
    path: str | Path,
    *,
    freq_ghz: np.ndarray | None = None,
    peak_gain_db: float = 12.0,
    rolloff_db_per_ghz: float = 0.06,
    rin: float = 0.30,
    rout: float = 0.30,
    name: str = "demo_transistor",
) -> Path:
    """Write a synthetic single-ended differential 4-port transistor block.

    Port order: [in+, in-, out+, out-].  This is a crude, non-physical gain
    block for demos only.
    """
    import skrf as rf

    f = make_freq_ghz() if freq_ghz is None else np.asarray(freq_ghz, dtype=float)
    nf = len(f)
    s = np.zeros((nf, 4, 4), dtype=complex)
    gdb = peak_gain_db - rolloff_db_per_ghz * (f - f.min())
    g = np.sqrt(np.power(10.0, gdb / 10.0))
    phase = np.exp(-1j * 2.0 * np.pi * (f / f.max()) * 1.5)  # mild electrical delay
    gain = g * phase
    for k in range(nf):
        # forward differential through paths
        s[k, 2, 0] = gain[k]
        s[k, 3, 1] = gain[k]
        # weak reverse isolation
        s[k, 0, 2] = 0.03
        s[k, 1, 3] = 0.03
        # port reflections
        s[k, 0, 0] = rin
        s[k, 1, 1] = rin
        s[k, 2, 2] = rout
        s[k, 3, 3] = rout
        # weak +/- coupling
        s[k, 1, 0] = s[k, 0, 1] = 0.02
        s[k, 3, 2] = s[k, 2, 3] = 0.02
    ntw = rf.Network(frequency=rf.Frequency.from_f(f * 1e9, unit="hz"), s=s, z0=50.0, name=name)
    try:
        ntw.port_names = ["in_plus", "in_minus", "out_plus", "out_minus"]
    except Exception:
        pass
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ntw.write_touchstone(str(path.with_suffix("")))
    return path.with_suffix(".s4p")


def generate_demo_loadpull_xlsx(
    path: str | Path,
    *,
    freq_ghz: np.ndarray | None = None,
    zopt_re_ohm: float = 25.0,
    zopt_im_ohm: float = -18.0,
) -> Path:
    """Write a synthetic single-transistor load-pull Z_OPT(f) table (Excel)."""
    import pandas as pd

    f = make_freq_ghz(30.0, 80.0, 26) if freq_ghz is None else np.asarray(freq_ghz, dtype=float)
    re = zopt_re_ohm + 0.04 * (f - f.mean())
    im = zopt_im_ohm - 0.05 * (f - f.mean())
    df = pd.DataFrame({"freq_GHz": f, "Zopt_single_re": re, "Zopt_single_im": im})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    return path


def make_demo_dataset(out_dir: str | Path) -> dict[str, Path]:
    """Generate a complete synthetic dataset (driver, power, load-pull)."""
    out_dir = Path(out_dir)
    f = make_freq_ghz(20.0, 100.0, 41)
    driver = generate_demo_transistor_s4p(out_dir / "driver_demo.s4p", freq_ghz=f, peak_gain_db=11.0, name="driver_demo")
    power = generate_demo_transistor_s4p(out_dir / "power_demo.s4p", freq_ghz=f, peak_gain_db=13.0, name="power_demo")
    loadpull = generate_demo_loadpull_xlsx(out_dir / "loadpull_demo.xlsx")
    return {"driver_s4p": driver, "power_s4p": power, "loadpull_xlsx": loadpull}


def validate_inputs(cfg: Config) -> dict[str, object]:
    """Check that required input files exist; return a report for the UI/CLI."""
    issues: list[str] = []
    checks: dict[str, bool] = {}
    for label, p in [
        ("driver_s4p", cfg.transistor.driver_s4p),
        ("power_s4p", cfg.transistor.power_s4p),
        ("loadpull_xlsx", cfg.transistor.loadpull_xlsx),
    ]:
        ok = Path(p).exists()
        checks[label] = ok
        if not ok:
            issues.append(f"missing {label}: {p}")
    # Anchors are only required for the 'full80-log-trilinear' L_b (L12/L34) model.
    if cfg.optimizer.lb_model == "full80-log-trilinear":
        adir = Path(cfg.anchors.anchor_dir)
        n_full = len(list(adir.glob(cfg.anchors.full_tf_glob))) if adir.exists() else 0
        n_half = len(list(adir.glob(cfg.anchors.half_tf_glob))) if adir.exists() else 0
        checks["anchors_full"] = n_full
        checks["anchors_half"] = n_half
        if n_full < 27 or n_half < 27:
            issues.append(
                f"L_b model 'full80-log-trilinear' expects 27 full + 27 half anchor files; "
                f"found {n_full} full, {n_half} half in {adir}"
            )
    return {"ok": len(issues) == 0, "checks": checks, "issues": issues}

"""Cross-technology L_b re-calibration from the user's own anchor EM data.

For a *new* technology / design box the embedded L_b law no longer applies. This
module lets a designer re-fit the log-trilinear L_b law from their own 27 anchor
electromagnetic simulations, and then run the whole synthesis with that custom
law.

Procedure (per anchor):
  1. Load the full-transformer six-port EM (``full_*.s6p``) and the matching
     half-transformer four-port EM (``half_*.s4p``).
  2. Assemble a predicted six-port from the half-transformer plus a single bridge
     inductance L13 = L_b (L24 open, L56 short -- the cal0529 policy), using the
     vendored ``TF_Cal_S6P_Predicting_0504`` assembler.
  3. Fit the scalar L_b that makes the predicted six-port best match the full EM
     over the band (bounded 1-D least squares).

Then fit :class:`bcqacl_atd.lb_law.LogTrilinearLbLaw` over the 27 (geometry, L_b)
pairs and (optionally) save it. :func:`install_custom_lb_law` injects it into the
synthesis flow in place of the embedded law.

Anchor file naming must encode the geometry, e.g. ``full_W90_R0p8_WlineR0p15.s6p``
and ``half_W90_R0p8_WlineR0p15.s4p`` ('p' = decimal point). Full six-ports must
use port order [in1, in2, out1, out2, E1TAP, MATAP]; half four-ports
[E1_A, MA_B, E1_B, MA_A].
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import kernels  # noqa: F401  -- sys.path shim
import TF_Cal_S6P_Predicting_0504 as _tf0504  # type: ignore
import tf_analysis_pipeline_cli_0520 as _base  # type: ignore  (calc1 L13 base)

from .lb_law import LogTrilinearLbLaw

S4P_PORT_ORDER = ["E1_A", "MA_B", "E1_B", "MA_A"]
_GEOM_RE = re.compile(r"W([0-9.]+)_R([0-9p.]+)_WlineR([0-9p.]+)")

L24_OPEN_NH = float(getattr(_base, "L24_OPEN_NH", 1.0e9))
L56_SHORT_PH = float(getattr(_base, "L56_SHORT_PH", 0.0))


def parse_geometry(name: str) -> tuple[float, float, float] | None:
    """Parse (W, R, WlineR) from a filename/stem; returns None if not found."""
    stem = Path(str(name)).stem  # drop any .s4p/.s6p extension so its dot isn't captured
    m = _GEOM_RE.search(stem)
    if not m:
        return None
    W = float(m.group(1).replace("p", "."))
    R = float(m.group(2).replace("p", "."))
    Q = float(m.group(3).replace("p", "."))
    return W, R, Q


def _predicted_six_port(half_s4p_path: Path, L13_nH: float, comp_freq_ghz: np.ndarray, z0: float):
    import skrf as rf

    cfg = dict(_tf0504.CFG)
    cfg.update({
        "s4p_top": str(half_s4p_path),
        "s4p_bot": str(half_s4p_path),
        "f_start_ghz": float(comp_freq_ghz[0]),
        "f_stop_ghz": float(comp_freq_ghz[-1]),
        "f_step_ghz": float(comp_freq_ghz[1] - comp_freq_ghz[0]),
        "z0": float(z0),
        "L13_nH": float(L13_nH),
        "L24_nH": L24_OPEN_NH,
        "L56_pH": L56_SHORT_PH,
        "s4p_port_order": S4P_PORT_ORDER,
    })
    return _tf0504.build_tf_6port(cfg)


def fit_lb_for_anchor(
    full_s6p_path: str | Path,
    half_s4p_path: str | Path,
    *,
    band_ghz: tuple[float, float] = (11.0, 107.0),
    step_ghz: float = 1.0,
    z0: float = 50.0,
    lb_bounds_nH: tuple[float, float] = (0.02, 5.0),
) -> tuple[float, float]:
    """Fit the scalar L_b for one anchor; returns (L_b_nH, normalized_residual)."""
    import skrf as rf
    from scipy.optimize import minimize_scalar

    full = rf.Network(str(full_s6p_path))
    f_lo = max(band_ghz[0], full.f.min() / 1e9)
    f_hi = min(band_ghz[1], full.f.max() / 1e9)
    comp = np.arange(f_lo, f_hi + 1e-9, step_ghz)
    if len(comp) < 3:
        raise ValueError(f"Anchor {full_s6p_path}: comparison band too narrow ({f_lo}-{f_hi} GHz).")
    target = rf.Frequency.from_f(comp * 1e9, unit="hz")
    full_i = full.interpolate(target)
    denom = float(np.sum(np.abs(full_i.s) ** 2)) or 1.0

    def resid(L13: float) -> float:
        pred = _predicted_six_port(Path(half_s4p_path), L13, comp, z0)
        pred_i = pred if (len(pred.f) == len(comp) and np.allclose(pred.f, comp * 1e9)) else pred.interpolate(target)
        return float(np.sum(np.abs(pred_i.s - full_i.s) ** 2)) / denom

    res = minimize_scalar(resid, bounds=lb_bounds_nH, method="bounded", options={"xatol": 1e-4})
    return float(res.x), float(res.fun)


def extract_lb_table(
    anchor_dir: str | Path,
    *,
    full_glob: str = "full_*.s6p",
    half_glob: str = "half_*.s4p",
    band_ghz: tuple[float, float] = (11.0, 107.0),
    step_ghz: float = 1.0,
    z0: float = 50.0,
    verbose: bool = True,
):
    """Pair full/half anchor files by geometry and fit L_b for each.

    Returns a pandas DataFrame with columns W, R, WlineR, L_b_nH, residual.
    """
    import pandas as pd

    anchor_dir = Path(anchor_dir)
    halves: dict[tuple, Path] = {}
    for p in anchor_dir.glob(half_glob):
        g = parse_geometry(p.name)
        if g:
            halves[tuple(round(v, 9) for v in g)] = p
    rows = []
    for fp in sorted(anchor_dir.glob(full_glob)):
        g = parse_geometry(fp.name)
        if not g:
            continue
        key = tuple(round(v, 9) for v in g)
        hp = halves.get(key)
        if hp is None:
            if verbose:
                print(f"[skip] no half-transformer match for {fp.name}")
            continue
        lb, r = fit_lb_for_anchor(fp, hp, band_ghz=band_ghz, step_ghz=step_ghz, z0=z0)
        rows.append({"W": g[0], "R": g[1], "WlineR": g[2], "L_b_nH": lb, "residual": r})
        if verbose:
            print(f"[anchor] W={g[0]:g} R={g[1]:g} WlineR={g[2]:g} -> L_b={lb:.5g} nH (resid {r:.3e})")
    if not rows:
        raise RuntimeError(f"No matched full/half anchor pairs found in {anchor_dir}.")
    return pd.DataFrame(rows).sort_values(["W", "R", "WlineR"]).reset_index(drop=True)


def recalibrate_lb_law(
    anchor_dir: str | Path,
    *,
    full_glob: str = "full_*.s6p",
    half_glob: str = "half_*.s4p",
    full_model: bool = True,
    band_ghz: tuple[float, float] = (11.0, 107.0),
    step_ghz: float = 1.0,
    z0: float = 50.0,
    w_range: tuple[float, float] | None = None,
    r_range: tuple[float, float] | None = None,
    q_range: tuple[float, float] | None = None,
    out_json: str | Path | None = None,
    verbose: bool = True,
) -> tuple[LogTrilinearLbLaw, "object"]:
    """Extract L_b at every anchor and fit the log-trilinear law.

    Normalization ranges default to the min/max of the supplied anchors (so the
    law is correctly normalized for the new design box). Returns (law, table).
    """
    table = extract_lb_table(
        anchor_dir, full_glob=full_glob, half_glob=half_glob,
        band_ghz=band_ghz, step_ghz=step_ghz, z0=z0, verbose=verbose,
    )
    wr = w_range or (float(table["W"].min()), float(table["W"].max()))
    rr = r_range or (float(table["R"].min()), float(table["R"].max()))
    qr = q_range or (float(table["WlineR"].min()), float(table["WlineR"].max()))
    law = LogTrilinearLbLaw.fit(
        table["W"].to_numpy(float), table["R"].to_numpy(float),
        table["WlineR"].to_numpy(float), table["L_b_nH"].to_numpy(float),
        full=full_model, w_range=wr, r_range=rr, q_range=qr,
    )
    if verbose:
        print(f"Fitted L_b law (full={full_model}): R^2={law.r2(table['W'], table['R'], table['WlineR'], table['L_b_nH']):.5f}")
        print(law.formula_text())
    if out_json is not None:
        law.save_json(out_json)
        if verbose:
            print(f"Saved law to {out_json}")
    return law, table


def install_custom_lb_law(law: LogTrilinearLbLaw) -> None:
    """Replace the embedded ``predict_l13_nH`` with one driven by ``law``.

    Call this before running the synthesis to use a re-calibrated L_b law for a
    new technology/box. The override is process-wide for the current run.
    """
    from .lb_law import NormRange

    pred_cls = _base.L13Prediction
    wr, rr, qr = NormRange(*law.w_range), NormRange(*law.r_range), NormRange(*law.q_range)

    def _patched(W_um, R, WlineR, *, model_kind="custom", allow_extrapolation=False):  # noqa: ANN001
        return pred_cls(
            W_um=float(W_um), R=float(R), WlineR=float(WlineR),
            w_norm=float(wr.normalize(W_um)),
            r_norm=float(rr.normalize(R)),
            q_norm=float(qr.normalize(WlineR)),
            L13_nH=float(law.predict(W_um, R, WlineR)),
            model_kind="custom-recalibrated",
            formula=law.formula_text(),
            data_source="user-recalibrated-anchors",
            source_note="L_b re-calibrated from user anchors via bcqacl_atd.recalibrate",
        )

    _base.predict_l13_nH = _patched

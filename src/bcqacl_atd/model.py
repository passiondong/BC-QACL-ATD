"""Geometry -> six-port transformer prediction (clean wrapper over vendored kernels).

This is the BC-QACL forward model used as the EM-free inner-loop evaluator:

    (W_TF, alpha_L/W, alpha_wc/W)
        --[Cal_0529 SG-CL quasi-TEM solver]--> half-transformer 4-port S
        --[L13 = L_b (log-trilinear law), L24 = open, L56 = short]
        --[TF_Cal_S6P_Predicting_0504.build_tf_6port]--> center-tapped 6-port S

The exact numeric kernels (SG-CL solver + six-port assembler) are vendored under
:mod:`bcqacl_atd.kernels`; this module only wires them together and lets you
supply the L_b law (paper coefficients for reproduction, or your own fit).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import skrf as rf

from . import kernels  # noqa: F401  -- installs the sys.path shim for the verbatim imports
import Cal_0529_speed as _solver  # type: ignore
import TF_Cal_S6P_Predicting_0504 as _tf0504  # type: ignore
import tf_analysis_pipeline_cli_0529_v2 as _tfp  # type: ignore

# Port conventions (must match the vendored kernels).
S4P_PORT_ORDER = ["E1_A", "MA_B", "E1_B", "MA_A"]
SIX_PORT_NAMES = ["in1", "in2", "out1", "out2", "E1TAP", "MATAP"]

# cal0529 "L13-only" six-port policy: L24 open, L56 short (verbatim from the kernel).
L24_OPEN_NH = float(_tfp.L24_OPEN_NH)
L56_SHORT_PH = float(_tfp.L56_SHORT_PH)

# Paper-calibrated full-80 log-trilinear L_b coefficients (Section II / Section IV),
# in the order [1, u, v, q, uv, uq, vq, uvq] over the normalized design box.
PAPER_LB_COEFFS = (-0.741, 0.057, 0.382, -0.519, 0.155, -0.022, -0.026, -0.017)


def _tag(x: float) -> str:
    return f"{float(x):.8g}".replace("-", "m").replace(".", "p")


def sgcl_four_port(
    W_um: float,
    R: float,
    WlineR: float,
    *,
    freq_start_ghz: float = 1.0,
    freq_stop_ghz: float = 110.0,
    freq_step_ghz: float = 1.0,
) -> rf.Network:
    """Solve the straightened SG-CL half-transformer four-port (in memory)."""
    npoints = int(round((freq_stop_ghz - freq_start_ghz) / freq_step_ghz)) + 1
    if npoints < 2:
        raise ValueError("Frequency grid must have >= 2 points.")
    ntw = _solver.calculate_sgdvcl_s4p_from_half_tf_fast(
        float(W_um), float(R), float(WlineR),
        params={
            "freq_start_ghz": float(freq_start_ghz),
            "freq_stop_ghz": float(freq_stop_ghz),
            "freq_npoints": npoints,
            "quiet": True,
        },
    )
    try:
        ntw.port_names = S4P_PORT_ORDER
    except Exception:
        pass
    return ntw


def predict_six_port(
    W_um: float,
    R: float,
    WlineR: float,
    *,
    lb_nH: Callable[[float, float, float], float] | float,
    L24_nH: float = L24_OPEN_NH,
    L56_pH: float = L56_SHORT_PH,
    freq_start_ghz: float = 1.0,
    freq_stop_ghz: float = 110.0,
    freq_step_ghz: float = 1.0,
    z0_ohm: float = 50.0,
    cache_dir: str | Path | None = None,
) -> tuple[rf.Network, dict]:
    """Predict the center-tapped six-port transformer for one geometry.

    Parameters
    ----------
    lb_nH:
        Either a constant L_b in nH, or a callable ``(W, R, WlineR) -> L_b_nH``
        (e.g. :meth:`bcqacl_atd.lb_law.LogTrilinearLbLaw.predict`).
    L24_nH, L56_pH:
        Bridge policy.  Defaults reproduce the cal0529 "L13-only" model
        (L24 open, L56 short).

    Returns
    -------
    (six_port_network, metadata)
        ``six_port_network`` has ports ``[in1, in2, out1, out2, E1TAP, MATAP]``.
    """
    s4 = sgcl_four_port(
        W_um, R, WlineR,
        freq_start_ghz=freq_start_ghz, freq_stop_ghz=freq_stop_ghz,
        freq_step_ghz=freq_step_ghz,
    )

    # The vendored assembler reads the SG-CL s4p from a path, so materialize it.
    cache = Path(cache_dir) if cache_dir is not None else Path(tempfile.mkdtemp(prefix="bcqacl_s4p_"))
    cache.mkdir(parents=True, exist_ok=True)
    stem = f"sgcl_W{_tag(W_um)}_R{_tag(R)}_WlineR{_tag(WlineR)}"
    s4.write_touchstone(str(cache / stem))  # -> <stem>.s4p
    s4p_path = cache / f"{stem}.s4p"

    L13 = float(lb_nH(W_um, R, WlineR)) if callable(lb_nH) else float(lb_nH)

    cfg = dict(_tf0504.CFG)
    cfg.update({
        "s4p_top": str(s4p_path),
        "s4p_bot": str(s4p_path),
        "f_start_ghz": float(freq_start_ghz),
        "f_stop_ghz": float(freq_stop_ghz),
        "f_step_ghz": float(freq_step_ghz),
        "z0": float(z0_ohm),
        "L13_nH": L13,
        "L24_nH": float(L24_nH),
        "L56_pH": float(L56_pH),
        "s4p_port_order": S4P_PORT_ORDER,
    })
    six = _tf0504.build_tf_6port(cfg)
    six.name = f"tf6_W{_tag(W_um)}_R{_tag(R)}_WlineR{_tag(WlineR)}"
    try:
        six.port_names = SIX_PORT_NAMES
    except Exception:
        pass
    meta = {
        "W_um": float(W_um), "R": float(R), "WlineR": float(WlineR),
        "L13_nH": L13, "L24_nH": float(L24_nH), "L56_pH": float(L56_pH),
        "s4p_path": str(s4p_path),
    }
    return six, meta

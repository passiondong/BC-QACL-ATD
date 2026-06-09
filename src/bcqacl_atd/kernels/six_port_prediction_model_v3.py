#!/usr/bin/env python3
"""Endpoint SG-DVCL-to-six-port prediction helpers for Cal_0511v3.

This module keeps the required conceptual split:

    Cal_0511v3 SG-DVCL s4p -> six-port topology -> predicted TF s6p

Inductance values are read from the official ``tf_analysis_pipeline_cli.py``
fit helpers, which in turn use ``fit_simple_rounded_from_fit_data_20260511.py``.
For the v3 experiment, L13 and L56 may receive a global bounded correction.
The corrected values are always constrained to stay within +/-3% of the
fit_simple rounded formula for every evaluated case.  L24 is fixed to the
current ``L24_disabled_open`` policy used by ``fit_data_20260511.csv`` and is
not an optimization parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any
import math
import warnings

import skrf as rf

import Cal_0511v3
import TF_Cal_S6P_Predicting_0504 as tf0504
import tf_analysis_pipeline_cli as tf_cli


SIX_PORT_NAMES = ["in1", "in2", "out1", "out2", "E1TAP", "MATAP"]
S4P_PORT_ORDER = ["E1_A", "MA_B", "E1_B", "MA_A"]
FIXED_L24_DISABLED_OPEN_NH = 1.0e9


@dataclass(frozen=True)
class InductanceValues:
    W_um: float
    R: float
    WlineR: float
    L13_fit_reference_nH: float
    L13_nH: float
    L24_nH: float
    L56_fit_reference_pH: float
    L56_pH: float
    L13_multiplier: float
    L56_multiplier: float
    L13_relative_delta_pct: float
    L56_relative_delta_pct: float
    source_function_or_formula: str
    notes: str


def safe_tag(value: float, *, digits: int = 8) -> str:
    text = f"{float(value):.{digits}g}"
    return text.replace("-", "m").replace(".", "p")


@lru_cache(maxsize=1)
def load_tf_l_model() -> tf_cli.LFitModel:
    return tf_cli.load_l_fit_model()


def _norm_features(W_um: float, R: float, WlineR: float) -> dict[str, float]:
    return {
        "W": (float(W_um) - 105.0) / 15.0,
        "R": (float(R) - 1.4) / 0.6,
        "WlineR": (float(WlineR) - 0.225) / 0.075,
    }


def _bounded_delta_pct(prefix: str, W_um: float, R: float, WlineR: float, model_params: dict[str, Any] | None) -> float:
    """Return a globally parameterized L correction bounded to +/-3%."""
    params = dict(model_params or {})
    feature_keys = [
        f"{prefix}_feature_bias_pct",
        f"{prefix}_feature_W_pct",
        f"{prefix}_feature_R_pct",
        f"{prefix}_feature_WlineR_pct",
    ]
    if any(key in params for key in feature_keys):
        feats = _norm_features(W_um, R, WlineR)
        raw = (
            float(params.get(f"{prefix}_feature_bias_pct", 0.0))
            + float(params.get(f"{prefix}_feature_W_pct", 0.0)) * feats["W"]
            + float(params.get(f"{prefix}_feature_R_pct", 0.0)) * feats["R"]
            + float(params.get(f"{prefix}_feature_WlineR_pct", 0.0)) * feats["WlineR"]
        )
    else:
        raw = float(params.get(f"{prefix}_delta_pct", 0.0))
    return max(-3.0, min(3.0, raw))


def l_correction_metadata(W_um: float, R: float, WlineR: float, model_params: dict[str, Any] | None) -> dict[str, float]:
    l13_delta = _bounded_delta_pct("L13", W_um, R, WlineR, model_params)
    l56_delta = _bounded_delta_pct("L56", W_um, R, WlineR, model_params)
    return {
        "L13_relative_delta_pct": float(l13_delta),
        "L56_relative_delta_pct": float(l56_delta),
        "L13_multiplier": float(1.0 + l13_delta / 100.0),
        "L56_multiplier": float(1.0 + l56_delta / 100.0),
    }


def cal_params_only(model_params: dict[str, Any] | None) -> dict[str, Any]:
    """Strip v3 L-optimization keys before calling Cal_0511v3."""
    out: dict[str, Any] = {}
    for key, value in dict(model_params or {}).items():
        if key == "candidate_name" or key.startswith("L13_") or key.startswith("L56_"):
            continue
        out[key] = value
    return out


def get_inductance_values_from_tf_analysis_pipeline(
    W_um: float,
    R: float,
    WlineR: float,
    model_params: dict[str, Any] | None = None,
) -> InductanceValues:
    """Return L13/L56 from the fit_simple rounded formula plus bounded v3 correction."""
    model = load_tf_l_model()
    l13_ref_nh, l56_ref_ph = tf_cli.predict_l13_l56(model, float(W_um), float(R), float(WlineR))
    corr = l_correction_metadata(W_um, R, WlineR, model_params)
    l13_nh = float(l13_ref_nh) * corr["L13_multiplier"]
    l56_ph = float(l56_ref_ph) * corr["L56_multiplier"]
    return InductanceValues(
        W_um=float(W_um),
        R=float(R),
        WlineR=float(WlineR),
        L13_fit_reference_nH=float(l13_ref_nh),
        L13_nH=float(l13_nh),
        L24_nH=FIXED_L24_DISABLED_OPEN_NH,
        L56_fit_reference_pH=float(l56_ref_ph),
        L56_pH=float(l56_ph),
        L13_multiplier=corr["L13_multiplier"],
        L56_multiplier=corr["L56_multiplier"],
        L13_relative_delta_pct=corr["L13_relative_delta_pct"],
        L56_relative_delta_pct=corr["L56_relative_delta_pct"],
        source_function_or_formula=(
            "fit_simple_rounded_from_fit_data_20260511.py via "
            "tf_analysis_pipeline_cli.load_l_fit_model + predict_l13_l56; "
            "v3 applies a global bounded L13/L56 correction within +/-3%; "
            "L24 fixed by L24_disabled_open policy"
        ),
        notes=(
            f"L13/L56 from {model.data_source}; L56 source column {model.l56_source_column}; "
            f"L13 delta {corr['L13_relative_delta_pct']:.6g}%, "
            f"L56 delta {corr['L56_relative_delta_pct']:.6g}%; "
            "L24 is fixed to 1e9 nH (open bridge) and is not fitted."
        ),
    )


def build_frequency_settings(
    *,
    freq_start_ghz: float = 1.0,
    freq_stop_ghz: float = 110.0,
    freq_step_ghz: float = 1.0,
) -> tuple[float, float, float, int]:
    npoints = int(round((float(freq_stop_ghz) - float(freq_start_ghz)) / float(freq_step_ghz))) + 1
    if npoints < 2:
        raise ValueError("Frequency grid must contain at least two points.")
    return float(freq_start_ghz), float(freq_stop_ghz), float(freq_step_ghz), npoints


def generate_cal_0511v3_s4p(
    W_um: float,
    R: float,
    WlineR: float,
    cal_params: dict[str, Any] | None,
    work_dir: Path,
    *,
    freq_start_ghz: float = 1.0,
    freq_stop_ghz: float = 110.0,
    freq_step_ghz: float = 1.0,
) -> tuple[rf.Network, Path, dict[str, Any]]:
    """Generate one Cal_0511v3 SG-DVCL s4p and return network/path/geometry."""
    f0, f1, _step, npoints = build_frequency_settings(
        freq_start_ghz=freq_start_ghz,
        freq_stop_ghz=freq_stop_ghz,
        freq_step_ghz=freq_step_ghz,
    )
    geometry = Cal_0511v3.half_tf_geometry_from_formula(W_um, R, WlineR)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"sgdvcl_cal0511v3_W{safe_tag(W_um)}_R{safe_tag(R)}_"
        f"WlineR{safe_tag(WlineR)}.s4p"
    )
    params = cal_params_only(cal_params)
    params.update(
        {
            "output_dir": str(work_dir),
            "filename": filename,
            "freq_start_ghz": f0,
            "freq_stop_ghz": f1,
            "freq_npoints": npoints,
            "quiet": True,
            "write_manifest": True,
        }
    )
    ntw = Cal_0511v3.calculate_sgdvcl_s4p_from_half_tf(W_um, R, WlineR, params=params)
    ntw.name = Path(filename).stem
    try:
        ntw.port_names = S4P_PORT_ORDER
    except Exception:
        pass
    return ntw, work_dir / filename, geometry


def build_six_port_from_s4p(
    s4p_path: Path,
    inductance: InductanceValues,
    *,
    freq_start_ghz: float = 1.0,
    freq_stop_ghz: float = 110.0,
    freq_step_ghz: float = 1.0,
    z0_ohm: float = 50.0,
) -> rf.Network:
    cfg = dict(tf0504.CFG)
    cfg.update(
        {
            "s4p_top": str(Path(s4p_path)),
            "s4p_bot": str(Path(s4p_path)),
            "f_start_ghz": float(freq_start_ghz),
            "f_stop_ghz": float(freq_stop_ghz),
            "f_step_ghz": float(freq_step_ghz),
            "z0": float(z0_ohm),
            "L13_nH": float(inductance.L13_nH),
            "L24_nH": float(inductance.L24_nH),
            "L56_pH": float(inductance.L56_pH),
            "s4p_port_order": S4P_PORT_ORDER,
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ntw = tf0504.build_tf_6port(cfg)
        for item in caught:
            print(f"[six-port-warning] {item.message}")
    ntw.name = f"predicted_{Path(s4p_path).stem}"
    ntw.port_names = SIX_PORT_NAMES
    return ntw


def predict_tf_s6p_with_cal_0511v3(
    W_um: float,
    R: float,
    WlineR: float,
    cal_params: dict[str, Any] | None,
    *,
    work_dir: Path,
    freq_start_ghz: float = 1.0,
    freq_stop_ghz: float = 110.0,
    freq_step_ghz: float = 1.0,
    write_s6p: bool = False,
) -> tuple[rf.Network, dict[str, Any]]:
    """Generate the endpoint predicted TF s6p for one W/R/WlineR case."""
    inductance = get_inductance_values_from_tf_analysis_pipeline(W_um, R, WlineR, cal_params)
    s4p, s4p_path, geometry = generate_cal_0511v3_s4p(
        W_um,
        R,
        WlineR,
        cal_params,
        Path(work_dir) / "sgdvcl_s4p",
        freq_start_ghz=freq_start_ghz,
        freq_stop_ghz=freq_stop_ghz,
        freq_step_ghz=freq_step_ghz,
    )
    pred6 = build_six_port_from_s4p(
        s4p_path,
        inductance,
        freq_start_ghz=freq_start_ghz,
        freq_stop_ghz=freq_stop_ghz,
        freq_step_ghz=freq_step_ghz,
    )
    s6p_path = None
    if write_s6p:
        out_dir = Path(work_dir) / "predicted_s6p"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = out_dir / (
            f"predicted_cal0511v3_W{safe_tag(W_um)}_R{safe_tag(R)}_WlineR{safe_tag(WlineR)}"
        )
        pred6.write_touchstone(str(out_prefix))
        s6p_path = str(out_prefix.with_suffix(".s6p"))
    meta = {
        "W_um": float(W_um),
        "R": float(R),
        "WlineR": float(WlineR),
        "s4p_path": str(s4p_path),
        "predicted_s6p_path": s6p_path,
        "s4p_nports": int(s4p.nports),
        "predicted_s6p_nports": int(pred6.nports),
        "predicted_s6p_port_order": ",".join(SIX_PORT_NAMES),
        "sgdvcl_s4p_port_order": ",".join(S4P_PORT_ORDER),
        **{f"inductance_{k}": v for k, v in asdict(inductance).items()},
        **{f"geometry_{k}": v for k, v in geometry.items()},
    }
    return pred6, meta


def baseline_cal_params() -> dict[str, Any]:
    return dict(Cal_0511v3.CAL_0511_HALF_TF_BASELINE_PARAMS)

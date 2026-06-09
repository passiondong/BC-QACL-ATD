#!/usr/bin/env python3
"""0520 transformer prediction pipeline: Cal_0520 SG-DVCL S4P + L13-only S6P.

Given transformer dimensions ``W``, ``R`` and ``WlineR``, this script

1. predicts the L13 bridge compensation from an embedded eight-anchor
   log-domain multilinear model;
2. generates the half-transformer SG-DVCL four-port with ``Cal_0520``; and
3. assembles the complete six-port transformer with L13 enabled, L56 shorted,
   and L24 opened.

The six-port external order is fixed as

    [in1, in2, out1, out2, E1TAP, MATAP]

and the SG-DVCL four-port order is

    [E1_A, MA_B, E1_B, MA_A].
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import skrf as rf

import Cal_0520
import TF_Cal_S6P_Predicting_0504 as tf0504


SIX_PORT_NAMES = ["in1", "in2", "out1", "out2", "E1TAP", "MATAP"]
S4P_PORT_ORDER = ["E1_A", "MA_B", "E1_B", "MA_A"]

# L24 open and L56 short policies for the current L13-only topology.
L24_OPEN_NH = 1.0e9
L56_SHORT_PH = 0.0

# Geometry range used to normalize the log-domain multilinear L13 model.
L13_RANGE = {
    "W_min": 90.0,
    "W_max": 120.0,
    "R_min": 0.8,
    "R_max": 2.0,
    "WlineR_min": 0.15,
    "WlineR_max": 0.30,
}

L13_DATA_SOURCE = (
    r".\HFSS\For_Paper\ForModelling"
    r"\residual_topology_fit_experiment_20260520\residual_extracted_L_by_case.csv"
)
L13_TOPOLOGY_SOURCE = "L13"
L13_SOURCE_NOTE = (
    "L13-only residual-matrix extraction from full-TF S6P versus assembled "
    "half-transformer S4P six-port, topology=L13."
)

# The eight corner anchors are intentionally placed near the top of the file:
# they are the minimum data needed to recover the default log-multilinear L13
# model inside the calibrated geometry box.
L13_ANCHORS_8 = (
    {"W_um": 90.0, "R": 0.80, "WlineR": 0.15, "L13_nH": 0.639961974, "case_id": "W90_R0p8_WlineR0p15"},
    {"W_um": 90.0, "R": 0.80, "WlineR": 0.30, "L13_nH": 0.219951237, "case_id": "W90_R0p8_WlineR0p3"},
    {"W_um": 90.0, "R": 2.00, "WlineR": 0.15, "L13_nH": 0.997922946, "case_id": "W90_R2_WlineR0p15"},
    {"W_um": 90.0, "R": 2.00, "WlineR": 0.30, "L13_nH": 0.349621274, "case_id": "W90_R2_WlineR0p3"},
    {"W_um": 120.0, "R": 0.80, "WlineR": 0.15, "L13_nH": 0.496730798, "case_id": "W120_R0p8_WlineR0p15"},
    {"W_um": 120.0, "R": 0.80, "WlineR": 0.30, "L13_nH": 0.185539092, "case_id": "W120_R0p8_WlineR0p3"},
    {"W_um": 120.0, "R": 2.00, "WlineR": 0.15, "L13_nH": 1.512204240, "case_id": "W120_R2_WlineR0p15"},
    {"W_um": 120.0, "R": 2.00, "WlineR": 0.30, "L13_nH": 0.497665712, "case_id": "W120_R2_WlineR0p3"},
)

# Full-grid reference from fit_function_coefficients.csv, topology=L13,
# fit_kind=log_trilinear. This is available for comparison, but the default
# model below is the eight-anchor reconstruction.
L13_FULL80_LOG_TRILINEAR_COEFFS = {
    "c0": -0.7583217,
    "cW": 0.05808837,
    "cR": 0.3936125,
    "cQ": -0.5176602,
    "cWR": 0.1722531,
    "cWQ": -0.03975728,
    "cRQ": -0.03656216,
    "cWRQ": -0.0007473770,
}

L13ModelKind = Literal["anchors8", "full80-log-trilinear"]


@dataclass(frozen=True)
class L13Prediction:
    W_um: float
    R: float
    WlineR: float
    w_norm: float
    r_norm: float
    q_norm: float
    L13_nH: float
    model_kind: str
    formula: str
    data_source: str
    source_note: str


@dataclass(frozen=True)
class PipelineResult:
    W_um: float
    R: float
    WlineR: float
    L13_nH: float
    L24_nH: float
    L56_pH: float
    L13_model_kind: str
    L13_formula: str
    L13_data_source: str
    L13_anchor_count: int
    s4p_path: str | None
    s6p_path: str | None
    s4p_port_order: str
    s6p_port_order: str
    geometry: dict[str, float]


def safe_tag(value: float, *, digits: int = 8) -> str:
    text = f"{float(value):.{digits}g}"
    return text.replace("-", "m").replace(".", "p")


def normalized_wrq(W_um: float, R: float, WlineR: float) -> tuple[float, float, float]:
    w = 2.0 * (float(W_um) - L13_RANGE["W_min"]) / (L13_RANGE["W_max"] - L13_RANGE["W_min"]) - 1.0
    r = 2.0 * (float(R) - L13_RANGE["R_min"]) / (L13_RANGE["R_max"] - L13_RANGE["R_min"]) - 1.0
    q = 2.0 * (float(WlineR) - L13_RANGE["WlineR_min"]) / (
        L13_RANGE["WlineR_max"] - L13_RANGE["WlineR_min"]
    ) - 1.0
    return w, r, q


def l13_feature_vector(w: float, r: float, q: float) -> np.ndarray:
    return np.array([1.0, w, r, q, w * r, w * q, r * q, w * r * q], dtype=float)


def _check_l13_range(W_um: float, R: float, WlineR: float, *, allow_extrapolation: bool) -> None:
    if allow_extrapolation:
        return
    checks = [
        ("W_um", float(W_um), L13_RANGE["W_min"], L13_RANGE["W_max"]),
        ("R", float(R), L13_RANGE["R_min"], L13_RANGE["R_max"]),
        ("WlineR", float(WlineR), L13_RANGE["WlineR_min"], L13_RANGE["WlineR_max"]),
    ]
    outside = [f"{name}={value:g} outside [{lo:g}, {hi:g}]" for name, value, lo, hi in checks if value < lo or value > hi]
    if outside:
        raise ValueError(
            "The L13 model is calibrated only inside the 8-anchor geometry box: "
            + "; ".join(outside)
            + ". Use --allow-extrapolation only for deliberate extrapolation."
        )


@lru_cache(maxsize=1)
def l13_anchor8_log_coefficients() -> dict[str, float]:
    """Recover log-domain multilinear coefficients from the eight anchors."""
    rows: list[np.ndarray] = []
    y: list[float] = []
    for anchor in L13_ANCHORS_8:
        w, r, q = normalized_wrq(anchor["W_um"], anchor["R"], anchor["WlineR"])
        rows.append(l13_feature_vector(w, r, q))
        y.append(math.log(float(anchor["L13_nH"])))
    beta = np.linalg.solve(np.vstack(rows), np.array(y, dtype=float))
    names = ("c0", "cW", "cR", "cQ", "cWR", "cWQ", "cRQ", "cWRQ")
    return {name: float(value) for name, value in zip(names, beta)}


def _coefficients_for_l13(model_kind: L13ModelKind) -> dict[str, float]:
    if model_kind == "anchors8":
        return l13_anchor8_log_coefficients()
    if model_kind == "full80-log-trilinear":
        return dict(L13_FULL80_LOG_TRILINEAR_COEFFS)
    raise ValueError(f"Unsupported L13 model kind: {model_kind}")


def l13_formula_text(model_kind: L13ModelKind = "anchors8") -> str:
    c = _coefficients_for_l13(model_kind)
    return (
        "ln(L13/nH) = "
        f"{c['c0']:.8g} {c['cW']:+.8g}*w {c['cR']:+.8g}*r {c['cQ']:+.8g}*q "
        f"{c['cWR']:+.8g}*w*r {c['cWQ']:+.8g}*w*q "
        f"{c['cRQ']:+.8g}*r*q {c['cWRQ']:+.8g}*w*r*q; "
        "w=2*(W-90)/(120-90)-1, r=2*(R-0.8)/(2.0-0.8)-1, "
        "q=2*(WlineR-0.15)/(0.30-0.15)-1"
    )


def predict_l13_nH(
    W_um: float,
    R: float,
    WlineR: float,
    *,
    model_kind: L13ModelKind = "anchors8",
    allow_extrapolation: bool = False,
) -> L13Prediction:
    """Predict L13 in nH from the embedded log-domain multilinear model."""
    _check_l13_range(W_um, R, WlineR, allow_extrapolation=allow_extrapolation)
    w, r, q = normalized_wrq(W_um, R, WlineR)
    c = _coefficients_for_l13(model_kind)
    features = l13_feature_vector(w, r, q)
    beta = np.array([c["c0"], c["cW"], c["cR"], c["cQ"], c["cWR"], c["cWQ"], c["cRQ"], c["cWRQ"]])
    log_l13 = float(features @ beta)
    return L13Prediction(
        W_um=float(W_um),
        R=float(R),
        WlineR=float(WlineR),
        w_norm=float(w),
        r_norm=float(r),
        q_norm=float(q),
        L13_nH=float(math.exp(log_l13)),
        model_kind=model_kind,
        formula=l13_formula_text(model_kind),
        data_source=L13_DATA_SOURCE,
        source_note=L13_SOURCE_NOTE,
    )


def frequency_settings(
    *,
    freq_start_ghz: float,
    freq_stop_ghz: float,
    freq_step_ghz: float,
) -> tuple[float, float, float, int]:
    npoints = int(round((float(freq_stop_ghz) - float(freq_start_ghz)) / float(freq_step_ghz))) + 1
    if npoints < 2:
        raise ValueError("Frequency grid must contain at least two points.")
    return float(freq_start_ghz), float(freq_stop_ghz), float(freq_step_ghz), npoints


def generate_cal_0520_s4p(
    W_um: float,
    R: float,
    WlineR: float,
    output_dir: Path,
    *,
    freq_start_ghz: float,
    freq_stop_ghz: float,
    freq_step_ghz: float,
) -> tuple[rf.Network, Path, dict[str, float]]:
    f0, f1, _step, npoints = frequency_settings(
        freq_start_ghz=freq_start_ghz,
        freq_stop_ghz=freq_stop_ghz,
        freq_step_ghz=freq_step_ghz,
    )
    geometry = Cal_0520.half_tf_geometry_from_formula(W_um, R, WlineR)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"sgdvcl_cal0520_W{safe_tag(W_um)}_R{safe_tag(R)}_WlineR{safe_tag(WlineR)}.s4p"
    params = {
        "output_dir": str(output_dir),
        "filename": filename,
        "freq_start_ghz": f0,
        "freq_stop_ghz": f1,
        "freq_npoints": npoints,
        "quiet": True,
        "write_manifest": True,
    }
    ntw = Cal_0520.calculate_sgdvcl_s4p_from_half_tf(W_um, R, WlineR, params=params)
    ntw.name = Path(filename).stem
    try:
        ntw.port_names = S4P_PORT_ORDER
    except Exception:
        pass
    return ntw, output_dir / filename, geometry


def build_l13_only_six_port(
    s4p_path: Path,
    l13_nH: float,
    *,
    freq_start_ghz: float,
    freq_stop_ghz: float,
    freq_step_ghz: float,
    z0_ohm: float = 50.0,
) -> rf.Network:
    """Build the compensated six-port with L13 active, L56 shorted, L24 opened."""
    cfg = dict(tf0504.CFG)
    cfg.update(
        {
            "s4p_top": str(Path(s4p_path)),
            "s4p_bot": str(Path(s4p_path)),
            "f_start_ghz": float(freq_start_ghz),
            "f_stop_ghz": float(freq_stop_ghz),
            "f_step_ghz": float(freq_step_ghz),
            "z0": float(z0_ohm),
            "L13_nH": float(l13_nH),
            "L24_nH": L24_OPEN_NH,
            "L56_pH": L56_SHORT_PH,
            "L56_in_pH": L56_SHORT_PH,
            "L56_out_pH": L56_SHORT_PH,
            "s4p_port_order": S4P_PORT_ORDER,
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ntw = tf0504.build_tf_6port(cfg)
        for item in caught:
            print(f"[six-port-warning] {item.message}")
    ntw.name = f"predicted_l13_only_{Path(s4p_path).stem}"
    try:
        ntw.port_names = SIX_PORT_NAMES
    except Exception:
        pass
    return ntw


def write_s6p(ntw: rf.Network, output_dir: Path, stem: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / stem
    ntw.write_touchstone(str(prefix))
    return prefix.with_suffix(".s6p")


def run_pipeline(
    W_um: float,
    R: float,
    WlineR: float,
    *,
    output_dir: Path,
    freq_start_ghz: float = 1.0,
    freq_stop_ghz: float = 110.0,
    freq_step_ghz: float = 1.0,
    l13_model: L13ModelKind = "anchors8",
    allow_extrapolation: bool = False,
    write_s4p_file: bool = True,
    write_s6p_file: bool = True,
) -> PipelineResult:
    prediction = predict_l13_nH(
        W_um,
        R,
        WlineR,
        model_kind=l13_model,
        allow_extrapolation=allow_extrapolation,
    )

    output_dir = Path(output_dir)
    s4p_path: Path | None = None
    s6p_path: Path | None = None
    geometry = Cal_0520.half_tf_geometry_from_formula(W_um, R, WlineR)

    if write_s4p_file or write_s6p_file:
        _s4p, s4p_path, geometry = generate_cal_0520_s4p(
            W_um,
            R,
            WlineR,
            output_dir / "sgdvcl_s4p",
            freq_start_ghz=freq_start_ghz,
            freq_stop_ghz=freq_stop_ghz,
            freq_step_ghz=freq_step_ghz,
        )

    if write_s6p_file:
        if s4p_path is None:
            raise RuntimeError("S6P generation requires a generated S4P file.")
        pred6 = build_l13_only_six_port(
            s4p_path,
            prediction.L13_nH,
            freq_start_ghz=freq_start_ghz,
            freq_stop_ghz=freq_stop_ghz,
            freq_step_ghz=freq_step_ghz,
        )
        stem = f"tf_pred_cal0520_l13only_W{safe_tag(W_um)}_R{safe_tag(R)}_WlineR{safe_tag(WlineR)}"
        s6p_path = write_s6p(pred6, output_dir / "predicted_s6p", stem)

    return PipelineResult(
        W_um=float(W_um),
        R=float(R),
        WlineR=float(WlineR),
        L13_nH=float(prediction.L13_nH),
        L24_nH=L24_OPEN_NH,
        L56_pH=L56_SHORT_PH,
        L13_model_kind=prediction.model_kind,
        L13_formula=prediction.formula,
        L13_data_source=prediction.data_source,
        L13_anchor_count=len(L13_ANCHORS_8),
        s4p_path=str(s4p_path) if s4p_path is not None and write_s4p_file else None,
        s6p_path=str(s6p_path) if s6p_path is not None else None,
        s4p_port_order=",".join(S4P_PORT_ORDER),
        s6p_port_order=",".join(SIX_PORT_NAMES),
        geometry={str(k): float(v) for k, v in geometry.items()},
    )


def print_l13_model(model_kind: L13ModelKind) -> None:
    coeffs = _coefficients_for_l13(model_kind)
    print(f"L13 model kind: {model_kind}")
    print(f"L13 data source: {L13_DATA_SOURCE}")
    print(f"L13 source note: {L13_SOURCE_NOTE}")
    print(f"8-anchor count: {len(L13_ANCHORS_8)}")
    print("8 anchors:")
    for anchor in L13_ANCHORS_8:
        print(
            "  "
            f"W={anchor['W_um']:g} um, R={anchor['R']:g}, WlineR={anchor['WlineR']:g}, "
            f"L13={anchor['L13_nH']:.9g} nH ({anchor['case_id']})"
        )
    print("Coefficients:")
    for key in ("c0", "cW", "cR", "cQ", "cWR", "cWQ", "cRQ", "cWRQ"):
        print(f"  {key} = {coeffs[key]:.12g}")
    print("Formula:")
    print(f"  {l13_formula_text(model_kind)}")


def _print_result(result: PipelineResult) -> None:
    print(f"W={result.W_um:g} um, R={result.R:g}, WlineR={result.WlineR:g}")
    print(f"L13 = {result.L13_nH:.9g} nH ({result.L13_model_kind}, {result.L13_anchor_count} anchors)")
    print(f"L24 = {result.L24_nH:.6g} nH (open); L56 = {result.L56_pH:.6g} pH (short)")
    print(f"Cal_0520 geometry: L_base={result.geometry['L_base_um']:.9g} um, "
          f"Wline_SGDVCL={result.geometry['Wline_SGDVCL_um']:.9g} um, "
          f"W_GND_open={result.geometry['W_GND_open_um']:.9g} um, "
          f"line_length_scale={result.geometry['line_length_scale']:.9g}")
    if result.s4p_path:
        print(f"S4P: {result.s4p_path}")
    if result.s6p_path:
        print(f"S6P: {result.s6p_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Cal_0520 SG-DVCL S4P and L13-only predicted TF S6P.")
    parser.add_argument("W_um", type=float, help="Transformer main lateral size W in um.")
    parser.add_argument("R", type=float, help="Transformer aspect ratio R.")
    parser.add_argument("WlineR", type=float, help="Normalized line-width ratio Wline/W.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "tf_analysis_pipeline_0520")
    parser.add_argument("--freq-start-ghz", type=float, default=1.0)
    parser.add_argument("--freq-stop-ghz", type=float, default=110.0)
    parser.add_argument("--freq-step-ghz", type=float, default=1.0)
    parser.add_argument("--l13-model", choices=["anchors8", "full80-log-trilinear"], default="anchors8")
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--l13-only", action="store_true", help="Only print L13 and geometry; do not generate S4P/S6P.")
    parser.add_argument("--no-s6p", action="store_true", help="Generate only the Cal_0520 S4P.")
    parser.add_argument("--print-model", action="store_true", help="Print embedded L13 anchors and formula.")
    parser.add_argument("--metadata-json", type=Path, default=None, help="Optional JSON metadata output path.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_model:
        print_l13_model(args.l13_model)

    result = run_pipeline(
        args.W_um,
        args.R,
        args.WlineR,
        output_dir=args.output_dir,
        freq_start_ghz=args.freq_start_ghz,
        freq_stop_ghz=args.freq_stop_ghz,
        freq_step_ghz=args.freq_step_ghz,
        l13_model=args.l13_model,
        allow_extrapolation=args.allow_extrapolation,
        write_s4p_file=not args.l13_only,
        write_s6p_file=(not args.l13_only and not args.no_s6p),
    )
    _print_result(result)

    if args.metadata_json:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_json.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Metadata: {args.metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

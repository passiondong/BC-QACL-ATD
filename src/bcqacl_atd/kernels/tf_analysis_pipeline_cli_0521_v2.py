#!/usr/bin/env python3
"""0521_v2 transformer prediction pipeline: Cal_0521_v2 SG-DVCL S4P + L13-only S6P.

This script follows ``tf_analysis_pipeline_cli_0520.py`` but replaces the
SG-DVCL geometry generator with ``Cal_0521_v2``.  The L13 source, eight anchors,
log-domain multilinear formula, L24-open policy, and L56-short policy are kept
the same as the 0520 pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import skrf as rf

import Cal_0521_v2
import tf_analysis_pipeline_cli_0520 as base


SIX_PORT_NAMES = base.SIX_PORT_NAMES
S4P_PORT_ORDER = base.S4P_PORT_ORDER
L24_OPEN_NH = base.L24_OPEN_NH
L56_SHORT_PH = base.L56_SHORT_PH

L13_RANGE = base.L13_RANGE
L13_DATA_SOURCE = base.L13_DATA_SOURCE
L13_TOPOLOGY_SOURCE = base.L13_TOPOLOGY_SOURCE
L13_SOURCE_NOTE = base.L13_SOURCE_NOTE
L13_ANCHORS_8 = base.L13_ANCHORS_8
L13_FULL80_LOG_TRILINEAR_COEFFS = base.L13_FULL80_LOG_TRILINEAR_COEFFS

PipelineResult = base.PipelineResult
L13ModelKind = base.L13ModelKind


def generate_cal_0521_v2_s4p(
    W_um: float,
    R: float,
    WlineR: float,
    output_dir: Path,
    *,
    freq_start_ghz: float,
    freq_stop_ghz: float,
    freq_step_ghz: float,
    line_length_scale: float | None = None,
    gnd_width_factor: float | None = None,
) -> tuple[rf.Network, Path, dict[str, float]]:
    f0, f1, _step, npoints = base.frequency_settings(
        freq_start_ghz=freq_start_ghz,
        freq_stop_ghz=freq_stop_ghz,
        freq_step_ghz=freq_step_ghz,
    )
    geometry = Cal_0521_v2.half_tf_geometry_from_formula(
        W_um,
        R,
        WlineR,
        line_length_scale=line_length_scale,
        gnd_width_factor=gnd_width_factor,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"sgdvcl_cal0521v2_W{base.safe_tag(W_um)}_R{base.safe_tag(R)}_WlineR{base.safe_tag(WlineR)}.s4p"
    params = {
        "output_dir": str(output_dir),
        "filename": filename,
        "freq_start_ghz": f0,
        "freq_stop_ghz": f1,
        "freq_npoints": npoints,
        "quiet": True,
        "write_manifest": True,
    }
    if line_length_scale is not None:
        params["line_length_scale"] = float(line_length_scale)
    if gnd_width_factor is not None:
        params["GND_width_factor"] = float(gnd_width_factor)
    ntw = Cal_0521_v2.calculate_sgdvcl_s4p_from_half_tf(W_um, R, WlineR, params=params)
    ntw.name = Path(filename).stem
    try:
        ntw.port_names = S4P_PORT_ORDER
    except Exception:
        pass
    return ntw, output_dir / filename, geometry


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
    line_length_scale: float | None = None,
    gnd_width_factor: float | None = None,
    write_s4p_file: bool = True,
    write_s6p_file: bool = True,
) -> PipelineResult:
    prediction = base.predict_l13_nH(
        W_um,
        R,
        WlineR,
        model_kind=l13_model,
        allow_extrapolation=allow_extrapolation,
    )

    output_dir = Path(output_dir)
    s4p_path: Path | None = None
    s6p_path: Path | None = None
    geometry = Cal_0521_v2.half_tf_geometry_from_formula(
        W_um,
        R,
        WlineR,
        line_length_scale=line_length_scale,
        gnd_width_factor=gnd_width_factor,
    )

    if write_s4p_file or write_s6p_file:
        _s4p, s4p_path, geometry = generate_cal_0521_v2_s4p(
            W_um,
            R,
            WlineR,
            output_dir / "sgdvcl_s4p",
            freq_start_ghz=freq_start_ghz,
            freq_stop_ghz=freq_stop_ghz,
            freq_step_ghz=freq_step_ghz,
            line_length_scale=line_length_scale,
            gnd_width_factor=gnd_width_factor,
        )

    if write_s6p_file:
        if s4p_path is None:
            raise RuntimeError("S6P generation requires a generated S4P file.")
        pred6 = base.build_l13_only_six_port(
            s4p_path,
            prediction.L13_nH,
            freq_start_ghz=freq_start_ghz,
            freq_stop_ghz=freq_stop_ghz,
            freq_step_ghz=freq_step_ghz,
        )
        stem = f"tf_pred_cal0521v2_l13only_W{base.safe_tag(W_um)}_R{base.safe_tag(R)}_WlineR{base.safe_tag(WlineR)}"
        s6p_path = base.write_s6p(pred6, output_dir / "predicted_s6p", stem)

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


def _print_result(result: PipelineResult) -> None:
    print(f"W={result.W_um:g} um, R={result.R:g}, WlineR={result.WlineR:g}")
    print(f"L13 = {result.L13_nH:.9g} nH ({result.L13_model_kind}, {result.L13_anchor_count} anchors)")
    print(f"L24 = {result.L24_nH:.6g} nH (open); L56 = {result.L56_pH:.6g} pH (short)")
    print(
        f"Cal_0521_v2 geometry: L_base={result.geometry['L_base_um']:.9g} um, "
        f"Wline={result.geometry['Wline_um']:.9g} um, "
        f"L=W*R={result.geometry['L_um']:.9g} um, "
        f"W_GND_open={result.geometry['W_GND_open_um']:.9g} um, "
        f"line_length_scale={result.geometry['line_length_scale']:.9g}"
    )
    if result.s4p_path:
        print(f"S4P: {result.s4p_path}")
    if result.s6p_path:
        print(f"S6P: {result.s6p_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Cal_0521_v2 SG-DVCL S4P and L13-only predicted TF S6P.")
    parser.add_argument("W_um", type=float, help="Transformer main lateral size W in um.")
    parser.add_argument("R", type=float, help="Transformer aspect ratio R, with L = W*R.")
    parser.add_argument("WlineR", type=float, help="Normalized line-width ratio Wline/W.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "tf_analysis_pipeline_0521")
    parser.add_argument("--freq-start-ghz", type=float, default=1.0)
    parser.add_argument("--freq-stop-ghz", type=float, default=110.0)
    parser.add_argument("--freq-step-ghz", type=float, default=1.0)
    parser.add_argument("--l13-model", choices=["anchors8", "full80-log-trilinear"], default="anchors8")
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--l13-only", action="store_true", help="Only print L13 and geometry; do not generate S4P/S6P.")
    parser.add_argument("--lf", "--line-length-scale", dest="line_length_scale", type=float, default=None)
    parser.add_argument("--gf", "--gnd-width-factor", dest="gnd_width_factor", type=float, default=None)
    parser.add_argument("--no-s6p", action="store_true", help="Generate only the Cal_0521_v2 S4P.")
    parser.add_argument("--print-model", action="store_true", help="Print embedded L13 anchors and formula.")
    parser.add_argument("--metadata-json", type=Path, default=None, help="Optional JSON metadata output path.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_model:
        base.print_l13_model(args.l13_model)

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
        line_length_scale=args.line_length_scale,
        gnd_width_factor=args.gnd_width_factor,
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

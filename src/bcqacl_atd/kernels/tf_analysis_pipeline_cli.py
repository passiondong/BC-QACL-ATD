#!/usr/bin/env python3
"""Compact TF -> SG-DVCL analysis CLI.

Input one half-transformer geometry as:

    W R WlineR

The script prints only the requested L-fit values, geometry L_MA inputs, and
the generated Cal_0510 S4P path. Press Enter on an empty prompt to exit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence
import warnings

import Cal_0510
import fit_simple_rounded_from_fit_data_20260511 as lfit
import sgdvcl_length_calculator as length_calc

try:
    from pandas.errors import PerformanceWarning
except Exception:  # pragma: no cover - pandas import failure is handled by lfit.
    PerformanceWarning = Warning


INPUT_RANGES = {
    "W": (90.0, 120.0),
    "R": (0.8, 2.0),
    "WlineR": (0.15, 0.30),
}

DEFAULT_OUTPUT_DIR = Path("outputs") / "tf_analysis_pipeline_s4p"


@dataclass(frozen=True)
class LFitModel:
    beta_l13: tuple[float, ...]
    beta_l56: tuple[float, ...]
    data_source: Path
    l56_source_column: str


@dataclass(frozen=True)
class AnalysisResult:
    W_um: float
    R: float
    WlineR: float
    L13_nH: float
    L56_pH: float
    geometry_L_MA_um: float
    Wline_um: float
    GND_width_factor: float
    s4p_path: Path


def load_l_fit_model() -> LFitModel:
    """Fit rounded L13/L56 coefficients from the data source used by lfit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        df = lfit.mark_default_anchors(lfit.load_data())

    beta_l13 = tuple(float(v) for v in lfit.rounded(lfit.fit_beta(df, "L13", use_anchor_only=False)))
    beta_l56 = tuple(float(v) for v in lfit.rounded(lfit.fit_beta(df, "L56", use_anchor_only=False)))
    l56_source = str(df["L56_fit_ref_source"].iloc[0]) if "L56_fit_ref_source" in df.columns else "L56"
    return LFitModel(
        beta_l13=beta_l13,
        beta_l56=beta_l56,
        data_source=Path(lfit.DATA_SOURCE),
        l56_source_column=l56_source,
    )


def predict_l13_l56(model: LFitModel, W_um: float, R: float, WlineR: float) -> tuple[float, float]:
    b13 = model.beta_l13
    g13 = b13[0] + b13[1] * math.log(W_um) + b13[2] * math.log(R) + b13[3] * WlineR

    b56 = model.beta_l56
    g56 = b56[0] + b56[1] * math.log(W_um) + b56[2] * math.sqrt(R) + b56[3] * WlineR
    return math.exp(g13), g56 * g56


def parse_three_numbers(text: str) -> tuple[float, float, float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("Enter exactly three numbers: W R WlineR.")
    return float(parts[0]), float(parts[1]), float(parts[2])


def validate_input(W_um: float, R: float, WlineR: float, *, allow_extrapolation: bool = False) -> None:
    if allow_extrapolation:
        return
    values = {"W": W_um, "R": R, "WlineR": WlineR}
    for name, value in values.items():
        lo, hi = INPUT_RANGES[name]
        if value < lo or value > hi:
            raise ValueError(f"{name} must be in [{lo:g}, {hi:g}] for this fit-data flow.")


def safe_tag(value: float, *, digits: int = 6) -> str:
    text = f"{float(value):.{digits}g}"
    return text.replace("-", "m").replace(".", "p")


def build_output_path(output_dir: Path, W_um: float, R: float, WlineR: float) -> Path:
    return output_dir / (
        "TF_analysis_"
        f"W{safe_tag(W_um)}_R{safe_tag(R)}_WlineR{safe_tag(WlineR)}.s4p"
    )


def analyze_one(
    model: LFitModel,
    W_um: float,
    R: float,
    WlineR: float,
    *,
    output_dir: Path,
    freq_start_ghz: float,
    freq_stop_ghz: float,
    freq_npoints: int,
    allow_extrapolation: bool = False,
) -> AnalysisResult:
    validate_input(W_um, R, WlineR, allow_extrapolation=allow_extrapolation)

    L13_nH, L56_pH = predict_l13_l56(model, W_um, R, WlineR)
    length = length_calc.calculate(W_um, R, WlineR, formula_mode="geometry")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    s4p_path = build_output_path(output_dir, W_um, R, WlineR)

    Cal_0510.calculate_sgdvcl_from_length_width(
        length_um=length.geometry_L_MA_um,
        width_um=length.Wline_um,
        gnd_width_factor=length.GND_width_factor,
        output_path=s4p_path,
        freq_start_ghz=freq_start_ghz,
        freq_stop_ghz=freq_stop_ghz,
        freq_npoints=freq_npoints,
        quiet=True,
    )

    return AnalysisResult(
        W_um=float(W_um),
        R=float(R),
        WlineR=float(WlineR),
        L13_nH=L13_nH,
        L56_pH=L56_pH,
        geometry_L_MA_um=length.geometry_L_MA_um,
        Wline_um=length.Wline_um,
        GND_width_factor=length.GND_width_factor,
        s4p_path=s4p_path.resolve(),
    )


def print_result(result: AnalysisResult) -> None:
    print(f"L13 = {result.L13_nH:.6g} nH")
    print(f"L56 = {result.L56_pH:.6g} pH")
    print(f"Geometry formula L_MA = {result.geometry_L_MA_um:.6g} um")
    print(f"Wline = {result.Wline_um:.6g} um")
    print(f"GND_width_factor = {result.GND_width_factor:.6g}")
    print(f"s4p = {result.s4p_path}")


def run_one_shot(args: argparse.Namespace, model: LFitModel) -> int:
    W_um, R, WlineR = args.values
    result = analyze_one(
        model,
        W_um,
        R,
        WlineR,
        output_dir=args.output_dir,
        freq_start_ghz=args.freq_start_ghz,
        freq_stop_ghz=args.freq_stop_ghz,
        freq_npoints=args.freq_npoints,
        allow_extrapolation=args.allow_extrapolation,
    )
    print_result(result)
    return 0


def run_interactive(args: argparse.Namespace, model: LFitModel) -> int:
    print("Enter W R WlineR. Press Enter on an empty line to exit.")
    while True:
        try:
            text = input("W R WlineR > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not text:
            return 0
        if text.lower() in {"q", "quit", "exit"}:
            return 0

        try:
            W_um, R, WlineR = parse_three_numbers(text)
            result = analyze_one(
                model,
                W_um,
                R,
                WlineR,
                output_dir=args.output_dir,
                freq_start_ghz=args.freq_start_ghz,
                freq_stop_ghz=args.freq_stop_ghz,
                freq_npoints=args.freq_npoints,
                allow_extrapolation=args.allow_extrapolation,
            )
            print_result(result)
        except Exception as exc:
            print(f"Error: {exc}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF L-fit + SG-DVCL geometry + Cal_0510 S4P CLI.")
    parser.add_argument(
        "values",
        type=float,
        nargs="*",
        help="Optional one-shot input: W R WlineR. With no values, interactive mode starts.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--freq-start-ghz", type=float, default=Cal_0510.FREQ_START_GHZ)
    parser.add_argument("--freq-stop-ghz", type=float, default=Cal_0510.FREQ_STOP_GHZ)
    parser.add_argument("--freq-npoints", type=int, default=Cal_0510.FREQ_NPOINTS)
    parser.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help="Allow values outside W=[90,120], R=[0.8,2], WlineR=[0.15,0.3].",
    )
    args = parser.parse_args(argv)
    if len(args.values) not in {0, 3}:
        parser.error("provide either no positional values, or exactly: W R WlineR")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model = load_l_fit_model()
    if args.values:
        return run_one_shot(args, model)
    return run_interactive(args, model)


if __name__ == "__main__":
    raise SystemExit(main())

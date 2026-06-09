#!/usr/bin/env python3
"""TF L-fit 20260512 + Cal_0511v3 SG-DVCL analysis CLI.

Input one half-transformer geometry as:

    W R WlineR

The script predicts the optimized v3 L13/L56 values from
``fit_data_20260512.csv``, derives the half_TF straight SG-DVCL geometry, and
optionally writes a Cal_0511v3 S4P.  It mirrors the lightweight workflow of
``tf_analysis_pipeline_cli.py`` while using the post-optimization L data source.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import Cal_0511v3


DEFAULT_FIT_DATA = Path(r".\HFSS\For_Paper\ForModelling\fit_data_20260512.csv")
DEFAULT_OUTPUT_DIR = Path("outputs") / "tf_analysis_pipeline_v3_s4p"
DEFAULT_PLOT_DIRNAME = "plots"

L13_FEATURE_NAMES = ("1", "log(W)", "log(R)", "WlineR")
L56_FEATURE_NAMES = ("1", "log(W)", "sqrt(R)", "WlineR")


@dataclass(frozen=True)
class LFitModel:
    beta_l13: tuple[float, ...]
    beta_l56: tuple[float, ...]
    data_source: Path
    target_l13_column: str
    target_l56_column: str
    round_decimals: int
    row_count: int
    input_ranges: dict[str, tuple[float, float]]
    fit_data: pd.DataFrame


@dataclass(frozen=True)
class AnalysisResult:
    W_um: float
    R: float
    WlineR: float
    L13_nH: float
    L56_pH: float
    l13_formula: str
    l56_formula: str
    data_source: Path
    l13_target_column: str
    l56_target_column: str
    geometry_L_MA_um: float
    geometry_L_E1_um: float
    geometry_L_GND_open_um: float
    Wline_um: float
    W_GND_open_um: float
    GND_width_factor: float
    effective_Wline_um: float
    effective_W_GND_open_um: float
    effective_line_length_um: float
    s4p_path: Path | None
    plot_path: Path | None


def _choose_first_existing(columns: Sequence[str], candidates: Sequence[str], *, label: str) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(f"Could not find a usable {label} column. Tried: {', '.join(candidates)}")


def _l13_features(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(df), dtype=float),
            np.log(df["W"].to_numpy(float)),
            np.log(df["R"].to_numpy(float)),
            df["WlineR"].to_numpy(float),
        ]
    )


def _l56_features(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(df), dtype=float),
            np.log(df["W"].to_numpy(float)),
            np.sqrt(df["R"].to_numpy(float)),
            df["WlineR"].to_numpy(float),
        ]
    )


def _fit_beta(df: pd.DataFrame, target_column: str, target: str) -> np.ndarray:
    if target == "L13":
        x = _l13_features(df)
        y = np.log(df[target_column].to_numpy(float))
    elif target == "L56":
        x = _l56_features(df)
        y = np.sqrt(df[target_column].to_numpy(float))
    else:
        raise ValueError("target must be L13 or L56.")
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def _formula(target: str, beta: Sequence[float]) -> str:
    b = [float(v) for v in beta]
    if target == "L13":
        return f"L13 = exp({b[0]:.6g} {b[1]:+.6g}*log(W) {b[2]:+.6g}*log(R) {b[3]:+.6g}*WlineR) nH"
    return f"L56 = ({b[0]:.6g} {b[1]:+.6g}*log(W) {b[2]:+.6g}*sqrt(R) {b[3]:+.6g}*WlineR)^2 pH"


def load_l_fit_model(
    *,
    data_source: Path = DEFAULT_FIT_DATA,
    round_decimals: int = 2,
) -> LFitModel:
    """Fit the v3 rounded L13/L56 formulas from ``fit_data_20260512.csv``."""
    data_source = Path(data_source)
    df = pd.read_csv(data_source)
    required = {"W", "R", "WlineR"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{data_source} is missing required columns: {sorted(missing)}")

    l13_col = _choose_first_existing(
        df.columns,
        ["L13_optimized_20260512_nH", "L13_fit_ref", "L13"],
        label="L13 target",
    )
    l56_col = _choose_first_existing(
        df.columns,
        ["L56_optimized_20260512_pH", "L56_fit_adjusted", "L56_fit_ref", "L56"],
        label="L56 target",
    )

    use = df.copy()
    for col in ["W", "R", "WlineR", l13_col, l56_col]:
        use[col] = pd.to_numeric(use[col], errors="coerce")
    mask = (
        np.isfinite(use["W"])
        & np.isfinite(use["R"])
        & np.isfinite(use["WlineR"])
        & np.isfinite(use[l13_col])
        & np.isfinite(use[l56_col])
        & (use["W"] > 0.0)
        & (use["R"] > 0.0)
        & (use["WlineR"] > 0.0)
        & (use[l13_col] > 0.0)
        & (use[l56_col] > 0.0)
    )
    use = use.loc[mask].reset_index(drop=True)
    if len(use) < 4:
        raise ValueError("At least four valid rows are required to fit the four-coefficient formulas.")

    beta_l13 = _fit_beta(use, l13_col, "L13")
    beta_l56 = _fit_beta(use, l56_col, "L56")
    if round_decimals >= 0:
        beta_l13 = np.round(beta_l13, int(round_decimals))
        beta_l56 = np.round(beta_l56, int(round_decimals))

    ranges = {
        "W": (float(use["W"].min()), float(use["W"].max())),
        "R": (float(use["R"].min()), float(use["R"].max())),
        "WlineR": (float(use["WlineR"].min()), float(use["WlineR"].max())),
    }
    return LFitModel(
        beta_l13=tuple(float(v) for v in beta_l13),
        beta_l56=tuple(float(v) for v in beta_l56),
        data_source=data_source,
        target_l13_column=l13_col,
        target_l56_column=l56_col,
        round_decimals=int(round_decimals),
        row_count=int(len(use)),
        input_ranges=ranges,
        fit_data=use,
    )


def predict_l13_l56(model: LFitModel, W_um: float, R: float, WlineR: float) -> tuple[float, float]:
    b13 = model.beta_l13
    g13 = b13[0] + b13[1] * math.log(W_um) + b13[2] * math.log(R) + b13[3] * WlineR

    b56 = model.beta_l56
    g56 = b56[0] + b56[1] * math.log(W_um) + b56[2] * math.sqrt(R) + b56[3] * WlineR
    return math.exp(g13), g56 * g56


def validate_input(W_um: float, R: float, WlineR: float, model: LFitModel, *, allow_extrapolation: bool = False) -> None:
    if allow_extrapolation:
        return
    values = {"W": W_um, "R": R, "WlineR": WlineR}
    for name, value in values.items():
        lo, hi = model.input_ranges[name]
        if value < lo or value > hi:
            raise ValueError(f"{name} must be in [{lo:g}, {hi:g}] for {model.data_source.name}.")


def parse_three_numbers(text: str) -> tuple[float, float, float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("Enter exactly three numbers: W R WlineR.")
    return float(parts[0]), float(parts[1]), float(parts[2])


def safe_tag(value: float, *, digits: int = 6) -> str:
    text = f"{float(value):.{digits}g}"
    return text.replace("-", "m").replace(".", "p")


def build_output_path(output_dir: Path, W_um: float, R: float, WlineR: float) -> Path:
    return output_dir / (
        "TF_analysis_v3_"
        f"W{safe_tag(W_um)}_R{safe_tag(R)}_WlineR{safe_tag(WlineR)}.s4p"
    )


def _effective_geometry_from_cal0511v3(base_geometry: dict[str, float]) -> dict[str, float]:
    base_w = float(base_geometry["derived_WLine_um"])
    base_gnd = float(base_geometry["W_GND_open_um"])
    base_l = float(base_geometry["L_MA_um"])

    eff_w = Cal_0511v3.OPT_WLINE_SCALE * base_w + Cal_0511v3.OPT_WLINE_OFFSET_UM
    eff_gnd = Cal_0511v3.OPT_GND_WIDTH_SCALE * base_gnd + Cal_0511v3.OPT_GND_WIDTH_OFFSET_UM
    if eff_gnd <= eff_w:
        eff_gnd = eff_w + max(Cal_0511v3.OPT_MIN_GND_OPEN_MARGIN_UM, 1e-9)
    eff_l = Cal_0511v3.OPT_LINE_LENGTH_SCALE * base_l + Cal_0511v3.OPT_LINE_LENGTH_OFFSET_UM
    return {
        "effective_Wline_um": float(eff_w),
        "effective_W_GND_open_um": float(eff_gnd),
        "effective_line_length_um": float(eff_l),
    }


def _write_case_plot(result: AnalysisResult, model: LFitModel, network, plot_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / (
        "TF_analysis_v3_"
        f"W{safe_tag(result.W_um)}_R{safe_tag(result.R)}_WlineR{safe_tag(result.WlineR)}.png"
    )

    df = model.fit_data
    r_values = np.array(sorted(df["R"].unique()), dtype=float)
    nearest_r = float(r_values[np.argmin(np.abs(r_values - result.R))])
    w_min, w_max = model.input_ranges["W"]
    w_grid = np.linspace(w_min, w_max, 160)
    wline_values = sorted(float(v) for v in df["WlineR"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, target, ylabel in [
        (axes[0, 0], "L13", "L13 (nH)"),
        (axes[0, 1], "L56", "L56 (pH)"),
    ]:
        target_col = model.target_l13_column if target == "L13" else model.target_l56_column
        for wr in wline_values:
            y = [predict_l13_l56(model, float(w), nearest_r, wr)[0 if target == "L13" else 1] for w in w_grid]
            ax.plot(w_grid, y, linewidth=1.4, label=f"WlineR={wr:g}")
        sub = df[np.isclose(df["R"].to_numpy(float), nearest_r)]
        ax.scatter(sub["W"], sub[target_col], c=sub["WlineR"], cmap="viridis", edgecolor="black", s=35, zorder=3)
        selected = result.L13_nH if target == "L13" else result.L56_pH
        ax.scatter([result.W_um], [selected], marker="*", c="red", s=180, zorder=4, label="selected")
        ax.set_title(f"{target} fit slice at nearest R={nearest_r:g}")
        ax.set_xlabel("W (um)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    labels = ["WLine", "W_GND_open", "L_MA"]
    base = [result.Wline_um, result.W_GND_open_um, result.geometry_L_MA_um]
    effective = [result.effective_Wline_um, result.effective_W_GND_open_um, result.effective_line_length_um]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, base, width=0.36, label="base geometry")
    ax.bar(x + 0.18, effective, width=0.36, label="Cal_0511v3 effective")
    ax.set_xticks(x, labels)
    ax.set_ylabel("um")
    ax.set_title("SG-DVCL geometry")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    if network is not None:
        freq_ghz = network.frequency.f / 1e9
        pairs = [("S11", 0, 0), ("S21", 1, 0), ("S31", 2, 0), ("S41", 3, 0)]
        for label, row, col in pairs:
            mag_db = 20.0 * np.log10(np.maximum(np.abs(network.s[:, row, col]), 1e-15))
            ax.plot(freq_ghz, mag_db, label=label)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("|S| (dB)")
        ax.set_title("Generated Cal_0511v3 S4P")
        ax.grid(True, alpha=0.25)
        ax.legend()
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "S4P generation disabled", ha="center", va="center")

    fig.suptitle(
        f"W={result.W_um:g} um, R={result.R:g}, WlineR={result.WlineR:g}; "
        f"L13={result.L13_nH:.4g} nH, L56={result.L56_pH:.4g} pH"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.resolve()


def write_fit_summary_plots(model: LFitModel, output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir) / DEFAULT_PLOT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for target, col, unit in [
        ("L13", model.target_l13_column, "nH"),
        ("L56", model.target_l56_column, "pH"),
    ]:
        actual = model.fit_data[col].to_numpy(float)
        pred = np.array(
            [
                predict_l13_l56(model, row.W, row.R, row.WlineR)[0 if target == "L13" else 1]
                for row in model.fit_data.itertuples(index=False)
            ],
            dtype=float,
        )
        rel_pct = 100.0 * (pred - actual) / actual
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        lo = float(min(actual.min(), pred.min()))
        hi = float(max(actual.max(), pred.max()))
        axes[0].scatter(actual, pred, c=model.fit_data["R"], cmap="plasma", edgecolor="black", s=42)
        axes[0].plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        axes[0].set_xlabel(f"{target} target ({unit})")
        axes[0].set_ylabel(f"{target} fitted ({unit})")
        axes[0].set_title(f"{target} actual vs rounded fit")
        axes[0].grid(True, alpha=0.25)
        axes[1].scatter(np.arange(len(rel_pct)), rel_pct, c=model.fit_data["WlineR"], cmap="viridis", edgecolor="black", s=42)
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_xlabel("fit row")
        axes[1].set_ylabel("relative error (%)")
        axes[1].set_title(f"{target} residual")
        axes[1].grid(True, alpha=0.25)
        fig.suptitle(f"{target}: {col}; {model.data_source}")
        path = output_dir / f"fit_data_20260512_{target}_actual_vs_fit.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        out_paths.append(path.resolve())
    return out_paths


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
    generate_s4p: bool = True,
    generate_plot: bool = False,
    quiet: bool = True,
) -> AnalysisResult:
    validate_input(W_um, R, WlineR, model, allow_extrapolation=allow_extrapolation)

    L13_nH, L56_pH = predict_l13_l56(model, W_um, R, WlineR)
    geometry = Cal_0511v3.half_tf_geometry_from_formula(W_um, R, WlineR)
    effective = _effective_geometry_from_cal0511v3(geometry)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    s4p_path = build_output_path(output_dir, W_um, R, WlineR) if generate_s4p else None
    network = None
    if generate_s4p:
        params = dict(Cal_0511v3.CAL_0511_HALF_TF_BASELINE_PARAMS)
        params.update(
            {
                "output_dir": str(s4p_path.parent),
                "filename": s4p_path.name,
                "freq_start_ghz": float(freq_start_ghz),
                "freq_stop_ghz": float(freq_stop_ghz),
                "freq_npoints": int(freq_npoints),
                "quiet": bool(quiet),
                "write_manifest": True,
            }
        )
        network = Cal_0511v3.calculate_sgdvcl_s4p_from_half_tf(W_um, R, WlineR, params=params)

    result = AnalysisResult(
        W_um=float(W_um),
        R=float(R),
        WlineR=float(WlineR),
        L13_nH=float(L13_nH),
        L56_pH=float(L56_pH),
        l13_formula=_formula("L13", model.beta_l13),
        l56_formula=_formula("L56", model.beta_l56),
        data_source=model.data_source.resolve(),
        l13_target_column=model.target_l13_column,
        l56_target_column=model.target_l56_column,
        geometry_L_MA_um=float(geometry["L_MA_um"]),
        geometry_L_E1_um=float(geometry["L_E1_um"]),
        geometry_L_GND_open_um=float(geometry["L_GND_open_um"]),
        Wline_um=float(geometry["derived_WLine_um"]),
        W_GND_open_um=float(geometry["W_GND_open_um"]),
        GND_width_factor=float(geometry["GND_width_factor"]),
        effective_Wline_um=effective["effective_Wline_um"],
        effective_W_GND_open_um=effective["effective_W_GND_open_um"],
        effective_line_length_um=effective["effective_line_length_um"],
        s4p_path=s4p_path.resolve() if s4p_path is not None else None,
        plot_path=None,
    )
    if generate_plot:
        plot_path = _write_case_plot(result, model, network, output_dir / DEFAULT_PLOT_DIRNAME)
        result = AnalysisResult(**{**result.__dict__, "plot_path": plot_path})
    return result


def print_model_summary(model: LFitModel) -> None:
    print(f"fit_data = {model.data_source.resolve()}")
    print(f"fit rows = {model.row_count}")
    print(f"L13 target = {model.target_l13_column}")
    print(f"L56 target = {model.target_l56_column}")
    print(f"L13 formula = {_formula('L13', model.beta_l13)}")
    print(f"L56 formula = {_formula('L56', model.beta_l56)}")
    print(
        "valid range = "
        f"W[{model.input_ranges['W'][0]:g},{model.input_ranges['W'][1]:g}], "
        f"R[{model.input_ranges['R'][0]:g},{model.input_ranges['R'][1]:g}], "
        f"WlineR[{model.input_ranges['WlineR'][0]:g},{model.input_ranges['WlineR'][1]:g}]"
    )


def print_result(result: AnalysisResult) -> None:
    print(f"L13 = {result.L13_nH:.6g} nH")
    print(f"L56 = {result.L56_pH:.6g} pH")
    print(f"Geometry formula L_MA = {result.geometry_L_MA_um:.6g} um")
    print(f"Geometry formula L_E1 = {result.geometry_L_E1_um:.6g} um")
    print(f"Geometry formula L_GND_open = {result.geometry_L_GND_open_um:.6g} um")
    print(f"Wline = {result.Wline_um:.6g} um")
    print(f"W_GND_open = {result.W_GND_open_um:.6g} um")
    print(f"GND_width_factor = {result.GND_width_factor:.6g}")
    print(f"Cal_0511v3 effective Wline = {result.effective_Wline_um:.6g} um")
    print(f"Cal_0511v3 effective W_GND_open = {result.effective_W_GND_open_um:.6g} um")
    print(f"Cal_0511v3 effective line_length = {result.effective_line_length_um:.6g} um")
    if result.s4p_path is not None:
        print(f"s4p = {result.s4p_path}")
    if result.plot_path is not None:
        print(f"plot = {result.plot_path}")


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
        generate_s4p=not args.no_s4p,
        generate_plot=args.plot,
        quiet=not args.verbose,
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
                generate_s4p=not args.no_s4p,
                generate_plot=args.plot,
                quiet=not args.verbose,
            )
            print_result(result)
        except Exception as exc:
            print(f"Error: {exc}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF L-fit 20260512 + Cal_0511v3 S4P CLI.")
    parser.add_argument(
        "values",
        type=float,
        nargs="*",
        help="Optional one-shot input: W R WlineR. With no values, interactive mode starts.",
    )
    parser.add_argument("--fit-data", type=Path, default=DEFAULT_FIT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--freq-start-ghz", type=float, default=Cal_0511v3.FREQ_START_GHZ)
    parser.add_argument("--freq-stop-ghz", type=float, default=Cal_0511v3.FREQ_STOP_GHZ)
    parser.add_argument("--freq-npoints", type=int, default=Cal_0511v3.FREQ_NPOINTS)
    parser.add_argument("--round-decimals", type=int, default=2, help="Round fitted coefficients; use -1 for raw.")
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--no-s4p", action="store_true", help="Only print L/geometry; do not generate Cal_0511v3 S4P.")
    parser.add_argument("--plot", action="store_true", help="Write a per-case L/geometry/S4P diagnostic PNG.")
    parser.add_argument("--plot-fit-summary", action="store_true", help="Write actual-vs-fit summary plots for L13 and L56.")
    parser.add_argument("--print-model", action="store_true", help="Print the fitted formulas and input ranges.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if len(args.values) not in {0, 3}:
        parser.error("provide either no positional values, or exactly: W R WlineR")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model = load_l_fit_model(data_source=args.fit_data, round_decimals=args.round_decimals)
    if args.print_model:
        print_model_summary(model)
    if args.plot_fit_summary:
        paths = write_fit_summary_plots(model, args.output_dir)
        for path in paths:
            print(f"fit_plot = {path}")
    if args.values:
        return run_one_shot(args, model)
    return run_interactive(args, model)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Bilinear predictor for L13/L24/L56 from four W/R corner values.

Default corner values:
  W=90,  R=1.3: L13=1.37, L24=0.14, L56=0.013
  W=90,  R=1.7: L13=1.66, L24=0.21, L56=1.98
  W=120, R=1.3: L13=1.67, L24=0.23, L56=4.625
  W=120, R=1.7: L13=2.24, L24=0.37, L56=5.1
  L56delta_pH = 5.0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from VCL_length_calculator import calculate_compensation_msl


DEFAULT_L56DELTA_PH = 5.0
DEFAULT_EXTRAPOLATION_FRACTION = None

DEFAULT_CORNERS = [
    {"W": 90.0, "R": 1.3, "L13": 1.37, "L24": 0.14, "L56": 0.013},
    {"W": 90.0, "R": 1.7, "L13": 1.66, "L24": 0.21, "L56": 1.98},
    {"W": 120.0, "R": 1.3, "L13": 1.67, "L24": 0.23, "L56": 4.625},
    {"W": 120.0, "R": 1.7, "L13": 2.24, "L24": 0.37, "L56": 5.1},
]


def _float_from(row: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return float(row[name])
    raise KeyError(f"Missing required value; tried keys: {', '.join(names)}")


def _normalize_corner(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "W": _float_from(row, "W", "w"),
        "R": _float_from(row, "R", "r"),
        "L13": _float_from(row, "L13", "L13_nH"),
        "L24": _float_from(row, "L24", "L24_nH"),
        "L56": _float_from(row, "L56", "L56_pH", "L56_sym"),
    }


def load_corners(path: str | Path) -> tuple[list[dict[str, float]], float]:
    """Load corner values from JSON or CSV.

    JSON format:
      {
        "L56delta_pH": 5.0,
        "corners": [
          {"W": 90, "R": 1.3, "L13": 1.37, "L24": 0.14, "L56": 0.013},
          ...
        ]
      }

    CSV columns:
      W,R,L13,L24,L56
    Optional CSV column:
      L56delta_pH
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_corners = payload.get("corners", payload)
        delta = float(payload.get("L56delta_pH", DEFAULT_L56DELTA_PH)) if isinstance(payload, dict) else DEFAULT_L56DELTA_PH
    else:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            raw_corners = list(csv.DictReader(handle))
        delta_values = [
            float(row["L56delta_pH"])
            for row in raw_corners
            if row.get("L56delta_pH") not in (None, "")
        ]
        delta = delta_values[0] if delta_values else DEFAULT_L56DELTA_PH

    corners = [_normalize_corner(row) for row in raw_corners]
    return corners, delta


def build_corner_model(
    corners: list[Mapping[str, Any]],
    L56delta_pH: float = DEFAULT_L56DELTA_PH,
    extrapolation_fraction: float | None = DEFAULT_EXTRAPOLATION_FRACTION,
) -> dict[str, Any]:
    """Validate four corners and convert them to a rectangular bilinear model."""
    normalized = [_normalize_corner(row) for row in corners]
    if len(normalized) != 4:
        raise ValueError(f"Exactly four corners are required; got {len(normalized)}.")

    w_values = sorted({row["W"] for row in normalized})
    r_values = sorted({row["R"] for row in normalized})
    if len(w_values) != 2 or len(r_values) != 2:
        raise ValueError("Corners must contain exactly two W values and two R values.")

    w_min, w_max = w_values
    r_min, r_max = r_values
    enforce_range = extrapolation_fraction is not None
    if enforce_range:
        w_margin = (w_max - w_min) * float(extrapolation_fraction)
        r_margin = (r_max - r_min) * float(extrapolation_fraction)
        w_allowed_min = w_min - w_margin
        w_allowed_max = w_max + w_margin
        r_allowed_min = r_min - r_margin
        r_allowed_max = r_max + r_margin
    else:
        w_allowed_min = -float("inf")
        w_allowed_max = float("inf")
        r_allowed_min = -float("inf")
        r_allowed_max = float("inf")
    by_wr = {(row["W"], row["R"]): row for row in normalized}
    required = {
        "Wmin_Rmin": (w_min, r_min),
        "Wmin_Rmax": (w_min, r_max),
        "Wmax_Rmin": (w_max, r_min),
        "Wmax_Rmax": (w_max, r_max),
    }
    for wr in required.values():
        if wr not in by_wr:
            raise ValueError(f"Missing corner W={wr[0]:g}, R={wr[1]:g}.")

    parameters = {}
    for name in ("L13", "L24", "L56"):
        parameters[name] = {key: by_wr[wr][name] for key, wr in required.items()}

    return {
        "W_min": w_min,
        "W_max": w_max,
        "W_allowed_min": w_allowed_min,
        "W_allowed_max": w_allowed_max,
        "R_min": r_min,
        "R_max": r_max,
        "R_allowed_min": r_allowed_min,
        "R_allowed_max": r_allowed_max,
        "extrapolation_fraction": None if extrapolation_fraction is None else float(extrapolation_fraction),
        "enforce_range": enforce_range,
        "L56delta_pH": float(L56delta_pH),
        "parameters": parameters,
    }


def _check_inside_range(W: float, R: float, model: Mapping[str, Any]) -> None:
    if not model.get("enforce_range", True):
        return
    if not (model["W_allowed_min"] <= W <= model["W_allowed_max"]):
        raise ValueError(
            f"W={W:g} is outside the allowed extrapolation range "
            f"[{model['W_allowed_min']:g}, {model['W_allowed_max']:g}]."
        )
    if not (model["R_allowed_min"] <= R <= model["R_allowed_max"]):
        raise ValueError(
            f"R={R:g} is outside the allowed extrapolation range "
            f"[{model['R_allowed_min']:g}, {model['R_allowed_max']:g}]."
        )


def _bilinear_value(corners: Mapping[str, float], W: float, R: float, model: Mapping[str, Any]) -> float:
    t = (W - model["W_min"]) / (model["W_max"] - model["W_min"])
    u = (R - model["R_min"]) / (model["R_max"] - model["R_min"])
    return (
        (1.0 - t) * (1.0 - u) * corners["Wmin_Rmin"]
        + (1.0 - t) * u * corners["Wmin_Rmax"]
        + t * (1.0 - u) * corners["Wmax_Rmin"]
        + t * u * corners["Wmax_Rmax"]
    )


def predict_l_params(
    W: float,
    R: float,
    corners: list[Mapping[str, Any]] | None = None,
    L56delta_pH: float = DEFAULT_L56DELTA_PH,
) -> dict[str, float]:
    """Return interpolated L parameters for target W/R.

    W and R may be inside or outside the four-corner rectangle; values outside
    the rectangle are linearly extrapolated from the same bilinear model.
    """
    model = build_corner_model(corners or DEFAULT_CORNERS, L56delta_pH)
    W = float(W)
    R = float(R)
    _check_inside_range(W, R, model)

    params = model["parameters"]
    msl = calculate_compensation_msl(W, R)
    length = float(msl["L_um"])
    chamfer_side_length = float(msl["chamfer_side_length_um"])
    w_port = float(msl["w_port_um"])
    distance_port = float(msl["distance_port_um"])
    real_coupling_length1 = float(msl["real_coupling_length1_um"])
    is_extrapolated = not (model["W_min"] <= W <= model["W_max"] and model["R_min"] <= R <= model["R_max"])
    return {
        "W": W,
        "R": R,
        "L13": _bilinear_value(params["L13"], W, R, model),
        "L24": _bilinear_value(params["L24"], W, R, model),
        "L56": _bilinear_value(params["L56"], W, R, model),
        "L13_nH": _bilinear_value(params["L13"], W, R, model),
        "L24_nH": _bilinear_value(params["L24"], W, R, model),
        "L56_pH": _bilinear_value(params["L56"], W, R, model),
        "L56delta_pH": float(model["L56delta_pH"]),
        "length": length,
        "w_line": float(msl["W_MSL_um"]),
        "l_compensate_line": float(msl["L_MSL_um"]),
        "chamfer_side_length": chamfer_side_length,
        "w_port": w_port,
        "distance_port": distance_port,
        "real_coupling_length1": real_coupling_length1,
        "L_um": length,
        "length_um": length,
        "W_MSL_um": float(msl["W_MSL_um"]),
        "L_MSL_um": float(msl["L_MSL_um"]),
        "chamfer_side_length_um": chamfer_side_length,
        "w_port_um": w_port,
        "distance_port_um": distance_port,
        "real_coupling_length1_um": real_coupling_length1,
        "is_extrapolated": is_extrapolated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict L13/L24/L56 by bilinear interpolation from four W/R corners.")
    parser.add_argument("--W", type=float, default=None, help="Target W. May be up to 15% outside the corner W range.")
    parser.add_argument("--R", type=float, default=None, help="Target R. May be up to 15% outside the corner R range.")
    parser.add_argument("--corners", default=None, help="Optional JSON/CSV file with four corner L values.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of aligned text.")
    return parser.parse_args()


QUIT_TOKENS = {"", "q", "quit", "exit"}


def prompt_float(
    name: str,
    value: float | None,
    min_value: float,
    max_value: float,
) -> float | None:
    if value is not None:
        return float(value)

    while True:
        try:
            text = input(f"Enter {name} [{min_value:g} to {max_value:g}] (Enter/q to quit): ").strip()
        except EOFError:
            raise ValueError(f"{name} is required. Use --{name} <value>, for example --{name} {min_value:g}.")
        if text.lower() in QUIT_TOKENS:
            return None
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            print(f"{name} must be a number.")


def print_result(result: Mapping[str, float], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"W={result['W']:g}, R={result['R']:g}")
        print(f"L13_nH: {result['L13_nH']:.12g}")
        print(f"L24_nH: {result['L24_nH']:.12g}")
        print(f"L56_pH: {result['L56_pH']:.12g}")
        print(f"L56delta_pH: {result['L56delta_pH']:.12g}")
        print(f"length: {result['length']:.12g}")
        print(f"w_line: {result['w_line']:.12g}")
        print(f"l_compensate_line: {result['l_compensate_line']:.12g}")
        print(f"chamfer_side_length: {result['chamfer_side_length']:.12g}")
        print(f"w_port: {result['w_port']:.12g}")
        print(f"distance_port: {result['distance_port']:.12g}")
        print(f"real_coupling_length1_um: {result['real_coupling_length1_um']:.12g}")
        print(f"is_extrapolated: {result['is_extrapolated']}")


def run_one_prediction(
    W: float,
    R: float,
    corners: list[Mapping[str, Any]],
    delta: float,
    as_json: bool,
) -> int:
    try:
        result = predict_l_params(W, R, corners, delta)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print_result(result, as_json)
    return 0


def run_interactive(corners: list[Mapping[str, Any]], delta: float, as_json: bool) -> int:
    model = build_corner_model(corners, delta)
    print("Bilinear L predictor")
    print(f"Corner W range: {model['W_min']:g} to {model['W_max']:g}")
    print(f"Corner R range: {model['R_min']:g} to {model['R_max']:g}")
    print(f"Allowed W range with 15% extrapolation: {model['W_allowed_min']:g} to {model['W_allowed_max']:g}")
    print(f"Allowed R range with 15% extrapolation: {model['R_allowed_min']:g} to {model['R_allowed_max']:g}")
    print("After each result, enter another W/R pair. Press Enter or type q/quit/exit to quit.")
    print()

    while True:
        try:
            W = prompt_float("W", None, model["W_allowed_min"], model["W_allowed_max"])
            if W is None:
                print("Exit.")
                return 0
            R = prompt_float("R", None, model["R_allowed_min"], model["R_allowed_max"])
            if R is None:
                print("Exit.")
                return 0
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        run_one_prediction(W, R, corners, delta, as_json)
        print()


def main() -> int:
    args = parse_args()
    if args.corners:
        corners, delta = load_corners(args.corners)
    else:
        corners, delta = DEFAULT_CORNERS, DEFAULT_L56DELTA_PH

    model = build_corner_model(corners, delta)
    if args.W is None and args.R is None:
        return run_interactive(corners, delta, args.json)
    if args.W is None or args.R is None:
        print("error: --W and --R must be provided together, or omit both for interactive mode.", file=sys.stderr)
        return 2

    try:
        W = prompt_float("W", args.W, model["W_allowed_min"], model["W_allowed_max"])
        R = prompt_float("R", args.R, model["R_allowed_min"], model["R_allowed_max"])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if W is None or R is None:
        print("Exit.")
        return 0
    return run_one_prediction(W, R, corners, delta, args.json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

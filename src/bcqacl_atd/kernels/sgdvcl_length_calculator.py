#!/usr/bin/env python3
"""Interactive SG-DVCL length calculator from Half_TF W/R/WlineR.

Run:

    python sgdvcl_length_calculator.py

Then repeatedly enter:

    W R WlineR

Example:

    100 1.4 0.25

Type q/quit/exit to leave.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


# =============================================================================
# User-editable parameters
# =============================================================================

# Formula mode:
#   "geometry" uses the port/tap geometry formula proposed by the user.
#   "fitted" uses the previous 87-case fitted formula.
#   "both" prints both formulas and uses DEFAULT_OUTPUT_FORMULA for SG-DVCL L.
FORMULA_MODE = "geometry"
DEFAULT_OUTPUT_FORMULA = "geometry"

# Half_TF fixed geometry parameters, in um.
W_PORT_UM = 10.0
W_PORT_DISTANCE_UM = 34.0

# SG-DVCL MQ ground opening width rule:
#   W_GND_open = GND_WIDTH_FACTOR * Wline
GND_WIDTH_FACTOR = 1.0

# SG-DVCL derived length rules:
#   L_E1 = L_MA - 2 um
#   L_GND_open = L_MA - 4 um
E1_LENGTH_SHORTEN_UM = 2.0
GND_OPEN_LENGTH_SHORTEN_UM = 4.0

# Previous 87-case fitted formula:
#   L = A0 + A_W*W + A_WR*(W*R) + A_WLINE*Wline
FITTED_A0 = 9.31324580296369
FITTED_A_W = 1.0741567944957495
FITTED_A_WR = 1.1870660145994067
FITTED_A_WLINE = -3.2203383564426735


# =============================================================================
# Calculator
# =============================================================================


@dataclass(frozen=True)
class SgdvclLengthResult:
    W_um: float
    R: float
    WlineR: float
    Wline_um: float
    W_full_length_um: float
    W_plus_WR_um: float
    distance_open_um: float
    length_long_um: float
    length_tap_out_um: float
    length_tap_in_um: float
    length_tap_middle_um: float
    length_port_in_um: float
    length_port_out_um: float
    length_port_middle_um: float
    geometry_L_MA_um: float
    fitted_L_MA_um: float
    selected_formula: str
    selected_L_MA_um: float
    selected_L_E1_um: float
    selected_L_GND_open_um: float
    GND_width_factor: float
    W_GND_open_um: float


def geometry_formula_length_um(W_um: float, R: float, Wline_um: float) -> dict[str, float]:
    distance_open_um = W_PORT_DISTANCE_UM - W_PORT_UM
    length_long_um = W_um * R
    length_tap_out_um = (W_um - W_PORT_UM) / 2.0
    length_tap_in_um = (W_um - 2.0 * Wline_um) / 2.0
    length_tap_middle_um = (length_tap_out_um + length_tap_in_um) / 2.0
    length_port_in_um = (W_um - 2.0 * Wline_um - distance_open_um) / 2.0
    length_port_out_um = (W_um - distance_open_um - 2.0 * W_PORT_UM) / 2.0
    length_port_middle_um = (length_port_in_um + length_port_out_um) / 2.0
    geometry_L_MA_um = length_long_um + length_tap_middle_um + length_port_middle_um

    return {
        "distance_open_um": distance_open_um,
        "length_long_um": length_long_um,
        "length_tap_out_um": length_tap_out_um,
        "length_tap_in_um": length_tap_in_um,
        "length_tap_middle_um": length_tap_middle_um,
        "length_port_in_um": length_port_in_um,
        "length_port_out_um": length_port_out_um,
        "length_port_middle_um": length_port_middle_um,
        "geometry_L_MA_um": geometry_L_MA_um,
    }


def fitted_formula_length_um(W_um: float, R: float, Wline_um: float) -> float:
    return FITTED_A0 + FITTED_A_W * W_um + FITTED_A_WR * (W_um * R) + FITTED_A_WLINE * Wline_um


def normalize_formula_name(name: str) -> str:
    out = name.strip().lower().replace("-", "_")
    aliases = {
        "g": "geometry",
        "geo": "geometry",
        "geometry_formula": "geometry",
        "f": "fitted",
        "fit": "fitted",
        "fitted_formula": "fitted",
        "all": "both",
    }
    out = aliases.get(out, out)
    if out not in {"geometry", "fitted", "both"}:
        raise ValueError("formula must be geometry, fitted, or both.")
    return out


def selected_formula_name(mode: str) -> str:
    mode = normalize_formula_name(mode)
    if mode == "both":
        return normalize_formula_name(DEFAULT_OUTPUT_FORMULA)
    return mode


def calculate(W_um: float, R: float, WlineR: float, *, formula_mode: str = FORMULA_MODE) -> SgdvclLengthResult:
    W_um = float(W_um)
    R = float(R)
    WlineR = float(WlineR)
    if W_um <= 0.0:
        raise ValueError("W must be positive.")
    if R <= 0.0:
        raise ValueError("R must be positive.")
    if WlineR <= 0.0:
        raise ValueError("WlineR must be positive.")
    if GND_WIDTH_FACTOR <= 0.0:
        raise ValueError("GND_WIDTH_FACTOR must be positive.")

    Wline_um = W_um * WlineR
    geom = geometry_formula_length_um(W_um, R, Wline_um)
    fitted_L_MA_um = fitted_formula_length_um(W_um, R, Wline_um)
    formula = selected_formula_name(formula_mode)
    selected_L_MA_um = geom["geometry_L_MA_um"] if formula == "geometry" else fitted_L_MA_um
    if selected_L_MA_um <= GND_OPEN_LENGTH_SHORTEN_UM:
        raise ValueError(f"Calculated L_MA is too small: {selected_L_MA_um:g} um.")

    return SgdvclLengthResult(
        W_um=W_um,
        R=R,
        WlineR=WlineR,
        Wline_um=Wline_um,
        W_full_length_um=W_um * R,
        W_plus_WR_um=W_um + W_um * R,
        distance_open_um=geom["distance_open_um"],
        length_long_um=geom["length_long_um"],
        length_tap_out_um=geom["length_tap_out_um"],
        length_tap_in_um=geom["length_tap_in_um"],
        length_tap_middle_um=geom["length_tap_middle_um"],
        length_port_in_um=geom["length_port_in_um"],
        length_port_out_um=geom["length_port_out_um"],
        length_port_middle_um=geom["length_port_middle_um"],
        geometry_L_MA_um=geom["geometry_L_MA_um"],
        fitted_L_MA_um=fitted_L_MA_um,
        selected_formula=formula,
        selected_L_MA_um=selected_L_MA_um,
        selected_L_E1_um=selected_L_MA_um - E1_LENGTH_SHORTEN_UM,
        selected_L_GND_open_um=selected_L_MA_um - GND_OPEN_LENGTH_SHORTEN_UM,
        GND_width_factor=GND_WIDTH_FACTOR,
        W_GND_open_um=GND_WIDTH_FACTOR * Wline_um,
    )


def fmt_um(value: float) -> str:
    return f"{value:.6g} um"


def print_result(result: SgdvclLengthResult, *, formula_mode: str) -> None:
    mode = normalize_formula_name(formula_mode)
    print("")
    print("Input")
    print(f"  W              = {fmt_um(result.W_um)}")
    print(f"  R              = {result.R:.6g}")
    print(f"  WlineR         = {result.WlineR:.6g}")
    print(f"  Wline          = {fmt_um(result.Wline_um)}")
    print(f"  W*R            = {fmt_um(result.W_full_length_um)}")
    print(f"  W + W*R        = {fmt_um(result.W_plus_WR_um)}")
    print("")
    print("Geometry components")
    print(f"  distance_open       = {fmt_um(result.distance_open_um)}")
    print(f"  length_long         = {fmt_um(result.length_long_um)}")
    print(f"  length_tap_out      = {fmt_um(result.length_tap_out_um)}")
    print(f"  length_tap_in       = {fmt_um(result.length_tap_in_um)}")
    print(f"  length_tap_middle   = {fmt_um(result.length_tap_middle_um)}")
    print(f"  length_port_in      = {fmt_um(result.length_port_in_um)}")
    print(f"  length_port_out     = {fmt_um(result.length_port_out_um)}")
    print(f"  length_port_middle  = {fmt_um(result.length_port_middle_um)}")
    print("")
    if mode in {"geometry", "both"}:
        print(f"Geometry formula L_MA = {fmt_um(result.geometry_L_MA_um)}")
        print("  simplified: L_MA = W*R + W - Wline - 19.5")
    if mode in {"fitted", "both"}:
        print(f"Fitted formula L_MA   = {fmt_um(result.fitted_L_MA_um)}")
        print(
            "  formula: L_MA = "
            f"{FITTED_A0:.6g} + {FITTED_A_W:.6g}*W "
            f"+ {FITTED_A_WR:.6g}*(W*R) {FITTED_A_WLINE:.6g}*Wline"
        )
    print("")
    print(f"Selected formula      = {result.selected_formula}")
    print(f"SG-DVCL L_MA          = {fmt_um(result.selected_L_MA_um)}")
    print(f"SG-DVCL L_E1          = {fmt_um(result.selected_L_E1_um)}")
    print(f"SG-DVCL L_GND_open    = {fmt_um(result.selected_L_GND_open_um)}")
    print(f"GND_width_factor      = {result.GND_width_factor:.6g}")
    print(f"SG-DVCL W_GND_open    = {fmt_um(result.W_GND_open_um)}")
    print("")


def parse_line(text: str) -> tuple[float, float, float]:
    cleaned = text.replace(",", " ").strip()
    parts = cleaned.split()
    if len(parts) != 3:
        raise ValueError("Please enter exactly three numbers: W R WlineR.")
    return float(parts[0]), float(parts[1]), float(parts[2])


def interactive_loop(formula_mode: str) -> None:
    print("SG-DVCL length calculator")
    print("Enter: W R WlineR")
    print("Example: 100 1.4 0.25")
    print("Type q/quit/exit to leave.")
    print("")
    while True:
        try:
            text = input("W R WlineR > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            break
        if not text:
            continue
        if text.lower() in {"q", "quit", "exit"}:
            break
        try:
            W_um, R, WlineR = parse_line(text)
            print_result(calculate(W_um, R, WlineR, formula_mode=formula_mode), formula_mode=formula_mode)
        except Exception as exc:
            print(f"Error: {exc}")
            print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formula",
        default=FORMULA_MODE,
        choices=("geometry", "fitted", "both"),
        help="Formula to print/use. Default comes from FORMULA_MODE at the top of the file.",
    )
    parser.add_argument("--W", type=float, help="One-shot W in um.")
    parser.add_argument("--R", type=float, help="One-shot R.")
    parser.add_argument("--WlineR", type=float, help="One-shot WlineR.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    one_shot = args.W is not None or args.R is not None or args.WlineR is not None
    if one_shot:
        if args.W is None or args.R is None or args.WlineR is None:
            raise SystemExit("--W, --R, and --WlineR must be provided together.")
        print_result(calculate(args.W, args.R, args.WlineR, formula_mode=args.formula), formula_mode=args.formula)
        return 0
    interactive_loop(args.formula)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

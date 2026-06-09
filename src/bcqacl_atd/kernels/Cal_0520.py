#!/usr/bin/env python3
"""Cal_0520 SG-DVCL S4P generator for the L13-only six-port TF model.

The half-transformer geometry is mapped to a straight SG-DVCL section with the
0520 length rule requested for the new residual-compensation experiment:

    Wline_SGDVCL = W * WlineR
    L_base       = W * R + W - 2 * Wline_SGDVCL - 12

The exported SG-DVCL geometry then uses

    L_MA         = L_base
    L_E1         = L_base - 2 um
    L_GND_open   = L_base - 4 um
    W_GND_open   = 1.7 * Wline_SGDVCL

and the modal line solver applies ``line_length_scale = 1.05`` internally.
Touchstone port order is fixed to the HFSS/SG-DVCL order:

    [E1_A, MA_B, E1_B, MA_A]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import skrf as rf

import Cal_0509


# =============================================================================
# Cal_0520 half_TF-to-SG-DVCL parameters
# =============================================================================

# Updated by optimize_cal_0520_l13only_tf_s6p.py from the six-port golden-v4
# endpoint fit.  Geometry and L13 remain fixed by the Cal_0520/tf_analysis
# 0520 policy; only numeric/mesh parameters below were optimized.

OPT_LINE_LENGTH_REFERENCE = "MA"
OPT_LINE_LENGTH_OFFSET_UM = 0.0
OPT_LINE_LENGTH_SCALE = 1.05

OPT_WLINE_SCALE = 1.0
OPT_WLINE_OFFSET_UM = 0.0

OPT_GND_WIDTH_SCALE = 1.0
OPT_GND_WIDTH_OFFSET_UM = 0.0
OPT_GND_WIDTH_FACTOR = 1.7

OPT_MIN_GND_OPEN_MARGIN_UM = 1e-3

OPT_M_MODES = 6
OPT_QUADRATURE_ORDER = 8
OPT_OUTER_DIRICHLET_PROJECTION_MODE = "x_seg"
OPT_TARGETED_REGION_SUBDIVISION_COUNTS = {
    1: 2,
    2: 2,
    3: 2,
    4: 2,
    5: 2,
}
OPT_ENABLE_DEFECT_EDGE_REFINEMENT = True
OPT_DEFECT_EDGE_REFINEMENT_SCALE = "eighth_line"
OPT_DEFECT_EDGE_REFINEMENT_STEPS_EACH_SIDE = 1
OPT_ENABLE_TARGETED_SLAB_Y_REFINEMENT = False
OPT_TARGETED_SLAB_Y_SPLIT_COUNTS = {
    "slab_b3": 1,
    "slab_b2": 1,
    "slab_b1": 1,
    "slab0": 1,
    "slab1": 1,
    "slab2": 1,
    "slab3": 1,
    "slab4": 1,
    "slab5": 1,
}

CAL_0520_HALF_TF_BASELINE_PARAMS: dict[str, Any] = {
    "line_length_reference": OPT_LINE_LENGTH_REFERENCE,
    "line_length_offset_um": OPT_LINE_LENGTH_OFFSET_UM,
    "line_length_scale": OPT_LINE_LENGTH_SCALE,
    "WLine_scale": OPT_WLINE_SCALE,
    "WLine_offset_um": OPT_WLINE_OFFSET_UM,
    "GND_width_scale": OPT_GND_WIDTH_SCALE,
    "GND_width_offset_um": OPT_GND_WIDTH_OFFSET_UM,
    "min_gnd_open_margin_um": OPT_MIN_GND_OPEN_MARGIN_UM,
    "M_modes": OPT_M_MODES,
    "quadrature_order": OPT_QUADRATURE_ORDER,
    "outer_dirichlet_projection_mode": OPT_OUTER_DIRICHLET_PROJECTION_MODE,
    "targeted_region_subdivision_counts": OPT_TARGETED_REGION_SUBDIVISION_COUNTS,
    "enable_defect_edge_refinement": OPT_ENABLE_DEFECT_EDGE_REFINEMENT,
    "defect_edge_refinement_scale": OPT_DEFECT_EDGE_REFINEMENT_SCALE,
    "defect_edge_refinement_steps_each_side": OPT_DEFECT_EDGE_REFINEMENT_STEPS_EACH_SIDE,
    "enable_targeted_slab_y_refinement": OPT_ENABLE_TARGETED_SLAB_Y_REFINEMENT,
    "targeted_slab_y_split_counts": OPT_TARGETED_SLAB_Y_SPLIT_COUNTS,
}

HALF_TF_FIXED_GEOMETRY_PARAMS = {
    "WLine_scale": OPT_WLINE_SCALE,
    "WLine_offset_um": OPT_WLINE_OFFSET_UM,
    "line_length_scale": OPT_LINE_LENGTH_SCALE,
    "line_length_offset_um": OPT_LINE_LENGTH_OFFSET_UM,
    "GND_width_scale": OPT_GND_WIDTH_SCALE,
    "GND_width_offset_um": OPT_GND_WIDTH_OFFSET_UM,
}

INPUT_SGDVCL_LENGTH_UM = 130.5
INPUT_SGDVCL_WIDTH_UM = 30.0
INPUT_GND_WIDTH_FACTOR = OPT_GND_WIDTH_FACTOR

OUTPUT_S4P_PATH = Path(
    r"."
    r"\PA Design\202409Tape_out\Paper writing\Code\V-MCLIN_Cal"
    r"\Cal_0520_outputs\Cal_0520_SGDVCL.s4p"
)

FREQ_START_GHZ = 1.0
FREQ_STOP_GHZ = 200.0
FREQ_NPOINTS = 331


def _validate_half_tf_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return Cal_0520 params after enforcing the fixed 0520 geometry policy."""
    checked = dict(params)
    if "targeted_region_subdivision_counts" in checked and checked["targeted_region_subdivision_counts"] is not None:
        checked["targeted_region_subdivision_counts"] = {
            int(k): int(v) for k, v in dict(checked["targeted_region_subdivision_counts"]).items()
        }
    if "targeted_slab_y_split_counts" in checked and checked["targeted_slab_y_split_counts"] is not None:
        checked["targeted_slab_y_split_counts"] = {
            str(k): int(v) for k, v in dict(checked["targeted_slab_y_split_counts"]).items()
        }
    for key, fixed_value in HALF_TF_FIXED_GEOMETRY_PARAMS.items():
        value = float(checked.get(key, fixed_value))
        if abs(value - float(fixed_value)) > 1e-15:
            raise ValueError(f"{key} is fixed at {fixed_value:g} for Cal_0520.")
        checked[key] = float(fixed_value)
    return checked


def half_tf_geometry_from_formula(W_um: float, R: float, WlineR: float) -> dict[str, float]:
    """Derive the Cal_0520 SG-DVCL base geometry from transformer dimensions."""
    W_um = float(W_um)
    R = float(R)
    WlineR = float(WlineR)
    Wline_SGDVCL_um = W_um * WlineR
    L_base_um = W_um * R + W_um - 2.0 * Wline_SGDVCL_um - 12.0
    if Wline_SGDVCL_um <= 0.0:
        raise ValueError("Wline_SGDVCL must be positive.")
    if L_base_um <= 4.0:
        raise ValueError(
            "Cal_0520 L_base must exceed 4 um so L_E1 and L_GND_open stay positive; "
            f"got L_base={L_base_um:g} um."
        )
    return {
        "W_um": W_um,
        "R": R,
        "WlineR": WlineR,
        "derived_WLine_um": Wline_SGDVCL_um,
        "Wline_SGDVCL_um": Wline_SGDVCL_um,
        "derived_SGDVCL_length_um": L_base_um,
        "L_base_um": L_base_um,
        "L_MA_um": L_base_um,
        "L_E1_um": L_base_um - 2.0,
        "L_GND_open_um": L_base_um - 4.0,
        "GND_width_factor": OPT_GND_WIDTH_FACTOR,
        "W_GND_open_um": OPT_GND_WIDTH_FACTOR * Wline_SGDVCL_um,
        "line_length_scale": OPT_LINE_LENGTH_SCALE,
        "effective_line_length_um": OPT_LINE_LENGTH_SCALE * L_base_um,
    }


def build_frequency_hz(
    *,
    freq_start_ghz: float = FREQ_START_GHZ,
    freq_stop_ghz: float = FREQ_STOP_GHZ,
    freq_npoints: int = FREQ_NPOINTS,
) -> tuple[float, ...]:
    """Return a GHz linspace converted to Hz."""
    if int(freq_npoints) < 2:
        raise ValueError("freq_npoints must be at least 2.")
    step = (float(freq_stop_ghz) - float(freq_start_ghz)) / float(int(freq_npoints) - 1)
    return tuple((float(freq_start_ghz) + idx * step) * 1e9 for idx in range(int(freq_npoints)))


def calculate_sgdvcl_from_length_width(
    *,
    length_um: float,
    width_um: float,
    gnd_width_factor: float = INPUT_GND_WIDTH_FACTOR,
    output_path: Path = OUTPUT_S4P_PATH,
    freq_start_ghz: float = FREQ_START_GHZ,
    freq_stop_ghz: float = FREQ_STOP_GHZ,
    freq_npoints: int = FREQ_NPOINTS,
    params: dict[str, Any] | None = None,
    quiet: bool = True,
) -> rf.Network:
    """Calculate one SG-DVCL S4P from a direct length and width."""
    length_um = float(length_um)
    width_um = float(width_um)
    gnd_width_factor = float(gnd_width_factor)
    output_path = Path(output_path)

    fit_params = dict(CAL_0520_HALF_TF_BASELINE_PARAMS)
    fit_params.update(params or {})
    fit_params = _validate_half_tf_params(fit_params)
    fit_params.update(
        {
            "frequency_hz": build_frequency_hz(
                freq_start_ghz=freq_start_ghz,
                freq_stop_ghz=freq_stop_ghz,
                freq_npoints=freq_npoints,
            ),
            "output_dir": str(output_path.parent),
            "filename": output_path.name,
            "quiet": bool(quiet),
            "write_manifest": True,
        }
    )

    return Cal_0509.calculate_sgdvcl_s4p(
        WLine_um=width_um,
        L_MA_um=length_um,
        L_E1_um=length_um - 2.0,
        L_GND_open_um=length_um - 4.0,
        W_GND_open_um=gnd_width_factor * width_um,
        GND_width_factor=gnd_width_factor,
        params=fit_params,
    )


def calculate_sgdvcl_s4p_from_geometry(
    WLine_um: float,
    L_MA_um: float,
    L_E1_um: float,
    L_GND_open_um: float,
    W_GND_open_um: float,
    params: dict[str, Any] | None = None,
) -> rf.Network:
    """Calculate a four-port SG-DVCL network from explicit base geometry."""
    fit_params = dict(CAL_0520_HALF_TF_BASELINE_PARAMS)
    fit_params.update(params or {})
    fit_params = _validate_half_tf_params(fit_params)
    output_dir = Path(fit_params.pop("output_dir", OUTPUT_S4P_PATH.parent))
    filename = fit_params.pop("filename", OUTPUT_S4P_PATH.name)
    if "frequency_hz" not in fit_params and "freq_list_Hz" not in fit_params:
        fit_params["frequency_hz"] = build_frequency_hz(
            freq_start_ghz=float(fit_params.pop("freq_start_ghz", FREQ_START_GHZ)),
            freq_stop_ghz=float(fit_params.pop("freq_stop_ghz", FREQ_STOP_GHZ)),
            freq_npoints=int(fit_params.pop("freq_npoints", FREQ_NPOINTS)),
        )
    fit_params.setdefault("quiet", True)
    fit_params.setdefault("write_manifest", True)

    return Cal_0509.calculate_sgdvcl_s4p(
        WLine_um=float(WLine_um),
        L_MA_um=float(L_MA_um),
        L_E1_um=float(L_E1_um),
        L_GND_open_um=float(L_GND_open_um),
        W_GND_open_um=float(W_GND_open_um),
        GND_width_factor=float(W_GND_open_um) / float(WLine_um),
        params={
            **fit_params,
            "output_dir": str(output_dir),
            "filename": filename,
        },
    )


def calculate_sgdvcl_s4p_from_half_tf(
    W_um: float,
    R: float,
    WlineR: float,
    params: dict[str, Any] | None = None,
) -> rf.Network:
    """Calculate a Cal_0520 half-TF-derived straight SG-DVCL four-port."""
    geometry = half_tf_geometry_from_formula(W_um, R, WlineR)
    return calculate_sgdvcl_s4p_from_geometry(
        WLine_um=geometry["derived_WLine_um"],
        L_MA_um=geometry["L_MA_um"],
        L_E1_um=geometry["L_E1_um"],
        L_GND_open_um=geometry["L_GND_open_um"],
        W_GND_open_um=geometry["W_GND_open_um"],
        params=params,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one Cal_0520 SG-DVCL S4P.")
    parser.add_argument("--length-um", type=float, default=INPUT_SGDVCL_LENGTH_UM, help="MA line length in um.")
    parser.add_argument("--width-um", type=float, default=INPUT_SGDVCL_WIDTH_UM, help="SG-DVCL line width in um.")
    parser.add_argument(
        "--gnd-width-factor",
        type=float,
        default=INPUT_GND_WIDTH_FACTOR,
        help="Ground opening width factor, W_GND_open = factor * WLine.",
    )
    parser.add_argument("--output-path", type=Path, default=OUTPUT_S4P_PATH, help="Output .s4p path.")
    parser.add_argument("--freq-start-ghz", type=float, default=FREQ_START_GHZ)
    parser.add_argument("--freq-stop-ghz", type=float, default=FREQ_STOP_GHZ)
    parser.add_argument("--freq-npoints", type=int, default=FREQ_NPOINTS)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    network = calculate_sgdvcl_from_length_width(
        length_um=args.length_um,
        width_um=args.width_um,
        gnd_width_factor=args.gnd_width_factor,
        output_path=args.output_path,
        freq_start_ghz=args.freq_start_ghz,
        freq_stop_ghz=args.freq_stop_ghz,
        freq_npoints=args.freq_npoints,
        quiet=not args.verbose,
    )
    print(args.output_path)
    print(
        "Generated Cal_0520 SG-DVCL S4P: "
        f"WLine={args.width_um:g} um, L_MA={args.length_um:g} um, "
        f"L_E1={args.length_um - 2.0:g} um, "
        f"L_GND_open={args.length_um - 4.0:g} um, "
        f"W_GND_open={args.gnd_width_factor * args.width_um:g} um, "
        f"line_length_scale={OPT_LINE_LENGTH_SCALE:g}, "
        f"nports={network.nports}, nfreq={len(network.frequency.f)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

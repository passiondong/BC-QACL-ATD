#!/usr/bin/env python3
"""Cal_0511v3 SG-DVCL S4P generator for six-port TF endpoint fitting.

Edit the user input block below, or pass ``--length-um`` and ``--width-um``
from the command line.  The script writes one Touchstone S4P file for the
vertical SG-DVCL geometry:

    MA length      = L
    E1 length      = L - 2 um
    MQ open length = L - 4 um
    MQ open width  = GND_width_factor * W

Touchstone port order is fixed to the HFSS export order:
    [E1_A, MA_B, E1_B, MA_A]

For half_TF fitting, the honest base geometry comes from
``sgdvcl_length_calculator.calculate(..., formula_mode="geometry")``:

    WLine = W * WlineR
    L_MA  = W*R + W - WLine - 19.5

Only the globally allowed scale factors may adjust this base geometry in the
solver:

    WLine_scale in [0.9, 1.1]
    line_length_scale in [0.9, 1.1]
    GND_width_scale in [1.0, 1.7]

All geometry offsets remain fixed at zero in the half_TF wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import skrf as rf

import Cal_0509
import sgdvcl_length_calculator as length_calc


# =============================================================================
# Final Cal_0511 half_TF fitting parameters
# =============================================================================

# These values are synchronized from the L20 endpoint six-port fit:
# outputs/cal_0511v3_six_port_tf_lfit_20pct_no_R0p5/best_cal_0511v3_l20_params.json
#
# Geometry source remains honest:
#   WLine_base = W * WlineR
#   L_MA_base  = W*R + W - WLine_base - 19.5
# from sgdvcl_length_calculator.calculate(..., formula_mode="geometry").
# The three scale factors below are global model-calibration factors explicitly
# allowed for the half_TF-to-straight-SG-DVCL comparison, not per-case tweaks.

# Propagation length reference.  MA is the physical long upper conductor in the
# straight SG-DVCL equivalent, so the modal line length follows L_MA rather than
# the shortened E1 or MQ opening.
OPT_LINE_LENGTH_REFERENCE = "MA"

# Geometry offsets are fixed at zero.  The old Cal_0510 +1 um length correction
# was an SG-DVCL-golden empirical extension and is not part of the half_TF
# geometric formula.
OPT_LINE_LENGTH_OFFSET_UM = 0.0

# Final search kept the derived half_TF length unchanged.  Allowed range was
# [0.9, 1.1], but the best strong-region objective selected 1.0.
OPT_LINE_LENGTH_SCALE = 1.0

# Final global line-width calibration from the R=0.5-excluded L20 endpoint fit.
# The balanced best selected the lower allowed bound.
OPT_WLINE_SCALE = 0.9
OPT_WLINE_OFFSET_UM = 0.0

# Final global MQ-opening calibration.  The search selected the upper allowed
# bound, meaning the straight-line model needs a wider effective ground opening
# to mimic the half_TF return/coupling environment.
OPT_GND_WIDTH_SCALE = 1.7
OPT_GND_WIDTH_OFFSET_UM = 0.0

# Numerical guard only: if a user gives too small a GND opening, keep it barely
# wider than WLine so the electrostatic decomposition remains valid.
OPT_MIN_GND_OPEN_MARGIN_UM = 1e-3

# Keep five modal terms.  Lower-mode ablation worsened the smoke objective by
# about 28%, so the extra modal content is still needed for coupled/through
# paths over 1-110 GHz.
OPT_M_MODES = 5

# Gauss-Legendre quadrature order.  Order 8 was sufficient after the x-mesh
# refinements; higher orders did not improve the selected objective enough to
# justify extra runtime.
OPT_QUADRATURE_ORDER = 8

# Per-x-segment outer Dirichlet projection won in the half_TF search.  This is
# more local than x_full and better matched the finite half_TF environment for
# the final straight-line approximation.
OPT_OUTER_DIRICHLET_PROJECTION_MODE = "x_seg"

# The endpoint L20 fit keeps the metal/opening center split while removing the
# far outer dielectric split in regions 1 and 5.
OPT_TARGETED_REGION_SUBDIVISION_COUNTS = {
    1: 1,  # Left far dielectric region.
    2: 2,  # Left opening-edge transition region: captures fringing variation.
    3: 2,  # Metal/opening center region: lets conductor charge split left/right.
    4: 2,  # Right opening-edge transition region: captures fringing variation.
    5: 1,  # Right far dielectric region.
}

# Defect/opening edge refinement is enabled for the endpoint L20 fit.
OPT_ENABLE_DEFECT_EDGE_REFINEMENT = True

# The endpoint fit selected a light eighth-line edge refinement.
OPT_DEFECT_EDGE_REFINEMENT_SCALE = "eighth_line"

# Kept for reproducibility if the switch above is enabled in experiments.
OPT_DEFECT_EDGE_REFINEMENT_STEPS_EACH_SIDE = 1

# Targeted slab-y refinement is disabled.  The expanded search tested Cal_0423's
# default slab2=3/slab4=3 plus 2/2, 4/4, 2/3, 3/2, single-slab and adjacent-slab
# variants; all worsened the smoke objective versus the simpler vertical stack.
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

# L20 endpoint compensation used by the six-port prediction helper.  These
# values are not part of the SG-DVCL S4P generator itself; they document the
# paired optimized inductance correction for Cal_0511v3 endpoint predictions.
OPT_L13_COMPENSATION_DELTA_PCT = 0.0
OPT_L56_COMPENSATION_DELTA_PCT = 20.0


CAL_0511_HALF_TF_BASELINE_PARAMS = {
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

# Backward-compatible alias for older local callers that imported the old name
# after copying Cal_0510.py.
CAL_0510_OPTIMIZED_PARAMS = CAL_0511_HALF_TF_BASELINE_PARAMS

HALF_TF_ALLOWED_SCALE_BOUNDS = {
    "WLine_scale": (0.9, 1.1),
    "line_length_scale": (0.9, 1.1),
    "GND_width_scale": (1.0, 1.7),
}

HALF_TF_FIXED_GEOMETRY_OFFSETS = {
    "WLine_offset_um": 0.0,
    "line_length_offset_um": 0.0,
    "GND_width_offset_um": 0.0,
}


# =============================================================================
# User entry: one-off SG-DVCL size and output
# =============================================================================

# These values are only for command-line one-off generation.  The half_TF wrapper
# ignores them and derives WLine/L_MA from W, R, WlineR using the formula above.

# SG-DVCL MA line length L in um.  The script derives E1=L-2 and MQ opening=L-4.
INPUT_SGDVCL_LENGTH_UM = 130.5

# SG-DVCL coupled-line width WLine in um.  Both MA and E1 metal widths use this W.
INPUT_SGDVCL_WIDTH_UM = 30

# MQ ground-opening width factor for one-off direct geometry generation.
INPUT_GND_WIDTH_FACTOR = 1.0

# Output Touchstone file requested for this Cal_0511v3 entry script.
OUTPUT_S4P_PATH = Path(
    r"."
    r"\PA Design\202409Tape_out\Paper writing\Code\V-MCLIN_Cal"
    r"\Cal_0511v3_outputs\Cal_0511v3_SGDVCL.s4p"
)

# Frequency sweep for the delivered S4P.  The fit objective used 1-110 GHz,
# while plots/evaluation were kept to 200 GHz; this file therefore exports
# the full 1-200 GHz band.
FREQ_START_GHZ = 1.0
FREQ_STOP_GHZ = 200.0
FREQ_NPOINTS = 331


def _validate_half_tf_params(params: dict) -> dict:
    """Return half_TF params after enforcing allowed scale and offset rules."""
    checked = dict(params)
    if "targeted_region_subdivision_counts" in checked and checked["targeted_region_subdivision_counts"] is not None:
        checked["targeted_region_subdivision_counts"] = {
            int(k): int(v) for k, v in dict(checked["targeted_region_subdivision_counts"]).items()
        }
    if "targeted_slab_y_split_counts" in checked and checked["targeted_slab_y_split_counts"] is not None:
        checked["targeted_slab_y_split_counts"] = {
            str(k): int(v) for k, v in dict(checked["targeted_slab_y_split_counts"]).items()
        }
    for key, fixed_value in HALF_TF_FIXED_GEOMETRY_OFFSETS.items():
        value = float(checked.get(key, fixed_value))
        if abs(value - fixed_value) > 1e-15:
            raise ValueError(f"{key} is fixed at {fixed_value:g} for Cal_0511 half_TF fitting.")
        checked[key] = fixed_value
    for key, (lo, hi) in HALF_TF_ALLOWED_SCALE_BOUNDS.items():
        value = float(checked.get(key, 1.0))
        if value < lo or value > hi:
            raise ValueError(f"{key}={value:g} is outside the allowed half_TF range [{lo:g}, {hi:g}].")
        checked[key] = value
    return checked


def half_tf_geometry_from_formula(W_um: float, R: float, WlineR: float) -> dict[str, float]:
    """Derive SG-DVCL base geometry from the formal half_TF geometry formula.

    Source: ``sgdvcl_length_calculator.calculate`` with ``formula_mode="geometry"``,
    which is also used by ``tf_analysis_pipeline_cli.py``.
    """
    result = length_calc.calculate(float(W_um), float(R), float(WlineR), formula_mode="geometry")
    return {
        "W_um": result.W_um,
        "R": result.R,
        "WlineR": result.WlineR,
        "derived_WLine_um": result.Wline_um,
        "derived_SGDVCL_length_um": result.geometry_L_MA_um,
        "L_MA_um": result.geometry_L_MA_um,
        "L_E1_um": result.selected_L_E1_um,
        "L_GND_open_um": result.selected_L_GND_open_um,
        "GND_width_factor": result.GND_width_factor,
        "W_GND_open_um": result.W_GND_open_um,
        "distance_open_um": result.distance_open_um,
        "length_long_um": result.length_long_um,
        "length_tap_middle_um": result.length_tap_middle_um,
        "length_port_middle_um": result.length_port_middle_um,
    }


def build_frequency_hz(
    *,
    freq_start_ghz: float = FREQ_START_GHZ,
    freq_stop_ghz: float = FREQ_STOP_GHZ,
    freq_npoints: int = FREQ_NPOINTS,
) -> tuple[float, ...]:
    """Return a GHz linspace converted to Hz."""
    if freq_npoints < 2:
        raise ValueError("freq_npoints must be at least 2.")
    step = (float(freq_stop_ghz) - float(freq_start_ghz)) / float(freq_npoints - 1)
    return tuple((float(freq_start_ghz) + idx * step) * 1e9 for idx in range(freq_npoints))


def calculate_sgdvcl_from_length_width(
    *,
    length_um: float,
    width_um: float,
    gnd_width_factor: float = INPUT_GND_WIDTH_FACTOR,
    output_path: Path = OUTPUT_S4P_PATH,
    freq_start_ghz: float = FREQ_START_GHZ,
    freq_stop_ghz: float = FREQ_STOP_GHZ,
    freq_npoints: int = FREQ_NPOINTS,
    params: dict | None = None,
    quiet: bool = True,
) -> rf.Network:
    """Calculate one SG-DVCL S4P from the user-facing length and width.

    Parameters
    ----------
    length_um:
        HFSS filename-style L, i.e. MA length in um.
    width_um:
        SG-DVCL line width WLine in um.
    gnd_width_factor:
        MQ opening width factor.  W_GND_open = gnd_width_factor * width_um.
    output_path:
        Full path of the Touchstone S4P to write.
    """
    length_um = float(length_um)
    width_um = float(width_um)
    gnd_width_factor = float(gnd_width_factor)
    output_path = Path(output_path)

    fit_params = dict(CAL_0511_HALF_TF_BASELINE_PARAMS)
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
    params: dict | None = None,
) -> rf.Network:
    """Calculate a 4-port SG-DVCL network from explicit base geometry."""
    fit_params = dict(CAL_0511_HALF_TF_BASELINE_PARAMS)
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
    params: dict | None = None,
) -> rf.Network:
    """Calculate a half_TF-derived straight SG-DVCL 4-port network.

    The base ``WLine`` and ``L_MA`` are derived from the formal half_TF formula.
    Any allowed scale factors are supplied through ``params`` and validated by
    ``_validate_half_tf_params``.
    """
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
    parser = argparse.ArgumentParser(description="Generate one Cal_0511v3 SG-DVCL S4P.")
    parser.add_argument("--length-um", type=float, default=INPUT_SGDVCL_LENGTH_UM, help="MA line length L in um.")
    parser.add_argument("--width-um", type=float, default=INPUT_SGDVCL_WIDTH_UM, help="SG-DVCL line width WLine in um.")
    parser.add_argument(
        "--gnd-width-factor",
        type=float,
        default=INPUT_GND_WIDTH_FACTOR,
        help="MQ opening width factor, W_GND_open = factor * WLine.",
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
        "Generated Cal_0511v3 SG-DVCL S4P: "
        f"WLine={args.width_um:g} um, L_MA={args.length_um:g} um, "
        f"L_E1={args.length_um - 2.0:g} um, "
        f"L_GND_open={args.length_um - 4.0:g} um, "
        f"W_GND_open={args.gnd_width_factor * args.width_um:g} um, "
        f"nports={network.nports}, nfreq={len(network.frequency.f)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

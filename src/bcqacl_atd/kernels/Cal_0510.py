#!/usr/bin/env python3
"""Cal_0510 SG-DVCL S4P generator.

Edit the user input block below, or pass ``--length-um`` and ``--width-um``
from the command line.  The script writes one Touchstone S4P file for the
vertical SG-DVCL geometry:

    MA length      = L
    E1 length      = L - 2 um
    MQ open length = L - 4 um
    MQ open width  = GND_width_factor * W

Touchstone port order is fixed to the HFSS export order:
    [E1_A, MA_B, E1_B, MA_A]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import skrf as rf

import Cal_0509


# =============================================================================
# User entry: SG-DVCL size and output
# =============================================================================

# SG-DVCL MA line length L in um.  The script derives E1=L-2 and MQ opening=L-4.
INPUT_SGDVCL_LENGTH_UM = 115.5

# SG-DVCL coupled-line width WLine in um.  Both MA and E1 metal widths use this W.
INPUT_SGDVCL_WIDTH_UM = 27

# MQ ground-opening width factor.  W_GND_open = GND_width_factor * WLine.
# Keep this explicit because the HFSS golden set swept 1.0, 1.3, 1.5, and 1.7.
INPUT_GND_WIDTH_FACTOR = 1.0

# Output Touchstone file requested for this Cal_0510 entry script.
OUTPUT_S4P_PATH = Path(
    r"."
    r"\PA Design\202409Tape_out\Paper writing\Code\V-MCLIN_Cal"
    r"\Cal_0510_outputs\Cal_0510_SGDVCL.s4p"
)

# Frequency sweep for the delivered S4P.  The fit objective used 1-110 GHz,
# while plots/evaluation were kept to 200 GHz; this file therefore exports
# the full 1-200 GHz band.
FREQ_START_GHZ = 1.0
FREQ_STOP_GHZ = 200.0
FREQ_NPOINTS = 331


# =============================================================================
# Optimized Cal_0510 parameters from the 140-case HFSS fit
# =============================================================================

# Use MA length as the propagation reference; physically, the optimized model
# follows the upper MA conductor length rather than E1 or MQ opening length.
OPT_LINE_LENGTH_REFERENCE = "MA"

# Add 1 um effective length to represent end/fringing-field electrical extension.
OPT_LINE_LENGTH_OFFSET_UM = 1.0

# Keep length scale at 1.0; no global stretch beyond the explicit +1 um offset.
OPT_LINE_LENGTH_SCALE = 1.0

# Keep metal width unchanged; the fit did not need an artificial WLine correction.
OPT_WLINE_SCALE = 1.0
OPT_WLINE_OFFSET_UM = 0.0

# Keep MQ opening width unchanged; do not fake accuracy by widening/narrowing GND.
OPT_GND_WIDTH_SCALE = 1.0
OPT_GND_WIDTH_OFFSET_UM = 0.0

# Numerical guard only: if a user gives too small a GND opening, keep it barely
# wider than WLine so the electrostatic decomposition remains valid.
OPT_MIN_GND_OPEN_MARGIN_UM = 1e-3

# Use 5 modes instead of the original 3; this was the main accuracy improvement
# for coupled/through S-parameters, at the cost of about 1.5x runtime.
OPT_M_MODES = 5

# Use Gauss-Legendre order 8 instead of original 12; the fit showed order 8 was
# enough after the mesh updates and helped recover runtime.
OPT_QUADRATURE_ORDER = 8

# Project the outer Dirichlet boundary on the full x span instead of per x-cell;
# this gives the far boundary a more global constraint and fitted HFSS better.
OPT_OUTER_DIRICHLET_PROJECTION_MODE = "x_full"

# Split each coarse lateral x region into two.  Region 3 is the metal/opening
# center region; its center cut was important.  Regions 1 and 5 are far outer
# dielectric regions; those cuts help modestly but are less physically critical.
OPT_TARGETED_REGION_SUBDIVISION_COUNTS = {
    1: 2,  # Left far dielectric region: improves global potential projection.
    2: 2,  # Left opening-edge transition region: captures fringing variation.
    3: 2,  # Metal/opening center region: lets conductor charge split left/right.
    4: 2,  # Right opening-edge transition region: captures fringing variation.
    5: 2,  # Right far dielectric region: improves global potential projection.
}

# Add x cuts around MQ defect/opening edges; this captures strong edge fringing.
OPT_ENABLE_DEFECT_EDGE_REFINEMENT = True

# Place the edge-refinement cuts at quarter-line-width spacing; this was better
# than the original eighth-line default in the full 140-case search.
OPT_DEFECT_EDGE_REFINEMENT_SCALE = "quarter_line"

# Use one extra refinement column on each side of the defect/opening edge.
OPT_DEFECT_EDGE_REFINEMENT_STEPS_EACH_SIDE = 1

# Disable the original slab-y targeted refinement; extra y splitting in slab2
# and slab4 made this SG-DVCL fit worse than the simpler vertical stack.
OPT_ENABLE_TARGETED_SLAB_Y_REFINEMENT = False


CAL_0510_OPTIMIZED_PARAMS = {
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

    params = dict(CAL_0510_OPTIMIZED_PARAMS)
    params.update(
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
        params=params,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one Cal_0510 SG-DVCL S4P.")
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
        "Generated SG-DVCL S4P: "
        f"WLine={args.width_um:g} um, L_MA={args.length_um:g} um, "
        f"L_E1={args.length_um - 2.0:g} um, "
        f"L_GND_open={args.length_um - 4.0:g} um, "
        f"W_GND_open={args.gnd_width_factor * args.width_um:g} um, "
        f"nports={network.nports}, nfreq={len(network.frequency.f)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate SG-DVCL half-transformer S4P with the 2026-05-04 fitted geometry.

This wrapper reuses the field solver and Touchstone export implementation in
``Cal_0423.py``.  It supports two parameter input modes:

  - transformer: input transformer width W and aspect ratio R, then derive
    W_line = 0.25 * W and line_length_um = 0.98 *
    real_coupling_length1_um(W, R, W_line)
  - coupled-line: directly input W_line and line_length_um

Both modes use W_GND_defect = 1.30 * W_line by default.

The resulting S4P is intended for the simplified six-port transformer model
where the compensation microstrip lines and L56delta are removed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import Cal_0423 as cal
from VCL_length_calculator import calculate_compensation_msl

#
DEFAULT_LENGTH_SCALE = 1
DEFAULT_SLOT_RATIO = 1.5

# DEFAULT_W_UM = 101
# DEFAULT_R = 1.65

DEFAULT_W_LINE_UM: float | None = 30
DEFAULT_LINE_LENGTH_UM: float | None = 210.1

DEFAULT_OUTPUT_DIR = Path("outputs") / "cal0504_s4p"
DEFAULT_SGDVCL_OUTPUT_DIR = Path("outputs") / "cal0509_sgdvcl_s4p"
DEFAULT_FREQ_START_GHZ = 1.0
DEFAULT_FREQ_STOP_GHZ = 200.0
DEFAULT_FREQ_NPOINTS = 20

SGDVCL_TOUCHSTONE_PORT_ORDER = ("E1_A", "MA_B", "E1_B", "MA_A")
SGDVCL_COUPLED_LINE_SECTION_ORDER = ("E1_A", "MA_A", "E1_B", "MA_B")

_CAL0509_CONFIG_OVERRIDE_KEYS = {
    "L",
    "quadrature_order",
    "plot_resolution",
    "outer_dirichlet_projection_mode",
    "enable_targeted_region_subdivision",
    "targeted_region_subdivision_counts",
    "enable_defect_edge_refinement",
    "defect_edge_refinement_scale",
    "defect_edge_refinement_steps_each_side",
    "enable_selective_slab_x_refinement",
    "x_refinement_slab_names",
    "M_modes",
    "enable_targeted_slab_y_refinement",
    "targeted_slab_y_split_counts",
    "freq_list_Hz",
    "z0_ref",
    "use_loss",
    "metal_sigma_by_conductor_S_per_m",
    "metal_roughness_um",
    "tan_delta_by_material",
    "tan_delta_eff_override",
    "y_levels",
    "slab_names",
}

_CAL0509_BLOCKED_CONFIG_KEYS = {
    "use_hfss_cprime_override",
    "use_hfss_cair_override",
    "Cprime_hfss",
    "Cair_hfss",
    "touchstone_port_perm",
    "touchstone_port_labels",
}



# ``None`` means that the default is resolved from W/R:
#   DEFAULT_W_LINE_UM(W) = 0.25 * W
#   DEFAULT_LINE_LENGTH_UM(W,R,W_line) = real_coupling_length1_um(W, R, W_line)
# The effective Cal_0423 line length is length_scale * DEFAULT_LINE_LENGTH_UM.
# Use ``--length-scale 1`` to pass the raw VCL_length_calculator length.



# 耦合线长度 [m] W90R1.3 160.1
# 耦合线长度 [m] W90R1.4 169.1
# 耦合线长度 [m] W90R1.5 178.1
# 耦合线长度 [m] W90R1.6 187.1
# 耦合线长度 [m] W90R1.7 196.1

# 耦合线长度 [m] W100R1.3 176.8
# 耦合线长度 [m] W100R1.4 186.8
# 耦合线长度 [m] W100R1.5 196.8
# 耦合线长度 [m] W100R1.6 206.8
# 耦合线长度 [m] W100R1.7 216.8

# 耦合线长度 [m]W110R1.3 193.5
# 耦合线长度 [m]W110R1.4 204.5
# 耦合线长度 [m]W110R1.5 215.5
# 耦合线长度 [m]W110R1.6 226.5
# 耦合线长度 [m]W110R1.7 237.5

# 耦合线长度 [m]W120R1.3 210.1
# 耦合线长度 [m]W120R1.4 222.1
# 耦合线长度 [m]W120R1.5 234.1
# 耦合线长度 [m]W120R1.6 246.1
# 耦合线长度 [m]W120R1.7 258.1


def safe_tag(value: float, digits: int = 5) -> str:
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def default_filename(w_um: float, r: float, length_scale: float, slot_ratio: float) -> str:
    return (
        f"Cal0504_SGDVCL_W{safe_tag(w_um)}_R{safe_tag(r)}_"
        f"ls{safe_tag(length_scale)}_slot{safe_tag(slot_ratio)}.s4p"
    )


def default_coupled_line_filename(w_line_um: float, line_length_um: float, slot_ratio: float) -> str:
    return (
        f"Cal0504_SGDVCL_Wline{safe_tag(w_line_um)}_"
        f"Len{safe_tag(line_length_um)}_slot{safe_tag(slot_ratio)}.s4p"
    )


def default_w_line_um(w_um: float) -> float:
    """Return the SG-DVCL line width derived from the transformer width."""
    return 0.25 * float(w_um)


def default_line_length_um(w_um: float, r: float, w_line_um: float | None = DEFAULT_W_LINE_UM) -> float:
    """Return raw ``real_coupling_length1_um`` from ``VCL_length_calculator``."""
    geom = calculate_compensation_msl(float(w_um), float(r), w_line_um=w_line_um)
    return float(geom["real_coupling_length1_um"])


def resolve_geometry_defaults(
    w_um: float,
    r: float,
    *,
    w_line_um: float | None = DEFAULT_W_LINE_UM,
    line_length_um: float | None = DEFAULT_LINE_LENGTH_UM,
    length_scale: float = DEFAULT_LENGTH_SCALE,
    slot_ratio: float = DEFAULT_SLOT_RATIO,
) -> dict[str, float | str]:
    """Resolve automatic Cal_0504 geometry from a transformer W/R pair.

    Manual values passed with ``--W-line`` or ``--line-length-um`` override
    these defaults.  The returned ``line_length_um`` is the effective value
    passed into ``Cal_0423``; ``raw_real_coupling_length1_um`` records the
    unscaled length from ``VCL_length_calculator``.
    """
    auto_geom = calculate_compensation_msl(float(w_um), float(r))
    auto_w_line_um = float(auto_geom["W_MSL_um"])
    effective_w_line_um = float(w_line_um) if w_line_um is not None else auto_w_line_um
    geom = calculate_compensation_msl(float(w_um), float(r), w_line_um=effective_w_line_um)
    raw_length_um = float(geom["real_coupling_length1_um"])
    effective_line_length_um = (
        float(line_length_um) if line_length_um is not None else float(length_scale) * raw_length_um
    )
    return {
        "input_mode": "transformer",
        "auto_w_line_um": auto_w_line_um,
        "W_line_um": effective_w_line_um,
        "W_line_source": "external/manual Wline" if w_line_um is not None else "0.25 * W fallback",
        "raw_real_coupling_length1_um": raw_length_um,
        "line_length_um": effective_line_length_um,
        "line_length_source": (
            "manual --line-length-um"
            if line_length_um is not None
            else "length_scale * real_coupling_length1_um(W,R,W_line)"
        ),
        "slot_ratio": float(slot_ratio),
        "W_GND_defect_um": float(slot_ratio) * effective_w_line_um,
    }


def resolve_coupled_line_geometry(
    *,
    w_line_um: float,
    line_length_um: float,
    slot_ratio: float = DEFAULT_SLOT_RATIO,
) -> dict[str, float | str]:
    """Resolve Cal_0504 geometry from direct coupled-line inputs."""
    effective_w_line_um = float(w_line_um)
    effective_line_length_um = float(line_length_um)
    if effective_w_line_um <= 0:
        raise ValueError("W_line_um must be positive.")
    if effective_line_length_um <= 0:
        raise ValueError("line_length_um must be positive.")
    if float(slot_ratio) < 0:
        raise ValueError("slot_ratio must be non-negative.")
    return {
        "input_mode": "coupled-line",
        "W_line_um": effective_w_line_um,
        "W_line_source": "manual/direct --W-line",
        "line_length_um": effective_line_length_um,
        "line_length_source": "manual/direct --line-length-um",
        "slot_ratio": float(slot_ratio),
        "W_GND_defect_um": float(slot_ratio) * effective_w_line_um,
    }


def build_freq_list_hz(
    *,
    freq_start_ghz: float = DEFAULT_FREQ_START_GHZ,
    freq_stop_ghz: float = DEFAULT_FREQ_STOP_GHZ,
    freq_step_ghz: float | None = None,
    freq_npoints: int = DEFAULT_FREQ_NPOINTS,
) -> tuple[float, ...]:
    if freq_step_ghz is None:
        if int(freq_npoints) <= 0:
            raise ValueError("freq_npoints must be positive.")
        if int(freq_npoints) == 1:
            return (float(freq_start_ghz) * 1e9,)
        start = float(freq_start_ghz)
        stop = float(freq_stop_ghz)
        step = (stop - start) / (int(freq_npoints) - 1)
        return tuple((start + idx * step) * 1e9 for idx in range(int(freq_npoints)))

    npoints = int(round((float(freq_stop_ghz) - float(freq_start_ghz)) / float(freq_step_ghz))) + 1
    if npoints <= 0:
        raise ValueError("Frequency range is empty.")
    return tuple((float(freq_start_ghz) + idx * float(freq_step_ghz)) * 1e9 for idx in range(npoints))


def build_config(
    *,
    w_um: float,
    r: float,
    output_dir: Path,
    filename: str | None = None,
    w_line_um: float | None = DEFAULT_W_LINE_UM,
    line_length_um: float | None = DEFAULT_LINE_LENGTH_UM,
    length_scale: float = DEFAULT_LENGTH_SCALE,
    slot_ratio: float = DEFAULT_SLOT_RATIO,
    freq_start_ghz: float = DEFAULT_FREQ_START_GHZ,
    freq_stop_ghz: float = DEFAULT_FREQ_STOP_GHZ,
    freq_step_ghz: float | None = None,
    freq_npoints: int = DEFAULT_FREQ_NPOINTS,
    m_modes: int | None = None,
    quiet: bool = True,
) -> tuple[cal.Config, dict[str, float | str]]:
    resolved = resolve_geometry_defaults(
        float(w_um),
        float(r),
        w_line_um=w_line_um,
        line_length_um=line_length_um,
        length_scale=length_scale,
        slot_ratio=slot_ratio,
    )
    effective_w_line_um = float(resolved["W_line_um"])
    raw_length_um = float(resolved["raw_real_coupling_length1_um"])
    effective_line_length_um = float(resolved["line_length_um"])
    slot_width_um = float(resolved["W_GND_defect_um"])

    freq_list_hz = build_freq_list_hz(
        freq_start_ghz=freq_start_ghz,
        freq_stop_ghz=freq_stop_ghz,
        freq_step_ghz=freq_step_ghz,
        freq_npoints=freq_npoints,
    )

    output_dir = Path(output_dir)
    out_name = filename or default_filename(w_um, r, length_scale, slot_ratio)
    params_kwargs = {
        "W_line": effective_w_line_um,
        "W_GND_defect": slot_width_um,
        "line_length_um": effective_line_length_um,
        "freq_list_Hz": freq_list_hz,
        "output_dir": str(output_dir),
        "export_touchstone_filename": out_name,
        "quiet": bool(quiet),
        "write_summary_json_enabled": False,
        "generate_geometry_plots": False,
        "generate_potential_plots": False,
        "print_geometry_table": False,
    }
    if m_modes is not None:
        params_kwargs["M_modes"] = int(m_modes)

    manifest = {
        "input_mode": str(resolved["input_mode"]),
        "W_um": float(w_um),
        "R": float(r),
        "W_line_um": effective_w_line_um,
        "auto_W_line_um": float(resolved["auto_w_line_um"]),
        "W_line_source": str(resolved["W_line_source"]),
        "raw_real_coupling_length1_um": raw_length_um,
        "length_scale": float(length_scale),
        "line_length_um": effective_line_length_um,
        "line_length_source": str(resolved["line_length_source"]),
        "slot_ratio": float(slot_ratio),
        "W_GND_defect_um": slot_width_um,
        "freq_start_ghz": float(freq_start_ghz),
        "freq_stop_ghz": float(freq_stop_ghz),
        "freq_step_ghz": "" if freq_step_ghz is None else float(freq_step_ghz),
        "freq_npoints": len(freq_list_hz),
        "freq_mode": "linspace" if freq_step_ghz is None else "step",
        "output_dir": str(output_dir),
        "touchstone_filename": out_name,
    }
    return cal.Config(**params_kwargs), manifest


def build_config_from_coupled_line(
    *,
    w_line_um: float,
    line_length_um: float,
    output_dir: Path,
    filename: str | None = None,
    slot_ratio: float = DEFAULT_SLOT_RATIO,
    freq_start_ghz: float = DEFAULT_FREQ_START_GHZ,
    freq_stop_ghz: float = DEFAULT_FREQ_STOP_GHZ,
    freq_step_ghz: float | None = None,
    freq_npoints: int = DEFAULT_FREQ_NPOINTS,
    m_modes: int | None = None,
    quiet: bool = True,
) -> tuple[cal.Config, dict[str, float | str]]:
    resolved = resolve_coupled_line_geometry(
        w_line_um=w_line_um,
        line_length_um=line_length_um,
        slot_ratio=slot_ratio,
    )
    effective_w_line_um = float(resolved["W_line_um"])
    effective_line_length_um = float(resolved["line_length_um"])
    slot_width_um = float(resolved["W_GND_defect_um"])
    freq_list_hz = build_freq_list_hz(
        freq_start_ghz=freq_start_ghz,
        freq_stop_ghz=freq_stop_ghz,
        freq_step_ghz=freq_step_ghz,
        freq_npoints=freq_npoints,
    )

    output_dir = Path(output_dir)
    out_name = filename or default_coupled_line_filename(
        effective_w_line_um,
        effective_line_length_um,
        slot_ratio,
    )
    params_kwargs = {
        "W_line": effective_w_line_um,
        "W_GND_defect": slot_width_um,
        "line_length_um": effective_line_length_um,
        "freq_list_Hz": freq_list_hz,
        "output_dir": str(output_dir),
        "export_touchstone_filename": out_name,
        "quiet": bool(quiet),
        "write_summary_json_enabled": False,
        "generate_geometry_plots": False,
        "generate_potential_plots": False,
        "print_geometry_table": False,
    }
    if m_modes is not None:
        params_kwargs["M_modes"] = int(m_modes)

    manifest = {
        "input_mode": str(resolved["input_mode"]),
        "W_line_um": effective_w_line_um,
        "W_line_source": str(resolved["W_line_source"]),
        "line_length_um": effective_line_length_um,
        "line_length_source": str(resolved["line_length_source"]),
        "slot_ratio": float(slot_ratio),
        "W_GND_defect_um": slot_width_um,
        "freq_start_ghz": float(freq_start_ghz),
        "freq_stop_ghz": float(freq_stop_ghz),
        "freq_step_ghz": "" if freq_step_ghz is None else float(freq_step_ghz),
        "freq_npoints": len(freq_list_hz),
        "freq_mode": "linspace" if freq_step_ghz is None else "step",
        "output_dir": str(output_dir),
        "touchstone_filename": out_name,
    }
    return cal.Config(**params_kwargs), manifest


def _sgdvcl_params(params: dict | None) -> dict:
    """Return a shallow copy of user SG-DVCL parameters with safe defaults."""
    merged = dict(params or {})
    for blocked in sorted(_CAL0509_BLOCKED_CONFIG_KEYS.intersection(merged)):
        raise ValueError(f"{blocked} is fixed for honest HFSS comparison and cannot be overridden.")
    merged.setdefault("line_length_reference", "MA")
    merged.setdefault("line_length_scale", 1.0)
    merged.setdefault("line_length_offset_um", 0.0)
    merged.setdefault("WLine_scale", 1.0)
    merged.setdefault("WLine_offset_um", 0.0)
    merged.setdefault("GND_width_scale", 1.0)
    merged.setdefault("GND_width_offset_um", 0.0)
    merged.setdefault("min_gnd_open_margin_um", 1e-3)
    return merged


def _sgdvcl_effective_length_um(
    *,
    L_MA_um: float,
    L_E1_um: float,
    L_GND_open_um: float,
    params: dict,
) -> tuple[float, str]:
    mode = str(params.get("line_length_reference", "MA")).strip().lower()
    if mode in {"ma", "l_ma"}:
        base = float(L_MA_um)
        description = "L_MA"
    elif mode in {"e1", "l_e1"}:
        base = float(L_E1_um)
        description = "L_E1"
    elif mode in {"gnd", "ground", "gnd_open", "l_gnd_open"}:
        base = float(L_GND_open_um)
        description = "L_GND_open"
    elif mode in {"mean", "average"}:
        base = (float(L_MA_um) + float(L_E1_um) + float(L_GND_open_um)) / 3.0
        description = "mean(L_MA,L_E1,L_GND_open)"
    elif mode == "weighted":
        w_ma = float(params.get("length_weight_MA", 1.0))
        w_e1 = float(params.get("length_weight_E1", 0.0))
        w_gnd = float(params.get("length_weight_GND_open", 0.0))
        denom = w_ma + w_e1 + w_gnd
        if abs(denom) < 1e-15:
            raise ValueError("weighted line length requires a non-zero weight sum.")
        base = (
            w_ma * float(L_MA_um)
            + w_e1 * float(L_E1_um)
            + w_gnd * float(L_GND_open_um)
        ) / denom
        description = (
            f"weighted(MA={w_ma:g},E1={w_e1:g},GND_open={w_gnd:g})"
        )
    else:
        raise ValueError(
            "line_length_reference must be MA, E1, GND_open, mean, or weighted."
        )

    effective = float(params["line_length_scale"]) * base + float(params["line_length_offset_um"])
    if effective <= 0.0:
        raise ValueError(f"Effective SG-DVCL line length must be positive, got {effective}.")
    return effective, description


def build_config_from_sgdvcl(
    *,
    WLine_um: float,
    L_MA_um: float,
    L_E1_um: float,
    L_GND_open_um: float,
    W_GND_open_um: float,
    GND_width_factor: float,
    params: dict | None = None,
    output_dir: Path = DEFAULT_SGDVCL_OUTPUT_DIR,
    filename: str | None = None,
) -> tuple[cal.Config, dict[str, float | str]]:
    """Build a Cal_0423 config from explicit HFSS SG-DVCL geometry.

    The underlying quasi-TEM model has one propagation length and one lateral
    ground-opening width.  The explicit HFSS lengths are still accepted and
    recorded; by default the propagation length is L_MA, with optional
    physically interpretable effective-length parameters for fitting.
    """
    fit_params = _sgdvcl_params(params)
    effective_length_um, length_source = _sgdvcl_effective_length_um(
        L_MA_um=float(L_MA_um),
        L_E1_um=float(L_E1_um),
        L_GND_open_um=float(L_GND_open_um),
        params=fit_params,
    )
    effective_w_line_um = (
        float(fit_params["WLine_scale"]) * float(WLine_um)
        + float(fit_params["WLine_offset_um"])
    )
    if effective_w_line_um <= 0.0:
        raise ValueError("Effective WLine must be positive.")

    effective_w_gnd_open_um = (
        float(fit_params["GND_width_scale"]) * float(W_GND_open_um)
        + float(fit_params["GND_width_offset_um"])
    )
    min_margin_um = float(fit_params["min_gnd_open_margin_um"])
    if effective_w_gnd_open_um <= effective_w_line_um:
        effective_w_gnd_open_um = effective_w_line_um + max(min_margin_um, 1e-9)

    if filename is None:
        filename = (
            "Cal0509_SGDVCL_"
            f"W{safe_tag(WLine_um)}_L{safe_tag(L_MA_um)}_"
            f"G{safe_tag(GND_width_factor, digits=3)}.s4p"
        )

    freq_list_hz = fit_params.get("frequency_hz", fit_params.get("freq_list_Hz"))
    if freq_list_hz is None:
        freq_list_hz = build_freq_list_hz(
            freq_start_ghz=float(fit_params.get("freq_start_ghz", DEFAULT_FREQ_START_GHZ)),
            freq_stop_ghz=float(fit_params.get("freq_stop_ghz", DEFAULT_FREQ_STOP_GHZ)),
            freq_step_ghz=fit_params.get("freq_step_ghz"),
            freq_npoints=int(fit_params.get("freq_npoints", DEFAULT_FREQ_NPOINTS)),
        )
    else:
        freq_list_hz = tuple(float(v) for v in freq_list_hz)

    params_kwargs = {
        "W_line": effective_w_line_um,
        "W_GND_defect": effective_w_gnd_open_um,
        "line_length_um": effective_length_um,
        "freq_list_Hz": tuple(freq_list_hz),
        "output_dir": str(Path(output_dir)),
        "export_touchstone_filename": filename,
        "quiet": bool(fit_params.get("quiet", True)),
        "write_summary_json_enabled": False,
        "generate_geometry_plots": False,
        "generate_potential_plots": False,
        "print_geometry_table": False,
        "touchstone_port_perm": (1, 2, 3, 0),
        "touchstone_port_labels": SGDVCL_TOUCHSTONE_PORT_ORDER,
    }

    if "m_modes" in fit_params and "M_modes" not in fit_params:
        params_kwargs["M_modes"] = int(fit_params["m_modes"])
    for key in sorted(_CAL0509_CONFIG_OVERRIDE_KEYS):
        if key in fit_params:
            params_kwargs[key] = fit_params[key]
    if "cross_section_half_width_um" in fit_params:
        params_kwargs["L"] = float(fit_params["cross_section_half_width_um"])

    manifest = {
        "input_mode": "sgdvcl",
        "WLine_um": float(WLine_um),
        "L_MA_um": float(L_MA_um),
        "L_E1_um": float(L_E1_um),
        "L_GND_open_um": float(L_GND_open_um),
        "W_GND_open_um": float(W_GND_open_um),
        "GND_width_factor": float(GND_width_factor),
        "effective_WLine_um": effective_w_line_um,
        "effective_W_GND_open_um": effective_w_gnd_open_um,
        "effective_line_length_um": effective_length_um,
        "line_length_source": length_source,
        "touchstone_port_order": ",".join(SGDVCL_TOUCHSTONE_PORT_ORDER),
        "coupled_line_section_order": ",".join(SGDVCL_COUPLED_LINE_SECTION_ORDER),
        "freq_npoints": len(tuple(freq_list_hz)),
        "output_dir": str(Path(output_dir)),
        "touchstone_filename": filename,
    }
    return cal.Config(**params_kwargs), manifest


def generate_s4p_from_sgdvcl(
    *,
    WLine_um: float,
    L_MA_um: float,
    L_E1_um: float,
    L_GND_open_um: float,
    W_GND_open_um: float,
    GND_width_factor: float,
    params: dict | None = None,
    output_dir: Path = DEFAULT_SGDVCL_OUTPUT_DIR,
    filename: str | None = None,
    write_manifest: bool = True,
) -> Path:
    config, manifest = build_config_from_sgdvcl(
        WLine_um=WLine_um,
        L_MA_um=L_MA_um,
        L_E1_um=L_E1_um,
        L_GND_open_um=L_GND_open_um,
        W_GND_open_um=W_GND_open_um,
        GND_width_factor=GND_width_factor,
        params=params,
        output_dir=Path(output_dir),
        filename=filename,
    )
    return export_config_s4p(config, manifest, write_manifest=write_manifest)


def calculate_sgdvcl_s4p(
    WLine_um: float,
    L_MA_um: float,
    L_E1_um: float,
    L_GND_open_um: float,
    W_GND_open_um: float,
    GND_width_factor: float,
    params: dict,
):
    """Calculate one SG-DVCL case and return it as a scikit-rf Network."""
    import skrf as rf

    fit_params = dict(params or {})
    output_dir = Path(fit_params.pop("output_dir", DEFAULT_SGDVCL_OUTPUT_DIR))
    filename = fit_params.pop("filename", None)
    path = generate_s4p_from_sgdvcl(
        WLine_um=WLine_um,
        L_MA_um=L_MA_um,
        L_E1_um=L_E1_um,
        L_GND_open_um=L_GND_open_um,
        W_GND_open_um=W_GND_open_um,
        GND_width_factor=GND_width_factor,
        params=fit_params,
        output_dir=output_dir,
        filename=filename,
        write_manifest=bool(fit_params.get("write_manifest", True)),
    )
    return rf.Network(str(path))


def export_config_s4p(
    params: cal.Config,
    manifest: dict[str, float | str],
    *,
    write_manifest: bool = True,
) -> Path:
    cal.ensure_output_dir(params.output_dir)
    bundle = cal.prepare_sweep_bundle(params)
    result = cal.export_sweep_s4p_from_modal_records(
        params,
        bundle.modal_records,
        params.line_length_um * cal.UM_TO_M,
    )
    out_path = Path(result["touchstone_path"])
    if write_manifest:
        manifest_path = out_path.with_suffix(".json")
        manifest["touchstone_path"] = str(out_path)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    return out_path


def generate_s4p(
    *,
    w_um: float,
    r: float,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    filename: str | None = None,
    w_line_um: float | None = DEFAULT_W_LINE_UM,
    line_length_um: float | None = DEFAULT_LINE_LENGTH_UM,
    length_scale: float = DEFAULT_LENGTH_SCALE,
    slot_ratio: float = DEFAULT_SLOT_RATIO,
    freq_start_ghz: float = DEFAULT_FREQ_START_GHZ,
    freq_stop_ghz: float = DEFAULT_FREQ_STOP_GHZ,
    freq_step_ghz: float | None = None,
    freq_npoints: int = DEFAULT_FREQ_NPOINTS,
    m_modes: int | None = None,
    quiet: bool = True,
    write_manifest: bool = True,
) -> Path:
    params, manifest = build_config(
        w_um=w_um,
        r=r,
        output_dir=Path(output_dir),
        filename=filename,
        w_line_um=w_line_um,
        line_length_um=line_length_um,
        length_scale=length_scale,
        slot_ratio=slot_ratio,
        freq_start_ghz=freq_start_ghz,
        freq_stop_ghz=freq_stop_ghz,
        freq_step_ghz=freq_step_ghz,
        freq_npoints=freq_npoints,
        m_modes=m_modes,
        quiet=quiet,
    )
    return export_config_s4p(params, manifest, write_manifest=write_manifest)


def generate_s4p_from_coupled_line(
    *,
    w_line_um: float,
    line_length_um: float,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    filename: str | None = None,
    slot_ratio: float = DEFAULT_SLOT_RATIO,
    freq_start_ghz: float = DEFAULT_FREQ_START_GHZ,
    freq_stop_ghz: float = DEFAULT_FREQ_STOP_GHZ,
    freq_step_ghz: float | None = None,
    freq_npoints: int = DEFAULT_FREQ_NPOINTS,
    m_modes: int | None = None,
    quiet: bool = True,
    write_manifest: bool = True,
) -> Path:
    params, manifest = build_config_from_coupled_line(
        w_line_um=w_line_um,
        line_length_um=line_length_um,
        output_dir=Path(output_dir),
        filename=filename,
        slot_ratio=slot_ratio,
        freq_start_ghz=freq_start_ghz,
        freq_stop_ghz=freq_stop_ghz,
        freq_step_ghz=freq_step_ghz,
        freq_npoints=freq_npoints,
        m_modes=m_modes,
        quiet=quiet,
    )
    return export_config_s4p(params, manifest, write_manifest=write_manifest)


def parse_wr_pair(text: str) -> tuple[float, float]:
    if "," in text:
        left, right = text.split(",", 1)
    elif ":" in text:
        left, right = text.split(":", 1)
    else:
        raise argparse.ArgumentTypeError("Use W,R or W:R, for example 100,1.5")
    return float(left), float(right)


def parse_line_pair(text: str) -> tuple[float, float]:
    if "," in text:
        left, right = text.split(",", 1)
    elif ":" in text:
        left, right = text.split(":", 1)
    else:
        raise argparse.ArgumentTypeError("Use W_line,line_length_um or W_line:line_length_um, for example 25.25,209.5")
    return float(left), float(right)


def iter_cases(args: argparse.Namespace) -> Iterable[tuple[float, float]]:
    if args.case:
        yield from args.case
    else:
        w_um = DEFAULT_W_UM if args.W is None else args.W
        r = DEFAULT_R if args.R is None else args.R
        yield float(w_um), float(r)


def iter_coupled_line_cases(args: argparse.Namespace) -> Iterable[tuple[float, float]]:
    if args.line_case:
        yield from args.line_case
    else:
        if args.W_line is None or args.line_length_um is None:
            raise ValueError(
                "coupled-line mode requires --W-line and --line-length-um, "
                "or at least one --line-case W_line,line_length_um."
            )
        yield float(args.W_line), float(args.line_length_um)


def infer_input_mode(args: argparse.Namespace) -> str:
    if args.input_mode is not None:
        return args.input_mode
    has_direct_line_inputs = args.W_line is not None and args.line_length_um is not None
    has_transformer_inputs = args.W is not None or args.R is not None or bool(args.case)
    if has_direct_line_inputs and not has_transformer_inputs:
        return "coupled-line"
    return "transformer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Cal_0504 SG-DVCL S4P files with two parameter input modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python Cal_0504.py --input-mode transformer --W 101 --R 1.65\n"
            "  python Cal_0504.py --input-mode coupled-line --W-line 25.25 --line-length-um 209.5\n"
            "  python Cal_0504.py --W-line 25.25 --line-length-um 209.5\n"
        ),
    )
    parser.add_argument(
        "--input-mode",
        "--mode",
        choices=("transformer", "coupled-line"),
        default=None,
        help=(
            "Parameter input mode. Default: infer coupled-line when only --W-line "
            "and --line-length-um are given; otherwise transformer."
        ),
    )
    parser.add_argument("--W", type=float, default=None, help="Transformer width in um for transformer mode.")
    parser.add_argument("--R", type=float, default=None, help="Transformer aspect ratio for transformer mode.")
    parser.add_argument(
        "--W-line",
        type=float,
        default=DEFAULT_W_LINE_UM,
        help=(
            "Coupled-line width in um. In coupled-line mode this is a required "
            "direct input; in transformer mode it is an optional manual override."
        ),
    )
    parser.add_argument(
        "--line-length-um",
        type=float,
        default=DEFAULT_LINE_LENGTH_UM,
        help=(
            "Physical coupled-line length in um. In coupled-line mode this is a "
            "required direct input; in transformer mode it is an optional manual override."
        ),
    )
    parser.add_argument(
        "--auto-W-line",
        action="store_true",
        help="Transformer mode only: ignore --W-line and use W_line=0.25*W.",
    )
    parser.add_argument(
        "--auto-line-length",
        action="store_true",
        help="Transformer mode only: ignore --line-length-um and use length_scale*real_coupling_length1_um(W,R).",
    )
    parser.add_argument(
        "--case",
        type=parse_wr_pair,
        action="append",
        help="Transformer-mode batch W,R pair. Can be repeated, e.g. --case 90,1.3 --case 120,1.7.",
    )
    parser.add_argument(
        "--line-case",
        type=parse_line_pair,
        action="append",
        help=(
            "Coupled-line-mode batch W_line,line_length_um pair. Can be repeated, "
            "e.g. --line-case 22.5,160.1 --line-case 30,258.1."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--filename", default=None, help="Output filename for single-case mode.")
    parser.add_argument("--length-scale", type=float, default=DEFAULT_LENGTH_SCALE)
    parser.add_argument("--slot-ratio", type=float, default=DEFAULT_SLOT_RATIO)
    parser.add_argument("--freq-start-ghz", type=float, default=DEFAULT_FREQ_START_GHZ)
    parser.add_argument("--freq-stop-ghz", type=float, default=DEFAULT_FREQ_STOP_GHZ)
    parser.add_argument(
        "--freq-npoints",
        type=int,
        default=DEFAULT_FREQ_NPOINTS,
        help="Number of linspace frequency points when --freq-step-ghz is omitted.",
    )
    parser.add_argument(
        "--freq-step-ghz",
        type=float,
        default=None,
        help="Use fixed frequency step instead of Cal_0423-style linspace points.",
    )
    parser.add_argument("--m-modes", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = []
    input_mode = infer_input_mode(args)
    if input_mode == "transformer":
        if args.line_case:
            raise ValueError("--line-case can only be used with --input-mode coupled-line.")
        cases = list(iter_cases(args))
        if args.filename and len(cases) > 1:
            raise ValueError("--filename can only be used with a single case.")
        for w_um, r in cases:
            w_line_um = None if args.auto_W_line else args.W_line
            line_length_um = None if args.auto_line_length else args.line_length_um
            out_path = generate_s4p(
                w_um=w_um,
                r=r,
                output_dir=args.output_dir,
                filename=args.filename,
                w_line_um=w_line_um,
                line_length_um=line_length_um,
                length_scale=args.length_scale,
                slot_ratio=args.slot_ratio,
                freq_start_ghz=args.freq_start_ghz,
                freq_stop_ghz=args.freq_stop_ghz,
                freq_step_ghz=args.freq_step_ghz,
                freq_npoints=args.freq_npoints,
                m_modes=args.m_modes,
                quiet=not args.verbose,
                write_manifest=not args.no_manifest,
            )
            paths.append(out_path)
            print(out_path)
    else:
        if args.case:
            raise ValueError("--case can only be used with --input-mode transformer.")
        if args.auto_W_line or args.auto_line_length:
            raise ValueError("--auto-W-line and --auto-line-length are only valid with --input-mode transformer.")
        line_cases = list(iter_coupled_line_cases(args))
        if args.filename and len(line_cases) > 1:
            raise ValueError("--filename can only be used with a single case.")
        for w_line_um, line_length_um in line_cases:
            out_path = generate_s4p_from_coupled_line(
                w_line_um=w_line_um,
                line_length_um=line_length_um,
                output_dir=args.output_dir,
                filename=args.filename,
                slot_ratio=args.slot_ratio,
                freq_start_ghz=args.freq_start_ghz,
                freq_stop_ghz=args.freq_stop_ghz,
                freq_step_ghz=args.freq_step_ghz,
                freq_npoints=args.freq_npoints,
                m_modes=args.m_modes,
                quiet=not args.verbose,
                write_manifest=not args.no_manifest,
            )
            paths.append(out_path)
            print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

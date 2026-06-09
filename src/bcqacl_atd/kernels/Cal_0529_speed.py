#!/usr/bin/env python3
"""Fast Cal_0529_v2 SG-DVCL S4P generator.

This is an accuracy-preserving speed wrapper around ``Cal_0529_v2``.  The
geometry rule and Cal_0509 electromagnetic/quasi-static solver settings are
unchanged; this file only avoids unnecessary Touchstone read-back and adds an
exact in-memory network cache for repeated geometries.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

import Cal_0509
import Cal_0529_v2


LPORT_UM = Cal_0529_v2.LPORT_UM
WPORT_UM = Cal_0529_v2.WPORT_UM
WIDTH_OPEN_UM = Cal_0529_v2.WIDTH_OPEN_UM

OPT_LINE_LENGTH_REFERENCE = Cal_0529_v2.OPT_LINE_LENGTH_REFERENCE
OPT_LINE_LENGTH_OFFSET_UM = Cal_0529_v2.OPT_LINE_LENGTH_OFFSET_UM
OPT_LINE_LENGTH_SCALE = Cal_0529_v2.OPT_LINE_LENGTH_SCALE
OPT_WLINE_SCALE = Cal_0529_v2.OPT_WLINE_SCALE
OPT_WLINE_OFFSET_UM = Cal_0529_v2.OPT_WLINE_OFFSET_UM
OPT_GND_WIDTH_SCALE = Cal_0529_v2.OPT_GND_WIDTH_SCALE
OPT_GND_WIDTH_OFFSET_UM = Cal_0529_v2.OPT_GND_WIDTH_OFFSET_UM
OPT_GND_OPEN_TO_INPUT_WLINE_FACTOR = Cal_0529_v2.OPT_GND_OPEN_TO_INPUT_WLINE_FACTOR
OPT_GND_WIDTH_FACTOR = Cal_0529_v2.OPT_GND_WIDTH_FACTOR
LF_RANGE = Cal_0529_v2.LF_RANGE
GF_RANGE = Cal_0529_v2.GF_RANGE

CAL_0529_V2_HALF_TF_BASELINE_PARAMS = Cal_0529_v2.CAL_0529_V2_HALF_TF_BASELINE_PARAMS
CAL_0521_V2_HALF_TF_BASELINE_PARAMS = CAL_0529_V2_HALF_TF_BASELINE_PARAMS
CAL_0521_HALF_TF_BASELINE_PARAMS = CAL_0529_V2_HALF_TF_BASELINE_PARAMS

INPUT_SGDVCL_LENGTH_UM = Cal_0529_v2.INPUT_SGDVCL_LENGTH_UM
INPUT_SGDVCL_WIDTH_UM = Cal_0529_v2.INPUT_SGDVCL_WIDTH_UM
INPUT_WLINE_SCALE = Cal_0529_v2.INPUT_WLINE_SCALE
INPUT_GND_OPEN_FACTOR = Cal_0529_v2.INPUT_GND_OPEN_FACTOR
INPUT_GND_WIDTH_FACTOR = Cal_0529_v2.INPUT_GND_WIDTH_FACTOR
OUTPUT_S4P_PATH = Cal_0529_v2.OUTPUT_S4P_PATH
FREQ_START_GHZ = Cal_0529_v2.FREQ_START_GHZ
FREQ_STOP_GHZ = Cal_0529_v2.FREQ_STOP_GHZ
FREQ_NPOINTS = Cal_0529_v2.FREQ_NPOINTS

FAST_NETWORK_CACHE_MAXSIZE = 2048
_FAST_NETWORK_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()


def clear_fast_network_cache() -> None:
    _FAST_NETWORK_CACHE.clear()


def _freeze_for_key(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_for_key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_key(v) for v in value)
    if isinstance(value, np.ndarray):
        return tuple(_freeze_for_key(v) for v in value.tolist())
    if isinstance(value, float):
        return round(float(value), 15)
    return value


def _fast_cache_get(key: tuple[Any, ...]):
    cached = _FAST_NETWORK_CACHE.get(key)
    if cached is None:
        return None
    _FAST_NETWORK_CACHE.move_to_end(key)
    return cached.copy()


def _fast_cache_put(key: tuple[Any, ...], ntw):
    _FAST_NETWORK_CACHE[key] = ntw.copy()
    _FAST_NETWORK_CACHE.move_to_end(key)
    while len(_FAST_NETWORK_CACHE) > FAST_NETWORK_CACHE_MAXSIZE:
        _FAST_NETWORK_CACHE.popitem(last=False)
    return ntw


def _network_from_config_in_memory(params: Any, *, name: str | None = None):
    import skrf as rf

    cal = Cal_0509.cal
    bundle = cal.prepare_sweep_bundle(params)
    length_m = float(params.line_length_um) * cal.UM_TO_M
    s4_list: list[np.ndarray] = []
    for modal_data in bundle.modal_records:
        _z4, s4 = cal.build_4port_ZS_from_modal(
            np.asarray(modal_data["T"], dtype=complex),
            np.asarray(modal_data["U"], dtype=complex),
            np.asarray(modal_data["gamma"], dtype=complex),
            length_m,
            params.z0_ref,
            params.touchstone_port_perm,
        )
        s4_list.append(np.asarray(s4, dtype=complex))

    freq = rf.Frequency.from_f(np.asarray(params.freq_list_Hz, dtype=float), unit="hz")
    ntw = rf.Network(
        frequency=freq,
        s=np.asarray(s4_list, dtype=complex),
        z0=float(params.z0_ref),
        name=name or Path(params.export_touchstone_filename).stem,
    )
    try:
        ntw.port_names = list(params.touchstone_port_labels)
    except Exception:
        pass
    return ntw


def _validate_half_tf_params(params: dict[str, Any]) -> dict[str, Any]:
    return Cal_0529_v2._validate_half_tf_params(params)


def half_tf_geometry_from_formula(
    W_um: float,
    R: float,
    WlineR: float,
    *,
    line_length_scale: float | None = None,
    gnd_width_factor: float | None = None,
) -> dict[str, float]:
    return Cal_0529_v2.half_tf_geometry_from_formula(
        W_um,
        R,
        WlineR,
        line_length_scale=line_length_scale,
        gnd_width_factor=gnd_width_factor,
    )


def build_frequency_hz(
    *,
    freq_start_ghz: float = FREQ_START_GHZ,
    freq_stop_ghz: float = FREQ_STOP_GHZ,
    freq_npoints: int = FREQ_NPOINTS,
) -> tuple[float, ...]:
    return Cal_0529_v2.build_frequency_hz(
        freq_start_ghz=freq_start_ghz,
        freq_stop_ghz=freq_stop_ghz,
        freq_npoints=freq_npoints,
    )


def calculate_sgdvcl_s4p_from_geometry_fast(
    WLine_um: float,
    L_MA_um: float,
    L_E1_um: float,
    L_GND_open_um: float,
    W_GND_open_um: float,
    params: dict[str, Any] | None = None,
    *,
    use_exact_network_cache: bool = True,
):
    fit_params = dict(CAL_0529_V2_HALF_TF_BASELINE_PARAMS)
    fit_params.update(params or {})
    fit_params = _validate_half_tf_params(fit_params)
    output_dir = Path(fit_params.pop("output_dir", OUTPUT_S4P_PATH.parent))
    filename = fit_params.pop("filename", OUTPUT_S4P_PATH.name)
    fit_params.pop("write_manifest", None)
    if "frequency_hz" not in fit_params and "freq_list_Hz" not in fit_params:
        fit_params["frequency_hz"] = build_frequency_hz(
            freq_start_ghz=float(fit_params.pop("freq_start_ghz", FREQ_START_GHZ)),
            freq_stop_ghz=float(fit_params.pop("freq_stop_ghz", FREQ_STOP_GHZ)),
            freq_npoints=int(fit_params.pop("freq_npoints", FREQ_NPOINTS)),
        )
    fit_params.setdefault("quiet", True)

    cache_key = (
        "cal0529v2_fast",
        round(float(WLine_um), 15),
        round(float(L_MA_um), 15),
        round(float(L_E1_um), 15),
        round(float(L_GND_open_um), 15),
        round(float(W_GND_open_um), 15),
        _freeze_for_key({k: v for k, v in fit_params.items() if k not in {"quiet"}}),
    )
    if use_exact_network_cache:
        cached = _fast_cache_get(cache_key)
        if cached is not None:
            return cached

    config, _manifest = Cal_0509.build_config_from_sgdvcl(
        WLine_um=float(WLine_um),
        L_MA_um=float(L_MA_um),
        L_E1_um=float(L_E1_um),
        L_GND_open_um=float(L_GND_open_um),
        W_GND_open_um=float(W_GND_open_um),
        GND_width_factor=float(W_GND_open_um) / float(WLine_um),
        params=fit_params,
        output_dir=output_dir,
        filename=filename,
    )
    ntw = _network_from_config_in_memory(config, name=Path(filename).stem)
    if use_exact_network_cache:
        return _fast_cache_put(cache_key, ntw).copy()
    return ntw


def write_sgdvcl_s4p_from_geometry_fast(
    WLine_um: float,
    L_MA_um: float,
    L_E1_um: float,
    L_GND_open_um: float,
    W_GND_open_um: float,
    params: dict[str, Any] | None = None,
) -> Path:
    fit_params = dict(CAL_0529_V2_HALF_TF_BASELINE_PARAMS)
    fit_params.update(params or {})
    fit_params = _validate_half_tf_params(fit_params)
    output_dir = Path(fit_params.pop("output_dir", OUTPUT_S4P_PATH.parent))
    filename = fit_params.pop("filename", OUTPUT_S4P_PATH.name)
    write_manifest = bool(fit_params.pop("write_manifest", True))
    if "frequency_hz" not in fit_params and "freq_list_Hz" not in fit_params:
        fit_params["frequency_hz"] = build_frequency_hz(
            freq_start_ghz=float(fit_params.pop("freq_start_ghz", FREQ_START_GHZ)),
            freq_stop_ghz=float(fit_params.pop("freq_stop_ghz", FREQ_STOP_GHZ)),
            freq_npoints=int(fit_params.pop("freq_npoints", FREQ_NPOINTS)),
        )
    fit_params.setdefault("quiet", True)
    return Cal_0509.generate_s4p_from_sgdvcl(
        WLine_um=float(WLine_um),
        L_MA_um=float(L_MA_um),
        L_E1_um=float(L_E1_um),
        L_GND_open_um=float(L_GND_open_um),
        W_GND_open_um=float(W_GND_open_um),
        GND_width_factor=float(W_GND_open_um) / float(WLine_um),
        params=fit_params,
        output_dir=output_dir,
        filename=filename,
        write_manifest=write_manifest,
    )


def calculate_sgdvcl_s4p_from_half_tf_fast(
    W_um: float,
    R: float,
    WlineR: float,
    params: dict[str, Any] | None = None,
    *,
    use_exact_network_cache: bool = True,
):
    fit_params = dict(CAL_0529_V2_HALF_TF_BASELINE_PARAMS)
    fit_params.update(params or {})
    fit_params = _validate_half_tf_params(fit_params)
    geometry = half_tf_geometry_from_formula(
        W_um,
        R,
        WlineR,
        line_length_scale=float(fit_params["line_length_scale"]),
        gnd_width_factor=float(fit_params["GND_width_factor"]),
    )
    return calculate_sgdvcl_s4p_from_geometry_fast(
        WLine_um=geometry["derived_WLine_um"],
        L_MA_um=geometry["L_MA_um"],
        L_E1_um=geometry["L_E1_um"],
        L_GND_open_um=geometry["L_GND_open_um"],
        W_GND_open_um=geometry["W_GND_open_um"],
        params=fit_params,
        use_exact_network_cache=use_exact_network_cache,
    )


def write_sgdvcl_s4p_from_half_tf_fast(
    W_um: float,
    R: float,
    WlineR: float,
    params: dict[str, Any] | None = None,
) -> Path:
    fit_params = dict(CAL_0529_V2_HALF_TF_BASELINE_PARAMS)
    fit_params.update(params or {})
    fit_params = _validate_half_tf_params(fit_params)
    geometry = half_tf_geometry_from_formula(
        W_um,
        R,
        WlineR,
        line_length_scale=float(fit_params["line_length_scale"]),
        gnd_width_factor=float(fit_params["GND_width_factor"]),
    )
    return write_sgdvcl_s4p_from_geometry_fast(
        WLine_um=geometry["derived_WLine_um"],
        L_MA_um=geometry["L_MA_um"],
        L_E1_um=geometry["L_E1_um"],
        L_GND_open_um=geometry["L_GND_open_um"],
        W_GND_open_um=geometry["W_GND_open_um"],
        params=fit_params,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one fast Cal_0529_v2 SG-DVCL S4P.")
    parser.add_argument("--length-um", type=float, default=INPUT_SGDVCL_LENGTH_UM)
    parser.add_argument("--width-um", type=float, default=INPUT_SGDVCL_WIDTH_UM)
    parser.add_argument("--gnd-open-factor", type=float, default=INPUT_GND_OPEN_FACTOR)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_S4P_PATH)
    parser.add_argument("--freq-start-ghz", type=float, default=FREQ_START_GHZ)
    parser.add_argument("--freq-stop-ghz", type=float, default=FREQ_STOP_GHZ)
    parser.add_argument("--freq-npoints", type=int, default=FREQ_NPOINTS)
    parser.add_argument("--write-only", action="store_true")
    parser.add_argument("--memory-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = {
        "frequency_hz": build_frequency_hz(
            freq_start_ghz=args.freq_start_ghz,
            freq_stop_ghz=args.freq_stop_ghz,
            freq_npoints=args.freq_npoints,
        ),
        "output_dir": str(args.output_path.parent),
        "filename": args.output_path.name,
        "quiet": True,
        "write_manifest": True,
    }
    if args.write_only:
        path = write_sgdvcl_s4p_from_geometry_fast(
            WLine_um=args.width_um,
            L_MA_um=args.length_um,
            L_E1_um=args.length_um - 2.0,
            L_GND_open_um=args.length_um - 4.0,
            W_GND_open_um=args.gnd_open_factor * args.width_um,
            params=params,
        )
        print(path)
        return 0
    ntw = calculate_sgdvcl_s4p_from_geometry_fast(
        WLine_um=args.width_um,
        L_MA_um=args.length_um,
        L_E1_um=args.length_um - 2.0,
        L_GND_open_um=args.length_um - 4.0,
        W_GND_open_um=args.gnd_open_factor * args.width_um,
        params=params,
    )
    if not args.memory_only:
        ntw.write_touchstone(str(args.output_path.with_suffix("")))
        print(args.output_path)
    else:
        print("<memory-only>")
    print(f"nports={ntw.nports}, nfreq={len(ntw.frequency.f)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

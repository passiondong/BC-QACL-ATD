#!/usr/bin/env python3
"""Generate three v3 transformer S6P blocks and cascade them with transistors.

Default topology:

    term1 -- IMN(out1), IMN(out2=GND), IMN(in1/in2) -- driver(in+/in-)
    driver(out+/out-) -- ISMN(in1/in2), ISMN(out1/out2) -- final(in+/in-)
    final(out+/out-) -- OMN(in1/in2), OMN(out1) -- term2, OMN(out2=GND)

For every transformer, E1TAP is grounded and MATAP is open in the PA cascade.
The generated S6P port order is:

    1=in1, 2=in2, 3=out1, 4=out2, 5=E1TAP, 6=MATAP
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf

import TF_Cal_S6P_Predicting_0504 as tf0504
import tf_analysis_pipeline_cli_v3 as tf_v3
from pa_synthesis.data_loaders import load_transistor_s4p
from pa_synthesis.network_utils import build_full_pa_network


ROOT = Path(__file__).resolve().parent
DEFAULT_TRANSISTOR_DIR = Path(
    r".\HFSS\For_Paper\ForModelling"
    r"\Predict_Model_Compare_metric\transistor_z4p_s4p"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "three_tf_v3_pa_cascade"

SIX_PORT_NAMES = ["in1", "in2", "out1", "out2", "E1TAP", "MATAP"]
S4P_PORT_ORDER = ["E1_A", "MA_B", "E1_B", "MA_A"]
FIXED_L24_DISABLED_OPEN_NH = 1.0e9
DEFAULT_VIEW_MIN_GHZ = 10.0
DEFAULT_VIEW_MAX_GHZ = 110.0
DEFAULT_MAG_FLOOR_DB = -30.0
PA_VIEW_SPARAMS = (("S11", 0, 0), ("S21", 1, 0), ("S22", 1, 1))


@dataclass(frozen=True)
class TransformerSpec:
    role: str
    W_um: float
    R: float
    WlineR: float

    @property
    def case_id(self) -> str:
        return f"{self.role}_W{tag(self.W_um)}_R{tag(self.R)}_WlineR{tag(self.WlineR)}"


@dataclass(frozen=True)
class TransformerBuild:
    spec: TransformerSpec
    L13_nH: float
    L56_pH: float
    geometry_L_MA_um: float
    Wline_um: float
    GND_width_factor: float
    s4p_path: Path
    s6p_path: Path
    network: rf.Network


DEFAULT_TF_SPECS = (
    TransformerSpec("input_match", 111.0, 1.47, 0.24),
    TransformerSpec("interstage_match", 106.0, 1.26, 0.23),
    TransformerSpec("output_match", 101.0, 1.65, 0.25),
)


def tag(value: float, *, digits: int = 8) -> str:
    text = f"{float(value):.{digits}g}"
    return text.replace("-", "m").replace(".", "p")


def db20(value: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(value), 1e-30))


def clipped_db20(value: np.ndarray, mag_floor_db: float) -> np.ndarray:
    return np.maximum(db20(value), float(mag_floor_db))


def frequency_view_mask(freq_ghz: np.ndarray, min_ghz: float, max_ghz: float) -> np.ndarray:
    eps = 1e-9
    return (freq_ghz >= float(min_ghz) - eps) & (freq_ghz <= float(max_ghz) + eps)


def infer_uniform_frequency_grid(freq_hz: np.ndarray) -> tuple[float, float, float, int]:
    freq_ghz = np.asarray(freq_hz, dtype=float) / 1e9
    if len(freq_ghz) < 2:
        raise ValueError("Frequency grid must contain at least two points.")
    diffs = np.diff(freq_ghz)
    step = float(np.median(diffs))
    if not np.allclose(diffs, step, rtol=1e-6, atol=1e-9):
        raise ValueError("Transformer S6P builder expects a uniform frequency grid.")
    return float(freq_ghz[0]), float(freq_ghz[-1]), step, int(len(freq_ghz))


def write_touchstone_with_port_comments(ntw: rf.Network, path_without_suffix: Path, port_names: Sequence[str]) -> Path:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    ntw.write_touchstone(str(path_without_suffix))
    out_path = path_without_suffix.with_suffix(f".s{ntw.nports}p")
    comments = "\n".join(f"! Port[{idx + 1}] = {name}" for idx, name in enumerate(port_names))
    text = out_path.read_text(encoding="utf-8", errors="ignore")
    if "! Port[1] =" not in text:
        out_path.write_text(f"{comments}\n{text}", encoding="utf-8")
    return out_path.resolve()


def build_transformer_s6p(
    spec: TransformerSpec,
    model: tf_v3.LFitModel,
    *,
    output_dir: Path,
    freq_start_ghz: float,
    freq_stop_ghz: float,
    freq_step_ghz: float,
    freq_npoints: int,
    allow_extrapolation: bool,
) -> TransformerBuild:
    result = tf_v3.analyze_one(
        model,
        spec.W_um,
        spec.R,
        spec.WlineR,
        output_dir=output_dir / "generated_s4p",
        freq_start_ghz=freq_start_ghz,
        freq_stop_ghz=freq_stop_ghz,
        freq_npoints=freq_npoints,
        allow_extrapolation=allow_extrapolation,
        generate_s4p=True,
        generate_plot=False,
        quiet=True,
    )
    if result.s4p_path is None:
        raise RuntimeError(f"{spec.case_id}: v3 S4P generation was disabled unexpectedly.")

    cfg = dict(tf0504.CFG)
    cfg.update(
        {
            "s4p_top": str(result.s4p_path),
            "s4p_bot": str(result.s4p_path),
            "f_start_ghz": float(freq_start_ghz),
            "f_stop_ghz": float(freq_stop_ghz),
            "f_step_ghz": float(freq_step_ghz),
            "z0": 50.0,
            "L13_nH": float(result.L13_nH),
            "L24_nH": FIXED_L24_DISABLED_OPEN_NH,
            "L56_pH": float(result.L56_pH),
            "s4p_port_order": S4P_PORT_ORDER,
        }
    )
    ntw = tf0504.build_tf_6port(cfg)
    ntw.name = spec.case_id
    ntw.port_names = list(SIX_PORT_NAMES)
    s6p_path = write_touchstone_with_port_comments(
        ntw,
        output_dir / "generated_s6p" / spec.case_id,
        SIX_PORT_NAMES,
    )
    return TransformerBuild(
        spec=spec,
        L13_nH=float(result.L13_nH),
        L56_pH=float(result.L56_pH),
        geometry_L_MA_um=float(result.geometry_L_MA_um),
        Wline_um=float(result.Wline_um),
        GND_width_factor=float(result.GND_width_factor),
        s4p_path=result.s4p_path,
        s6p_path=s6p_path,
        network=ntw,
    )


def sparam_frame(
    ntw: rf.Network,
    *,
    min_ghz: float,
    max_ghz: float,
    mag_floor_db: float,
) -> pd.DataFrame:
    freq_ghz = ntw.f / 1e9
    mask = frequency_view_mask(freq_ghz, min_ghz, max_ghz)
    data: dict[str, np.ndarray] = {"frequency_ghz": freq_ghz[mask]}
    for label, i, j in PA_VIEW_SPARAMS:
        value = ntw.s[:, i, j][mask]
        data[f"{label}_mag_db"] = clipped_db20(value, mag_floor_db)
        data[f"{label}_phase_deg"] = np.unwrap(np.angle(value)) * 180.0 / np.pi
        data[f"{label}_real"] = np.real(value)
        data[f"{label}_imag"] = np.imag(value)
    return pd.DataFrame(data)


def write_key_frequency_summary(
    ntw: rf.Network,
    out_path: Path,
    key_freqs_ghz: Sequence[float],
    *,
    min_ghz: float,
    max_ghz: float,
    mag_floor_db: float,
) -> None:
    freq_ghz = ntw.f / 1e9
    rows: list[dict[str, float]] = []
    for target in key_freqs_ghz:
        if float(target) < float(min_ghz) or float(target) > float(max_ghz):
            continue
        idx = int(np.argmin(np.abs(freq_ghz - float(target))))
        row: dict[str, float] = {
            "target_frequency_ghz": float(target),
            "actual_frequency_ghz": float(freq_ghz[idx]),
        }
        for label, i, j in PA_VIEW_SPARAMS:
            value = ntw.s[idx, i, j]
            row[f"{label}_mag_db"] = float(clipped_db20(np.asarray([value]), mag_floor_db)[0])
            row[f"{label}_phase_deg"] = float(np.angle(value, deg=True))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def plot_pa_sparams(
    ntw: rf.Network,
    out_path: Path,
    *,
    min_ghz: float,
    max_ghz: float,
    mag_floor_db: float,
) -> None:
    freq_ghz = ntw.f / 1e9
    mask = frequency_view_mask(freq_ghz, min_ghz, max_ghz)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True, constrained_layout=True)
    for label, i, j in PA_VIEW_SPARAMS:
        value = ntw.s[:, i, j][mask]
        axes[0].plot(freq_ghz[mask], clipped_db20(value, mag_floor_db), label=label)
        axes[1].plot(freq_ghz[mask], np.unwrap(np.angle(value)) * 180.0 / np.pi, label=label)
    axes[0].set_ylabel("Magnitude (dB)")
    axes[1].set_ylabel("Unwrapped phase (deg)")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[0].set_title(f"Cascaded PA S-parameters, {min_ghz:g}-{max_ghz:g} GHz")
    axes[0].set_ylim(bottom=mag_floor_db)
    axes[0].set_xlim(min_ghz, max_ghz)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_transformer_sparams(
    builds: Sequence[TransformerBuild],
    out_path: Path,
    *,
    min_ghz: float,
    max_ghz: float,
    mag_floor_db: float,
) -> None:
    fig, axes = plt.subplots(len(builds), 1, figsize=(9.5, 3.3 * len(builds)), sharex=True, constrained_layout=True)
    if len(builds) == 1:
        axes = [axes]
    for ax, build in zip(axes, builds):
        ntw = build.network
        freq_ghz = ntw.f / 1e9
        mask = frequency_view_mask(freq_ghz, min_ghz, max_ghz)
        for label, i, j in [
            ("S11", 0, 0),
            ("S33", 2, 2),
            ("S31", 2, 0),
            ("S42", 3, 1),
            ("S55", 4, 4),
            ("S66", 5, 5),
        ]:
            ax.plot(freq_ghz[mask], clipped_db20(ntw.s[:, i, j][mask], mag_floor_db), label=label)
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(build.spec.case_id)
        ax.set_ylim(bottom=mag_floor_db)
        ax.set_xlim(min_ghz, max_ghz)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=6, fontsize=8)
    axes[-1].set_xlabel("Frequency (GHz)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_manifest(
    *,
    out_path: Path,
    builds: Sequence[TransformerBuild],
    driver_path: Path,
    final_path: Path,
    pa_s2p_path: Path,
    pa_csv_path: Path,
    pa_summary_path: Path,
    pa_plot_path: Path,
    tf_plot_path: Path,
    freq_start_ghz: float,
    freq_stop_ghz: float,
    freq_step_ghz: float,
    view_min_ghz: float,
    view_max_ghz: float,
    mag_floor_db: float,
) -> None:
    data = {
        "frequency": {
            "start_ghz": freq_start_ghz,
            "stop_ghz": freq_stop_ghz,
            "step_ghz": freq_step_ghz,
        },
        "view": {
            "sparameters": [item[0] for item in PA_VIEW_SPARAMS],
            "min_ghz": view_min_ghz,
            "max_ghz": view_max_ghz,
            "mag_floor_db": mag_floor_db,
            "note": "CSV and PNG view outputs omit S12, keep only this frequency span, and clip dB magnitudes below the floor.",
        },
        "transistor_files": {
            "driver": str(driver_path),
            "final": str(final_path),
        },
        "port_convention": {
            "transformer_s6p": "1=in1, 2=in2, 3=out1, 4=out2, 5=E1TAP, 6=MATAP",
            "transistor_s4p": "1=in_plus, 2=in_minus, 3=out_plus, 4=out_minus",
            "cascade": (
                "IMN out1=term1, IMN out2=ground, IMN in1/in2=driver inputs; "
                "driver outputs=ISMN in1/in2; ISMN out1/out2=final inputs; "
                "final outputs=OMN in1/in2; OMN out1=term2, OMN out2=ground; "
                "all E1TAP=ground and MATAP=open"
            ),
        },
        "transformers": [
            {
                "role": build.spec.role,
                "W_um": build.spec.W_um,
                "R": build.spec.R,
                "WlineR": build.spec.WlineR,
                "L13_nH": build.L13_nH,
                "L56_pH": build.L56_pH,
                "L24_nH": FIXED_L24_DISABLED_OPEN_NH,
                "geometry_L_MA_um": build.geometry_L_MA_um,
                "Wline_um": build.Wline_um,
                "GND_width_factor": build.GND_width_factor,
                "s4p_path": str(build.s4p_path),
                "s6p_path": str(build.s6p_path),
            }
            for build in builds
        ],
        "outputs": {
            "pa_s2p": str(pa_s2p_path),
            "pa_sparameter_csv": str(pa_csv_path),
            "pa_key_frequency_summary_csv": str(pa_summary_path),
            "pa_plot": str(pa_plot_path),
            "transformer_plot": str(tf_plot_path),
        },
    }
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    driver_path = Path(args.transistor_dir) / "driver_2x12_single_ended_z0_50.s4p"
    final_path = Path(args.transistor_dir) / "final_2x18_single_ended_z0_50.s4p"
    driver = load_transistor_s4p(driver_path, z0=50.0, name="driver_2x12")
    final = load_transistor_s4p(final_path, target_freq_hz=driver.f, z0=50.0, name="final_2x18")
    freq_start_ghz, freq_stop_ghz, freq_step_ghz, freq_npoints = infer_uniform_frequency_grid(driver.f)

    model = tf_v3.load_l_fit_model(data_source=args.fit_data, round_decimals=args.round_decimals)
    builds = [
        build_transformer_s6p(
            spec,
            model,
            output_dir=out_dir,
            freq_start_ghz=freq_start_ghz,
            freq_stop_ghz=freq_stop_ghz,
            freq_step_ghz=freq_step_ghz,
            freq_npoints=freq_npoints,
            allow_extrapolation=args.allow_extrapolation,
        )
        for spec in DEFAULT_TF_SPECS
    ]

    pa = build_full_pa_network(
        freq_hz=driver.f,
        driver_s4p=driver,
        final_s4p=final,
        imn=builds[0].network,
        ismn=builds[1].network,
        omn=builds[2].network,
        z0=50.0,
        include_dc_blocks=args.include_dc_blocks,
    )
    pa.name = "three_tf_v3_pa_cascade"
    pa.port_names = ["term1_50ohm", "term2_50ohm"]

    pa_s2p_path = write_touchstone_with_port_comments(
        pa,
        out_dir / "three_tf_v3_pa_cascade",
        pa.port_names,
    )
    pa_csv_path = out_dir / "three_tf_v3_pa_cascade_sparams.csv"
    sparam_frame(
        pa,
        min_ghz=args.view_min_ghz,
        max_ghz=args.view_max_ghz,
        mag_floor_db=args.mag_floor_db,
    ).to_csv(pa_csv_path, index=False, encoding="utf-8-sig")
    pa_summary_path = out_dir / "three_tf_v3_pa_cascade_key_frequency_summary.csv"
    write_key_frequency_summary(
        pa,
        pa_summary_path,
        args.key_freq_ghz,
        min_ghz=args.view_min_ghz,
        max_ghz=args.view_max_ghz,
        mag_floor_db=args.mag_floor_db,
    )

    pa_plot_path = out_dir / "three_tf_v3_pa_cascade_sparams.png"
    tf_plot_path = out_dir / "three_tf_v3_transformer_s6p_selected_sparams.png"
    plot_pa_sparams(
        pa,
        pa_plot_path,
        min_ghz=args.view_min_ghz,
        max_ghz=args.view_max_ghz,
        mag_floor_db=args.mag_floor_db,
    )
    plot_transformer_sparams(
        builds,
        tf_plot_path,
        min_ghz=args.view_min_ghz,
        max_ghz=args.view_max_ghz,
        mag_floor_db=args.mag_floor_db,
    )

    build_rows = []
    for build in builds:
        row = {
            **asdict(build.spec),
            "case_id": build.spec.case_id,
            "L13_nH": build.L13_nH,
            "L56_pH": build.L56_pH,
            "L24_nH": FIXED_L24_DISABLED_OPEN_NH,
            "geometry_L_MA_um": build.geometry_L_MA_um,
            "Wline_um": build.Wline_um,
            "GND_width_factor": build.GND_width_factor,
            "s4p_path": str(build.s4p_path),
            "s6p_path": str(build.s6p_path),
        }
        build_rows.append(row)
    pd.DataFrame(build_rows).to_csv(out_dir / "three_tf_v3_transformer_build_summary.csv", index=False, encoding="utf-8-sig")

    write_manifest(
        out_path=out_dir / "three_tf_v3_pa_cascade_manifest.json",
        builds=builds,
        driver_path=driver_path,
        final_path=final_path,
        pa_s2p_path=pa_s2p_path,
        pa_csv_path=pa_csv_path.resolve(),
        pa_summary_path=pa_summary_path.resolve(),
        pa_plot_path=pa_plot_path.resolve(),
        tf_plot_path=tf_plot_path.resolve(),
        freq_start_ghz=freq_start_ghz,
        freq_stop_ghz=freq_stop_ghz,
        freq_step_ghz=freq_step_ghz,
        view_min_ghz=args.view_min_ghz,
        view_max_ghz=args.view_max_ghz,
        mag_floor_db=args.mag_floor_db,
    )
    print(out_dir.resolve())
    return out_dir.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fit-data", type=Path, default=tf_v3.DEFAULT_FIT_DATA)
    parser.add_argument("--transistor-dir", type=Path, default=DEFAULT_TRANSISTOR_DIR)
    parser.add_argument("--round-decimals", type=int, default=2)
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--include-dc-blocks", action="store_true")
    parser.add_argument(
        "--key-freq-ghz",
        type=float,
        nargs="*",
        default=[10.0, 30.0, 60.0, 80.0, 110.0],
        help="Frequencies summarized in the key-frequency CSV.",
    )
    parser.add_argument("--view-min-ghz", type=float, default=DEFAULT_VIEW_MIN_GHZ)
    parser.add_argument("--view-max-ghz", type=float, default=DEFAULT_VIEW_MAX_GHZ)
    parser.add_argument("--mag-floor-db", type=float, default=DEFAULT_MAG_FLOOR_DB)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Global predicted PA optimization using the Cal_0521 L13-only TF core.

This is the Cal_0521 analogue of ``optimize_predicted_pa_global_center_band.py``.
The PA-level objective, search grid, ranking logic, transistor cascade, and
Zin/ZOPT metric are intentionally kept unchanged.  Only the inner-loop
transformer builder is replaced:

    tf_analysis_pipeline_cli_v3  ->  tf_analysis_pipeline_cli_0521

The Cal_0521 core generates the SG-DVCL S4P and the L13-only six-port
prediction model for each queried W/R/WlineR geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import skrf as rf

import tf_analysis_pipeline_cli_0521 as tf0521
from optimize_predicted_pa_cascade_cmaes import Combo, Evaluator, Triple, cma_es
from optimize_predicted_pa_global_center_band import (
    GLOBAL_GRIDS,
    CenterBandEvaluator,
    add_rank,
    combo_to_normalized_global,
    nearest_triple,
    normalized_to_combo_global,
    plot_top20,
)
from pa_synthesis.data_loaders import load_loadpull_zopt, load_transistor_s4p
from pa_synthesis.network_utils import align_network
from run_three_tf_v3_pa_cascade import SIX_PORT_NAMES, infer_uniform_frequency_grid


ROOT = Path(__file__).resolve().parent
DEFAULT_METRIC_DIR = Path(r".\HFSS\For_Paper\ForModelling\Predict_Model_Compare_metric")
DEFAULT_TRANSISTOR_DIR = DEFAULT_METRIC_DIR / "transistor_z4p_s4p"
DEFAULT_LOADPULL_XLSX = DEFAULT_METRIC_DIR / "loadpull_marker_stats_30_80GHz.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "predicted_pa_global_center_band_16_19_cal0521"


class Cal0521PredictedBuilder:
    """Build and cache Cal_0521 predicted six-port transformer networks."""

    def __init__(
        self,
        *,
        output_dir: Path,
        freq_hz: np.ndarray,
        allow_extrapolation: bool,
        l13_model: str,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.freq_hz = np.asarray(freq_hz, dtype=float)
        self.allow_extrapolation = bool(allow_extrapolation)
        self.l13_model = str(l13_model)
        self.freq_start_ghz, self.freq_stop_ghz, self.freq_step_ghz, self.freq_npoints = infer_uniform_frequency_grid(
            self.freq_hz
        )
        self.cache: dict[tuple[str, tuple[float, float, float]], rf.Network] = {}
        self.metadata_cache: dict[tuple[str, tuple[float, float, float]], dict[str, object]] = {}

    def get(self, role: str, triple: Triple) -> rf.Network:
        key = (role, triple.key())
        if key in self.cache:
            return self.cache[key]

        role_dir = self.output_dir / role
        result = tf0521.run_pipeline(
            triple.W,
            triple.R,
            triple.WlineR,
            output_dir=role_dir,
            freq_start_ghz=self.freq_start_ghz,
            freq_stop_ghz=self.freq_stop_ghz,
            freq_step_ghz=self.freq_step_ghz,
            l13_model=self.l13_model,
            allow_extrapolation=self.allow_extrapolation,
            write_s4p_file=True,
            write_s6p_file=True,
        )
        if result.s6p_path is None:
            raise RuntimeError(f"Cal_0521 S6P generation failed for {role} {triple.label()}.")

        ntw = rf.Network(result.s6p_path)
        ntw = align_network(ntw, self.freq_hz, f"{role}_{triple.label()}_cal0521")
        ntw.port_names = list(SIX_PORT_NAMES)
        self.cache[key] = ntw
        self.metadata_cache[key] = {
            "role": role,
            "W": triple.W,
            "R": triple.R,
            "WlineR": triple.WlineR,
            "L13_nH": result.L13_nH,
            "L24_nH": result.L24_nH,
            "L56_pH": result.L56_pH,
            "s4p_path": result.s4p_path,
            "s6p_path": result.s6p_path,
            **{f"geometry_{k}": v for k, v in result.geometry.items()},
        }
        return ntw

    def write_metadata(self, out_path: Path) -> None:
        rows = list(self.metadata_cache.values())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "predicted_transformers"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    driver = load_transistor_s4p(
        Path(args.transistor_dir) / "driver_2x12_single_ended_z0_50.s4p",
        z0=50.0,
        name="driver_2x12",
    )
    final = load_transistor_s4p(
        Path(args.transistor_dir) / "final_2x18_single_ended_z0_50.s4p",
        target_freq_hz=driver.f,
        z0=50.0,
        name="final_2x18",
    )
    loadpull = load_loadpull_zopt(args.loadpull_xlsx)

    builder = Cal0521PredictedBuilder(
        output_dir=pred_dir,
        freq_hz=driver.f,
        allow_extrapolation=args.allow_extrapolation,
        l13_model=args.l13_model,
    )
    evaluator = Evaluator(
        builder=builder,
        driver=driver,
        final=final,
        loadpull_zopt=loadpull.zopt_single,
        loadpull_freq_hz=loadpull.freq_hz,
    )
    center_eval = CenterBandEvaluator(
        evaluator=evaluator,
        full_lo=args.full_lo_ghz,
        full_hi=args.full_hi_ghz,
        gain_min=args.gain_min_db,
        gain_max=args.gain_max_db,
        zin_weight=args.zin_weight,
    )

    def objective(x: np.ndarray) -> float:
        return float(center_eval.evaluate(normalized_to_combo_global(x))["center_band_score"])

    target_combo = Combo(
        nearest_triple("input_match", 114.0, 1.45, 0.22),
        nearest_triple("interstage_match", 106.5, 1.20, 0.24),
        nearest_triple("output_match", 103.0, 1.80, 0.24),
    )
    initial = [
        target_combo,
        Combo(
            nearest_triple("input_match", 108.5, 1.44, 0.22),
            nearest_triple("interstage_match", 106.5, 1.20, 0.22),
            nearest_triple("output_match", 102.5, 1.80, 0.24),
        ),
        Combo(
            nearest_triple("input_match", 114.0, 1.40, 0.21),
            nearest_triple("interstage_match", 103.0, 1.10, 0.18),
            nearest_triple("output_match", 112.0, 2.00, 0.24),
        ),
        Combo(
            nearest_triple("input_match", 110.0, 1.40, 0.15),
            nearest_triple("interstage_match", 103.0, 1.26, 0.23),
            nearest_triple("output_match", 100.0, 2.00, 0.25),
        ),
    ]
    rng = np.random.default_rng(args.seed)
    while len(initial) < args.restarts:
        initial.append(normalized_to_combo_global(rng.random(9)))

    traces: list[dict[str, float]] = []
    evals_per_restart = max(args.popsize, int(np.ceil(args.max_evals / max(1, args.restarts))))
    for idx in range(args.restarts):
        trace, best_x, best_score = cma_es(
            objective,
            x0=combo_to_normalized_global(initial[idx]),
            sigma0=args.sigma,
            max_evals=evals_per_restart,
            popsize=args.popsize,
            seed=args.seed + idx * 4211,
            restart_index=idx + 1,
        )
        traces.extend(trace)
        center_eval.evaluate(normalized_to_combo_global(best_x), source=f"center_restart_{idx + 1}_best")
        print(f"restart {idx + 1}/{args.restarts}: best_score={best_score:.3f}, unique={len(center_eval.cache)}", flush=True)

    center_eval.evaluate(target_combo, source="requested_anchor")
    ranked = add_rank(pd.DataFrame(center_eval.cache.values()))
    ranked.to_csv(out_dir / "center_band_ranked_candidates_all.csv", index=False, encoding="utf-8-sig")
    ranked.head(20).to_csv(out_dir / "center_band_top20_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(traces).to_csv(out_dir / "center_band_cmaes_trace.csv", index=False, encoding="utf-8-sig")
    builder.write_metadata(out_dir / "cal0521_transformer_build_metadata.csv")
    plot_top20(ranked, center_eval, out_dir)

    anchor_mask = (
        np.isclose(ranked["input_W"], target_combo.imn.W)
        & np.isclose(ranked["input_R"], target_combo.imn.R)
        & np.isclose(ranked["input_WlineR"], target_combo.imn.WlineR)
        & np.isclose(ranked["interstage_W"], target_combo.ismn.W)
        & np.isclose(ranked["interstage_R"], target_combo.ismn.R)
        & np.isclose(ranked["interstage_WlineR"], target_combo.ismn.WlineR)
        & np.isclose(ranked["output_W"], target_combo.omn.W)
        & np.isclose(ranked["output_R"], target_combo.omn.R)
        & np.isclose(ranked["output_WlineR"], target_combo.omn.WlineR)
    )
    anchor = ranked.loc[anchor_mask].iloc[0].to_dict()
    pd.DataFrame([anchor]).to_csv(out_dir / "requested_anchor_rank.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "core": "Cal_0521 L13-only six-port prediction model",
        "cal_0521_path": str((ROOT / "Cal_0521.py").resolve()),
        "pipeline": str((ROOT / "tf_analysis_pipeline_cli_0521.py").resolve()),
        "unique_candidates": int(len(ranked)),
        "full_band_ghz": [float(args.full_lo_ghz), float(args.full_hi_ghz)],
        "gain_window_db": [float(args.gain_min_db), float(args.gain_max_db)],
        "l13_model": args.l13_model,
        "grid": {
            role: {
                key: [float(values[0]), float(values[-1]), float(values[1] - values[0]) if len(values) > 1 else 0.0]
                for key, values in grids.items()
            }
            for role, grids in GLOBAL_GRIDS.items()
        },
        "full_30_90_ok_count": int(ranked["full_30_90_ok"].sum()),
        "best": ranked.iloc[0].to_dict(),
        "requested_anchor": anchor,
        "top20_csv": str((out_dir / "center_band_top20_candidates.csv").resolve()),
        "all_ranked_csv": str((out_dir / "center_band_ranked_candidates_all.csv").resolve()),
        "anchor_csv": str((out_dir / "requested_anchor_rank.csv").resolve()),
    }
    (out_dir / "center_band_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"output_dir={out_dir.resolve()}")
    print(f"unique_candidates={len(ranked)}")
    print(f"full_30_90_ok_count={int(ranked['full_30_90_ok'].sum())}")
    print("best")
    print(
        ranked.iloc[0][
            [
                "rank",
                "input_W",
                "input_R",
                "input_WlineR",
                "interstage_W",
                "interstage_R",
                "interstage_WlineR",
                "output_W",
                "output_R",
                "output_WlineR",
                "full_30_90_ok",
                "centered_window_lower_ghz",
                "centered_window_upper_ghz",
                "centered_window_width_ghz",
                "full_violation_rms_db",
                "full_violation_max_db",
                "full_gain_min_db",
                "full_gain_max_db",
                "omn_zin_rms_to_zopt_ohm",
            ]
        ]
    )
    print("requested_anchor")
    print(
        pd.Series(anchor)[
            [
                "rank",
                "input_W",
                "input_R",
                "input_WlineR",
                "interstage_W",
                "interstage_R",
                "interstage_WlineR",
                "output_W",
                "output_R",
                "output_WlineR",
                "full_30_90_ok",
                "centered_window_lower_ghz",
                "centered_window_upper_ghz",
                "centered_window_width_ghz",
                "full_violation_rms_db",
                "full_violation_max_db",
                "full_gain_min_db",
                "full_gain_max_db",
                "omn_zin_rms_to_zopt_ohm",
            ]
        ]
    )
    return out_dir.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transistor-dir", type=Path, default=DEFAULT_TRANSISTOR_DIR)
    parser.add_argument("--loadpull-xlsx", type=Path, default=DEFAULT_LOADPULL_XLSX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--l13-model", choices=["anchors8", "full80-log-trilinear"], default="anchors8")
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--full-lo-ghz", type=float, default=30.0)
    parser.add_argument("--full-hi-ghz", type=float, default=90.0)
    parser.add_argument("--gain-min-db", type=float, default=16.0)
    parser.add_argument("--gain-max-db", type=float, default=19.0)
    parser.add_argument("--zin-weight", type=float, default=1.0)
    parser.add_argument("--max-evals", type=int, default=900)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--popsize", type=int, default=18)
    parser.add_argument("--sigma", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

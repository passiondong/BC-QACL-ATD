#!/usr/bin/env python3
"""Global PA synthesis with Cal_0529_v2/Cal_0529_speed and Ltot <= 480 um.

The final Pareto objectives match the Cal_0521_v2 length-constrained synthesis:

* maximize relative-peak -3 dB overlap inside 30-80 GHz,
* minimize OMN Zin/ZOPT RMS error,
* maximize S21 peak within the 16.9-20.1 dB gain window.

The requested manuscript geometry box is used only for reporting whether any
archive Pareto-front point falls in that practical sub-box.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import skrf as rf

import Cal_0529_speed
import tf_analysis_pipeline_cli_0529_v2 as tf0529
from optimize_predicted_pa_cascade_cmaes import Combo, Triple, cma_es
from optimize_predicted_pa_global_anchor_objective_cal0521_v2 import FastPAEvaluator
from optimize_predicted_pa_global_center_band import (
    GLOBAL_GRIDS,
    combo_to_normalized_global,
    nearest_triple,
    normalized_to_combo_global,
)
from optimize_predicted_pa_global_relative_3db_len485_pareto_cal0524_speed import (
    GlobalLengthParetoEvaluator,
    add_basic_rank,
    add_percentile_rank,
    combo_from_row,
    combo_length_um,
    coordinate_polish,
    grid_count_manifest,
    nondominated_mask,
    plot_pareto,
    plot_top_s21,
    seed_combos_from_csv,
)
from optimize_predicted_pa_local_anchor_range_cal0521_v2 import infer_uniform_frequency_grid
from pa_synthesis.data_loaders import load_loadpull_zopt, load_transistor_s4p
from pa_synthesis.network_utils import align_network
from run_three_tf_v3_pa_cascade import SIX_PORT_NAMES


ROOT = Path(__file__).resolve().parent
DEFAULT_METRIC_DIR = Path(r".\HFSS\For_Paper\ForModelling\Predict_Model_Compare_metric")
DEFAULT_TRANSISTOR_DIR = DEFAULT_METRIC_DIR / "transistor_z4p_s4p"
DEFAULT_LOADPULL_XLSX = DEFAULT_METRIC_DIR / "loadpull_marker_stats_30_80GHz.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "predicted_pa_global_relative_3db_len480_pareto_cal0529_speed_20260608"

ROLES = ["input_match", "interstage_match", "output_match"]
ATTRS = ["W", "R", "WlineR"]

REQUESTED_BOX = {
    "input_match": {"W": (111.0, 117.0), "R": (1.32, 1.42), "WlineR": (0.238, 0.258)},
    "interstage_match": {"W": (103.0, 106.0), "R": (1.21, 1.31), "WlineR": (0.22, 0.24)},
    "output_match": {"W": (100.0, 103.0), "R": (1.70, 1.80), "WlineR": (0.24, 0.26)},
}


def combo_in_requested_box(combo: Combo) -> bool:
    triples = {"input_match": combo.imn, "interstage_match": combo.ismn, "output_match": combo.omn}
    for role, triple in triples.items():
        for attr, value in zip(ATTRS, triple.key()):
            lo, hi = REQUESTED_BOX[role][attr]
            if not (lo - 1e-12 <= float(value) <= hi + 1e-12):
                return False
    return True


def row_in_requested_box(row: pd.Series) -> bool:
    return combo_in_requested_box(combo_from_row(row))


def random_combos_global(count: int, seed: int, *, length_limit_um: float | None = None) -> list[Combo]:
    rng = np.random.default_rng(seed)
    out: list[Combo] = []
    attempts = 0
    max_attempts = max(5000, count * 250)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        triples = []
        for role in ROLES:
            grids = GLOBAL_GRIDS[role]
            triples.append(
                Triple(
                    float(rng.choice(grids["W"])),
                    float(rng.choice(grids["R"])),
                    float(rng.choice(grids["WlineR"])),
                )
            )
        combo = Combo(*triples)
        if length_limit_um is None or combo_length_um(combo) <= length_limit_um + 1e-12:
            out.append(combo)
    return out


def random_combos_requested_box(count: int, seed: int, *, length_limit_um: float | None = None) -> list[Combo]:
    rng = np.random.default_rng(seed)
    out: list[Combo] = []
    attempts = 0
    max_attempts = max(5000, count * 250)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        triples = []
        for role in ROLES:
            ranges = REQUESTED_BOX[role]
            vals = []
            for attr in ATTRS:
                grid = GLOBAL_GRIDS[role][attr]
                lo, hi = ranges[attr]
                sub = grid[(grid >= lo - 1e-12) & (grid <= hi + 1e-12)]
                vals.append(float(rng.choice(sub)))
            triples.append(Triple(*vals))
        combo = Combo(*triples)
        if length_limit_um is None or combo_length_um(combo) <= length_limit_um + 1e-12:
            out.append(combo)
    return out


class ReuseCal0529SpeedPredictedBuilder:
    """S6P builder using Cal_0529_speed write-only S4P generation."""

    def __init__(
        self,
        *,
        output_dir: Path,
        reuse_roots: Sequence[Path],
        freq_hz: np.ndarray,
        allow_extrapolation: bool,
        l13_model: str,
        line_length_scale: float | None,
        gnd_width_factor: float | None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.reuse_roots = [Path(root) for root in reuse_roots if Path(root).exists()]
        self.freq_hz = np.asarray(freq_hz, dtype=float)
        self.allow_extrapolation = bool(allow_extrapolation)
        self.l13_model = str(l13_model)
        self.line_length_scale = None if line_length_scale is None else float(line_length_scale)
        self.gnd_width_factor = None if gnd_width_factor is None else float(gnd_width_factor)
        self.freq_start_ghz, self.freq_stop_ghz, self.freq_step_ghz, self.freq_npoints = infer_uniform_frequency_grid(
            self.freq_hz
        )
        self.cache: dict[tuple[str, tuple[float, float, float]], rf.Network] = {}
        self.metadata_cache: dict[tuple[str, tuple[float, float, float]], dict[str, object]] = {}
        self.reused_count = 0
        self.generated_count = 0
        self.s4p_generation_seconds = 0.0
        self.s6p_assembly_seconds = 0.0

    @staticmethod
    def _s6p_stem(triple: Triple) -> str:
        return (
            "tf_pred_cal0529v2_l13only_"
            f"W{tf0529.base.safe_tag(triple.W)}_"
            f"R{tf0529.base.safe_tag(triple.R)}_"
            f"WlineR{tf0529.base.safe_tag(triple.WlineR)}"
        )

    def _find_existing_s6p(self, role: str, triple: Triple) -> Path | None:
        rel = Path(role) / "predicted_s6p" / f"{self._s6p_stem(triple)}.s6p"
        for root in [self.output_dir, *self.reuse_roots]:
            for candidate in [Path(root) / rel, Path(root) / "predicted_transformers" / rel]:
                if candidate.exists():
                    return candidate
        return None

    def get(self, role: str, triple: Triple) -> rf.Network:
        key = (role, triple.key())
        if key in self.cache:
            return self.cache[key]

        existing = self._find_existing_s6p(role, triple)
        if existing is not None:
            ntw = rf.Network(str(existing))
            ntw = align_network(ntw, self.freq_hz, f"{role}_{triple.label()}_cal0529_reused")
            ntw.port_names = list(SIX_PORT_NAMES)
            self.cache[key] = ntw
            self.metadata_cache[key] = {
                "role": role,
                "W": triple.W,
                "R": triple.R,
                "WlineR": triple.WlineR,
                "s6p_path": str(existing),
                "reuse": True,
            }
            self.reused_count += 1
            return ntw

        role_dir = self.output_dir / role
        s4p_dir = role_dir / "sgdvcl_s4p"
        s4p_dir.mkdir(parents=True, exist_ok=True)
        prediction = tf0529.base.predict_l13_nH(
            triple.W,
            triple.R,
            triple.WlineR,
            model_kind=self.l13_model,
            allow_extrapolation=self.allow_extrapolation,
        )
        params = {
            "frequency_hz": tuple(float(v) for v in self.freq_hz),
            "output_dir": str(s4p_dir),
            "filename": (
                "sgdvcl_cal0529speed_"
                f"W{tf0529.base.safe_tag(triple.W)}_"
                f"R{tf0529.base.safe_tag(triple.R)}_"
                f"WlineR{tf0529.base.safe_tag(triple.WlineR)}.s4p"
            ),
            "quiet": True,
            "write_manifest": True,
        }
        if self.line_length_scale is not None:
            params["line_length_scale"] = float(self.line_length_scale)
        if self.gnd_width_factor is not None:
            params["GND_width_factor"] = float(self.gnd_width_factor)

        geometry = Cal_0529_speed.half_tf_geometry_from_formula(
            triple.W,
            triple.R,
            triple.WlineR,
            line_length_scale=self.line_length_scale,
            gnd_width_factor=self.gnd_width_factor,
        )
        t0 = time.perf_counter()
        s4p_path = Cal_0529_speed.write_sgdvcl_s4p_from_half_tf_fast(
            triple.W,
            triple.R,
            triple.WlineR,
            params=params,
        )
        self.s4p_generation_seconds += time.perf_counter() - t0

        t1 = time.perf_counter()
        pred6 = tf0529.base.build_l13_only_six_port(
            s4p_path,
            prediction.L13_nH,
            freq_start_ghz=self.freq_start_ghz,
            freq_stop_ghz=self.freq_stop_ghz,
            freq_step_ghz=self.freq_step_ghz,
        )
        stem = self._s6p_stem(triple)
        s6p_path = tf0529.base.write_s6p(pred6, role_dir / "predicted_s6p", stem)
        self.s6p_assembly_seconds += time.perf_counter() - t1

        ntw = align_network(pred6, self.freq_hz, f"{role}_{triple.label()}_cal0529_generated")
        ntw.port_names = list(SIX_PORT_NAMES)
        self.cache[key] = ntw
        self.metadata_cache[key] = {
            "role": role,
            "W": triple.W,
            "R": triple.R,
            "WlineR": triple.WlineR,
            "L13_nH": float(prediction.L13_nH),
            "L24_nH": tf0529.L24_OPEN_NH,
            "L56_pH": tf0529.L56_SHORT_PH,
            "s4p_path": str(s4p_path),
            "s6p_path": str(s6p_path),
            "reuse": False,
            **{f"geometry_{k}": v for k, v in geometry.items()},
        }
        self.generated_count += 1
        return ntw

    def write_metadata(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(list(self.metadata_cache.values())).to_csv(out_path, index=False, encoding="utf-8-sig")


def add_requested_box_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["in_requested_box"] = [row_in_requested_box(row) for _, row in out.iterrows()]
    return out


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

    builder = ReuseCal0529SpeedPredictedBuilder(
        output_dir=pred_dir,
        reuse_roots=list(args.reuse_dir or []),
        freq_hz=driver.f,
        allow_extrapolation=args.allow_extrapolation,
        l13_model=args.l13_model,
        line_length_scale=args.line_length_scale,
        gnd_width_factor=args.gnd_width_factor,
    )
    pa_eval = FastPAEvaluator(
        builder=builder,
        driver=driver,
        final=final,
        loadpull_zopt=loadpull.zopt_single,
        loadpull_freq_hz=loadpull.freq_hz,
    )
    evalr = GlobalLengthParetoEvaluator(
        pa_eval=pa_eval,
        target_lo=args.target_lo_ghz,
        target_hi=args.target_hi_ghz,
        fallback_lo=args.fallback_lo_ghz,
        fallback_hi=args.fallback_hi_ghz,
        peak_lo=args.peak_lo_ghz,
        peak_hi=args.peak_hi_ghz,
        gain_min=args.gain_min_db,
        gain_max=args.gain_max_db,
        length_limit_um=args.length_limit_um,
    )

    def objective(x: np.ndarray) -> float:
        return float(evalr.evaluate(normalized_to_combo_global(x))["global_len485_score"])

    seed_combos = [
        Combo(
            nearest_triple("input_match", 116.0, 1.39, 0.25),
            nearest_triple("interstage_match", 106.0, 1.28, 0.22),
            nearest_triple("output_match", 102.0, 1.78, 0.25),
        ),
        Combo(
            nearest_triple("input_match", 115.0, 1.39, 0.24),
            nearest_triple("interstage_match", 106.0, 1.28, 0.22),
            nearest_triple("output_match", 102.0, 1.78, 0.25),
        ),
        Combo(
            nearest_triple("input_match", 112.5, 1.37, 0.23),
            nearest_triple("interstage_match", 106.5, 1.25, 0.225),
            nearest_triple("output_match", 101.5, 1.94, 0.225),
        ),
    ]
    seed_combos.extend(seed_combos_from_csv(Path(args.previous_top_csv), args.previous_seed_count))
    seed_combos.extend(random_combos_requested_box(args.requested_random_seed_count, args.seed + 100, length_limit_um=args.length_limit_um))
    seed_combos.extend(random_combos_global(args.random_seed_count, args.seed + 200, length_limit_um=args.length_limit_um))

    unique_seed: dict[tuple, Combo] = {}
    for combo in seed_combos:
        unique_seed[combo.key()] = combo
    seed_combos = list(unique_seed.values())

    t_start = time.perf_counter()
    for idx, combo in enumerate(seed_combos, start=1):
        evalr.evaluate(combo, source="seed")
        if idx % 50 == 0:
            print(f"seed eval {idx}/{len(seed_combos)}: unique={len(evalr.cache)}", flush=True)

    rng = np.random.default_rng(args.seed)
    while len(seed_combos) < args.restarts:
        seed_combos.append(normalized_to_combo_global(rng.random(9)))

    traces: list[dict[str, float]] = []
    evals_per_restart = max(args.popsize, int(np.ceil(args.max_evals / max(1, args.restarts))))
    for idx in range(args.restarts):
        trace, best_x, best_score = cma_es(
            objective,
            x0=combo_to_normalized_global(seed_combos[idx % len(seed_combos)]),
            sigma0=args.sigma,
            max_evals=evals_per_restart,
            popsize=args.popsize,
            seed=args.seed + idx * 4211,
            restart_index=idx + 1,
        )
        traces.extend(trace)
        evalr.evaluate(normalized_to_combo_global(best_x), source=f"restart_{idx + 1}_best")
        print(f"restart {idx + 1}/{args.restarts}: best_score={best_score:.3f}, unique={len(evalr.cache)}", flush=True)

    ranked_pre = add_basic_rank(pd.DataFrame(evalr.cache.values()))
    coordinate_polish(evalr, ranked_pre, top_k=args.polish_top_k, rounds=args.polish_rounds, radius=args.polish_radius)

    ranked = add_requested_box_flag(add_basic_rank(pd.DataFrame(evalr.cache.values())))
    feasible = ranked.loc[ranked["hard_feasible"]].copy()
    if feasible.empty:
        raise RuntimeError("No hard-feasible candidates found.")
    front_raw = feasible.loc[nondominated_mask(feasible)].copy()
    front = add_requested_box_flag(add_percentile_rank(front_raw))
    requested_front = front.loc[front["in_requested_box"]].copy()
    requested_feasible = feasible.loc[feasible["in_requested_box"]].copy()
    requested_feasible_top = add_percentile_rank(requested_feasible) if not requested_feasible.empty else requested_feasible

    ranked.to_csv(out_dir / "global_len480_cal0529_ranked_candidates_all.csv", index=False, encoding="utf-8-sig")
    feasible.to_csv(out_dir / "global_len480_cal0529_hard_feasible_candidates.csv", index=False, encoding="utf-8-sig")
    front.to_csv(out_dir / "global_len480_cal0529_pareto_front_equal_priority.csv", index=False, encoding="utf-8-sig")
    requested_front.to_csv(out_dir / "global_len480_cal0529_pareto_front_in_requested_box.csv", index=False, encoding="utf-8-sig")
    requested_feasible_top.to_csv(out_dir / "global_len480_cal0529_requested_box_hard_feasible_ranked.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(traces).to_csv(out_dir / "global_len480_cal0529_cmaes_trace.csv", index=False, encoding="utf-8-sig")
    builder.write_metadata(out_dir / "cal0529_speed_global_len480_transformer_build_metadata.csv")

    if not args.no_plots:
        plot_pareto(ranked, front, out_dir)
        plot_top_s21(front, evalr, out_dir, n=min(args.top_n, 20))

    grid_counts, full_count = grid_count_manifest()
    elapsed = time.perf_counter() - t_start
    manifest = {
        "core": "Cal_0529_speed write-only S4P + Cal_0529_v2 L13-only six-port assembly",
        "global_box": "W 90-120 step 0.5, R 0.8-2.0 step 0.01, WlineR 0.15-0.30 step 0.005",
        "requested_box": REQUESTED_BOX,
        "objective": "CMA-ES scalar search; final hard filtering + Pareto: maximize 30-80 GHz relative-3dB overlap, minimize OMN Zin/ZOPT RMS, maximize S21 peak",
        "target_band_ghz": [float(args.target_lo_ghz), float(args.target_hi_ghz)],
        "gain_window_db": [float(args.gain_min_db), float(args.gain_max_db)],
        "length_limit_um": float(args.length_limit_um),
        "grid_counts": grid_counts,
        "global_pa_candidate_space_count": int(full_count),
        "unique_candidates_evaluated": int(len(ranked)),
        "hard_feasible_candidate_count": int(len(feasible)),
        "pareto_front_count": int(len(front)),
        "requested_box_hard_feasible_count": int(len(requested_feasible)),
        "requested_box_pareto_front_count": int(len(requested_front)),
        "reuse_count": int(builder.reused_count),
        "generated_count": int(builder.generated_count),
        "s4p_generation_seconds_total": float(builder.s4p_generation_seconds),
        "s6p_assembly_seconds_total": float(builder.s6p_assembly_seconds),
        "elapsed_seconds": float(elapsed),
        "cmaes": {
            "max_evals": int(args.max_evals),
            "restarts": int(args.restarts),
            "popsize_lambda": int(args.popsize),
            "mu": int(args.popsize // 2),
            "sigma0": float(args.sigma),
            "evals_per_restart": int(evals_per_restart),
            "initial_covariance": "identity in normalized 9-D space",
        },
        "best_equal_priority_front": front.iloc[0].to_dict(),
        "requested_box_front_best": None if requested_front.empty else requested_front.iloc[0].to_dict(),
        "requested_box_feasible_best": None if requested_feasible_top.empty else requested_feasible_top.iloc[0].to_dict(),
    }
    (out_dir / "global_len480_cal0529_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    show_cols = [
        "selected_rank",
        "input_W",
        "input_R",
        "input_WlineR",
        "interstage_W",
        "interstage_R",
        "interstage_WlineR",
        "output_W",
        "output_R",
        "output_WlineR",
        "target_overlap_ghz",
        "relative_3db_lower_ghz",
        "relative_3db_upper_ghz",
        "s21_peak_db",
        "omn_zin_rms_to_zopt_ohm",
        "length_total_um",
        "in_requested_box",
        "equal_priority_score",
    ]
    print(f"output_dir={out_dir.resolve()}")
    print(f"global_pa_candidate_space_count={full_count}")
    print(f"unique_candidates={len(ranked)}")
    print(f"hard_feasible={len(feasible)}")
    print(f"pareto_front={len(front)}")
    print(f"requested_box_hard_feasible={len(requested_feasible)}")
    print(f"requested_box_pareto_front={len(requested_front)}")
    print(f"generated_count={builder.generated_count}")
    print(f"reuse_count={builder.reused_count}")
    print("front_top")
    print(front.head(min(args.top_n, 20))[show_cols].to_string(index=False))
    if not requested_front.empty:
        print("requested_box_front_top")
        print(requested_front.head(min(args.top_n, 20))[show_cols].to_string(index=False))
    else:
        print("requested_box_front_top=<empty>")
        if not requested_feasible_top.empty:
            print("requested_box_feasible_top")
            print(requested_feasible_top.head(min(args.top_n, 20))[show_cols].to_string(index=False))
    return out_dir.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transistor-dir", type=Path, default=DEFAULT_TRANSISTOR_DIR)
    parser.add_argument("--loadpull-xlsx", type=Path, default=DEFAULT_LOADPULL_XLSX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reuse-dir", type=Path, action="append", default=[])
    parser.add_argument("--previous-top-csv", type=Path, default=ROOT / "outputs" / "final_candidate_global_len480_pareto_w116_w106_w102_synthesis_validation_20260607" / "global_len480_pareto_front_used_for_fig_pareto_front.csv")
    parser.add_argument("--previous-seed-count", type=int, default=30)
    parser.add_argument("--requested-random-seed-count", type=int, default=350)
    parser.add_argument("--random-seed-count", type=int, default=250)
    parser.add_argument("--l13-model", choices=["anchors8", "full80-log-trilinear"], default="anchors8")
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--lf", "--line-length-scale", dest="line_length_scale", type=float, default=None)
    parser.add_argument("--gf", "--gnd-width-factor", dest="gnd_width_factor", type=float, default=None)
    parser.add_argument("--target-lo-ghz", type=float, default=30.0)
    parser.add_argument("--target-hi-ghz", type=float, default=80.0)
    parser.add_argument("--fallback-lo-ghz", type=float, default=33.0)
    parser.add_argument("--fallback-hi-ghz", type=float, default=77.0)
    parser.add_argument("--peak-lo-ghz", type=float, default=25.0)
    parser.add_argument("--peak-hi-ghz", type=float, default=90.0)
    parser.add_argument("--gain-min-db", type=float, default=16.9)
    parser.add_argument("--gain-max-db", type=float, default=20.1)
    parser.add_argument("--length-limit-um", type=float, default=480.0)
    parser.add_argument("--max-evals", type=int, default=600)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.26)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--polish-rounds", type=int, default=2)
    parser.add_argument("--polish-top-k", type=int, default=30)
    parser.add_argument("--polish-radius", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

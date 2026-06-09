#!/usr/bin/env python3
"""Global Cal_0521_v2 PA synthesis with the local anchor-box objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimize_predicted_pa_cascade_cmaes import Combo, Triple, clipped_db20, cma_es
from optimize_predicted_pa_global_center_band import (
    GLOBAL_GRIDS,
    centered_window_metrics,
    mask_range,
    _triple_from_role_values,
    combo_to_normalized_global,
    nearest_triple,
    normalized_to_combo_global,
)
from optimize_predicted_pa_local_anchor_range_cal0521 import add_local_rank
from optimize_predicted_pa_local_anchor_range_cal0521_v2 import Cal0521V2PredictedBuilder
from pa_synthesis.data_loaders import load_loadpull_zopt, load_transistor_s4p
from pa_synthesis.network_utils import align_network, transformer_single_input_impedance
from run_pa_synthesis_pipeline_0504 import build_full_pa_y_fast, s_from_y_2port


ROOT = Path(__file__).resolve().parent
DEFAULT_METRIC_DIR = Path(r".\HFSS\For_Paper\ForModelling\Predict_Model_Compare_metric")
DEFAULT_TRANSISTOR_DIR = DEFAULT_METRIC_DIR / "transistor_z4p_s4p"
DEFAULT_LOADPULL_XLSX = DEFAULT_METRIC_DIR / "loadpull_marker_stats_30_80GHz.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "predicted_pa_global_anchor_objective_cal0521_v2_16_20"


class FastPAEvaluator:
    """PA evaluator using direct nodal-Y reduction instead of scikit-rf Circuit."""

    def __init__(
        self,
        *,
        builder: Cal0521V2PredictedBuilder,
        driver,
        final,
        loadpull_zopt: np.ndarray,
        loadpull_freq_hz: np.ndarray,
    ) -> None:
        self.builder = builder
        self.driver = driver
        self.final = final
        self.freq_hz = driver.f
        self.loadpull_zopt = loadpull_zopt
        self.loadpull_freq_hz = loadpull_freq_hz
        self.omn_zin_cache: dict[tuple[float, float, float], dict[str, float]] = {}
        self.s21_cache: dict[tuple, np.ndarray] = {}

    def combo_s21_db(self, combo: Combo) -> np.ndarray:
        key = combo.key()
        if key in self.s21_cache:
            return self.s21_cache[key]

        imn = align_network(self.builder.get("input_match", combo.imn), self.freq_hz, "imn_tf6")
        ismn = align_network(self.builder.get("interstage_match", combo.ismn), self.freq_hz, "ismn_tf6")
        omn = align_network(self.builder.get("output_match", combo.omn), self.freq_hz, "omn_tf6")
        y2 = build_full_pa_y_fast(
            driver_s4p=self.driver,
            final_s4p=self.final,
            imn=imn,
            ismn=ismn,
            omn=omn,
        )
        s2 = s_from_y_2port(y2, z0=50.0)
        s21_db = clipped_db20(s2[:, 1, 0])
        self.s21_cache[key] = s21_db
        return s21_db

    def omn_zin_metrics(self, triple: Triple) -> dict[str, float]:
        key = triple.key()
        if key in self.omn_zin_cache:
            return self.omn_zin_cache[key]
        omn = self.builder.get("output_match", triple)
        omn_lp = align_network(omn, self.loadpull_freq_hz, f"omn_{triple.label()}_loadpull")
        _, z_single = transformer_single_input_impedance(omn_lp, load_ohm=50.0)
        delta = z_single - self.loadpull_zopt
        metrics = {
            "omn_zin_rms_to_zopt_ohm": float(np.sqrt(np.mean(np.abs(delta) ** 2))),
            "omn_zin_mean_abs_to_zopt_ohm": float(np.mean(np.abs(delta))),
            "omn_zin_max_abs_to_zopt_ohm": float(np.max(np.abs(delta))),
        }
        self.omn_zin_cache[key] = metrics
        return metrics


class FastCenterBandEvaluator:
    def __init__(
        self,
        *,
        evaluator: FastPAEvaluator,
        full_lo: float,
        full_hi: float,
        gain_min: float,
        gain_max: float,
        zin_weight: float,
    ) -> None:
        self.evaluator = evaluator
        self.full_lo = float(full_lo)
        self.full_hi = float(full_hi)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.zin_weight = float(zin_weight)
        self.cache: dict[tuple, dict[str, float | str | bool]] = {}

    @property
    def full_span(self) -> float:
        return self.full_hi - self.full_lo

    def evaluate(self, combo: Combo, *, source: str = "fast_center") -> dict[str, float | str | bool]:
        key = combo.key()
        if key in self.cache:
            row = dict(self.cache[key])
            row["source"] = source
            return row

        freq_ghz = self.evaluator.freq_hz / 1e9
        s21_db = self.evaluator.combo_s21_db(combo)
        metrics = centered_window_metrics(
            freq_ghz,
            s21_db,
            full_lo=self.full_lo,
            full_hi=self.full_hi,
            gain_min=self.gain_min,
            gain_max=self.gain_max,
        )
        zin = self.evaluator.omn_zin_metrics(combo.omn)
        missing = max(0.0, self.full_span - float(metrics["centered_window_width_ghz"]))
        score = (
            5000.0 * float(metrics["full_violation_rms_db"]) ** 2
            + 140.0 * float(metrics["full_violation_max_db"])
            + 55.0 * missing
            + self.zin_weight * float(zin["omn_zin_rms_to_zopt_ohm"])
            + 0.25 * float(metrics["full_gain_ripple_db"])
        )
        row: dict[str, float | str | bool] = {
            "source": source,
            "combo_id": combo.label(),
            "input_W": combo.imn.W,
            "input_R": combo.imn.R,
            "input_WlineR": combo.imn.WlineR,
            "interstage_W": combo.ismn.W,
            "interstage_R": combo.ismn.R,
            "interstage_WlineR": combo.ismn.WlineR,
            "output_W": combo.omn.W,
            "output_R": combo.omn.R,
            "output_WlineR": combo.omn.WlineR,
            "center_band_score": float(score),
        }
        row.update(metrics)
        row.update(zin)
        self.cache[key] = dict(row)
        return row


class FastLocalAchievementEvaluator:
    """Local-anchor score with fast PA-level S21 reuse."""

    def __init__(
        self,
        *,
        center_eval: FastCenterBandEvaluator,
        core_lo: float,
        core_hi: float,
        zin_soft_ohm: float,
    ) -> None:
        self.center_eval = center_eval
        self.core_lo = float(core_lo)
        self.core_hi = float(core_hi)
        self.zin_soft_ohm = float(zin_soft_ohm)
        self.cache: dict[tuple, dict[str, float | str | bool]] = {}

    def evaluate(self, combo: Combo, *, source: str = "fast_local") -> dict[str, float | str | bool]:
        key = combo.key()
        if key in self.cache:
            row = dict(self.cache[key])
            row["source"] = source
            return row

        row = dict(self.center_eval.evaluate(combo, source=source))
        freq_ghz = self.center_eval.evaluator.freq_hz / 1e9
        s21_db = self.center_eval.evaluator.combo_s21_db(combo)
        core = centered_window_metrics(
            freq_ghz,
            s21_db,
            full_lo=self.core_lo,
            full_hi=self.core_hi,
            gain_min=self.center_eval.gain_min,
            gain_max=self.center_eval.gain_max,
        )
        for k, v in core.items():
            row[f"core_{k}"] = v

        full_missing = max(0.0, self.center_eval.full_span - float(row["centered_window_width_ghz"]))
        core_span = self.core_hi - self.core_lo
        core_missing = max(0.0, core_span - float(core["centered_window_width_ghz"]))
        zin_excess = max(0.0, float(row["omn_zin_rms_to_zopt_ohm"]) - self.zin_soft_ohm)
        row["local_anchor_box_score"] = float(
            900.0 * float(core["full_violation_rms_db"]) ** 2
            + 90.0 * float(core["full_violation_max_db"])
            + 70.0 * core_missing
            + 35.0 * full_missing
            + 0.8 * zin_excess
            + 0.10 * float(row["full_gain_ripple_db"])
        )
        self.cache[key] = dict(row)
        return row


def plot_global_top20(rows: pd.DataFrame, center_eval: FastCenterBandEvaluator, out_dir: Path) -> None:
    top = rows.head(20)
    fig, axes = plt.subplots(5, 4, figsize=(16, 13), sharex=True, sharey=True, constrained_layout=True)
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        combo = Combo(
            Triple(row["input_W"], row["input_R"], row["input_WlineR"]),
            Triple(row["interstage_W"], row["interstage_R"], row["interstage_WlineR"]),
            Triple(row["output_W"], row["output_R"], row["output_WlineR"]),
        )
        freq = center_eval.evaluator.freq_hz / 1e9
        s21 = center_eval.evaluator.combo_s21_db(combo)
        mask = mask_range(freq, center_eval.full_lo - 8.0, center_eval.full_hi + 8.0)
        ax.plot(freq[mask], s21[mask], linewidth=1.0)
        ax.axhspan(center_eval.gain_min, center_eval.gain_max, color="green", alpha=0.12)
        ax.axvline(center_eval.full_lo, color="black", linewidth=0.7)
        ax.axvline(center_eval.full_hi, color="black", linewidth=0.7)
        ax.axvspan(row["centered_window_lower_ghz"], row["centered_window_upper_ghz"], color="steelblue", alpha=0.08)
        ax.set_title(
            f"#{int(row['local_rank'])} {row['centered_window_lower_ghz']:.0f}-{row['centered_window_upper_ghz']:.0f} "
            f"Zin={row['omn_zin_rms_to_zopt_ohm']:.2f}",
            fontsize=8,
        )
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("GHz")
    for ax in axes[:, 0]:
        ax.set_ylabel("S21 (dB)")
    fig.savefig(out_dir / "global_anchor_objective_top20_s21.png", dpi=180)
    plt.close(fig)


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

    builder = Cal0521V2PredictedBuilder(
        output_dir=pred_dir,
        freq_hz=driver.f,
        allow_extrapolation=args.allow_extrapolation,
        l13_model=args.l13_model,
        line_length_scale=args.line_length_scale,
        gnd_width_factor=args.gnd_width_factor,
    )
    evaluator = FastPAEvaluator(
        builder=builder,
        driver=driver,
        final=final,
        loadpull_zopt=loadpull.zopt_single,
        loadpull_freq_hz=loadpull.freq_hz,
    )
    center_eval = FastCenterBandEvaluator(
        evaluator=evaluator,
        full_lo=args.full_lo_ghz,
        full_hi=args.full_hi_ghz,
        gain_min=args.gain_min_db,
        gain_max=args.gain_max_db,
        zin_weight=args.zin_weight,
    )
    local_eval = FastLocalAchievementEvaluator(
        center_eval=center_eval,
        core_lo=args.core_lo_ghz,
        core_hi=args.core_hi_ghz,
        zin_soft_ohm=args.zin_soft_ohm,
    )

    def objective(x: np.ndarray) -> float:
        return float(local_eval.evaluate(normalized_to_combo_global(x))["local_anchor_box_score"])

    requested_anchor = Combo(
        nearest_triple("input_match", 114.0, 1.45, 0.22),
        nearest_triple("interstage_match", 106.5, 1.20, 0.24),
        nearest_triple("output_match", 103.0, 1.80, 0.24),
    )
    local_best_seed = Combo(
        nearest_triple("input_match", 109.5, 1.37, 0.22),
        nearest_triple("interstage_match", 104.0, 1.20, 0.22),
        nearest_triple("output_match", 103.0, 1.80, 0.24),
    )
    initial = [
        local_best_seed,
        requested_anchor,
        Combo(
            nearest_triple("input_match", 108.5, 1.40, 0.25),
            nearest_triple("interstage_match", 104.0, 1.29, 0.22),
            nearest_triple("output_match", 100.0, 1.80, 0.24),
        ),
        Combo(
            nearest_triple("input_match", 97.5, 1.59, 0.265),
            nearest_triple("interstage_match", 100.0, 1.14, 0.18),
            nearest_triple("output_match", 104.0, 2.00, 0.19),
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
        local_eval.evaluate(normalized_to_combo_global(best_x), source=f"global_v2_restart_{idx + 1}_best")
        print(f"restart {idx + 1}/{args.restarts}: best_score={best_score:.3f}, unique={len(local_eval.cache)}", flush=True)

    local_eval.evaluate(requested_anchor, source="requested_anchor")
    local_eval.evaluate(local_best_seed, source="local_best_seed")
    ranked = add_local_rank(pd.DataFrame(local_eval.cache.values()))
    ranked.to_csv(out_dir / "global_anchor_objective_v2_ranked_candidates_all.csv", index=False, encoding="utf-8-sig")
    ranked.head(30).to_csv(out_dir / "global_anchor_objective_v2_top30_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(traces).to_csv(out_dir / "global_anchor_objective_v2_cmaes_trace.csv", index=False, encoding="utf-8-sig")
    builder.write_metadata(out_dir / "cal0521_v2_global_transformer_build_metadata.csv")
    plot_global_top20(ranked.head(20), center_eval, out_dir)

    def find_combo(combo: Combo) -> dict:
        mask = (
            np.isclose(ranked["input_W"], combo.imn.W)
            & np.isclose(ranked["input_R"], combo.imn.R)
            & np.isclose(ranked["input_WlineR"], combo.imn.WlineR)
            & np.isclose(ranked["interstage_W"], combo.ismn.W)
            & np.isclose(ranked["interstage_R"], combo.ismn.R)
            & np.isclose(ranked["interstage_WlineR"], combo.ismn.WlineR)
            & np.isclose(ranked["output_W"], combo.omn.W)
            & np.isclose(ranked["output_R"], combo.omn.R)
            & np.isclose(ranked["output_WlineR"], combo.omn.WlineR)
        )
        return ranked.loc[mask].iloc[0].to_dict()

    anchor_row = find_combo(requested_anchor)
    local_best_row = find_combo(local_best_seed)
    pd.DataFrame([anchor_row]).to_csv(out_dir / "requested_anchor_global_v2_rank.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([local_best_row]).to_csv(out_dir / "local_best_seed_global_v2_rank.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "core": "Cal_0521_v2 L13-only six-port prediction model",
        "objective": "global search using 16-20 dB centered coverage, 35-85 GHz core fallback, Zin/ZOPT secondary",
        "unique_candidates": int(len(ranked)),
        "full_band_ghz": [float(args.full_lo_ghz), float(args.full_hi_ghz)],
        "core_band_ghz": [float(args.core_lo_ghz), float(args.core_hi_ghz)],
        "gain_window_db": [float(args.gain_min_db), float(args.gain_max_db)],
        "line_length_scale": args.line_length_scale,
        "gnd_width_factor": args.gnd_width_factor,
        "best": ranked.iloc[0].to_dict(),
        "requested_anchor": anchor_row,
        "local_best_seed": local_best_row,
    }
    (out_dir / "global_anchor_objective_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"output_dir={out_dir.resolve()}")
    print(f"unique_candidates={len(ranked)}")
    print("best")
    print(ranked.iloc[0][[
        "local_rank",
        "input_W",
        "input_R",
        "input_WlineR",
        "interstage_W",
        "interstage_R",
        "interstage_WlineR",
        "output_W",
        "output_R",
        "output_WlineR",
        "core_centered_window_lower_ghz",
        "core_centered_window_upper_ghz",
        "core_centered_window_width_ghz",
        "centered_window_lower_ghz",
        "centered_window_upper_ghz",
        "centered_window_width_ghz",
        "full_violation_rms_db",
        "full_gain_min_db",
        "full_gain_max_db",
        "omn_zin_rms_to_zopt_ohm",
        "local_anchor_box_score",
    ]])
    print("local_best_seed")
    print(pd.Series(local_best_row)[["local_rank", "centered_window_lower_ghz", "centered_window_upper_ghz", "centered_window_width_ghz", "full_violation_rms_db", "omn_zin_rms_to_zopt_ohm", "local_anchor_box_score"]])
    print("requested_anchor")
    print(pd.Series(anchor_row)[["local_rank", "centered_window_lower_ghz", "centered_window_upper_ghz", "centered_window_width_ghz", "full_violation_rms_db", "omn_zin_rms_to_zopt_ohm", "local_anchor_box_score"]])
    return out_dir.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transistor-dir", type=Path, default=DEFAULT_TRANSISTOR_DIR)
    parser.add_argument("--loadpull-xlsx", type=Path, default=DEFAULT_LOADPULL_XLSX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--l13-model", choices=["anchors8", "full80-log-trilinear"], default="anchors8")
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--lf", "--line-length-scale", dest="line_length_scale", type=float, default=None)
    parser.add_argument("--gf", "--gnd-width-factor", dest="gnd_width_factor", type=float, default=None)
    parser.add_argument("--full-lo-ghz", type=float, default=30.0)
    parser.add_argument("--full-hi-ghz", type=float, default=90.0)
    parser.add_argument("--core-lo-ghz", type=float, default=35.0)
    parser.add_argument("--core-hi-ghz", type=float, default=85.0)
    parser.add_argument("--gain-min-db", type=float, default=16.0)
    parser.add_argument("--gain-max-db", type=float, default=20.0)
    parser.add_argument("--zin-weight", type=float, default=1.0)
    parser.add_argument("--zin-soft-ohm", type=float, default=13.0)
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

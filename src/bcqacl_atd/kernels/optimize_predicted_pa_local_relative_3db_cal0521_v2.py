#!/usr/bin/env python3
"""Cal_0521_v2 local PA synthesis using peak-relative 3-dB bandwidth.

Search box:

* IMN:  W=108..114, R=1.37..1.56, WlineR=0.22..0.25
* ISMN: W=104..107, R=1.20..1.30, WlineR=0.22..0.24
* OMN:  W=100..103, R=1.70..1.80, WlineR=0.24..0.25

Objective requested on 2026-05-22:

1. Use peak-relative -3 dB bandwidth, not the absolute 16-20 dB centered
   window from earlier runs.
2. Prefer coverage of 30-80 GHz. If that cannot be achieved, shrink toward
   33-77 GHz.
3. Keep S21 gain within 16-20 dB with a +/-0.1 dB tolerance.
4. Prefer lower OMN Zin/ZOPT RMS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimize_predicted_pa_cascade_cmaes import Combo, Triple, cma_es
from optimize_predicted_pa_global_anchor_objective_cal0521_v2 import FastPAEvaluator
from optimize_predicted_pa_global_center_band import mask_range
from optimize_predicted_pa_local_30_80_cal0521_v2 import (
    DEFAULT_REUSE_DIRS,
    ReuseCal0521V2PredictedBuilder,
)
from optimize_predicted_pa_local_anchor_range_cal0521 import (
    LOCAL_GRIDS,
    combo_to_normalized_local,
    nearest_triple,
    normalized_to_combo_local,
)
from pa_synthesis.data_loaders import load_loadpull_zopt, load_transistor_s4p


ROOT = Path(__file__).resolve().parent
DEFAULT_METRIC_DIR = Path(r".\HFSS\For_Paper\ForModelling\Predict_Model_Compare_metric")
DEFAULT_TRANSISTOR_DIR = DEFAULT_METRIC_DIR / "transistor_z4p_s4p"
DEFAULT_LOADPULL_XLSX = DEFAULT_METRIC_DIR / "loadpull_marker_stats_30_80GHz.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "predicted_pa_local_relative_3db_cal0521_v2_20260522"


def interp_crossing(f1: float, y1: float, f2: float, y2: float, threshold: float) -> float:
    if abs(y2 - y1) < 1e-15:
        return float(f1)
    return float(f1 + (threshold - y1) * (f2 - f1) / (y2 - y1))


def conventional_3db_band(freq_ghz: np.ndarray, s21_db: np.ndarray, threshold_db: float, peak_idx: int) -> dict[str, float]:
    left = float(freq_ghz[0])
    for idx in range(peak_idx, 0, -1):
        if s21_db[idx - 1] < threshold_db <= s21_db[idx]:
            left = interp_crossing(freq_ghz[idx - 1], s21_db[idx - 1], freq_ghz[idx], s21_db[idx], threshold_db)
            break

    right = float(freq_ghz[-1])
    for idx in range(peak_idx, len(freq_ghz) - 1):
        if s21_db[idx] >= threshold_db > s21_db[idx + 1]:
            right = interp_crossing(freq_ghz[idx], s21_db[idx], freq_ghz[idx + 1], s21_db[idx + 1], threshold_db)
            break

    return {
        "relative_3db_lower_ghz": float(left),
        "relative_3db_upper_ghz": float(right),
        "relative_3db_bandwidth_ghz": float(right - left),
    }


def centered_relative_window(
    freq_ghz: np.ndarray,
    s21_db: np.ndarray,
    *,
    lo: float,
    hi: float,
    threshold_db: float,
) -> dict[str, float | bool]:
    center = 0.5 * (lo + hi)
    max_half = 0.5 * (hi - lo)
    step = float(np.median(np.diff(freq_ghz))) if len(freq_ghz) > 1 else 0.5
    best_half = 0.0
    best_lo = center
    best_hi = center
    for half in np.arange(max_half, -0.5 * step, -step):
        sub_lo = center - half
        sub_hi = center + half
        mask = mask_range(freq_ghz, sub_lo, sub_hi)
        if np.any(mask) and np.all(s21_db[mask] >= threshold_db - 1e-12):
            best_half = float(half)
            best_lo = float(sub_lo)
            best_hi = float(sub_hi)
            break
    band_mask = mask_range(freq_ghz, lo, hi)
    violation = np.maximum(threshold_db - s21_db[band_mask], 0.0)
    return {
        "ok": bool(np.all(violation <= 1e-12)),
        "centered_lower_ghz": best_lo,
        "centered_upper_ghz": best_hi,
        "centered_width_ghz": 2.0 * best_half,
        "relative_violation_rms_db": float(np.sqrt(np.mean(violation * violation))),
        "relative_violation_max_db": float(np.max(violation)),
    }


def gain_window_violation(s21_db: np.ndarray, mask: np.ndarray, gain_min: float, gain_max: float) -> dict[str, float | bool]:
    sub = s21_db[mask]
    violation = np.maximum(gain_min - sub, 0.0) + np.maximum(sub - gain_max, 0.0)
    return {
        "gain_window_ok": bool(np.all(violation <= 1e-12)),
        "gain_window_violation_rms_db": float(np.sqrt(np.mean(violation * violation))),
        "gain_window_violation_max_db": float(np.max(violation)),
        "gain_min_db": float(np.min(sub)),
        "gain_max_db": float(np.max(sub)),
        "gain_ripple_db": float(np.max(sub) - np.min(sub)),
    }


def relative_3db_metrics(
    freq_ghz: np.ndarray,
    s21_db: np.ndarray,
    *,
    target_lo: float,
    target_hi: float,
    fallback_lo: float,
    fallback_hi: float,
    peak_lo: float,
    peak_hi: float,
    gain_min: float,
    gain_max: float,
) -> dict[str, float | bool]:
    peak_mask = mask_range(freq_ghz, peak_lo, peak_hi)
    peak_freqs = freq_ghz[peak_mask]
    peak_gains = s21_db[peak_mask]
    local_peak_idx = int(np.argmax(peak_gains))
    peak_idx = int(np.where(peak_mask)[0][local_peak_idx])
    peak_db = float(s21_db[peak_idx])
    peak_freq = float(freq_ghz[peak_idx])
    threshold = peak_db - 3.0

    target = centered_relative_window(freq_ghz, s21_db, lo=target_lo, hi=target_hi, threshold_db=threshold)
    fallback = centered_relative_window(freq_ghz, s21_db, lo=fallback_lo, hi=fallback_hi, threshold_db=threshold)
    target_gain = gain_window_violation(s21_db, mask_range(freq_ghz, target_lo, target_hi), gain_min, gain_max)
    fallback_gain = gain_window_violation(s21_db, mask_range(freq_ghz, fallback_lo, fallback_hi), gain_min, gain_max)
    peak_gain_violation = max(gain_min - peak_db, 0.0, peak_db - gain_max)

    out: dict[str, float | bool] = {
        "s21_peak_db": peak_db,
        "s21_peak_frequency_ghz": peak_freq,
        "s21_relative_3db_threshold_db": float(threshold),
        "peak_gain_window_violation_db": float(peak_gain_violation),
    }
    out.update(conventional_3db_band(freq_ghz, s21_db, threshold, peak_idx))
    for prefix, metrics in [
        ("target_30_80", target),
        ("fallback_33_77", fallback),
        ("target_30_80_abs", target_gain),
        ("fallback_33_77_abs", fallback_gain),
    ]:
        for key, value in metrics.items():
            out[f"{prefix}_{key}"] = value
    return out


class Relative3dBEvaluator:
    def __init__(
        self,
        *,
        pa_eval: FastPAEvaluator,
        target_lo: float,
        target_hi: float,
        fallback_lo: float,
        fallback_hi: float,
        peak_lo: float,
        peak_hi: float,
        gain_min: float,
        gain_max: float,
        zin_weight: float,
        gain_weight: float,
    ) -> None:
        self.pa_eval = pa_eval
        self.target_lo = float(target_lo)
        self.target_hi = float(target_hi)
        self.fallback_lo = float(fallback_lo)
        self.fallback_hi = float(fallback_hi)
        self.peak_lo = float(peak_lo)
        self.peak_hi = float(peak_hi)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.zin_weight = float(zin_weight)
        self.gain_weight = float(gain_weight)
        self.cache: dict[tuple, dict[str, float | str | bool]] = {}

    @property
    def target_span(self) -> float:
        return self.target_hi - self.target_lo

    @property
    def fallback_span(self) -> float:
        return self.fallback_hi - self.fallback_lo

    def evaluate(self, combo: Combo, *, source: str = "relative_3db") -> dict[str, float | str | bool]:
        key = combo.key()
        if key in self.cache:
            row = dict(self.cache[key])
            row["source"] = source
            return row

        freq = self.pa_eval.freq_hz / 1e9
        s21 = self.pa_eval.combo_s21_db(combo)
        metrics = relative_3db_metrics(
            freq,
            s21,
            target_lo=self.target_lo,
            target_hi=self.target_hi,
            fallback_lo=self.fallback_lo,
            fallback_hi=self.fallback_hi,
            peak_lo=self.peak_lo,
            peak_hi=self.peak_hi,
            gain_min=self.gain_min,
            gain_max=self.gain_max,
        )
        zin = self.pa_eval.omn_zin_metrics(combo.omn)
        target_missing = max(0.0, self.target_span - float(metrics["target_30_80_centered_width_ghz"]))
        fallback_missing = max(0.0, self.fallback_span - float(metrics["fallback_33_77_centered_width_ghz"]))
        gain_penalty = (
            float(metrics["peak_gain_window_violation_db"])
            + float(metrics["fallback_33_77_abs_gain_window_violation_rms_db"])
            + 0.35 * float(metrics["target_30_80_abs_gain_window_violation_rms_db"])
        )
        score = (
            5000.0 * float(metrics["target_30_80_relative_violation_rms_db"]) ** 2
            + 350.0 * float(metrics["target_30_80_relative_violation_max_db"])
            + 90.0 * target_missing
            + 45.0 * fallback_missing
            + self.gain_weight * gain_penalty
            + self.zin_weight * float(zin["omn_zin_rms_to_zopt_ohm"])
            + 0.10 * float(metrics["fallback_33_77_abs_gain_ripple_db"])
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
            "target_missing_ghz": float(target_missing),
            "fallback_missing_ghz": float(fallback_missing),
            "relative_3db_score": float(score),
        }
        row.update(metrics)
        row.update(zin)
        self.cache[key] = dict(row)
        return row


def add_rank(rows: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy()
    data["target_ok_flag"] = data["target_30_80_ok"].astype(int)
    data["fallback_ok_flag"] = data["fallback_33_77_ok"].astype(int)
    data["peak_gain_ok_flag"] = (data["peak_gain_window_violation_db"] <= 1e-12).astype(int)
    data["fallback_abs_gain_ok_flag"] = data["fallback_33_77_abs_gain_window_ok"].astype(int)
    data = data.sort_values(
        [
            "target_ok_flag",
            "target_30_80_centered_width_ghz",
            "fallback_ok_flag",
            "fallback_33_77_centered_width_ghz",
            "peak_gain_ok_flag",
            "fallback_abs_gain_ok_flag",
            "peak_gain_window_violation_db",
            "fallback_33_77_abs_gain_window_violation_rms_db",
            "target_30_80_abs_gain_window_violation_rms_db",
            "omn_zin_rms_to_zopt_ohm",
            "relative_3db_score",
        ],
        ascending=[False, False, False, False, False, False, True, True, True, True, True],
    ).reset_index(drop=True)
    data.insert(0, "rank", np.arange(1, len(data) + 1))
    return data


def combo_from_row(row: pd.Series) -> Combo:
    return Combo(
        Triple(float(row["input_W"]), float(row["input_R"]), float(row["input_WlineR"])),
        Triple(float(row["interstage_W"]), float(row["interstage_R"]), float(row["interstage_WlineR"])),
        Triple(float(row["output_W"]), float(row["output_R"]), float(row["output_WlineR"])),
    )


def seed_combos_from_previous(path: Path, limit: int) -> list[Combo]:
    if not path.exists():
        return []
    df = pd.read_csv(path).head(limit)
    out: list[Combo] = []
    for _, row in df.iterrows():
        out.append(combo_from_row(row))
    return out


def random_combos(count: int, seed: int) -> list[Combo]:
    rng = np.random.default_rng(seed)
    roles = ["input_match", "interstage_match", "output_match"]
    out: list[Combo] = []
    for _ in range(count):
        triples = []
        for role in roles:
            grids = LOCAL_GRIDS[role]
            triples.append(
                Triple(
                    float(rng.choice(grids["W"])),
                    float(rng.choice(grids["R"])),
                    float(rng.choice(grids["WlineR"])),
                )
            )
        out.append(Combo(*triples))
    return out


def coordinate_polish(evalr: Relative3dBEvaluator, ranked: pd.DataFrame, *, top_k: int, rounds: int) -> None:
    roles = ["input_match", "interstage_match", "output_match"]
    attrs = ["W", "R", "WlineR"]
    for round_idx in range(rounds):
        base = add_rank(pd.DataFrame(evalr.cache.values())).head(top_k)
        before = len(evalr.cache)
        for _, row in base.iterrows():
            combo = combo_from_row(row)
            triples_by_role = {
                "input_match": combo.imn,
                "interstage_match": combo.ismn,
                "output_match": combo.omn,
            }
            for role_idx, role in enumerate(roles):
                for attr in attrs:
                    for value in LOCAL_GRIDS[role][attr]:
                        current = triples_by_role[role]
                        vals = {"W": current.W, "R": current.R, "WlineR": current.WlineR}
                        vals[attr] = float(value)
                        new_triples = [triples_by_role[r] for r in roles]
                        new_triples[role_idx] = Triple(vals["W"], vals["R"], vals["WlineR"])
                        evalr.evaluate(Combo(*new_triples), source=f"polish{round_idx + 1}")
        print(f"polish {round_idx + 1}/{rounds}: added {len(evalr.cache) - before}, unique={len(evalr.cache)}", flush=True)


def plot_top(rows: pd.DataFrame, evalr: Relative3dBEvaluator, out_dir: Path, n: int = 20) -> None:
    top = rows.head(n)
    fig, axes = plt.subplots(5, 4, figsize=(16, 13), sharex=True, sharey=True, constrained_layout=True)
    freq = evalr.pa_eval.freq_hz / 1e9
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        combo = combo_from_row(row)
        s21 = evalr.pa_eval.combo_s21_db(combo)
        mask = mask_range(freq, 25.0, 90.0)
        threshold = float(row["s21_relative_3db_threshold_db"])
        ax.plot(freq[mask], s21[mask], linewidth=1.0)
        ax.axhline(threshold, color="black", linewidth=0.65, linestyle="--")
        ax.axvspan(evalr.target_lo, evalr.target_hi, color="steelblue", alpha=0.08)
        ax.axvspan(evalr.fallback_lo, evalr.fallback_hi, color="green", alpha=0.07)
        ax.set_title(
            f"#{int(row['rank'])} rel {row['relative_3db_lower_ghz']:.1f}-{row['relative_3db_upper_ghz']:.1f} "
            f"Z={row['omn_zin_rms_to_zopt_ohm']:.2f}",
            fontsize=7.5,
        )
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("GHz")
    for ax in axes[:, 0]:
        ax.set_ylabel("S21 (dB)")
    fig.savefig(out_dir / "relative_3db_top20_s21.png", dpi=180)
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

    builder = ReuseCal0521V2PredictedBuilder(
        output_dir=pred_dir,
        reuse_roots=args.reuse_root,
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
    evalr = Relative3dBEvaluator(
        pa_eval=pa_eval,
        target_lo=args.target_lo_ghz,
        target_hi=args.target_hi_ghz,
        fallback_lo=args.fallback_lo_ghz,
        fallback_hi=args.fallback_hi_ghz,
        peak_lo=args.peak_lo_ghz,
        peak_hi=args.peak_hi_ghz,
        gain_min=args.gain_min_db - args.gain_tolerance_db,
        gain_max=args.gain_max_db + args.gain_tolerance_db,
        zin_weight=args.zin_weight,
        gain_weight=args.gain_weight,
    )

    seeds = [
        Combo(
            nearest_triple("input_match", 111.5, 1.46, 0.22),
            nearest_triple("interstage_match", 106.0, 1.22, 0.22),
            nearest_triple("output_match", 102.5, 1.80, 0.24),
        ),
        Combo(
            nearest_triple("input_match", 109.5, 1.37, 0.22),
            nearest_triple("interstage_match", 104.0, 1.20, 0.22),
            nearest_triple("output_match", 103.0, 1.80, 0.24),
        ),
        Combo(
            nearest_triple("input_match", 108.0, 1.56, 0.22),
            nearest_triple("interstage_match", 107.0, 1.20, 0.22),
            nearest_triple("output_match", 102.0, 1.80, 0.24),
        ),
    ]
    seeds += seed_combos_from_previous(Path(args.previous_top_csv), args.previous_seed_count)
    seeds += random_combos(args.random_seed_count, args.seed + 99)

    seen_seed_keys = set()
    unique_seeds: list[Combo] = []
    for combo in seeds:
        if combo.key() not in seen_seed_keys:
            seen_seed_keys.add(combo.key())
            unique_seeds.append(combo)
            evalr.evaluate(combo, source="seed")

    traces: list[dict[str, float]] = []
    cma_seeds = unique_seeds[: max(args.restarts, 1)]
    rng = np.random.default_rng(args.seed)
    while len(cma_seeds) < args.restarts:
        cma_seeds.append(normalized_to_combo_local(rng.random(9)))
    evals_per_restart = max(args.popsize, int(np.ceil(args.max_evals / max(1, args.restarts))))
    for idx in range(args.restarts):
        trace, best_x, best_score = cma_es(
            lambda x: float(evalr.evaluate(normalized_to_combo_local(x), source=f"restart_{idx + 1}")["relative_3db_score"]),
            x0=combo_to_normalized_local(cma_seeds[idx]),
            sigma0=args.sigma,
            max_evals=evals_per_restart,
            popsize=args.popsize,
            seed=args.seed + idx * 4211,
            restart_index=idx + 1,
        )
        traces.extend(trace)
        evalr.evaluate(normalized_to_combo_local(best_x), source=f"restart_{idx + 1}_best")
        ranked_now = add_rank(pd.DataFrame(evalr.cache.values()))
        best = ranked_now.iloc[0]
        print(
            f"restart {idx + 1}/{args.restarts}: best_score={best_score:.3f}, unique={len(evalr.cache)}, "
            f"rank1={best['input_W']:g}/{best['interstage_W']:g}/{best['output_W']:g} "
            f"rel={best['relative_3db_lower_ghz']:.1f}-{best['relative_3db_upper_ghz']:.1f} "
            f"Z={best['omn_zin_rms_to_zopt_ohm']:.2f}",
            flush=True,
        )

    ranked = add_rank(pd.DataFrame(evalr.cache.values()))
    if args.polish_rounds > 0:
        coordinate_polish(evalr, ranked, top_k=args.polish_top_k, rounds=args.polish_rounds)
        ranked = add_rank(pd.DataFrame(evalr.cache.values()))

    ranked.to_csv(out_dir / "relative_3db_cal0521_v2_ranked_candidates_all.csv", index=False, encoding="utf-8-sig")
    ranked.head(args.top_n).to_csv(out_dir / f"relative_3db_cal0521_v2_top{args.top_n}_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(traces).to_csv(out_dir / "relative_3db_cal0521_v2_cmaes_trace.csv", index=False, encoding="utf-8-sig")
    builder.write_metadata(out_dir / "relative_3db_cal0521_v2_transformer_build_metadata.csv")
    plot_top(ranked, evalr, out_dir, n=min(20, args.top_n))

    requested = ranked[
        np.isclose(ranked["input_W"], 111.5)
        & np.isclose(ranked["input_R"], 1.46)
        & np.isclose(ranked["input_WlineR"], 0.22)
        & np.isclose(ranked["interstage_W"], 106.0)
        & np.isclose(ranked["interstage_R"], 1.22)
        & np.isclose(ranked["interstage_WlineR"], 0.22)
        & np.isclose(ranked["output_W"], 102.5)
        & np.isclose(ranked["output_R"], 1.80)
        & np.isclose(ranked["output_WlineR"], 0.24)
    ]
    if not requested.empty:
        requested.to_csv(out_dir / "requested_rank1_geometry_relative_3db_rank.csv", index=False, encoding="utf-8-sig")

    counts = {role: {key: int(len(values)) for key, values in grids.items()} for role, grids in LOCAL_GRIDS.items()}
    total = 1
    for grids in counts.values():
        for count in grids.values():
            total *= count
    manifest = {
        "core": "Cal_0521_v2 L13-only six-port prediction model",
        "objective": "peak-relative -3 dB bandwidth covers 30-80 GHz; fallback 33-77 GHz; S21 peak/window tolerance 16-20 dB +/-0.1 dB; lower Zin/ZOPT RMS preferred",
        "grid_counts": counts,
        "candidate_space_count": int(total),
        "unique_candidates_evaluated": int(len(ranked)),
        "reuse_count": int(builder.reused_count),
        "generated_count": int(builder.generated_count),
        "target_band_ghz": [float(args.target_lo_ghz), float(args.target_hi_ghz)],
        "fallback_band_ghz": [float(args.fallback_lo_ghz), float(args.fallback_hi_ghz)],
        "peak_search_band_ghz": [float(args.peak_lo_ghz), float(args.peak_hi_ghz)],
        "gain_window_with_tolerance_db": [float(args.gain_min_db - args.gain_tolerance_db), float(args.gain_max_db + args.gain_tolerance_db)],
        "best": ranked.iloc[0].to_dict(),
        "requested_geometry": requested.iloc[0].to_dict() if not requested.empty else None,
    }
    (out_dir / "relative_3db_cal0521_v2_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"output_dir={out_dir.resolve()}")
    print(f"unique_candidates={len(ranked)}")
    print(f"reuse_count={builder.reused_count}")
    print(f"generated_count={builder.generated_count}")
    columns = [
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
        "s21_peak_db",
        "s21_peak_frequency_ghz",
        "relative_3db_lower_ghz",
        "relative_3db_upper_ghz",
        "relative_3db_bandwidth_ghz",
        "target_30_80_ok",
        "target_30_80_centered_width_ghz",
        "fallback_33_77_ok",
        "fallback_33_77_centered_width_ghz",
        "peak_gain_window_violation_db",
        "fallback_33_77_abs_gain_window_violation_rms_db",
        "omn_zin_rms_to_zopt_ohm",
        "relative_3db_score",
    ]
    print("top")
    print(ranked.head(min(args.top_n, 20))[columns].to_string(index=False))
    if not requested.empty:
        print("requested_geometry")
        print(requested.iloc[0][columns].to_string())
    return out_dir.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transistor-dir", type=Path, default=DEFAULT_TRANSISTOR_DIR)
    parser.add_argument("--loadpull-xlsx", type=Path, default=DEFAULT_LOADPULL_XLSX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reuse-root", type=Path, action="append", default=DEFAULT_REUSE_DIRS)
    parser.add_argument("--previous-top-csv", type=Path, default=ROOT / "outputs" / "predicted_pa_local_30_80_cal0521_v2_20260521" / "local_30_80_cal0521_v2_top30_bandwidth_zin_priority.csv")
    parser.add_argument("--previous-seed-count", type=int, default=30)
    parser.add_argument("--random-seed-count", type=int, default=350)
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
    parser.add_argument("--gain-min-db", type=float, default=16.0)
    parser.add_argument("--gain-max-db", type=float, default=20.0)
    parser.add_argument("--gain-tolerance-db", type=float, default=0.1)
    parser.add_argument("--zin-weight", type=float, default=1.0)
    parser.add_argument("--gain-weight", type=float, default=450.0)
    parser.add_argument("--max-evals", type=int, default=2600)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--sigma", type=float, default=0.24)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--polish-rounds", type=int, default=2)
    parser.add_argument("--polish-top-k", type=int, default=35)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Global predicted PA optimization with a center-shrinking gain window.

Goal:
    Prefer S21 in [16, 19] dB across 30-90 GHz.  If no candidate satisfies
    the full band, rank candidates by the widest symmetric band around 60 GHz
    that stays inside [16, 19] dB, e.g. 33-87 GHz.

The search range is:
    W      90..120
    R      0.8..2.0
    WlineR 0.15..0.30

For practical repeatability, CMA-ES samples normalized variables and snaps to:
    W step 0.5, R step 0.01, WlineR step 0.005

OMN Zin is compared to load-pull ZOPT by complex RMS of Zin_single - ZOPT_single.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tf_analysis_pipeline_cli_v3 as tf_v3
from optimize_predicted_pa_cascade_cmaes import (
    Combo,
    Evaluator,
    PredictedBuilder,
    Triple,
    clipped_db20,
    cma_es,
    db20,
    mask_range,
)
from pa_synthesis.data_loaders import load_loadpull_zopt, load_transistor_s4p
from pa_synthesis.network_utils import build_full_pa_network


ROOT = Path(__file__).resolve().parent
DEFAULT_METRIC_DIR = Path(r".\HFSS\For_Paper\ForModelling\Predict_Model_Compare_metric")
DEFAULT_TRANSISTOR_DIR = DEFAULT_METRIC_DIR / "transistor_z4p_s4p"
DEFAULT_LOADPULL_XLSX = DEFAULT_METRIC_DIR / "loadpull_marker_stats_30_80GHz.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "predicted_pa_global_center_band_16_19"

GLOBAL_GRIDS = {
    "input_match": {
        "W": np.arange(90.0, 120.0 + 0.25, 0.5),
        "R": np.round(np.arange(0.8, 2.0 + 0.005, 0.01), 10),
        "WlineR": np.round(np.arange(0.15, 0.30 + 0.0025, 0.005), 10),
    },
    "interstage_match": {
        "W": np.arange(90.0, 120.0 + 0.25, 0.5),
        "R": np.round(np.arange(0.8, 2.0 + 0.005, 0.01), 10),
        "WlineR": np.round(np.arange(0.15, 0.30 + 0.0025, 0.005), 10),
    },
    "output_match": {
        "W": np.arange(90.0, 120.0 + 0.25, 0.5),
        "R": np.round(np.arange(0.8, 2.0 + 0.005, 0.01), 10),
        "WlineR": np.round(np.arange(0.15, 0.30 + 0.0025, 0.005), 10),
    },
}


def _triple_from_role_values(role: str, vals: np.ndarray) -> Triple:
    grids = GLOBAL_GRIDS[role]
    out = []
    for value, key in zip(vals, ("W", "R", "WlineR")):
        grid = grids[key]
        idx = int(np.rint(float(np.clip(value, 0.0, 1.0)) * (len(grid) - 1)))
        idx = max(0, min(len(grid) - 1, idx))
        out.append(float(grid[idx]))
    return Triple(*out)


def normalized_to_combo_global(x: np.ndarray) -> Combo:
    x = np.asarray(x, dtype=float)
    return Combo(
        _triple_from_role_values("input_match", x[0:3]),
        _triple_from_role_values("interstage_match", x[3:6]),
        _triple_from_role_values("output_match", x[6:9]),
    )


def triple_to_norm(role: str, triple: Triple) -> list[float]:
    vals = []
    for value, key in zip(triple.key(), ("W", "R", "WlineR")):
        grid = GLOBAL_GRIDS[role][key]
        idx = int(np.argmin(np.abs(grid - float(value))))
        vals.append(idx / (len(grid) - 1))
    return vals


def combo_to_normalized_global(combo: Combo) -> np.ndarray:
    return np.asarray(
        triple_to_norm("input_match", combo.imn)
        + triple_to_norm("interstage_match", combo.ismn)
        + triple_to_norm("output_match", combo.omn),
        dtype=float,
    )


def nearest_triple(role: str, W: float, R: float, WlineR: float) -> Triple:
    vals = []
    for value, key in zip((W, R, WlineR), ("W", "R", "WlineR")):
        grid = GLOBAL_GRIDS[role][key]
        vals.append(float(grid[int(np.argmin(np.abs(grid - float(value))))]))
    return Triple(*vals)


def centered_window_metrics(
    freq_ghz: np.ndarray,
    s21_db: np.ndarray,
    *,
    full_lo: float,
    full_hi: float,
    gain_min: float,
    gain_max: float,
) -> dict[str, float | bool]:
    freq = np.asarray(freq_ghz, dtype=float)
    mag = np.asarray(s21_db, dtype=float)
    center = 0.5 * (full_lo + full_hi)
    max_half_width = 0.5 * (full_hi - full_lo)
    full_mask = mask_range(freq, full_lo, full_hi)
    full_mag = mag[full_mask]
    full_violation = np.maximum(gain_min - full_mag, 0.0) + np.maximum(full_mag - gain_max, 0.0)

    best_half = 0.0
    best_lo = center
    best_hi = center
    # Work on the native frequency grid, normally 0.5 GHz.
    step = float(np.median(np.diff(freq))) if len(freq) > 1 else 0.5
    for half_width in np.arange(max_half_width, -0.5 * step, -step):
        lo = center - half_width
        hi = center + half_width
        mask = mask_range(freq, lo, hi)
        if not np.any(mask):
            continue
        sub = mag[mask]
        if np.all((sub >= gain_min) & (sub <= gain_max)):
            best_half = float(half_width)
            best_lo = float(lo)
            best_hi = float(hi)
            break

    centered_width = 2.0 * best_half
    centered_mask = mask_range(freq, best_lo, best_hi) if centered_width > 0 else np.zeros_like(freq, dtype=bool)
    centered_mag = mag[centered_mask] if np.any(centered_mask) else np.asarray([], dtype=float)
    if centered_mag.size:
        centered_gain_min = float(np.min(centered_mag))
        centered_gain_max = float(np.max(centered_mag))
        centered_gain_ripple = centered_gain_max - centered_gain_min
    else:
        centered_gain_min = float("nan")
        centered_gain_max = float("nan")
        centered_gain_ripple = float("nan")
    return {
        "full_30_90_ok": bool(np.all(full_violation <= 1e-12)),
        "centered_window_width_ghz": float(centered_width),
        "centered_window_lower_ghz": float(best_lo),
        "centered_window_upper_ghz": float(best_hi),
        "full_violation_rms_db": float(np.sqrt(np.mean(full_violation * full_violation))),
        "full_violation_mean_db": float(np.mean(full_violation)),
        "full_violation_max_db": float(np.max(full_violation)),
        "full_gain_min_db": float(np.min(full_mag)),
        "full_gain_max_db": float(np.max(full_mag)),
        "full_gain_mean_db": float(np.mean(full_mag)),
        "full_gain_ripple_db": float(np.max(full_mag) - np.min(full_mag)),
        "centered_gain_min_db": centered_gain_min,
        "centered_gain_max_db": centered_gain_max,
        "centered_gain_ripple_db": centered_gain_ripple,
    }


class CenterBandEvaluator:
    def __init__(
        self,
        *,
        evaluator: Evaluator,
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

    def evaluate(self, combo: Combo, *, source: str = "cmaes_center") -> dict[str, float | str | bool]:
        key = combo.key()
        if key in self.cache:
            row = dict(self.cache[key])
            row["source"] = source
            return row

        imn = self.evaluator.builder.get("input_match", combo.imn)
        ismn = self.evaluator.builder.get("interstage_match", combo.ismn)
        omn = self.evaluator.builder.get("output_match", combo.omn)
        pa = build_full_pa_network(
            freq_hz=self.evaluator.freq_hz,
            driver_s4p=self.evaluator.driver,
            final_s4p=self.evaluator.final,
            imn=imn,
            ismn=ismn,
            omn=omn,
            z0=50.0,
            include_dc_blocks=False,
        )
        freq_ghz = pa.f / 1e9
        s21_db = db20(pa.s[:, 1, 0])
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


def add_rank(rows: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy()
    data["full_ok_flag"] = data["full_30_90_ok"].astype(int)
    data = data.sort_values(
        [
            "full_ok_flag",
            "centered_window_width_ghz",
            "full_violation_rms_db",
            "full_violation_max_db",
            "omn_zin_rms_to_zopt_ohm",
            "full_gain_ripple_db",
            "center_band_score",
        ],
        ascending=[False, False, True, True, True, True, True],
    ).reset_index(drop=True)
    data.insert(0, "rank", np.arange(1, len(data) + 1))
    return data


def plot_top20(rows: pd.DataFrame, center_eval: CenterBandEvaluator, out_dir: Path) -> None:
    top = rows.head(20)
    fig, axes = plt.subplots(5, 4, figsize=(16, 13), sharex=True, sharey=True, constrained_layout=True)
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        combo = Combo(
            Triple(row["input_W"], row["input_R"], row["input_WlineR"]),
            Triple(row["interstage_W"], row["interstage_R"], row["interstage_WlineR"]),
            Triple(row["output_W"], row["output_R"], row["output_WlineR"]),
        )
        imn = center_eval.evaluator.builder.get("input_match", combo.imn)
        ismn = center_eval.evaluator.builder.get("interstage_match", combo.ismn)
        omn = center_eval.evaluator.builder.get("output_match", combo.omn)
        pa = build_full_pa_network(
            freq_hz=center_eval.evaluator.freq_hz,
            driver_s4p=center_eval.evaluator.driver,
            final_s4p=center_eval.evaluator.final,
            imn=imn,
            ismn=ismn,
            omn=omn,
            z0=50.0,
            include_dc_blocks=False,
        )
        freq = pa.f / 1e9
        s21 = clipped_db20(pa.s[:, 1, 0])
        mask = mask_range(freq, center_eval.full_lo - 8.0, center_eval.full_hi + 8.0)
        ax.plot(freq[mask], s21[mask], linewidth=1.0)
        ax.axhspan(center_eval.gain_min, center_eval.gain_max, color="green", alpha=0.12)
        ax.axvline(center_eval.full_lo, color="black", linewidth=0.7)
        ax.axvline(center_eval.full_hi, color="black", linewidth=0.7)
        ax.axvspan(row["centered_window_lower_ghz"], row["centered_window_upper_ghz"], color="steelblue", alpha=0.08)
        ax.set_title(
            f"#{int(row['rank'])} {row['centered_window_lower_ghz']:.0f}-{row['centered_window_upper_ghz']:.0f} "
            f"Zin={row['omn_zin_rms_to_zopt_ohm']:.2f}",
            fontsize=8,
        )
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("GHz")
    for ax in axes[:, 0]:
        ax.set_ylabel("S21 (dB)")
    fig.savefig(out_dir / "center_band_top20_s21.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "predicted_transformers"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    driver = load_transistor_s4p(Path(args.transistor_dir) / "driver_2x12_single_ended_z0_50.s4p", z0=50.0, name="driver_2x12")
    final = load_transistor_s4p(
        Path(args.transistor_dir) / "final_2x18_single_ended_z0_50.s4p",
        target_freq_hz=driver.f,
        z0=50.0,
        name="final_2x18",
    )
    loadpull = load_loadpull_zopt(args.loadpull_xlsx)
    model = tf_v3.load_l_fit_model(data_source=args.fit_data, round_decimals=args.round_decimals)
    builder = PredictedBuilder(model=model, output_dir=pred_dir, freq_hz=driver.f, allow_extrapolation=args.allow_extrapolation)
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
        "unique_candidates": int(len(ranked)),
        "full_band_ghz": [float(args.full_lo_ghz), float(args.full_hi_ghz)],
        "gain_window_db": [float(args.gain_min_db), float(args.gain_max_db)],
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
    print(ranked.iloc[0][[
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
    ]])
    print("requested_anchor")
    print(pd.Series(anchor)[[
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
    ]])
    return out_dir.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transistor-dir", type=Path, default=DEFAULT_TRANSISTOR_DIR)
    parser.add_argument("--loadpull-xlsx", type=Path, default=DEFAULT_LOADPULL_XLSX)
    parser.add_argument("--fit-data", type=Path, default=tf_v3.DEFAULT_FIT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--round-decimals", type=int, default=2)
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
    parser.add_argument("--seed", type=int, default=20260513)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

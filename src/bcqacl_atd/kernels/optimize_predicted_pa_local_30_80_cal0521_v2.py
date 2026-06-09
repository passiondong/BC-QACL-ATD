#!/usr/bin/env python3
"""Cal_0521_v2 local PA synthesis for the 30-80 GHz target.

Search box:

* IMN:  W=108..114, R=1.37..1.56, WlineR=0.22..0.25
* ISMN: W=104..107, R=1.20..1.30, WlineR=0.22..0.24
* OMN:  W=100..103, R=1.70..1.80, WlineR=0.24..0.25

The score follows the requested priority:

1. Keep PA S21 inside 16-20 dB over the centered target band 30-80 GHz.
2. If this cannot be fully achieved, shrink symmetrically toward 33-77 GHz.
3. Prefer lower OMN Zin-vs-ZOPT RMS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf

import tf_analysis_pipeline_cli_0521_v2 as tf0521v2
from optimize_predicted_pa_cascade_cmaes import Combo, Triple, cma_es
from optimize_predicted_pa_global_center_band import centered_window_metrics, mask_range
from optimize_predicted_pa_global_anchor_objective_cal0521_v2 import (
    FastCenterBandEvaluator,
    FastPAEvaluator,
)
from optimize_predicted_pa_local_anchor_range_cal0521 import (
    LOCAL_GRIDS,
    combo_to_normalized_local,
    nearest_triple,
    normalized_to_combo_local,
)
from optimize_predicted_pa_local_anchor_range_cal0521_v2 import Cal0521V2PredictedBuilder
from pa_synthesis.data_loaders import load_loadpull_zopt, load_transistor_s4p
from pa_synthesis.network_utils import align_network
from run_three_tf_v3_pa_cascade import SIX_PORT_NAMES


ROOT = Path(__file__).resolve().parent
DEFAULT_METRIC_DIR = Path(r".\HFSS\For_Paper\ForModelling\Predict_Model_Compare_metric")
DEFAULT_TRANSISTOR_DIR = DEFAULT_METRIC_DIR / "transistor_z4p_s4p"
DEFAULT_LOADPULL_XLSX = DEFAULT_METRIC_DIR / "loadpull_marker_stats_30_80GHz.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "predicted_pa_local_30_80_cal0521_v2_20260521"
DEFAULT_REUSE_DIRS = [
    ROOT
    / "outputs"
    / "predicted_pa_local_anchor_range_cal0521_v2_16_20_20260521"
    / "predicted_transformers",
    ROOT
    / "outputs"
    / "predicted_pa_global_anchor_objective_cal0521_v2_16_20_fast_20260521"
    / "predicted_transformers",
]


class ReuseCal0521V2PredictedBuilder(Cal0521V2PredictedBuilder):
    """Cal_0521_v2 builder with file-level S6P reuse from earlier runs."""

    def __init__(self, *, reuse_roots: Sequence[Path], **kwargs) -> None:
        super().__init__(**kwargs)
        self.reuse_roots = [Path(root) for root in reuse_roots if Path(root).exists()]
        self.reused_count = 0
        self.generated_count = 0

    @staticmethod
    def _s6p_stem(triple: Triple) -> str:
        return (
            "tf_pred_cal0521v2_l13only_"
            f"W{tf0521v2.base.safe_tag(triple.W)}_"
            f"R{tf0521v2.base.safe_tag(triple.R)}_"
            f"WlineR{tf0521v2.base.safe_tag(triple.WlineR)}"
        )

    def _find_existing_s6p(self, role: str, triple: Triple) -> Path | None:
        rel = Path(role) / "predicted_s6p" / f"{self._s6p_stem(triple)}.s6p"
        for root in [self.output_dir, *self.reuse_roots]:
            candidate = Path(root) / rel
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
            ntw = align_network(ntw, self.freq_hz, f"{role}_{triple.label()}_cal0521v2_reused")
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

        ntw = super().get(role, triple)
        self.metadata_cache[key]["reuse"] = False
        self.generated_count += 1
        return ntw


class RequestedLocalEvaluator:
    """Requested 30-80 / 33-77 local-box score using fast PA S21 reuse."""

    def __init__(
        self,
        *,
        center_eval: FastCenterBandEvaluator,
        core_lo: float,
        core_hi: float,
        zin_weight: float,
        ripple_weight: float,
    ) -> None:
        self.center_eval = center_eval
        self.core_lo = float(core_lo)
        self.core_hi = float(core_hi)
        self.zin_weight = float(zin_weight)
        self.ripple_weight = float(ripple_weight)
        self.cache: dict[tuple, dict[str, float | str | bool]] = {}

    @property
    def full_span(self) -> float:
        return self.center_eval.full_hi - self.center_eval.full_lo

    @property
    def core_span(self) -> float:
        return self.core_hi - self.core_lo

    def evaluate(self, combo: Combo, *, source: str = "local_30_80") -> dict[str, float | str | bool]:
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

        full_missing = max(0.0, self.full_span - float(row["centered_window_width_ghz"]))
        core_missing = max(0.0, self.core_span - float(core["centered_window_width_ghz"]))
        row["target_30_80_ok"] = bool(row["full_30_90_ok"])
        row["fallback_33_77_ok"] = bool(core["full_30_90_ok"])
        row["target_missing_ghz"] = float(full_missing)
        row["fallback_missing_ghz"] = float(core_missing)
        row["requested_local_30_80_score"] = float(
            2000.0 * float(row["full_violation_rms_db"]) ** 2
            + 180.0 * float(row["full_violation_max_db"])
            + 90.0 * full_missing
            + 45.0 * core_missing
            + self.zin_weight * float(row["omn_zin_rms_to_zopt_ohm"])
            + self.ripple_weight * float(row["full_gain_ripple_db"])
        )
        self.cache[key] = dict(row)
        return row


def add_requested_rank(rows: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy()
    data["target_ok_flag"] = data["target_30_80_ok"].astype(int)
    data["fallback_ok_flag"] = data["fallback_33_77_ok"].astype(int)
    data = data.sort_values(
        [
            "target_ok_flag",
            "centered_window_width_ghz",
            "fallback_ok_flag",
            "core_centered_window_width_ghz",
            "full_violation_rms_db",
            "core_full_violation_rms_db",
            "omn_zin_rms_to_zopt_ohm",
            "full_gain_ripple_db",
            "requested_local_30_80_score",
        ],
        ascending=[False, False, False, False, True, True, True, True, True],
    ).reset_index(drop=True)
    data.insert(0, "rank", np.arange(1, len(data) + 1))
    return data


def combo_from_row(row: pd.Series) -> Combo:
    return Combo(
        Triple(float(row["input_W"]), float(row["input_R"]), float(row["input_WlineR"])),
        Triple(float(row["interstage_W"]), float(row["interstage_R"]), float(row["interstage_WlineR"])),
        Triple(float(row["output_W"]), float(row["output_R"]), float(row["output_WlineR"])),
    )


def plot_top20(rows: pd.DataFrame, local_eval: RequestedLocalEvaluator, out_dir: Path) -> None:
    top = rows.head(20)
    fig, axes = plt.subplots(5, 4, figsize=(16, 13), sharex=True, sharey=True, constrained_layout=True)
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        combo = combo_from_row(row)
        freq = local_eval.center_eval.evaluator.freq_hz / 1e9
        s21 = local_eval.center_eval.evaluator.combo_s21_db(combo)
        mask = mask_range(freq, local_eval.center_eval.full_lo - 5.0, local_eval.center_eval.full_hi + 5.0)
        ax.plot(freq[mask], s21[mask], linewidth=1.0)
        ax.axhspan(local_eval.center_eval.gain_min, local_eval.center_eval.gain_max, color="green", alpha=0.12)
        ax.axvline(local_eval.center_eval.full_lo, color="black", linewidth=0.7)
        ax.axvline(local_eval.center_eval.full_hi, color="black", linewidth=0.7)
        ax.axvspan(row["centered_window_lower_ghz"], row["centered_window_upper_ghz"], color="steelblue", alpha=0.08)
        ax.set_title(
            f"#{int(row['rank'])} {row['centered_window_lower_ghz']:.0f}-{row['centered_window_upper_ghz']:.0f} "
            f"Z={row['omn_zin_rms_to_zopt_ohm']:.2f}",
            fontsize=8,
        )
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("GHz")
    for ax in axes[:, 0]:
        ax.set_ylabel("S21 (dB)")
    fig.savefig(out_dir / "local_30_80_top20_s21.png", dpi=180)
    plt.close(fig)


def grid_counts() -> tuple[dict[str, dict[str, int]], int]:
    counts = {role: {key: int(len(values)) for key, values in grids.items()} for role, grids in LOCAL_GRIDS.items()}
    total = 1
    for grids in counts.values():
        for count in grids.values():
            total *= count
    return counts, int(total)


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

    reuse_roots = list(args.reuse_dir or [])
    builder = ReuseCal0521V2PredictedBuilder(
        output_dir=pred_dir,
        reuse_roots=reuse_roots,
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
    local_eval = RequestedLocalEvaluator(
        center_eval=center_eval,
        core_lo=args.core_lo_ghz,
        core_hi=args.core_hi_ghz,
        zin_weight=args.zin_weight,
        ripple_weight=args.ripple_weight,
    )

    def objective(x: np.ndarray) -> float:
        return float(local_eval.evaluate(normalized_to_combo_local(x))["requested_local_30_80_score"])

    seed_combos = [
        Combo(
            nearest_triple("input_match", 109.5, 1.37, 0.22),
            nearest_triple("interstage_match", 104.0, 1.20, 0.22),
            nearest_triple("output_match", 103.0, 1.80, 0.24),
        ),
        Combo(
            nearest_triple("input_match", 114.0, 1.45, 0.22),
            nearest_triple("interstage_match", 106.5, 1.20, 0.24),
            nearest_triple("output_match", 103.0, 1.80, 0.24),
        ),
        Combo(
            nearest_triple("input_match", 114.0, 1.37, 0.225),
            nearest_triple("interstage_match", 104.0, 1.20, 0.22),
            nearest_triple("output_match", 103.0, 1.80, 0.24),
        ),
        Combo(
            nearest_triple("input_match", 108.5, 1.42, 0.22),
            nearest_triple("interstage_match", 105.5, 1.20, 0.22),
            nearest_triple("output_match", 103.0, 1.80, 0.24),
        ),
        Combo(
            nearest_triple("input_match", 112.0, 1.37, 0.225),
            nearest_triple("interstage_match", 106.0, 1.21, 0.225),
            nearest_triple("output_match", 103.0, 1.80, 0.245),
        ),
    ]
    rng = np.random.default_rng(args.seed)
    while len(seed_combos) < args.restarts:
        seed_combos.append(normalized_to_combo_local(rng.random(9)))

    traces: list[dict[str, float]] = []
    evals_per_restart = max(args.popsize, int(np.ceil(args.max_evals / max(1, args.restarts))))
    for idx in range(args.restarts):
        trace, best_x, best_score = cma_es(
            objective,
            x0=combo_to_normalized_local(seed_combos[idx]),
            sigma0=args.sigma,
            max_evals=evals_per_restart,
            popsize=args.popsize,
            seed=args.seed + idx * 4211,
            restart_index=idx + 1,
        )
        traces.extend(trace)
        local_eval.evaluate(normalized_to_combo_local(best_x), source=f"restart_{idx + 1}_best")
        print(f"restart {idx + 1}/{args.restarts}: best_score={best_score:.3f}, unique={len(local_eval.cache)}", flush=True)

    for idx, combo in enumerate(seed_combos[:5], start=1):
        local_eval.evaluate(combo, source=f"seed_combo_{idx}")

    ranked = add_requested_rank(pd.DataFrame(local_eval.cache.values()))
    ranked.to_csv(out_dir / "local_30_80_cal0521_v2_ranked_candidates_all.csv", index=False, encoding="utf-8-sig")
    ranked.head(args.top_n).to_csv(out_dir / f"local_30_80_cal0521_v2_top{args.top_n}_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(traces).to_csv(out_dir / "local_30_80_cal0521_v2_cmaes_trace.csv", index=False, encoding="utf-8-sig")
    builder.write_metadata(out_dir / "cal0521_v2_local_30_80_transformer_build_metadata.csv")
    plot_top20(ranked.head(20), local_eval, out_dir)

    counts, total = grid_counts()
    manifest = {
        "core": "Cal_0521_v2 L13-only six-port prediction model",
        "objective": "requested local-box 16-20 dB centered coverage, primary 30-80 GHz, fallback 33-77 GHz, Zin/ZOPT secondary",
        "grid_counts": counts,
        "candidate_space_count": total,
        "unique_candidates_evaluated": int(len(ranked)),
        "reuse_roots": [str(Path(p).resolve()) for p in reuse_roots],
        "reuse_count": int(builder.reused_count),
        "generated_count": int(builder.generated_count),
        "target_band_ghz": [float(args.full_lo_ghz), float(args.full_hi_ghz)],
        "fallback_band_ghz": [float(args.core_lo_ghz), float(args.core_hi_ghz)],
        "gain_window_db": [float(args.gain_min_db), float(args.gain_max_db)],
        "weights": {
            "zin_weight": float(args.zin_weight),
            "ripple_weight": float(args.ripple_weight),
        },
        "best": ranked.iloc[0].to_dict(),
    }
    (out_dir / "local_30_80_cal0521_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    show_cols = [
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
        "target_30_80_ok",
        "centered_window_lower_ghz",
        "centered_window_upper_ghz",
        "centered_window_width_ghz",
        "fallback_33_77_ok",
        "core_centered_window_lower_ghz",
        "core_centered_window_upper_ghz",
        "core_centered_window_width_ghz",
        "full_violation_rms_db",
        "full_gain_min_db",
        "full_gain_max_db",
        "omn_zin_rms_to_zopt_ohm",
        "requested_local_30_80_score",
    ]
    print(f"output_dir={out_dir.resolve()}")
    print(f"candidate_space_count={total}")
    print(f"unique_candidates={len(ranked)}")
    print(f"reuse_count={builder.reused_count}")
    print(f"generated_count={builder.generated_count}")
    print("best")
    print(ranked.iloc[0][show_cols])
    print("top")
    print(ranked.head(min(args.top_n, 20))[show_cols].to_string(index=False))
    return out_dir.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transistor-dir", type=Path, default=DEFAULT_TRANSISTOR_DIR)
    parser.add_argument("--loadpull-xlsx", type=Path, default=DEFAULT_LOADPULL_XLSX)
    parser.add_argument("--reuse-dir", type=Path, action="append", default=DEFAULT_REUSE_DIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--l13-model", choices=["anchors8", "full80-log-trilinear"], default="anchors8")
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--lf", "--line-length-scale", dest="line_length_scale", type=float, default=None)
    parser.add_argument("--gf", "--gnd-width-factor", dest="gnd_width_factor", type=float, default=None)
    parser.add_argument("--full-lo-ghz", type=float, default=30.0)
    parser.add_argument("--full-hi-ghz", type=float, default=80.0)
    parser.add_argument("--core-lo-ghz", type=float, default=33.0)
    parser.add_argument("--core-hi-ghz", type=float, default=77.0)
    parser.add_argument("--gain-min-db", type=float, default=16.0)
    parser.add_argument("--gain-max-db", type=float, default=20.0)
    parser.add_argument("--zin-weight", type=float, default=1.0)
    parser.add_argument("--ripple-weight", type=float, default=0.10)
    parser.add_argument("--max-evals", type=int, default=2500)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--sigma", type=float, default=0.24)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

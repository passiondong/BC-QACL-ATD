#!/usr/bin/env python3
"""Restricted Cal_0521 PA synthesis in the anchor-near geometry box.

Compared with the global Cal_0521 run, this script keeps the same PA cascade
and center-band objective family but restricts the decision variables to the
user-specified box:

    IMN:  W 108..114, R 1.37..1.56, WlineR 0.22..0.25
    ISMN: W 104..107, R 1.20..1.30, WlineR 0.22..0.24
    OMN:  W 100..103, R 1.70..1.80, WlineR 0.24..0.25

The primary engineering target is S21 inside 16-20 dB and the widest centered
gain-window band around 60 GHz, preferably covering 30-90 GHz and otherwise
shrinking toward the core 35-85 GHz band.  Zin/ZOPT RMS is retained as a
secondary discriminator.
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
    CenterBandEvaluator,
    add_rank,
    centered_window_metrics,
    clipped_db20,
    mask_range,
    plot_top20,
)
from optimize_predicted_pa_global_center_band_cal0521 import Cal0521PredictedBuilder
from pa_synthesis.data_loaders import load_loadpull_zopt, load_transistor_s4p
from pa_synthesis.network_utils import align_network, build_full_pa_network
from run_three_tf_v3_pa_cascade import SIX_PORT_NAMES


ROOT = Path(__file__).resolve().parent
DEFAULT_METRIC_DIR = Path(r".\HFSS\For_Paper\ForModelling\Predict_Model_Compare_metric")
DEFAULT_TRANSISTOR_DIR = DEFAULT_METRIC_DIR / "transistor_z4p_s4p"
DEFAULT_LOADPULL_XLSX = DEFAULT_METRIC_DIR / "loadpull_marker_stats_30_80GHz.xlsx"
DEFAULT_REUSE_DIR = ROOT / "outputs" / "predicted_pa_global_center_band_16_19_cal0521_20260521" / "predicted_transformers"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "predicted_pa_local_anchor_range_cal0521_16_20"


LOCAL_GRIDS = {
    "input_match": {
        "W": np.arange(108.0, 114.0 + 0.25, 0.5),
        "R": np.round(np.arange(1.37, 1.56 + 0.005, 0.01), 10),
        "WlineR": np.round(np.arange(0.22, 0.25 + 0.0025, 0.005), 10),
    },
    "interstage_match": {
        "W": np.arange(104.0, 107.0 + 0.25, 0.5),
        "R": np.round(np.arange(1.20, 1.30 + 0.005, 0.01), 10),
        "WlineR": np.round(np.arange(0.22, 0.24 + 0.0025, 0.005), 10),
    },
    "output_match": {
        "W": np.arange(100.0, 103.0 + 0.25, 0.5),
        "R": np.round(np.arange(1.70, 1.80 + 0.005, 0.01), 10),
        "WlineR": np.round(np.arange(0.24, 0.25 + 0.0025, 0.005), 10),
    },
}


def _triple_from_role_values(role: str, vals: np.ndarray) -> Triple:
    grids = LOCAL_GRIDS[role]
    out = []
    for value, key in zip(vals, ("W", "R", "WlineR")):
        grid = grids[key]
        idx = int(np.rint(float(np.clip(value, 0.0, 1.0)) * (len(grid) - 1)))
        idx = max(0, min(len(grid) - 1, idx))
        out.append(float(grid[idx]))
    return Triple(*out)


def normalized_to_combo_local(x: np.ndarray) -> Combo:
    x = np.asarray(x, dtype=float)
    return Combo(
        _triple_from_role_values("input_match", x[0:3]),
        _triple_from_role_values("interstage_match", x[3:6]),
        _triple_from_role_values("output_match", x[6:9]),
    )


def triple_to_norm(role: str, triple: Triple) -> list[float]:
    vals = []
    for value, key in zip(triple.key(), ("W", "R", "WlineR")):
        grid = LOCAL_GRIDS[role][key]
        idx = int(np.argmin(np.abs(grid - float(value))))
        vals.append(idx / (len(grid) - 1))
    return vals


def combo_to_normalized_local(combo: Combo) -> np.ndarray:
    return np.asarray(
        triple_to_norm("input_match", combo.imn)
        + triple_to_norm("interstage_match", combo.ismn)
        + triple_to_norm("output_match", combo.omn),
        dtype=float,
    )


def nearest_triple(role: str, W: float, R: float, WlineR: float) -> Triple:
    vals = []
    for value, key in zip((W, R, WlineR), ("W", "R", "WlineR")):
        grid = LOCAL_GRIDS[role][key]
        vals.append(float(grid[int(np.argmin(np.abs(grid - float(value))))]))
    return Triple(*vals)


class ReuseCal0521PredictedBuilder(Cal0521PredictedBuilder):
    """Cal_0521 builder that can reuse previously generated S6P files."""

    def __init__(self, *, reuse_roots: Sequence[Path], **kwargs) -> None:
        super().__init__(**kwargs)
        self.reuse_roots = [Path(root) for root in reuse_roots if Path(root).exists()]
        self.reused_count = 0
        self.generated_count = 0

    @staticmethod
    def _s6p_stem(triple: Triple) -> str:
        return (
            "tf_pred_cal0521_l13only_"
            f"W{tf0521.base.safe_tag(triple.W)}_"
            f"R{tf0521.base.safe_tag(triple.R)}_"
            f"WlineR{tf0521.base.safe_tag(triple.WlineR)}"
        )

    def _find_existing_s6p(self, role: str, triple: Triple) -> Path | None:
        rel = Path(role) / "predicted_s6p" / f"{self._s6p_stem(triple)}.s6p"
        for root in [self.output_dir, *self.reuse_roots]:
            path = Path(root) / rel
            if path.exists():
                return path
        return None

    def get(self, role: str, triple: Triple) -> rf.Network:
        key = (role, triple.key())
        if key in self.cache:
            return self.cache[key]

        existing = self._find_existing_s6p(role, triple)
        if existing is not None:
            ntw = rf.Network(str(existing))
            ntw = align_network(ntw, self.freq_hz, f"{role}_{triple.label()}_cal0521_reused")
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


class LocalAchievementEvaluator:
    """Variant score that favors 16-20 dB centered coverage and 35-85 GHz core."""

    def __init__(
        self,
        *,
        center_eval: CenterBandEvaluator,
        core_lo: float,
        core_hi: float,
        zin_soft_ohm: float,
    ) -> None:
        self.center_eval = center_eval
        self.core_lo = float(core_lo)
        self.core_hi = float(core_hi)
        self.zin_soft_ohm = float(zin_soft_ohm)
        self.cache: dict[tuple, dict[str, float | str | bool]] = {}

    def evaluate(self, combo: Combo, *, source: str = "local_cmaes") -> dict[str, float | str | bool]:
        key = combo.key()
        if key in self.cache:
            row = dict(self.cache[key])
            row["source"] = source
            return row

        row = dict(self.center_eval.evaluate(combo, source=source))
        evaluator = self.center_eval.evaluator
        imn = evaluator.builder.get("input_match", combo.imn)
        ismn = evaluator.builder.get("interstage_match", combo.ismn)
        omn = evaluator.builder.get("output_match", combo.omn)
        pa = build_full_pa_network(
            freq_hz=evaluator.freq_hz,
            driver_s4p=evaluator.driver,
            final_s4p=evaluator.final,
            imn=imn,
            ismn=ismn,
            omn=omn,
            z0=50.0,
            include_dc_blocks=False,
        )
        freq_ghz = pa.f / 1e9
        s21_db = clipped_db20(pa.s[:, 1, 0])
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
        # This score is deliberately lexicographic-like:
        # first make 35-85 GHz work, then expand toward 30-90 GHz, then use Zin.
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


def add_local_rank(rows: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy()
    data["core_full_ok_flag"] = data["core_full_30_90_ok"].astype(int)
    data["full_ok_flag"] = data["full_30_90_ok"].astype(int)
    data = data.sort_values(
        [
            "core_full_ok_flag",
            "core_centered_window_width_ghz",
            "full_ok_flag",
            "centered_window_width_ghz",
            "core_full_violation_rms_db",
            "full_violation_rms_db",
            "omn_zin_rms_to_zopt_ohm",
            "local_anchor_box_score",
        ],
        ascending=[False, False, False, False, True, True, True, True],
    ).reset_index(drop=True)
    data.insert(0, "local_rank", np.arange(1, len(data) + 1))
    return data


def plot_local_top20(rows: pd.DataFrame, center_eval: CenterBandEvaluator, out_dir: Path) -> None:
    renamed = rows.rename(columns={"local_rank": "rank"}).copy()
    plot_top20(renamed, center_eval, out_dir)
    old = out_dir / "center_band_top20_s21.png"
    if old.exists():
        old.rename(out_dir / "local_top20_s21.png")


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

    builder = ReuseCal0521PredictedBuilder(
        output_dir=pred_dir,
        reuse_roots=[Path(args.reuse_dir)] if args.reuse_dir else [],
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
    local_eval = LocalAchievementEvaluator(
        center_eval=center_eval,
        core_lo=args.core_lo_ghz,
        core_hi=args.core_hi_ghz,
        zin_soft_ohm=args.zin_soft_ohm,
    )

    def objective(x: np.ndarray) -> float:
        return float(local_eval.evaluate(normalized_to_combo_local(x))["local_anchor_box_score"])

    anchor = Combo(
        nearest_triple("input_match", 114.0, 1.45, 0.22),
        nearest_triple("interstage_match", 106.5, 1.20, 0.24),
        nearest_triple("output_match", 103.0, 1.80, 0.24),
    )
    initial = [
        anchor,
        Combo(nearest_triple("input_match", 114.0, 1.45, 0.22), nearest_triple("interstage_match", 106.5, 1.20, 0.24), nearest_triple("output_match", 102.5, 1.80, 0.25)),
        Combo(nearest_triple("input_match", 112.0, 1.45, 0.23), nearest_triple("interstage_match", 106.0, 1.25, 0.23), nearest_triple("output_match", 103.0, 1.78, 0.24)),
        Combo(nearest_triple("input_match", 108.0, 1.56, 0.25), nearest_triple("interstage_match", 104.0, 1.30, 0.24), nearest_triple("output_match", 100.0, 1.70, 0.25)),
    ]
    rng = np.random.default_rng(args.seed)
    while len(initial) < args.restarts:
        initial.append(normalized_to_combo_local(rng.random(9)))

    traces: list[dict[str, float]] = []
    evals_per_restart = max(args.popsize, int(np.ceil(args.max_evals / max(1, args.restarts))))
    for idx in range(args.restarts):
        trace, best_x, best_score = cma_es(
            objective,
            x0=combo_to_normalized_local(initial[idx]),
            sigma0=args.sigma,
            max_evals=evals_per_restart,
            popsize=args.popsize,
            seed=args.seed + idx * 4211,
            restart_index=idx + 1,
        )
        traces.extend(trace)
        local_eval.evaluate(normalized_to_combo_local(best_x), source=f"local_restart_{idx + 1}_best")
        print(f"restart {idx + 1}/{args.restarts}: best_score={best_score:.3f}, unique={len(local_eval.cache)}", flush=True)

    local_eval.evaluate(anchor, source="requested_anchor")
    ranked = add_local_rank(pd.DataFrame(local_eval.cache.values()))
    ranked.to_csv(out_dir / "local_anchor_range_ranked_candidates_all.csv", index=False, encoding="utf-8-sig")
    ranked.head(30).to_csv(out_dir / "local_anchor_range_top30_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(traces).to_csv(out_dir / "local_anchor_range_cmaes_trace.csv", index=False, encoding="utf-8-sig")
    builder.write_metadata(out_dir / "cal0521_local_transformer_build_metadata.csv")
    plot_local_top20(ranked.head(20), center_eval, out_dir)

    anchor_mask = (
        np.isclose(ranked["input_W"], anchor.imn.W)
        & np.isclose(ranked["input_R"], anchor.imn.R)
        & np.isclose(ranked["input_WlineR"], anchor.imn.WlineR)
        & np.isclose(ranked["interstage_W"], anchor.ismn.W)
        & np.isclose(ranked["interstage_R"], anchor.ismn.R)
        & np.isclose(ranked["interstage_WlineR"], anchor.ismn.WlineR)
        & np.isclose(ranked["output_W"], anchor.omn.W)
        & np.isclose(ranked["output_R"], anchor.omn.R)
        & np.isclose(ranked["output_WlineR"], anchor.omn.WlineR)
    )
    anchor_row = ranked.loc[anchor_mask].iloc[0].to_dict()
    pd.DataFrame([anchor_row]).to_csv(out_dir / "requested_anchor_local_rank.csv", index=False, encoding="utf-8-sig")

    counts = {
        role: {key: int(len(values)) for key, values in grids.items()}
        for role, grids in LOCAL_GRIDS.items()
    }
    total = 1
    for grids in counts.values():
        for count in grids.values():
            total *= count
    manifest = {
        "core": "Cal_0521 L13-only six-port prediction model",
        "objective": "local anchor-box 16-20 dB centered coverage, 35-85 GHz core fallback, Zin/ZOPT secondary",
        "candidate_space_count": int(total),
        "unique_candidates": int(len(ranked)),
        "reuse_count": int(builder.reused_count),
        "generated_count": int(builder.generated_count),
        "full_band_ghz": [float(args.full_lo_ghz), float(args.full_hi_ghz)],
        "core_band_ghz": [float(args.core_lo_ghz), float(args.core_hi_ghz)],
        "gain_window_db": [float(args.gain_min_db), float(args.gain_max_db)],
        "best": ranked.iloc[0].to_dict(),
        "requested_anchor": anchor_row,
    }
    (out_dir / "local_anchor_range_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"output_dir={out_dir.resolve()}")
    print(f"unique_candidates={len(ranked)}")
    print(f"reuse_count={builder.reused_count}")
    print(f"generated_count={builder.generated_count}")
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
    print("requested_anchor")
    print(pd.Series(anchor_row)[[
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
    return out_dir.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transistor-dir", type=Path, default=DEFAULT_TRANSISTOR_DIR)
    parser.add_argument("--loadpull-xlsx", type=Path, default=DEFAULT_LOADPULL_XLSX)
    parser.add_argument("--reuse-dir", type=Path, default=DEFAULT_REUSE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--l13-model", choices=["anchors8", "full80-log-trilinear"], default="anchors8")
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--full-lo-ghz", type=float, default=30.0)
    parser.add_argument("--full-hi-ghz", type=float, default=90.0)
    parser.add_argument("--core-lo-ghz", type=float, default=35.0)
    parser.add_argument("--core-hi-ghz", type=float, default=85.0)
    parser.add_argument("--gain-min-db", type=float, default=16.0)
    parser.add_argument("--gain-max-db", type=float, default=20.0)
    parser.add_argument("--zin-weight", type=float, default=1.0)
    parser.add_argument("--zin-soft-ohm", type=float, default=13.0)
    parser.add_argument("--max-evals", type=int, default=1200)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.28)
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

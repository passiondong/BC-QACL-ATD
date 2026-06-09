#!/usr/bin/env python3
"""Global Cal_0524-speed PA synthesis with a 485 um length constraint.

The search box is the full transformer design box:

    W_TF      in [90, 120] um, step 0.5 um
    alpha_L/W in [0.8, 2.0], step 0.01
    alpha_wc/W in [0.15, 0.30], step 0.005

The scalar CMA-ES score is only used to steer the search. Final reporting uses
hard filtering and a three-objective Pareto front:

    maximize  relative-peak 3-dB overlap inside 30-80 GHz
    minimize  OMN Zin/ZOPT RMS error
    maximize  S21 peak, with peak constrained to 16.9-20.1 dB

The total matching length sum(W_TF * alpha_L/W) is a hard constraint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf

import Cal_0524_speed
import tf_analysis_pipeline_cli_0521_v2 as tf0521v2
from optimize_predicted_pa_cascade_cmaes import Combo, Triple, cma_es
from optimize_predicted_pa_global_anchor_objective_cal0521_v2 import FastPAEvaluator
from optimize_predicted_pa_global_center_band import (
    GLOBAL_GRIDS,
    combo_to_normalized_global,
    nearest_triple,
    normalized_to_combo_global,
)
from optimize_predicted_pa_local_anchor_range_cal0521_v2 import infer_uniform_frequency_grid
from optimize_predicted_pa_local_relative_3db_cal0521_v2 import relative_3db_metrics
from pa_synthesis.data_loaders import load_loadpull_zopt, load_transistor_s4p
from pa_synthesis.network_utils import align_network
from run_three_tf_v3_pa_cascade import SIX_PORT_NAMES


ROOT = Path(__file__).resolve().parent
DEFAULT_METRIC_DIR = Path(r".\HFSS\For_Paper\ForModelling\Predict_Model_Compare_metric")
DEFAULT_TRANSISTOR_DIR = DEFAULT_METRIC_DIR / "transistor_z4p_s4p"
DEFAULT_LOADPULL_XLSX = DEFAULT_METRIC_DIR / "loadpull_marker_stats_30_80GHz.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "predicted_pa_global_relative_3db_len485_pareto_cal0524_speed_20260524"
DEFAULT_REUSE_DIRS = [
    ROOT / "outputs" / "predicted_pa_global_anchor_objective_cal0521_v2_16_20_fast_20260521" / "predicted_transformers",
    ROOT / "outputs" / "predicted_pa_local_relative_3db_len485_pareto_cal0521_v2_20260523" / "predicted_transformers",
    ROOT / "outputs" / "predicted_pa_local_30_80_cal0521_v2_20260521" / "predicted_transformers",
]

ROLES = ["input_match", "interstage_match", "output_match"]
ATTRS = ["W", "R", "WlineR"]

LOCAL_BOX = {
    "input_match": {"W": (108.0, 114.0), "R": (1.37, 1.56), "WlineR": (0.22, 0.25)},
    "interstage_match": {"W": (104.0, 107.0), "R": (1.20, 1.30), "WlineR": (0.22, 0.24)},
    "output_match": {"W": (100.0, 103.0), "R": (1.70, 1.80), "WlineR": (0.24, 0.25)},
}


def combo_length_um(combo: Combo) -> float:
    return float(combo.imn.W * combo.imn.R + combo.ismn.W * combo.ismn.R + combo.omn.W * combo.omn.R)


def length_from_row(row: pd.Series) -> float:
    return float(row["input_W"] * row["input_R"] + row["interstage_W"] * row["interstage_R"] + row["output_W"] * row["output_R"])


def target_overlap(lo: float, hi: float, target_lo: float, target_hi: float) -> float:
    return float(max(0.0, min(float(hi), target_hi) - max(float(lo), target_lo)))


def combo_from_row(row: pd.Series) -> Combo:
    return Combo(
        Triple(float(row["input_W"]), float(row["input_R"]), float(row["input_WlineR"])),
        Triple(float(row["interstage_W"]), float(row["interstage_R"]), float(row["interstage_WlineR"])),
        Triple(float(row["output_W"]), float(row["output_R"]), float(row["output_WlineR"])),
    )


def combo_in_requested_local_box(combo: Combo) -> bool:
    triples = {"input_match": combo.imn, "interstage_match": combo.ismn, "output_match": combo.omn}
    for role, triple in triples.items():
        for attr, value in zip(ATTRS, triple.key()):
            lo, hi = LOCAL_BOX[role][attr]
            if not (lo - 1e-12 <= float(value) <= hi + 1e-12):
                return False
    return True


def row_in_requested_local_box(row: pd.Series) -> bool:
    return combo_in_requested_local_box(combo_from_row(row))


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


def random_combos_local_box(count: int, seed: int, *, length_limit_um: float | None = None) -> list[Combo]:
    rng = np.random.default_rng(seed)
    out: list[Combo] = []
    attempts = 0
    max_attempts = max(5000, count * 250)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        triples = []
        for role in ROLES:
            ranges = LOCAL_BOX[role]
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


def seed_combos_from_csv(path: Path, limit: int) -> list[Combo]:
    if not path.exists():
        return []
    df = pd.read_csv(path).head(limit)
    out: list[Combo] = []
    for _, row in df.iterrows():
        try:
            out.append(combo_from_row(row))
        except KeyError:
            continue
    return out


def nondominated_mask(df: pd.DataFrame) -> np.ndarray:
    values = np.column_stack(
        [
            df["target_overlap_ghz"].to_numpy(float),
            -df["omn_zin_rms_to_zopt_ohm"].to_numpy(float),
            df["s21_peak_db"].to_numpy(float),
        ]
    )
    keep = np.ones(len(df), dtype=bool)
    for idx, row in enumerate(values):
        dominated = np.all(values >= row - 1e-12, axis=1) & np.any(values > row + 1e-12, axis=1)
        if dominated.any():
            keep[idx] = False
    return keep


def add_percentile_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["P_bw"] = (-out["target_overlap_ghz"]).rank(method="average", pct=True)
    out["P_zin"] = out["omn_zin_rms_to_zopt_ohm"].rank(method="average", pct=True)
    out["P_peak"] = (-out["s21_peak_db"]).rank(method="average", pct=True)
    out["equal_priority_score"] = (out["P_bw"] + out["P_zin"] + out["P_peak"]) / 3.0
    out = out.sort_values(
        ["equal_priority_score", "target_overlap_ghz", "omn_zin_rms_to_zopt_ohm", "s21_peak_db", "length_total_um"],
        ascending=[True, False, True, False, True],
    ).reset_index(drop=True)
    out.insert(0, "selected_rank", np.arange(1, len(out) + 1))
    return out


def add_basic_rank(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["hard_feasible_flag"] = data["hard_feasible"].astype(int)
    data["length_ok_flag"] = data["length_ok"].astype(int)
    data["peak_gain_ok_flag"] = (data["peak_gain_window_violation_db"].fillna(1e9) <= 1e-12).astype(int)
    data["target_overlap_sort"] = data["target_overlap_ghz"].fillna(-1.0)
    data["zin_sort"] = data["omn_zin_rms_to_zopt_ohm"].fillna(1e9)
    data["peak_sort"] = data["s21_peak_db"].fillna(-1e9)
    data = data.sort_values(
        [
            "hard_feasible_flag",
            "length_ok_flag",
            "peak_gain_ok_flag",
            "target_overlap_sort",
            "zin_sort",
            "peak_sort",
            "global_len485_score",
        ],
        ascending=[False, False, False, False, True, False, True],
    ).reset_index(drop=True)
    data.insert(0, "rank", np.arange(1, len(data) + 1))
    return data.drop(columns=["target_overlap_sort", "zin_sort", "peak_sort"])


class ReuseCal0524SpeedPredictedBuilder:
    """S6P builder using Cal_0524_speed generation and file-level reuse."""

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
            "tf_pred_cal0521v2_l13only_"
            f"W{tf0521v2.base.safe_tag(triple.W)}_"
            f"R{tf0521v2.base.safe_tag(triple.R)}_"
            f"WlineR{tf0521v2.base.safe_tag(triple.WlineR)}"
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
            ntw = align_network(ntw, self.freq_hz, f"{role}_{triple.label()}_cal0524speed_reused")
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
        prediction = tf0521v2.base.predict_l13_nH(
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
                "sgdvcl_cal0524speed_"
                f"W{tf0521v2.base.safe_tag(triple.W)}_"
                f"R{tf0521v2.base.safe_tag(triple.R)}_"
                f"WlineR{tf0521v2.base.safe_tag(triple.WlineR)}.s4p"
            ),
            "quiet": True,
            "write_manifest": True,
        }
        if self.line_length_scale is not None:
            params["line_length_scale"] = float(self.line_length_scale)
        if self.gnd_width_factor is not None:
            params["GND_width_factor"] = float(self.gnd_width_factor)

        geometry = Cal_0524_speed.half_tf_geometry_from_formula(
            triple.W,
            triple.R,
            triple.WlineR,
            line_length_scale=self.line_length_scale,
            gnd_width_factor=self.gnd_width_factor,
        )
        t0 = time.perf_counter()
        s4p_path = Cal_0524_speed.write_sgdvcl_s4p_from_half_tf_fast(
            triple.W,
            triple.R,
            triple.WlineR,
            params=params,
        )
        self.s4p_generation_seconds += time.perf_counter() - t0

        t1 = time.perf_counter()
        pred6 = tf0521v2.base.build_l13_only_six_port(
            s4p_path,
            prediction.L13_nH,
            freq_start_ghz=self.freq_start_ghz,
            freq_stop_ghz=self.freq_stop_ghz,
            freq_step_ghz=self.freq_step_ghz,
        )
        stem = self._s6p_stem(triple)
        s6p_path = tf0521v2.base.write_s6p(pred6, role_dir / "predicted_s6p", stem)
        self.s6p_assembly_seconds += time.perf_counter() - t1

        ntw = align_network(pred6, self.freq_hz, f"{role}_{triple.label()}_cal0524speed_generated")
        ntw.port_names = list(SIX_PORT_NAMES)
        self.cache[key] = ntw
        self.metadata_cache[key] = {
            "role": role,
            "W": triple.W,
            "R": triple.R,
            "WlineR": triple.WlineR,
            "L13_nH": float(prediction.L13_nH),
            "L24_nH": tf0521v2.L24_OPEN_NH,
            "L56_pH": tf0521v2.L56_SHORT_PH,
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


class GlobalLengthParetoEvaluator:
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
        length_limit_um: float,
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
        self.length_limit_um = float(length_limit_um)
        self.cache: dict[tuple, dict[str, float | str | bool]] = {}
        self.length_short_circuit_count = 0

    @property
    def target_span(self) -> float:
        return self.target_hi - self.target_lo

    def _base_row(self, combo: Combo, source: str, length_total: float, score: float) -> dict[str, float | str | bool]:
        return {
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
            "length_total_um": float(length_total),
            "length_excess_um": max(0.0, float(length_total) - self.length_limit_um),
            "length_ok": bool(length_total <= self.length_limit_um + 1e-12),
            "in_requested_local_box": combo_in_requested_local_box(combo),
            "global_len485_score": float(score),
        }

    def evaluate(self, combo: Combo, *, source: str = "global_len485") -> dict[str, float | str | bool]:
        key = combo.key()
        if key in self.cache:
            row = dict(self.cache[key])
            row["source"] = source
            return row

        length_total = combo_length_um(combo)
        length_excess = max(0.0, length_total - self.length_limit_um)
        if length_excess > 0.0:
            self.length_short_circuit_count += 1
            score = 1.0e7 * (length_excess / 10.0) ** 2 + 1.0e6
            row = self._base_row(combo, source, length_total, score)
            row.update(
                {
                    "hard_feasible": False,
                    "s21_peak_db": np.nan,
                    "s21_peak_frequency_ghz": np.nan,
                    "s21_relative_3db_threshold_db": np.nan,
                    "relative_3db_lower_ghz": np.nan,
                    "relative_3db_upper_ghz": np.nan,
                    "relative_3db_width_ghz": np.nan,
                    "target_overlap_ghz": np.nan,
                    "peak_gain_window_violation_db": np.nan,
                    "target_30_80_relative_violation_rms_db": np.nan,
                    "target_30_80_relative_violation_max_db": np.nan,
                    "omn_zin_rms_to_zopt_ohm": np.nan,
                    "omn_zin_mean_abs_to_zopt_ohm": np.nan,
                    "omn_zin_max_abs_to_zopt_ohm": np.nan,
                }
            )
            self.cache[key] = dict(row)
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
        overlap = target_overlap(
            float(metrics["relative_3db_lower_ghz"]),
            float(metrics["relative_3db_upper_ghz"]),
            self.target_lo,
            self.target_hi,
        )
        gain_excess = float(metrics["peak_gain_window_violation_db"])
        target_shortfall = max(0.0, self.target_span - overlap)
        score = (
            5.0e5 * gain_excess**2
            + 110.0 * target_shortfall
            + 40.0 * float(metrics["target_30_80_relative_violation_rms_db"])
            + float(zin["omn_zin_rms_to_zopt_ohm"])
            - 2.0 * float(metrics["s21_peak_db"])
        )
        row = self._base_row(combo, source, length_total, score)
        row.update(metrics)
        row.update(zin)
        row["target_overlap_ghz"] = float(overlap)
        row["hard_feasible"] = bool(row["length_ok"] and gain_excess <= 1e-12)
        self.cache[key] = dict(row)
        return row


def neighbor_values(role: str, attr: str, value: float, radius: int) -> list[float]:
    grid = GLOBAL_GRIDS[role][attr]
    idx = int(np.argmin(np.abs(grid - float(value))))
    lo = max(0, idx - int(radius))
    hi = min(len(grid) - 1, idx + int(radius))
    return [float(v) for v in grid[lo : hi + 1]]


def coordinate_polish(evalr: GlobalLengthParetoEvaluator, rows: pd.DataFrame, *, top_k: int, rounds: int, radius: int) -> None:
    for round_idx in range(int(rounds)):
        base = rows.loc[rows["length_ok"]].head(int(top_k))
        before = len(evalr.cache)
        for _, row in base.iterrows():
            combo = combo_from_row(row)
            triples_by_role = {"input_match": combo.imn, "interstage_match": combo.ismn, "output_match": combo.omn}
            for role_idx, role in enumerate(ROLES):
                current = triples_by_role[role]
                for attr in ATTRS:
                    for value in neighbor_values(role, attr, getattr(current, attr), radius):
                        vals = {"W": current.W, "R": current.R, "WlineR": current.WlineR}
                        vals[attr] = float(value)
                        new_triples = [triples_by_role[r] for r in ROLES]
                        new_triples[role_idx] = Triple(vals["W"], vals["R"], vals["WlineR"])
                        new_combo = Combo(*new_triples)
                        evalr.evaluate(new_combo, source=f"polish{round_idx + 1}")
        rows = add_basic_rank(pd.DataFrame(evalr.cache.values()))
        print(f"polish {round_idx + 1}/{rounds}: added {len(evalr.cache) - before}, unique={len(evalr.cache)}", flush=True)


def apply_tmtt_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.4,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.75,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ["pdf", "png", "svg"]:
        kwargs: dict[str, object] = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out_dir / f"{stem}.{ext}", **kwargs)


def plot_pareto(df: pd.DataFrame, front: pd.DataFrame, out_dir: Path) -> None:
    apply_tmtt_style()
    feasible = df.loc[df["hard_feasible"]].copy()
    sample = feasible.sample(min(len(feasible), 3500), random_state=20260524) if len(feasible) else feasible
    local_front = front.loc[front["in_requested_local_box"]].copy()

    fig = plt.figure(figsize=(4.1, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        sample["target_overlap_ghz"],
        sample["omn_zin_rms_to_zopt_ohm"],
        sample["s21_peak_db"],
        s=5,
        c="#9A9A9A",
        alpha=0.14,
        depthshade=False,
        label="Feasible candidates",
    )
    ax.scatter(
        front["target_overlap_ghz"],
        front["omn_zin_rms_to_zopt_ohm"],
        front["s21_peak_db"],
        s=22,
        facecolors="white",
        edgecolors="#0F7BC4",
        linewidths=0.85,
        depthshade=False,
        label="Pareto front",
    )
    if not local_front.empty:
        ax.scatter(
            local_front["target_overlap_ghz"],
            local_front["omn_zin_rms_to_zopt_ohm"],
            local_front["s21_peak_db"],
            marker="*",
            s=100,
            c="#D95218",
            edgecolors="black",
            linewidths=0.45,
            depthshade=False,
            label="Local-box front",
        )
    ax.set_xlabel("30-80 GHz overlap\nbandwidth (GHz)", labelpad=7)
    ax.set_ylabel("Zin/ZOPT RMS (Ohm)", labelpad=8)
    ax.set_zlabel("S21 peak (dB)", labelpad=7)
    ax.view_init(elev=23, azim=-53)
    ax.grid(True)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), fontsize=6.2, frameon=True)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_alpha(0.86)
    for text in leg.get_texts():
        text.set_fontweight("bold")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    save_all(fig, out_dir, "fig_global_len485_3d_pareto")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.1, 2.25))
    ax.scatter(
        feasible["target_overlap_ghz"],
        feasible["omn_zin_rms_to_zopt_ohm"],
        s=6,
        c="#B5B5B5",
        alpha=0.18,
        linewidths=0.0,
        label="Feasible candidates",
    )
    sc = ax.scatter(
        front["target_overlap_ghz"],
        front["omn_zin_rms_to_zopt_ohm"],
        s=26,
        c=front["s21_peak_db"],
        cmap="Blues",
        edgecolors="#0F7BC4",
        linewidths=0.55,
        label="Pareto front",
        zorder=3,
    )
    if not local_front.empty:
        ax.scatter(
            local_front["target_overlap_ghz"],
            local_front["omn_zin_rms_to_zopt_ohm"],
            marker="*",
            s=105,
            c="#D95218",
            edgecolors="black",
            linewidths=0.45,
            label="Local-box front",
            zorder=4,
        )
    cb = fig.colorbar(sc, ax=ax, pad=0.015)
    cb.set_label("S21 peak (dB)", fontweight="bold")
    for tick in cb.ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.set_xlabel("30-80 GHz overlap bandwidth (GHz)")
    ax.set_ylabel("Zin/ZOPT RMS (Ohm)")
    ax.grid(True, which="major", alpha=0.25)
    ax.minorticks_off()
    leg = ax.legend(loc="best", fontsize=6.0, frameon=True)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_alpha(0.88)
    for text in leg.get_texts():
        text.set_fontweight("bold")
    save_all(fig, out_dir, "fig_global_len485_2d_pareto")
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 2.75))
    ax3 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    ax3.scatter(
        sample["target_overlap_ghz"],
        sample["omn_zin_rms_to_zopt_ohm"],
        sample["s21_peak_db"],
        s=5,
        c="#B0B0B0",
        alpha=0.13,
        depthshade=False,
        label="Feasible candidates",
    )
    ax3.scatter(
        front["target_overlap_ghz"],
        front["omn_zin_rms_to_zopt_ohm"],
        front["s21_peak_db"],
        s=20,
        facecolors="white",
        edgecolors="#0F7BC4",
        linewidths=0.8,
        depthshade=False,
        label="Pareto front",
    )
    if not local_front.empty:
        ax3.scatter(
            local_front["target_overlap_ghz"],
            local_front["omn_zin_rms_to_zopt_ohm"],
            local_front["s21_peak_db"],
            marker="*",
            s=110,
            c="#D95218",
            edgecolors="black",
            linewidths=0.45,
            depthshade=False,
            label="Local-box front",
        )
    ax3.set_xlabel("30-80 GHz overlap\nbandwidth (GHz)", labelpad=7)
    ax3.set_ylabel("Zin/ZOPT RMS (Ohm)", labelpad=8)
    ax3.set_zlabel("S21 peak (dB)", labelpad=7)
    ax3.view_init(elev=23, azim=-53)
    ax3.grid(True)
    ax2.scatter(feasible["target_overlap_ghz"], feasible["omn_zin_rms_to_zopt_ohm"], s=6, c="#B5B5B5", alpha=0.18, linewidths=0.0)
    sc2 = ax2.scatter(
        front["target_overlap_ghz"],
        front["omn_zin_rms_to_zopt_ohm"],
        s=25,
        c=front["s21_peak_db"],
        cmap="Blues",
        edgecolors="#0F7BC4",
        linewidths=0.55,
        zorder=3,
    )
    if not local_front.empty:
        ax2.scatter(
            local_front["target_overlap_ghz"],
            local_front["omn_zin_rms_to_zopt_ohm"],
            marker="*",
            s=105,
            c="#D95218",
            edgecolors="black",
            linewidths=0.45,
            zorder=4,
        )
    ax2.set_xlabel("30-80 GHz overlap bandwidth (GHz)")
    ax2.set_ylabel("Zin/ZOPT RMS (Ohm)")
    ax2.grid(True, which="major", alpha=0.25)
    ax2.minorticks_off()
    cb2 = fig.colorbar(sc2, ax=ax2, pad=0.015)
    cb2.set_label("S21 peak (dB)", fontweight="bold")
    for tick in cb2.ax.get_yticklabels():
        tick.set_fontweight("bold")
    save_all(fig, out_dir, "fig_global_len485_pareto_surface_readable")
    plt.close(fig)


def plot_top_s21(rows: pd.DataFrame, evalr: GlobalLengthParetoEvaluator, out_dir: Path, n: int = 20) -> None:
    apply_tmtt_style()
    top = rows.head(min(n, len(rows)))
    cols = 4
    rows_n = int(np.ceil(len(top) / cols)) if len(top) else 1
    fig, axes = plt.subplots(rows_n, cols, figsize=(10.0, max(2.6, 2.15 * rows_n)), sharex=True, sharey=True)
    axes_arr = np.asarray(axes).ravel()
    freq = evalr.pa_eval.freq_hz / 1e9
    mask = (freq >= 25.0) & (freq <= 90.0)
    for ax in axes_arr:
        ax.axis("off")
    for ax, (_, row) in zip(axes_arr, top.iterrows()):
        ax.axis("on")
        combo = combo_from_row(row)
        s21 = evalr.pa_eval.combo_s21_db(combo)
        ax.plot(freq[mask], s21[mask], color="#D95218", linewidth=1.0)
        ax.axvspan(evalr.target_lo, evalr.target_hi, color="#0F7BC4", alpha=0.08)
        ax.axhspan(evalr.gain_min, evalr.gain_max, color="#B5B5B5", alpha=0.12)
        ax.axhline(float(row["s21_relative_3db_threshold_db"]), color="black", linestyle="--", linewidth=0.65)
        local_tag = " local" if bool(row["in_requested_local_box"]) else ""
        ax.set_title(
            f"#{int(row['selected_rank'])}{local_tag} ov={row['target_overlap_ghz']:.1f} "
            f"Z={row['omn_zin_rms_to_zopt_ohm']:.2f} L={row['length_total_um']:.1f}",
            fontsize=7.2,
        )
        ax.grid(True, which="major", alpha=0.25)
        ax.minorticks_off()
    for ax in axes_arr[-cols:]:
        ax.set_xlabel("Frequency (GHz)")
    for ax in axes_arr[::cols]:
        ax.set_ylabel("S21 (dB)")
    fig.tight_layout()
    save_all(fig, out_dir, "fig_global_len485_pareto_top_s21")
    plt.close(fig)


def grid_count_manifest() -> tuple[dict[str, dict[str, list[float] | int]], int]:
    counts: dict[str, dict[str, list[float] | int]] = {}
    total = 1
    for role, grids in GLOBAL_GRIDS.items():
        role_count = 1
        counts[role] = {}
        for attr, values in grids.items():
            arr = np.asarray(values, dtype=float)
            counts[role][attr] = [float(arr[0]), float(arr[-1]), float(arr[1] - arr[0]) if len(arr) > 1 else 0.0, int(len(arr))]
            role_count *= int(len(arr))
        counts[role]["role_candidate_count"] = int(role_count)
        total *= role_count
    return counts, int(total)


def write_report(out_dir: Path, manifest: dict[str, object], front: pd.DataFrame, local_front: pd.DataFrame) -> None:
    best = front.iloc[0] if len(front) else None
    lines = [
        "# Global-Box Length-Constrained Cal_0524-Speed PA Synthesis",
        "",
        "This run uses the full transformer geometry box, not the previous local box.",
        "",
        "## Search Space",
        "",
        "- `W_TF`: 90-120 um, step 0.5 um.",
        "- `alpha_L/W` (`R`): 0.8-2.0, step 0.01.",
        "- `alpha_wc/W` (`WlineR`): 0.15-0.30, step 0.005.",
        "- Hard total-length constraint: `sum_i W_TF,i * alpha_L/W_i <= 485 um`.",
        "",
        "## Objective",
        "",
        "CMA-ES is steered by a scalar score that penalizes length violation, peak-gain violation outside 16.9-20.1 dB, missing 30-80 GHz relative-peak 3-dB overlap, and OMN `Zin/ZOPT` RMS. Final selection is reported by the three-objective Pareto front: maximize 30-80 GHz overlap bandwidth, minimize OMN `Zin/ZOPT` RMS, and maximize S21 peak within the allowed gain window.",
        "",
        "## Local-Box Check",
        "",
        "The requested local box is checked only after global optimization. A point is marked as local-box front if all three transformers fall inside:",
        "",
        "- IMN: W 108-114, R 1.37-1.56, WlineR 0.22-0.25",
        "- ISMN: W 104-107, R 1.20-1.30, WlineR 0.22-0.24",
        "- OMN: W 100-103, R 1.70-1.80, WlineR 0.24-0.25",
        "",
        "## Key Results",
        "",
        f"- Unique PA candidates evaluated: {manifest['unique_candidates_evaluated']}",
        f"- Hard-feasible candidates: {manifest['hard_feasible_candidate_count']}",
        f"- Pareto-front candidates: {manifest['pareto_front_count']}",
        f"- Pareto-front candidates inside requested local box: {manifest['local_box_pareto_front_count']}",
        f"- Reused S6P networks: {manifest['reuse_count']}",
        f"- Newly generated S6P networks: {manifest['generated_count']}",
        f"- Length short-circuit candidates: {manifest['length_short_circuit_count']}",
        "",
    ]
    if best is not None:
        lines += [
            "## Equal-Priority Rank-1 on Pareto Front",
            "",
            f"- IMN: W{best['input_W']:.3g} R{best['input_R']:.3g} WlineR{best['input_WlineR']:.3g}",
            f"- ISMN: W{best['interstage_W']:.3g} R{best['interstage_R']:.3g} WlineR{best['interstage_WlineR']:.3g}",
            f"- OMN: W{best['output_W']:.3g} R{best['output_R']:.3g} WlineR{best['output_WlineR']:.3g}",
            f"- 30-80 GHz overlap bandwidth: {best['target_overlap_ghz']:.2f} GHz",
            f"- Relative-peak 3-dB band: {best['relative_3db_lower_ghz']:.2f}-{best['relative_3db_upper_ghz']:.2f} GHz",
            f"- S21 peak: {best['s21_peak_db']:.2f} dB @ {best['s21_peak_frequency_ghz']:.1f} GHz",
            f"- OMN Zin/ZOPT RMS: {best['omn_zin_rms_to_zopt_ohm']:.3f} ohm",
            f"- Total matching length: {best['length_total_um']:.2f} um",
            f"- Inside requested local box: {bool(best['in_requested_local_box'])}",
            "",
        ]
    if not local_front.empty:
        local_table = local_front.head(10)[
            [
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
                "omn_zin_rms_to_zopt_ohm",
                "s21_peak_db",
                "length_total_um",
            ]
        ].to_csv(index=False).strip()
        lines += [
            "## Best Local-Box Pareto Points",
            "",
            "```csv",
            local_table,
            "```",
            "",
        ]
    lines += [
        "## Output Files",
        "",
        "- `global_len485_ranked_candidates_all.csv`",
        "- `global_len485_hard_feasible_candidates.csv`",
        "- `global_len485_pareto_front_equal_priority.csv`",
        "- `global_len485_pareto_front_in_requested_local_box.csv`",
        "- `fig_global_len485_3d_pareto.pdf/png/svg`",
        "- `fig_global_len485_2d_pareto.pdf/png/svg`",
        "- `fig_global_len485_pareto_surface_readable.pdf/png/svg`",
        "- `fig_global_len485_pareto_top_s21.pdf/png/svg`",
    ]
    (out_dir / "global_len485_pareto_synthesis_report.md").write_text("\n".join(lines), encoding="utf-8")


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

    builder = ReuseCal0524SpeedPredictedBuilder(
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
            nearest_triple("input_match", 113.0, 1.44, 0.22),
            nearest_triple("interstage_match", 106.5, 1.24, 0.22),
            nearest_triple("output_match", 101.5, 1.80, 0.24),
        ),
        Combo(
            nearest_triple("input_match", 112.5, 1.37, 0.23),
            nearest_triple("interstage_match", 106.5, 1.25, 0.225),
            nearest_triple("output_match", 101.5, 1.94, 0.225),
        ),
        Combo(
            nearest_triple("input_match", 114.0, 1.45, 0.22),
            nearest_triple("interstage_match", 106.5, 1.20, 0.24),
            nearest_triple("output_match", 103.0, 1.80, 0.24),
        ),
        Combo(
            nearest_triple("input_match", 109.5, 1.37, 0.22),
            nearest_triple("interstage_match", 104.0, 1.20, 0.22),
            nearest_triple("output_match", 103.0, 1.80, 0.24),
        ),
    ]
    seed_combos.extend(seed_combos_from_csv(Path(args.previous_top_csv), args.previous_seed_count))
    seed_combos.extend(random_combos_local_box(args.local_random_seed_count, args.seed + 100, length_limit_um=args.length_limit_um))
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

    ranked = add_basic_rank(pd.DataFrame(evalr.cache.values()))
    feasible = ranked.loc[ranked["hard_feasible"]].copy()
    if feasible.empty:
        raise RuntimeError("No hard-feasible candidates found. Increase max_evals or relax constraints.")
    front_raw = feasible.loc[nondominated_mask(feasible)].copy()
    front = add_percentile_rank(front_raw)
    local_front = front.loc[front["in_requested_local_box"]].copy()

    ranked.to_csv(out_dir / "global_len485_ranked_candidates_all.csv", index=False, encoding="utf-8-sig")
    feasible.to_csv(out_dir / "global_len485_hard_feasible_candidates.csv", index=False, encoding="utf-8-sig")
    front.to_csv(out_dir / "global_len485_pareto_front_equal_priority.csv", index=False, encoding="utf-8-sig")
    local_front.to_csv(out_dir / "global_len485_pareto_front_in_requested_local_box.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(traces).to_csv(out_dir / "global_len485_cmaes_trace.csv", index=False, encoding="utf-8-sig")
    builder.write_metadata(out_dir / "cal0524_speed_global_len485_transformer_build_metadata.csv")

    plot_pareto(ranked, front, out_dir)
    plot_top_s21(front, evalr, out_dir, n=min(args.top_n, 20))

    grid_counts, full_count = grid_count_manifest()
    elapsed = time.perf_counter() - t_start
    manifest = {
        "core": "Cal_0524_speed write-only S4P + Cal_0521_v2 L13-only six-port assembly",
        "global_box": "W 90-120 step 0.5, R 0.8-2.0 step 0.01, WlineR 0.15-0.30 step 0.005",
        "objective": "CMA-ES scalar search; final hard filtering + Pareto: maximize 30-80 GHz relative-3dB overlap, minimize OMN Zin/ZOPT RMS, maximize S21 peak",
        "target_band_ghz": [float(args.target_lo_ghz), float(args.target_hi_ghz)],
        "fallback_band_ghz": [float(args.fallback_lo_ghz), float(args.fallback_hi_ghz)],
        "gain_window_db": [float(args.gain_min_db), float(args.gain_max_db)],
        "length_limit_um": float(args.length_limit_um),
        "grid_counts": grid_counts,
        "global_pa_candidate_space_count": int(full_count),
        "unique_candidates_evaluated": int(len(ranked)),
        "hard_feasible_candidate_count": int(len(feasible)),
        "pareto_front_count": int(len(front)),
        "local_box_pareto_front_count": int(len(local_front)),
        "local_box_hard_feasible_count": int(feasible["in_requested_local_box"].sum()),
        "reuse_roots": [str(Path(p).resolve()) for p in list(args.reuse_dir or [])],
        "reuse_count": int(builder.reused_count),
        "generated_count": int(builder.generated_count),
        "length_short_circuit_count": int(evalr.length_short_circuit_count),
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
        "local_box_front_best": None if local_front.empty else local_front.iloc[0].to_dict(),
    }
    (out_dir / "global_len485_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    write_report(out_dir, manifest, front, local_front)

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
        "in_requested_local_box",
        "equal_priority_score",
    ]
    print(f"output_dir={out_dir.resolve()}")
    print(f"global_pa_candidate_space_count={full_count}")
    print(f"unique_candidates={len(ranked)}")
    print(f"hard_feasible={len(feasible)}")
    print(f"pareto_front={len(front)}")
    print(f"local_box_pareto_front={len(local_front)}")
    print(f"reuse_count={builder.reused_count}")
    print(f"generated_count={builder.generated_count}")
    print(f"length_short_circuit_count={evalr.length_short_circuit_count}")
    print("front_top")
    print(front.head(min(args.top_n, 20))[show_cols].to_string(index=False))
    if not local_front.empty:
        print("local_box_front_top")
        print(local_front.head(min(args.top_n, 20))[show_cols].to_string(index=False))
    else:
        print("local_box_front_top=<empty>")
    return out_dir.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transistor-dir", type=Path, default=DEFAULT_TRANSISTOR_DIR)
    parser.add_argument("--loadpull-xlsx", type=Path, default=DEFAULT_LOADPULL_XLSX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reuse-dir", type=Path, action="append", default=DEFAULT_REUSE_DIRS)
    parser.add_argument("--previous-top-csv", type=Path, default=ROOT / "outputs" / "predicted_pa_local_relative_3db_len485_pareto_cal0521_v2_20260523" / "global_len485_pareto_front_equal_priority.csv")
    parser.add_argument("--previous-seed-count", type=int, default=50)
    parser.add_argument("--random-seed-count", type=int, default=450)
    parser.add_argument("--local-random-seed-count", type=int, default=350)
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
    parser.add_argument("--length-limit-um", type=float, default=485.0)
    parser.add_argument("--max-evals", type=int, default=2600)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--sigma", type=float, default=0.26)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--polish-rounds", type=int, default=2)
    parser.add_argument("--polish-top-k", type=int, default=40)
    parser.add_argument("--polish-radius", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

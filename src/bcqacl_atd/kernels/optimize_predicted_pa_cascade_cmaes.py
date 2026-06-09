#!/usr/bin/env python3
"""CMA-ES search of predicted three-transformer PA cascades.

Search variables are the three transformer geometry triples:

    input_match:      W, R, WlineR
    interstage_match: W, R, WlineR
    output_match:     W, R, WlineR

The CMA-ES samples continuous normalized variables, then snaps them to the
requested discrete design grid:

    W      = 90..120 step 1
    R      = 0.8..2.0 step 0.1
    WlineR = 0.15..0.30 step 0.01

Each snapped geometry is converted to a predicted SG-DVCL S4P through
``tf_analysis_pipeline_cli_v3.py``, assembled into the standard six-port
transformer with compensation inductors, then cascaded with the two transistor
S4P files.  The objective rewards S21 staying between 16 and 20 dB, first over
30-80 GHz and then, if possible, extending inside 30-90 GHz.  It also penalizes
the output-match balun input impedance RMS error to load-pull ZOPT.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf

import tf_analysis_pipeline_cli_v3 as tf_v3
from pa_synthesis.data_loaders import load_loadpull_zopt, load_transistor_s4p
from pa_synthesis.network_utils import align_network, build_full_pa_network, transformer_single_input_impedance
from run_three_tf_v3_pa_cascade import SIX_PORT_NAMES, TransformerSpec, build_transformer_s6p, infer_uniform_frequency_grid


ROOT = Path(__file__).resolve().parent
DEFAULT_METRIC_DIR = Path(r".\HFSS\For_Paper\ForModelling\Predict_Model_Compare_metric")
DEFAULT_TRANSISTOR_DIR = DEFAULT_METRIC_DIR / "transistor_z4p_s4p"
DEFAULT_LOADPULL_XLSX = DEFAULT_METRIC_DIR / "loadpull_marker_stats_30_80GHz.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "predicted_pa_cmaes_gain_window"

W_VALUES = np.arange(90.0, 120.0 + 0.5, 1.0)
R_VALUES = np.round(np.arange(0.8, 2.0 + 0.05, 0.1), 10)
WLINER_VALUES = np.round(np.arange(0.15, 0.30 + 0.005, 0.01), 10)

GAIN_MIN_DB = 16.0
GAIN_MAX_DB = 20.0
BASE_MIN_GHZ = 30.0
BASE_MAX_GHZ = 80.0
EXT_MAX_GHZ = 90.0
MAX_USEFUL_WIDTH_GHZ = EXT_MAX_GHZ - BASE_MIN_GHZ
VIEW_MIN_GHZ = 10.0
VIEW_MAX_GHZ = 110.0
MAG_FLOOR_DB = -30.0


@dataclass(frozen=True)
class Triple:
    W: float
    R: float
    WlineR: float

    def key(self) -> tuple[float, float, float]:
        return (round(float(self.W), 6), round(float(self.R), 6), round(float(self.WlineR), 6))

    def label(self) -> str:
        return f"W{self.W:g}R{self.R:g}WlineR{self.WlineR:g}"


@dataclass(frozen=True)
class Combo:
    imn: Triple
    ismn: Triple
    omn: Triple

    def key(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        return (self.imn.key(), self.ismn.key(), self.omn.key())

    def label(self) -> str:
        return f"IMN_{self.imn.label()}__ISMN_{self.ismn.label()}__OMN_{self.omn.label()}"


def tag(value: float, *, digits: int = 8) -> str:
    return f"{float(value):.{digits}g}".replace("-", "m").replace(".", "p")


def db20(value: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(value), 1e-30))


def clipped_db20(value: np.ndarray, floor_db: float = MAG_FLOOR_DB) -> np.ndarray:
    return np.maximum(db20(value), floor_db)


def mask_range(freq_ghz: np.ndarray, lo: float, hi: float) -> np.ndarray:
    eps = 1e-9
    return (freq_ghz >= lo - eps) & (freq_ghz <= hi + eps)


def nearest_grid(value: float, grid: np.ndarray) -> float:
    idx = int(np.argmin(np.abs(grid - float(value))))
    return float(grid[idx])


def snap_triple(W: float, R: float, WlineR: float) -> Triple:
    return Triple(nearest_grid(W, W_VALUES), nearest_grid(R, R_VALUES), nearest_grid(WlineR, WLINER_VALUES))


def normalized_to_combo(x: np.ndarray) -> Combo:
    vals = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    triples: list[Triple] = []
    grids = (W_VALUES, R_VALUES, WLINER_VALUES)
    for offset in (0, 3, 6):
        raw = []
        for local, grid in enumerate(grids):
            idx = int(np.rint(vals[offset + local] * (len(grid) - 1)))
            idx = max(0, min(len(grid) - 1, idx))
            raw.append(float(grid[idx]))
        triples.append(Triple(*raw))
    return Combo(triples[0], triples[1], triples[2])


def combo_to_normalized(combo: Combo) -> np.ndarray:
    vals: list[float] = []
    for triple in (combo.imn, combo.ismn, combo.omn):
        for value, grid in zip(triple.key(), (W_VALUES, R_VALUES, WLINER_VALUES)):
            idx = int(np.argmin(np.abs(grid - float(value))))
            vals.append(idx / (len(grid) - 1))
    return np.asarray(vals, dtype=float)


def widest_true_band(freq_ghz: np.ndarray, keep: np.ndarray) -> tuple[float, float, float]:
    freq = np.asarray(freq_ghz, dtype=float)
    flags = np.asarray(keep, dtype=bool)
    best_width = 0.0
    best_lo = float("nan")
    best_hi = float("nan")
    start: int | None = None
    for idx, flag in enumerate(flags.tolist() + [False]):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            end = idx - 1
            width = float(freq[end] - freq[start])
            if width > best_width:
                best_width = width
                best_lo = float(freq[start])
                best_hi = float(freq[end])
            start = None
    return best_width, best_lo, best_hi


def gain_window_metrics(freq_ghz: np.ndarray, s21_db: np.ndarray) -> dict[str, float | bool]:
    freq = np.asarray(freq_ghz, dtype=float)
    mag = np.asarray(s21_db, dtype=float)

    base = mask_range(freq, BASE_MIN_GHZ, BASE_MAX_GHZ)
    ext = mask_range(freq, BASE_MIN_GHZ, EXT_MAX_GHZ)
    base_violation = np.maximum(GAIN_MIN_DB - mag[base], 0.0) + np.maximum(mag[base] - GAIN_MAX_DB, 0.0)
    ext_violation = np.maximum(GAIN_MIN_DB - mag[ext], 0.0) + np.maximum(mag[ext] - GAIN_MAX_DB, 0.0)
    in_window = (mag >= GAIN_MIN_DB) & (mag <= GAIN_MAX_DB)

    width80, lo80, hi80 = widest_true_band(freq[base], in_window[base])
    width90, lo90, hi90 = widest_true_band(freq[ext], in_window[ext])
    width90 = min(float(width90), MAX_USEFUL_WIDTH_GHZ)
    full80_ok = bool(np.all(base_violation <= 1e-12))

    peak_mask = mask_range(freq, VIEW_MIN_GHZ, VIEW_MAX_GHZ)
    peak_idx_local = int(np.nanargmax(mag[peak_mask]))
    peak_freqs = freq[peak_mask]
    peak_mags = mag[peak_mask]
    center90 = 0.5 * (lo90 + hi90) if np.isfinite(lo90) and np.isfinite(hi90) else float("nan")
    rel90 = 100.0 * width90 / center90 if np.isfinite(center90) and center90 > 0 else 0.0

    return {
        "full_30_80_gain_window_ok": full80_ok,
        "gain_window_width_30_80_ghz": float(width80),
        "gain_window_lower_30_80_ghz": float(lo80),
        "gain_window_upper_30_80_ghz": float(hi80),
        "gain_window_width_30_90_ghz": float(width90),
        "gain_window_lower_30_90_ghz": float(lo90),
        "gain_window_upper_30_90_ghz": float(hi90),
        "gain_window_relative_bandwidth_30_90_pct": float(rel90),
        "gain_violation_rms_30_80_db": float(np.sqrt(np.mean(base_violation * base_violation))),
        "gain_violation_mean_30_80_db": float(np.mean(base_violation)),
        "gain_violation_max_30_80_db": float(np.max(base_violation)),
        "gain_violation_rms_30_90_db": float(np.sqrt(np.mean(ext_violation * ext_violation))),
        "s21_peak_db_10_110": float(peak_mags[peak_idx_local]),
        "s21_peak_frequency_ghz_10_110": float(peak_freqs[peak_idx_local]),
    }


class PredictedBuilder:
    def __init__(
        self,
        *,
        model: tf_v3.LFitModel,
        output_dir: Path,
        freq_hz: np.ndarray,
        allow_extrapolation: bool,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.freq_hz = np.asarray(freq_hz, dtype=float)
        self.allow_extrapolation = allow_extrapolation
        self.freq_start_ghz, self.freq_stop_ghz, self.freq_step_ghz, self.freq_npoints = infer_uniform_frequency_grid(
            self.freq_hz
        )
        self.cache: dict[tuple[str, tuple[float, float, float]], rf.Network] = {}

    def get(self, role: str, triple: Triple) -> rf.Network:
        key = (role, triple.key())
        if key in self.cache:
            return self.cache[key]
        spec = TransformerSpec(role, triple.W, triple.R, triple.WlineR)
        build = build_transformer_s6p(
            spec,
            self.model,
            output_dir=self.output_dir,
            freq_start_ghz=self.freq_start_ghz,
            freq_stop_ghz=self.freq_stop_ghz,
            freq_step_ghz=self.freq_step_ghz,
            freq_npoints=self.freq_npoints,
            allow_extrapolation=self.allow_extrapolation,
        )
        ntw = align_network(build.network, self.freq_hz, f"{role}_{triple.label()}")
        ntw.port_names = list(SIX_PORT_NAMES)
        self.cache[key] = ntw
        return ntw


class Evaluator:
    def __init__(
        self,
        *,
        builder: PredictedBuilder,
        driver: rf.Network,
        final: rf.Network,
        loadpull_zopt: np.ndarray,
        loadpull_freq_hz: np.ndarray,
    ) -> None:
        self.builder = builder
        self.driver = driver
        self.final = final
        self.freq_hz = driver.f
        self.loadpull_zopt = loadpull_zopt
        self.loadpull_freq_hz = loadpull_freq_hz
        self.combo_cache: dict[tuple, dict[str, float | str | bool]] = {}
        self.omn_zin_cache: dict[tuple[float, float, float], dict[str, float]] = {}

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

    def evaluate(self, combo: Combo, *, source: str = "cmaes") -> dict[str, float | str | bool]:
        key = combo.key()
        if key in self.combo_cache:
            out = dict(self.combo_cache[key])
            out["source"] = source
            return out

        imn = self.builder.get("input_match", combo.imn)
        ismn = self.builder.get("interstage_match", combo.ismn)
        omn = self.builder.get("output_match", combo.omn)
        pa = build_full_pa_network(
            freq_hz=self.freq_hz,
            driver_s4p=self.driver,
            final_s4p=self.final,
            imn=imn,
            ismn=ismn,
            omn=omn,
            z0=50.0,
            include_dc_blocks=False,
        )
        freq_ghz = pa.f / 1e9
        s21_db = db20(pa.s[:, 1, 0])
        metrics = gain_window_metrics(freq_ghz, s21_db)
        metrics.update(self.omn_zin_metrics(combo.omn))

        missing_80 = max(0.0, (BASE_MAX_GHZ - BASE_MIN_GHZ) - float(metrics["gain_window_width_30_80_ghz"]))
        width_reward = float(metrics["gain_window_width_30_90_ghz"]) if metrics["full_30_80_gain_window_ok"] else float(
            metrics["gain_window_width_30_80_ghz"]
        )
        score = (
            600.0 * float(metrics["gain_violation_rms_30_80_db"]) ** 2
            + 80.0 * float(metrics["gain_violation_max_30_80_db"])
            + 35.0 * missing_80
            + 0.35 * float(metrics["omn_zin_rms_to_zopt_ohm"])
            - 5.0 * min(width_reward, MAX_USEFUL_WIDTH_GHZ)
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
            "objective_score": float(score),
        }
        row.update(metrics)
        self.combo_cache[key] = dict(row)
        return row


def cma_es(
    objective: Callable[[np.ndarray], float],
    *,
    x0: np.ndarray,
    sigma0: float,
    max_evals: int,
    popsize: int,
    seed: int,
    restart_index: int,
) -> tuple[list[dict[str, float]], np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = len(x0)
    lam = int(popsize)
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = float(np.sum(weights) ** 2 / np.sum(weights**2))

    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

    mean = np.clip(np.asarray(x0, dtype=float), 0.0, 1.0)
    sigma = float(sigma0)
    pc = np.zeros(n)
    ps = np.zeros(n)
    B = np.eye(n)
    D = np.ones(n)
    C = np.eye(n)
    invsqrtC = np.eye(n)
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    trace: list[dict[str, float]] = []
    evals = 0
    best_x = mean.copy()
    best_score = float("inf")
    generation = 0
    while evals < max_evals:
        generation += 1
        arz = rng.standard_normal((lam, n))
        ary = arz @ (B * D).T
        arx = np.clip(mean + sigma * ary, 0.0, 1.0)
        scores = np.asarray([objective(x) for x in arx], dtype=float)
        evals += lam
        order = np.argsort(scores)
        arx = arx[order]
        ary = ary[order]
        arz = arz[order]
        scores = scores[order]
        if float(scores[0]) < best_score:
            best_score = float(scores[0])
            best_x = arx[0].copy()

        old_mean = mean.copy()
        mean = np.sum(arx[:mu] * weights[:, None], axis=0)
        y_w = (mean - old_mean) / max(sigma, 1e-12)
        z_w = np.sum(arz[:mu] * weights[:, None], axis=0)
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ z_w)
        norm_ps = float(np.linalg.norm(ps))
        hsig = norm_ps / np.sqrt(1 - (1 - cs) ** (2 * evals / lam)) / chi_n < (1.4 + 2 / (n + 1))
        pc = (1 - cc) * pc + float(hsig) * np.sqrt(cc * (2 - cc) * mueff) * y_w
        artmp = ary[:mu]
        C = (
            (1 - c1 - cmu + (1 - float(hsig)) * c1 * cc * (2 - cc)) * C
            + c1 * np.outer(pc, pc)
            + cmu * np.einsum("i,ij,ik->jk", weights, artmp, artmp)
        )
        sigma *= float(np.exp((cs / damps) * (norm_ps / chi_n - 1)))
        C = np.triu(C) + np.triu(C, 1).T
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        D = np.sqrt(eigvals)
        B = eigvecs
        invsqrtC = B @ np.diag(1.0 / D) @ B.T
        _ = invsqrtC  # kept for readability and future diagnostics
        trace.append(
            {
                "restart": float(restart_index),
                "generation": float(generation),
                "evaluations": float(evals),
                "best_score": float(best_score),
                "generation_best_score": float(scores[0]),
                "generation_median_score": float(np.median(scores)),
                "sigma": float(sigma),
            }
        )
    return trace, best_x, best_score


def anchor_combos() -> list[tuple[str, Combo]]:
    exact1 = Combo(Triple(110.0, 1.4, 0.15), Triple(103.0, 1.26, 0.23), Triple(100.0, 2.0, 0.25))
    exact2 = Combo(Triple(114.0, 1.37, 0.248), Triple(106.0, 1.26, 0.23), Triple(102.0, 1.75, 0.25))
    quant1 = Combo(
        snap_triple(110.0, 1.4, 0.15),
        snap_triple(103.0, 1.26, 0.23),
        snap_triple(100.0, 2.0, 0.25),
    )
    quant2 = Combo(
        snap_triple(114.0, 1.37, 0.248),
        snap_triple(106.0, 1.26, 0.23),
        snap_triple(102.0, 1.75, 0.25),
    )
    return [
        ("anchor1_exact", exact1),
        ("anchor1_quantized_to_grid", quant1),
        ("anchor2_exact", exact2),
        ("anchor2_quantized_to_grid", quant2),
    ]


def plot_top_s21(rows: pd.DataFrame, evaluator: Evaluator, out_dir: Path, n: int = 20) -> None:
    top = rows.sort_values("rank_score_tuple").head(n)
    fig, axes = plt.subplots(5, 4, figsize=(16, 13), sharex=True, sharey=True, constrained_layout=True)
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        combo = Combo(
            Triple(row["input_W"], row["input_R"], row["input_WlineR"]),
            Triple(row["interstage_W"], row["interstage_R"], row["interstage_WlineR"]),
            Triple(row["output_W"], row["output_R"], row["output_WlineR"]),
        )
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
        freq = pa.f / 1e9
        s21 = clipped_db20(pa.s[:, 1, 0])
        mask = mask_range(freq, 25.0, 95.0)
        ax.plot(freq[mask], s21[mask], linewidth=1.1)
        ax.axhspan(GAIN_MIN_DB, GAIN_MAX_DB, color="green", alpha=0.12)
        ax.axvline(BASE_MIN_GHZ, color="black", linewidth=0.6)
        ax.axvline(BASE_MAX_GHZ, color="black", linewidth=0.6)
        ax.axvline(EXT_MAX_GHZ, color="gray", linewidth=0.6, linestyle="--")
        ax.set_title(f"#{int(row['rank'])}: {row['gain_window_width_30_90_ghz']:.1f}GHz, Zin {row['omn_zin_rms_to_zopt_ohm']:.1f}", fontsize=8)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("GHz")
    for ax in axes[:, 0]:
        ax.set_ylabel("S21 dB")
    fig.savefig(out_dir / "top20_s21_gain_window.png", dpi=180)
    plt.close(fig)


def add_ranking_columns(rows: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy()
    # Lexicographic rank helper: feasible first, extended width high, base violation low,
    # then OMN Zin RMS low.  The scalar objective is still included for CMA diagnostics.
    data["rank_full80_flag"] = data["full_30_80_gain_window_ok"].astype(int)
    data = data.sort_values(
        [
            "rank_full80_flag",
            "gain_window_width_30_90_ghz",
            "gain_window_width_30_80_ghz",
            "gain_violation_rms_30_80_db",
            "omn_zin_rms_to_zopt_ohm",
            "objective_score",
        ],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)
    data.insert(0, "rank", np.arange(1, len(data) + 1))
    data["rank_score_tuple"] = np.arange(len(data))
    return data


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "predicted_transformers"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    driver_path = Path(args.transistor_dir) / "driver_2x12_single_ended_z0_50.s4p"
    final_path = Path(args.transistor_dir) / "final_2x18_single_ended_z0_50.s4p"
    driver = load_transistor_s4p(driver_path, z0=50.0, name="driver_2x12")
    final = load_transistor_s4p(final_path, target_freq_hz=driver.f, z0=50.0, name="final_2x18")
    loadpull = load_loadpull_zopt(args.loadpull_xlsx)
    model = tf_v3.load_l_fit_model(data_source=args.fit_data, round_decimals=args.round_decimals)
    builder = PredictedBuilder(
        model=model,
        output_dir=pred_dir,
        freq_hz=driver.f,
        allow_extrapolation=args.allow_extrapolation,
    )
    evaluator = Evaluator(
        builder=builder,
        driver=driver,
        final=final,
        loadpull_zopt=loadpull.zopt_single,
        loadpull_freq_hz=loadpull.freq_hz,
    )

    def evaluate_x(x: np.ndarray) -> float:
        combo = normalized_to_combo(x)
        return float(evaluator.evaluate(combo, source="cmaes")["objective_score"])

    initial = [
        Combo(snap_triple(110, 1.4, 0.15), snap_triple(103, 1.26, 0.23), snap_triple(100, 2.0, 0.25)),
        Combo(snap_triple(114, 1.37, 0.248), snap_triple(106, 1.26, 0.23), snap_triple(102, 1.75, 0.25)),
        Combo(snap_triple(105, 1.4, 0.23), snap_triple(105, 1.3, 0.23), snap_triple(101, 1.8, 0.25)),
    ]
    rng = np.random.default_rng(args.seed)
    while len(initial) < args.restarts:
        initial.append(normalized_to_combo(rng.random(9)))

    traces: list[dict[str, float]] = []
    evals_per_restart = max(args.popsize, int(np.ceil(args.max_evals / max(1, args.restarts))))
    for idx in range(args.restarts):
        x0 = combo_to_normalized(initial[idx])
        trace, best_x, best_score = cma_es(
            evaluate_x,
            x0=x0,
            sigma0=args.sigma,
            max_evals=evals_per_restart,
            popsize=args.popsize,
            seed=args.seed + 1009 * idx,
            restart_index=idx + 1,
        )
        traces.extend(trace)
        best_combo = normalized_to_combo(best_x)
        evaluator.evaluate(best_combo, source=f"cmaes_restart_{idx + 1}_best")
        print(f"restart {idx + 1}/{args.restarts}: best_score={best_score:.3f}, unique={len(evaluator.combo_cache)}", flush=True)

    for source, combo in anchor_combos():
        evaluator.evaluate(combo, source=source)

    archive = pd.DataFrame(evaluator.combo_cache.values())
    ranked = add_ranking_columns(archive)
    ranked.to_csv(out_dir / "cmaes_ranked_candidates_all.csv", index=False, encoding="utf-8-sig")
    ranked.head(20).to_csv(out_dir / "cmaes_top20_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(traces).to_csv(out_dir / "cmaes_trace.csv", index=False, encoding="utf-8-sig")

    anchor_rows = []
    for source, combo in anchor_combos():
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
        row = ranked.loc[mask].iloc[0].to_dict()
        row["anchor_name"] = source
        anchor_rows.append(row)
    pd.DataFrame(anchor_rows).to_csv(out_dir / "anchor_candidate_ranks.csv", index=False, encoding="utf-8-sig")

    aggregate = {
        "evaluated_unique_candidates": int(len(ranked)),
        "requested_design_grid": {
            "W": "90..120 step 1",
            "R": "0.8..2.0 step 0.1",
            "WlineR": "0.15..0.30 step 0.01",
        },
        "objective": {
            "gain_window_db": [GAIN_MIN_DB, GAIN_MAX_DB],
            "base_band_ghz": [BASE_MIN_GHZ, BASE_MAX_GHZ],
            "extension_band_ghz": [BASE_MIN_GHZ, EXT_MAX_GHZ],
            "max_useful_width_ghz": MAX_USEFUL_WIDTH_GHZ,
            "note": "CMA-ES archive ranking, not exhaustive enumeration of the 6448^3 grid.",
        },
        "best": ranked.iloc[0].to_dict(),
        "anchors": pd.DataFrame(anchor_rows)[
            [
                "anchor_name",
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
                "full_30_80_gain_window_ok",
                "gain_window_width_30_90_ghz",
                "gain_violation_rms_30_80_db",
                "omn_zin_rms_to_zopt_ohm",
            ]
        ].to_dict(orient="records"),
    }
    (out_dir / "cmaes_manifest.json").write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    trace_df = pd.DataFrame(traces)
    for restart, sub in trace_df.groupby("restart"):
        ax.plot(sub["evaluations"], sub["best_score"], label=f"restart {int(restart)}")
    ax.set_xlabel("CMA-ES evaluations per restart")
    ax.set_ylabel("Best objective score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "cmaes_trace.png", dpi=180)
    plt.close(fig)

    plot_top_s21(ranked, evaluator, out_dir, n=min(20, len(ranked)))

    print(f"output_dir={out_dir.resolve()}")
    print(f"unique_candidates={len(ranked)}")
    print("top20_csv", out_dir / "cmaes_top20_candidates.csv")
    print("anchor_ranks_csv", out_dir / "anchor_candidate_ranks.csv")
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
        "full_30_80_gain_window_ok",
        "gain_window_width_30_90_ghz",
        "gain_violation_rms_30_80_db",
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
    parser.add_argument("--max-evals", type=int, default=900)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--popsize", type=int, default=18)
    parser.add_argument("--sigma", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=20260513)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

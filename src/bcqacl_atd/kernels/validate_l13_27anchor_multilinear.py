#!/usr/bin/env python3
"""Validate 27-anchor calibration for the Lb/L13 multilinear law.

This script uses the residual-extracted L13 values from the 2026-05-20
residual topology experiment.  The intended 3x3x3 anchor grid is

    W      = {90, 105, 120}
    R      = {0.8, 1.4, 2.0}
    WlineR = {0.15, 0.225, 0.30}

If an intended anchor is not present in the simulated 80-case grid, it is
replaced by the nearest available simulated value.  Ties are resolved toward
the lower available value.  The fitted coordinates are the actual substituted
coordinates, not the unavailable intended coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(r".\HFSS\For_Paper\ForModelling")
DEFAULT_SOURCE = BASE_DIR / "residual_topology_fit_experiment_20260520" / "residual_extracted_L_by_case.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "l13_27anchor_multilinear_validation_20260521"

TARGET_W = (90.0, 105.0, 120.0)
TARGET_R = (0.8, 1.4, 2.0)
TARGET_Q = (0.15, 0.225, 0.30)

NORM_RANGES = {
    "W": (90.0, 120.0),
    "R": (0.8, 2.0),
    "WlineR": (0.15, 0.30),
}


def parse_case_id(case_id: str) -> tuple[float, float, float]:
    match = re.fullmatch(r"W([0-9.]+)_R([0-9p.]+)_WlineR([0-9p.]+)", str(case_id))
    if not match:
        raise ValueError(f"Cannot parse case_id={case_id!r}")
    W = float(match.group(1))
    R = float(match.group(2).replace("p", "."))
    q = float(match.group(3).replace("p", "."))
    return W, R, q


def nearest_lower_tie(value: float, available: np.ndarray) -> float:
    available = np.asarray(sorted(float(v) for v in available), dtype=float)
    distances = np.abs(available - float(value))
    min_distance = np.min(distances)
    candidates = available[np.isclose(distances, min_distance, rtol=0.0, atol=1e-12)]
    return float(np.min(candidates))


def normalized(values: np.ndarray, key: str) -> np.ndarray:
    lo, hi = NORM_RANGES[key]
    return 2.0 * (np.asarray(values, dtype=float) - lo) / (hi - lo) - 1.0


def feature_matrix(df: pd.DataFrame, *, full: bool) -> np.ndarray:
    u = normalized(df["W"].to_numpy(dtype=float), "W")
    v = normalized(df["R"].to_numpy(dtype=float), "R")
    q = normalized(df["WlineR"].to_numpy(dtype=float), "WlineR")
    if full:
        return np.column_stack([np.ones(len(df)), u, v, q, u * v, u * q, v * q, u * v * q])
    return np.column_stack([np.ones(len(df)), u, v, q])


def fit_log_model(train: pd.DataFrame, *, full: bool) -> np.ndarray:
    X = feature_matrix(train, full=full)
    y_log = np.log(train["L_b_nH"].to_numpy(dtype=float))
    beta, *_ = np.linalg.lstsq(X, y_log, rcond=None)
    return beta


def predict_log_model(df: pd.DataFrame, beta: np.ndarray, *, full: bool) -> np.ndarray:
    return np.exp(feature_matrix(df, full=full) @ np.asarray(beta, dtype=float))


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((yhat - y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan


def nmse_stats(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = float(np.mean((y - np.mean(y)) ** 2))
    if denom <= 0:
        return {"nmse_mean": math.nan, "nmse_median": math.nan}
    point_nmse = (yhat - y) ** 2 / denom
    return {
        "nmse_mean": float(np.mean(point_nmse)),
        "nmse_median": float(np.median(point_nmse)),
    }


def build_dataset(source: Path) -> pd.DataFrame:
    raw = pd.read_csv(source)
    l13 = raw[raw["topology"].eq("L13")].copy()
    if l13.empty:
        raise RuntimeError("No topology == 'L13' rows found.")
    coords = np.array([parse_case_id(case_id) for case_id in l13["case_id"]], dtype=float)
    l13["W"] = coords[:, 0]
    l13["R"] = coords[:, 1]
    l13["WlineR"] = coords[:, 2]
    l13["L_b_nH"] = l13["L13_nH"].astype(float)
    return l13[["case_id", "W", "R", "WlineR", "L_b_nH"]].sort_values(["W", "R", "WlineR"]).reset_index(drop=True)


def select_anchors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_W = df["W"].unique()
    available_R = df["R"].unique()
    available_Q = df["WlineR"].unique()
    rows = []
    replacements = []
    for target_w in TARGET_W:
        actual_w = nearest_lower_tie(target_w, available_W)
        for target_r in TARGET_R:
            actual_r = nearest_lower_tie(target_r, available_R)
            for target_q in TARGET_Q:
                actual_q = nearest_lower_tie(target_q, available_Q)
                match = df[
                    np.isclose(df["W"], actual_w)
                    & np.isclose(df["R"], actual_r)
                    & np.isclose(df["WlineR"], actual_q)
                ]
                if len(match) != 1:
                    raise RuntimeError(
                        f"Expected one case for W={actual_w}, R={actual_r}, WlineR={actual_q}, got {len(match)}."
                    )
                item = match.iloc[0].to_dict()
                item.update(
                    {
                        "target_W": target_w,
                        "target_R": target_r,
                        "target_WlineR": target_q,
                    }
                )
                rows.append(item)
                replacements.append(
                    {
                        "target_W": target_w,
                        "target_R": target_r,
                        "target_WlineR": target_q,
                        "used_W": actual_w,
                        "used_R": actual_r,
                        "used_WlineR": actual_q,
                        "case_id": item["case_id"],
                        "is_replaced": (
                            not np.isclose(target_w, actual_w)
                            or not np.isclose(target_r, actual_r)
                            or not np.isclose(target_q, actual_q)
                        ),
                    }
                )
    anchors = pd.DataFrame(rows).drop_duplicates("case_id").reset_index(drop=True)
    replacement_table = pd.DataFrame(replacements)
    if len(anchors) != 27:
        raise RuntimeError(f"Anchor selection produced {len(anchors)} unique cases, expected 27.")
    return anchors, replacement_table


def formula_text(beta: np.ndarray, *, full: bool) -> str:
    if full:
        names = ["1", "u", "v", "q", "uv", "uq", "vq", "uvq"]
    else:
        names = ["1", "u", "v", "q"]
    parts = [f"{coef:+.10g}{name}" for coef, name in zip(beta, names)]
    return "ln(L_b/nH) = " + " ".join(parts)


def plot_predicted_vs_extracted(results: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.1), dpi=180)
    for role, color, marker in [("train27", "#1f77b4", "o"), ("validation", "#d65f5f", "s")]:
        sub = results[results["role"].eq(role)]
        ax.scatter(
            sub["L_b_nH"],
            sub["pred_full_nH"],
            s=42,
            c=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.6,
            label=f"{role} (n={len(sub)})",
            alpha=0.9,
        )
    lo = float(min(results["L_b_nH"].min(), results["pred_full_nH"].min()))
    hi = float(max(results["L_b_nH"].max(), results["pred_full_nH"].max()))
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.1, label="ideal")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Residual-extracted $L_b$ (nH)")
    ax.set_ylabel("Predicted $L_b$ by 27-anchor fit (nH)")
    ax.set_title("27-anchor log-domain multilinear calibration")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "L_b_27anchor_predicted_vs_extracted.png")
    fig.savefig(out_dir / "L_b_27anchor_predicted_vs_extracted.pdf")
    plt.close(fig)


def write_report(out_dir: Path, metrics: dict, full_beta: np.ndarray, first_beta: np.ndarray) -> None:
    lines = [
        "# 27-Anchor Lb Multilinear Validation",
        "",
        "Normalization:",
        "",
        "- `u = 2*(W - 90)/(120 - 90) - 1`",
        "- `v = 2*(R - 0.8)/(2.0 - 0.8) - 1`",
        "- `q = 2*(WlineR - 0.15)/(0.30 - 0.15) - 1`",
        "",
        "Anchor replacement rule: nearest available simulated grid point; exact ties use the lower available value.",
        "",
        "## Full Multilinear Model",
        "",
        f"`{formula_text(full_beta, full=True)}`",
        "",
        f"- Training R2: {metrics['full_train_R2']:.8f}",
        f"- Validation NMSE mean: {metrics['full_val_nmse_mean']:.8g}",
        f"- Validation NMSE median: {metrics['full_val_nmse_median']:.8g}",
        f"- Validation case count: {metrics['validation_n']}",
        f"- Three-way interaction coefficient c_uvq: {metrics['full_c_uvq']:.10g}",
        "",
        "## First-Order Separable Model",
        "",
        f"`{formula_text(first_beta, full=False)}`",
        "",
        f"- Training R2: {metrics['first_train_R2']:.8f}",
        f"- Validation NMSE mean: {metrics['first_val_nmse_mean']:.8g}",
        f"- Validation NMSE median: {metrics['first_val_nmse_median']:.8g}",
    ]
    (out_dir / "README_l13_27anchor_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 27-anchor Lb/L13 log-domain multilinear law.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(args.source)
    train, replacements = select_anchors(df)
    train_ids = set(train["case_id"])
    val = df[~df["case_id"].isin(train_ids)].copy().reset_index(drop=True)
    train = train.copy().reset_index(drop=True)

    full_beta = fit_log_model(train, full=True)
    first_beta = fit_log_model(train, full=False)

    train["role"] = "train27"
    val["role"] = "validation"
    results = pd.concat([train, val], ignore_index=True)
    results["pred_full_nH"] = predict_log_model(results, full_beta, full=True)
    results["pred_first_order_nH"] = predict_log_model(results, first_beta, full=False)
    results["err_full_nH"] = results["pred_full_nH"] - results["L_b_nH"]
    results["err_first_order_nH"] = results["pred_first_order_nH"] - results["L_b_nH"]

    train_mask = results["role"].eq("train27").to_numpy()
    val_mask = results["role"].eq("validation").to_numpy()
    y_train = results.loc[train_mask, "L_b_nH"].to_numpy(dtype=float)
    y_val = results.loc[val_mask, "L_b_nH"].to_numpy(dtype=float)
    full_train = results.loc[train_mask, "pred_full_nH"].to_numpy(dtype=float)
    first_train = results.loc[train_mask, "pred_first_order_nH"].to_numpy(dtype=float)
    full_val = results.loc[val_mask, "pred_full_nH"].to_numpy(dtype=float)
    first_val = results.loc[val_mask, "pred_first_order_nH"].to_numpy(dtype=float)
    full_nmse = nmse_stats(y_val, full_val)
    first_nmse = nmse_stats(y_val, first_val)

    val_denom = float(np.mean((y_val - np.mean(y_val)) ** 2))
    results["point_nmse_full_validation_denom"] = (results["pred_full_nH"] - results["L_b_nH"]) ** 2 / val_denom
    results["point_nmse_first_order_validation_denom"] = (
        (results["pred_first_order_nH"] - results["L_b_nH"]) ** 2 / val_denom
    )

    metrics = {
        "source": str(args.source),
        "train_n": int(train_mask.sum()),
        "validation_n": int(val_mask.sum()),
        "full_coefficients": {name: float(value) for name, value in zip(["c0", "c_u", "c_v", "c_q", "c_uv", "c_uq", "c_vq", "c_uvq"], full_beta)},
        "first_order_coefficients": {name: float(value) for name, value in zip(["c0", "c_u", "c_v", "c_q"], first_beta)},
        "full_train_R2": r2_score(y_train, full_train),
        "first_train_R2": r2_score(y_train, first_train),
        "full_val_R2": r2_score(y_val, full_val),
        "first_val_R2": r2_score(y_val, first_val),
        "full_val_nmse_mean": full_nmse["nmse_mean"],
        "full_val_nmse_median": full_nmse["nmse_median"],
        "first_val_nmse_mean": first_nmse["nmse_mean"],
        "first_val_nmse_median": first_nmse["nmse_median"],
        "full_c_uvq": float(full_beta[7]),
    }

    replacements.to_csv(out_dir / "L_b_27anchor_replacement_table.csv", index=False)
    results.to_csv(out_dir / "L_b_27anchor_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "L_b_27anchor_metrics.csv", index=False)
    (out_dir / "L_b_27anchor_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_predicted_vs_extracted(results, out_dir)
    write_report(out_dir, metrics, full_beta, first_beta)

    print(f"Output directory: {out_dir}")
    print("27-anchor replacement rule: nearest neighbor; ties use the lower available value.")
    print("Replacements used: W=105 -> 100, WlineR=0.225 -> 0.20; R targets are exact.")
    print("Full log-domain multilinear coefficients:")
    for key, value in metrics["full_coefficients"].items():
        print(f"  {key} = {value:.12g}")
    print(f"Full model train R2 = {metrics['full_train_R2']:.8f}")
    print(f"Full model validation NMSE median = {metrics['full_val_nmse_median']:.8g}")
    print(f"Full model validation NMSE mean = {metrics['full_val_nmse_mean']:.8g}")
    print(f"Validation point count = {metrics['validation_n']}")
    print(f"First-order separable model train R2 = {metrics['first_train_R2']:.8f}")
    print(f"First-order separable model validation NMSE median = {metrics['first_val_nmse_median']:.8g}")
    print(f"First-order separable model validation NMSE mean = {metrics['first_val_nmse_mean']:.8g}")
    print(f"c_uvq = {metrics['full_c_uvq']:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

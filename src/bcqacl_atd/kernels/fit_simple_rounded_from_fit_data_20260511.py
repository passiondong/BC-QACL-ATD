from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path


DEPS = Path(r".\HFSS\For_Paper\ForModelling\_codex_pydeps_lfit")
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_SOURCE = Path(r".\HFSS\For_Paper\ForModelling\fit_data_20260511.csv")
OUTPUT_ROOT = Path(r".\HFSS\For_Paper\ForModelling")
OUTPUT_DIR = OUTPUT_ROOT / ("simple_rounded_fit_from_fit_data_20260511_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
COEFF_DECIMALS = 2


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_SOURCE)
    df = df.sort_values(["R", "W", "WlineR"]).reset_index(drop=True)
    df["L13_fit_ref"] = df["L13"].astype(float)
    # This locked file contains the floating-adjusted target from the previous step.
    # Use it for L56 if present, while keeping the original L56 columns for audit.
    if "L56_fit_adjusted" in df.columns:
        df["L56_fit_ref"] = df["L56_fit_adjusted"].astype(float)
        df["L56_fit_ref_source"] = "L56_fit_adjusted"
    else:
        df["L56_fit_ref"] = df["L56"].astype(float)
        df["L56_fit_ref_source"] = "L56"
    return df


def l13_features(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(df), dtype=float),
            np.log(df["W"].to_numpy(float)),
            np.log(df["R"].to_numpy(float)),
            df["WlineR"].to_numpy(float),
        ]
    )


def l56_features(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(df), dtype=float),
            np.log(df["W"].to_numpy(float)),
            np.sqrt(df["R"].to_numpy(float)),
            df["WlineR"].to_numpy(float),
        ]
    )


FEATURES = {
    "L13": ["1", "log(W)", "log(R)", "WlineR"],
    "L56": ["1", "log(W)", "sqrt(R)", "WlineR"],
}


def feature_matrix(df: pd.DataFrame, target: str) -> np.ndarray:
    return l13_features(df) if target == "L13" else l56_features(df)


def transformed_target(df: pd.DataFrame, target: str) -> np.ndarray:
    if target == "L13":
        return np.log(df["L13_fit_ref"].to_numpy(float))
    if target == "L56":
        return np.sqrt(df["L56_fit_ref"].to_numpy(float))
    raise KeyError(target)


def inverse_target(z: np.ndarray, target: str) -> np.ndarray:
    if target == "L13":
        return np.exp(z)
    if target == "L56":
        return z * z
    raise KeyError(target)


def ref_values(df: pd.DataFrame, target: str) -> np.ndarray:
    return df[f"{target}_fit_ref"].to_numpy(float)


def fit_beta(df: pd.DataFrame, target: str, use_anchor_only: bool = False) -> np.ndarray:
    fit_df = df[df["is_default_anchor"]].copy() if use_anchor_only else df
    beta, *_ = np.linalg.lstsq(feature_matrix(fit_df, target), transformed_target(fit_df, target), rcond=None)
    return beta.astype(float)


def predict_from_beta(df: pd.DataFrame, target: str, beta: np.ndarray) -> np.ndarray:
    return inverse_target(feature_matrix(df, target) @ beta, target)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    abs_err = np.abs(err)
    rel = abs_err / np.maximum(np.abs(y), 1e-15) * 100.0
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(err * err))),
        "max_abs_error": float(np.max(abs_err)),
        "mean_relative_error_pct": float(np.mean(rel)),
        "max_relative_error_pct": float(np.max(rel)),
        "R2": float(1.0 - ss_res / ss_tot),
        "min_prediction": float(np.min(pred)),
        "negative_prediction_count": int(np.sum(pred < 0.0)),
    }


def rounded(beta: np.ndarray) -> np.ndarray:
    return np.round(beta.astype(float), COEFF_DECIMALS)


def formula(target: str, beta: np.ndarray) -> str:
    b = [round(float(v), COEFF_DECIMALS) for v in beta]
    if target == "L13":
        return f"L13 = exp({b[0]:.2f} {b[1]:+.2f}*log(W) {b[2]:+.2f}*log(R) {b[3]:+.2f}*WlineR)"
    return f"L56 = ({b[0]:.2f} {b[1]:+.2f}*log(W) {b[2]:+.2f}*sqrt(R) {b[3]:+.2f}*WlineR)^2"


def mark_default_anchors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    w_min, w_max = float(out["W"].min()), float(out["W"].max())
    r_min, r_max = float(out["R"].min()), float(out["R"].max())
    wr_min, wr_max = float(out["WlineR"].min()), float(out["WlineR"].max())
    out["is_default_anchor"] = (
        out["W"].isin([w_min, w_max])
        & out["R"].isin([r_min, r_max])
        & out["WlineR"].isin([wr_min, wr_max])
    )
    return out


def write_audit(df: pd.DataFrame) -> None:
    rows = [
        {"item": "data_source", "value": str(DATA_SOURCE)},
        {"item": "rows", "value": len(df)},
        {"item": "coefficient_decimal_places", "value": COEFF_DECIMALS},
        {"item": "L13_fit_ref_source", "value": "L13"},
        {"item": "L56_fit_ref_source", "value": df["L56_fit_ref_source"].iloc[0]},
        {"item": "R_values", "value": ",".join(map(str, sorted(df["R"].unique())))},
        {"item": "W_values", "value": ",".join(map(str, sorted(df["W"].unique())))},
        {"item": "WlineR_values", "value": ",".join(map(str, sorted(df["WlineR"].unique())))},
        {"item": "default_anchor_count", "value": int(df["is_default_anchor"].sum())},
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "input_data_audit.csv", index=False)


def fit_all_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    coef_rows = []
    for fit_scope, use_anchor_only in [("all_data", False), ("default_8_anchor_only", True)]:
        for target in ["L13", "L56"]:
            beta_raw = fit_beta(df, target, use_anchor_only=use_anchor_only)
            beta_round = rounded(beta_raw)
            for feature, raw_value, rounded_value in zip(FEATURES[target], beta_raw, beta_round):
                coef_rows.append(
                    {
                        "fit_scope": fit_scope,
                        "target": target,
                        "feature": feature,
                        "raw_full_precision": float(raw_value),
                        "rounded_2dp": float(rounded_value),
                        "unit": "log(nH)" if target == "L13" else "sqrt(pH)",
                    }
                )
            for coeff_kind, beta in [("raw_full_precision", beta_raw), ("rounded_2dp", beta_round)]:
                pred = predict_from_beta(df, target, beta)
                y = ref_values(df, target)
                row = {
                    "fit_scope": fit_scope,
                    "target": target,
                    "unit": "nH" if target == "L13" else "pH",
                    "coeff_kind": coeff_kind,
                    "k": len(beta),
                    "formula": formula(target, beta),
                    "fit_rows": int(df["is_default_anchor"].sum() if use_anchor_only else len(df)),
                }
                row.update(metrics(y, pred))
                summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    coefs = pd.DataFrame(coef_rows)
    summary.to_csv(OUTPUT_DIR / "fit_error_summary.csv", index=False)
    coefs.to_csv(OUTPUT_DIR / "fit_coefficients_rounded_2dp.csv", index=False)
    return summary, coefs


def write_predictions(df: pd.DataFrame) -> None:
    out = df.copy()
    for fit_scope, use_anchor_only in [("all_data", False), ("default_8_anchor_only", True)]:
        for target in ["L13", "L56"]:
            beta = rounded(fit_beta(df, target, use_anchor_only=use_anchor_only))
            pred = predict_from_beta(df, target, beta)
            ref = ref_values(df, target)
            out[f"{target}_{fit_scope}_rounded_pred"] = pred
            out[f"{target}_{fit_scope}_rounded_residual"] = pred - ref
            out[f"{target}_{fit_scope}_rounded_abs_error"] = np.abs(pred - ref)
    out.to_csv(OUTPUT_DIR / "fit_predictions_by_case.csv", index=False)
    out[out["is_default_anchor"]].to_csv(OUTPUT_DIR / "default_8_anchor_points.csv", index=False)


def layer_colors(values) -> dict[float, str]:
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    return {float(v): palette[i % len(palette)] for i, v in enumerate(sorted(values))}


def json_dumps(obj) -> str:
    return json.dumps(obj, allow_nan=False)


def html_page(title: str, note: str, traces: list[dict], layout: dict) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; }}
    #plot {{ width: 100vw; height: 88vh; }}
    .note {{ padding: 10px 14px; font-size: 13px; }}
    code {{ background: #f3f4f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <div class="note">{note}</div>
  <div id="plot"></div>
  <script>
    const traces = {json_dumps(traces)};
    const layout = {json_dumps(layout)};
    Plotly.newPlot("plot", traces, layout, {{responsive: true}});
  </script>
</body>
</html>
"""


def make_grid_predictions(target: str, beta: np.ndarray, W_values, R_value, WlineR_values) -> np.ndarray:
    rows = [(float(W), float(R_value), float(wr)) for wr in WlineR_values for W in W_values]
    grid = pd.DataFrame(rows, columns=["W", "R", "WlineR"])
    return predict_from_beta(grid, target, beta).reshape(len(WlineR_values), len(W_values))


def write_2d_html(df: pd.DataFrame, target: str, beta: np.ndarray, mark_anchors: bool = False) -> None:
    r_values = sorted(df["R"].unique())
    w_values = sorted(df["W"].unique())
    w_colors = layer_colors(w_values)
    n_cols = 2
    n_rows = 2
    traces = []
    wr_grid = np.linspace(float(df["WlineR"].min()), float(df["WlineR"].max()), 160)
    for idx, r in enumerate(r_values):
        xaxis = "x" if idx == 0 else f"x{idx + 1}"
        yaxis = "y" if idx == 0 else f"y{idx + 1}"
        for w in w_values:
            sub = df[(df["R"] == r) & (df["W"] == w)].sort_values("WlineR")
            line = make_grid_predictions(target, beta, [float(w)], float(r), wr_grid).reshape(-1)
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": f"W={w:g}",
                    "x": wr_grid.tolist(),
                    "y": line.tolist(),
                    "xaxis": xaxis,
                    "yaxis": yaxis,
                    "line": {"color": w_colors[float(w)], "width": 2.0},
                    "legendgroup": f"W={w:g}",
                    "showlegend": idx == 0,
                }
            )
            traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "name": f"data W={w:g}",
                    "x": sub["WlineR"].astype(float).tolist(),
                    "y": sub[f"{target}_fit_ref"].astype(float).tolist(),
                    "xaxis": xaxis,
                    "yaxis": yaxis,
                    "marker": {"color": w_colors[float(w)], "size": 7, "line": {"color": "white", "width": 1}},
                    "legendgroup": f"W={w:g}",
                    "showlegend": False,
                }
            )
            if mark_anchors:
                anc = sub[sub["is_default_anchor"]]
                if not anc.empty:
                    traces.append(
                        {
                            "type": "scatter",
                            "mode": "markers",
                            "name": "8 anchors",
                            "x": anc["WlineR"].astype(float).tolist(),
                            "y": anc[f"{target}_fit_ref"].astype(float).tolist(),
                            "xaxis": xaxis,
                            "yaxis": yaxis,
                            "marker": {
                                "color": w_colors[float(w)],
                                "symbol": "star",
                                "size": 15,
                                "line": {"color": "black", "width": 2},
                            },
                            "legendgroup": "anchors",
                            "showlegend": idx == 0 and float(w) == float(w_values[0]),
                        }
                    )
    annotations = []
    layout_axes = {}
    for idx, r in enumerate(r_values):
        suffix = "" if idx == 0 else str(idx + 1)
        layout_axes[f"xaxis{suffix}"] = {"title": "WlineR", "range": [float(df["WlineR"].min()) - 0.006, float(df["WlineR"].max()) + 0.006]}
        layout_axes[f"yaxis{suffix}"] = {"title": f"{target} ({'nH' if target == 'L13' else 'pH'})", "rangemode": "tozero"}
        annotations.append(
            {
                "text": f"R={r:g}",
                "xref": "x domain" if idx == 0 else f"x{idx + 1} domain",
                "yref": "y domain" if idx == 0 else f"y{idx + 1} domain",
                "x": 0.5,
                "y": 1.08,
                "showarrow": False,
                "font": {"size": 14},
            }
        )
    title_suffix = "anchor marked" if mark_anchors else "2D slices"
    layout = {
        "title": {"text": f"{target}: rounded simple fit {title_suffix}", "x": 0.5},
        "grid": {"rows": n_rows, "columns": n_cols, "pattern": "independent"},
        "height": 820,
        "legend": {"orientation": "h", "y": 1.08},
        "annotations": annotations,
        "margin": {"l": 72, "r": 36, "t": 86, "b": 60},
        **layout_axes,
    }
    note = f"<b>{target}</b> formula: <code>{formula(target, beta)}</code>. Data source: <code>{DATA_SOURCE}</code>."
    if mark_anchors:
        note += " Star markers are the default 8 anchor points."
    stem = f"{target}_rounded_simple_fit_2d_slices"
    if mark_anchors:
        stem += "_anchor_marked"
    (OUTPUT_DIR / f"{stem}.html").write_text(html_page(f"{target} {title_suffix}", note, traces, layout), encoding="utf-8")


def write_3d_html(df: pd.DataFrame, target: str, beta: np.ndarray, mark_anchors: bool = False) -> None:
    r_values = sorted(df["R"].unique())
    w_grid = np.linspace(float(df["W"].min()), float(df["W"].max()), 42)
    wr_grid = np.linspace(float(df["WlineR"].min()), float(df["WlineR"].max()), 42)
    W_grid, WR_grid = np.meshgrid(w_grid, wr_grid)
    colorscales = ["Viridis", "Cividis", "Plasma", "Magma"]
    traces = []
    for idx, r in enumerate(r_values):
        rows = pd.DataFrame({"W": W_grid.ravel(), "R": float(r), "WlineR": WR_grid.ravel()})
        z = predict_from_beta(rows, target, beta).reshape(W_grid.shape)
        traces.append(
            {
                "type": "surface",
                "name": f"fit R={r:g}",
                "x": w_grid.tolist(),
                "y": wr_grid.tolist(),
                "z": z.tolist(),
                "opacity": 0.48,
                "colorscale": colorscales[idx % len(colorscales)],
                "showscale": False,
            }
        )
        sub = df[df["R"] == r]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": f"data R={r:g}",
                "x": sub["W"].astype(float).tolist(),
                "y": sub["WlineR"].astype(float).tolist(),
                "z": sub[f"{target}_fit_ref"].astype(float).tolist(),
                "marker": {"size": 4, "color": "black", "symbol": "circle"},
            }
        )
    if mark_anchors:
        anc = df[df["is_default_anchor"]]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": "default 8 anchors",
                "x": anc["W"].astype(float).tolist(),
                "y": anc["WlineR"].astype(float).tolist(),
                "z": anc[f"{target}_fit_ref"].astype(float).tolist(),
                "marker": {"size": 7, "color": "#ffcc00", "symbol": "diamond", "line": {"color": "black", "width": 2}},
            }
        )
    layout = {
        "title": {"text": f"{target}: rounded simple 3D fit by R layer", "x": 0.5},
        "scene": {
            "xaxis": {"title": "W"},
            "yaxis": {"title": "WlineR"},
            "zaxis": {"title": f"{target} ({'nH' if target == 'L13' else 'pH'})"},
            "camera": {"eye": {"x": 1.65, "y": -1.65, "z": 1.0}},
        },
        "legend": {"orientation": "h", "y": -0.03},
        "margin": {"l": 0, "r": 0, "t": 62, "b": 0},
    }
    note = f"<b>{target}</b> formula: <code>{formula(target, beta)}</code>. Data source: <code>{DATA_SOURCE}</code>."
    if mark_anchors:
        note += " Diamond markers are the default 8 anchor points."
    stem = f"{target}_rounded_simple_fit_3d_interactive"
    if mark_anchors:
        stem += "_anchor_marked"
    (OUTPUT_DIR / f"{stem}.html").write_text(html_page(f"{target} 3D", note, traces, layout), encoding="utf-8")


def write_anchor_marked_png(df: pd.DataFrame, target: str, beta: np.ndarray) -> None:
    r_values = sorted(df["R"].unique())
    w_values = sorted(df["W"].unique())
    colors = layer_colors(w_values)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4), sharex=True)
    axes_flat = axes.ravel()
    wr_grid = np.linspace(float(df["WlineR"].min()), float(df["WlineR"].max()), 160)
    for idx, r in enumerate(r_values):
        ax = axes_flat[idx]
        for w in w_values:
            sub = df[(df["R"] == r) & (df["W"] == w)].sort_values("WlineR")
            yfit = make_grid_predictions(target, beta, [float(w)], float(r), wr_grid).reshape(-1)
            ax.plot(wr_grid, yfit, color=colors[float(w)], lw=1.9, label=f"W={w:g}")
            ax.scatter(sub["WlineR"], sub[f"{target}_fit_ref"], color=colors[float(w)], s=28, edgecolor="white", linewidth=0.7)
            anc = sub[sub["is_default_anchor"]]
            if not anc.empty:
                ax.scatter(anc["WlineR"], anc[f"{target}_fit_ref"], color=colors[float(w)], marker="*", s=170, edgecolor="black", linewidth=1.3, zorder=4)
        ax.set_title(f"R={r:g}", fontsize=11)
        ax.set_xlabel("WlineR")
        ax.set_ylabel(f"{target} ({'nH' if target == 'L13' else 'pH'})")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False)
    fig.suptitle(f"{target}: rounded simple fit with default 8 anchors marked", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUTPUT_DIR / f"{target}_rounded_simple_fit_anchor_marked.png", dpi=180)
    plt.close(fig)


def write_prediction_scripts(summary: pd.DataFrame, coefs: pd.DataFrame) -> None:
    all_data_coefs = {}
    anchor_coefs = {}
    for target in ["L13", "L56"]:
        sub = coefs[(coefs["fit_scope"] == "all_data") & (coefs["target"] == target)].sort_values("feature")
        # Preserve feature order explicitly.
        vals = []
        for feature in FEATURES[target]:
            vals.append(float(coefs[(coefs["fit_scope"] == "all_data") & (coefs["target"] == target) & (coefs["feature"] == feature)]["rounded_2dp"].iloc[0]))
        all_data_coefs[target] = vals
        vals = []
        for feature in FEATURES[target]:
            vals.append(float(coefs[(coefs["fit_scope"] == "default_8_anchor_only") & (coefs["target"] == target) & (coefs["feature"] == feature)]["rounded_2dp"].iloc[0]))
        anchor_coefs[target] = vals

    direct_text = f'''from __future__ import annotations

import math


COEFFICIENTS = {json.dumps(all_data_coefs, indent=4)}


def predict_L13_L56(W, R, WlineR):
    W = float(W)
    R = float(R)
    WlineR = float(WlineR)
    b13 = COEFFICIENTS["L13"]
    g13 = b13[0] + b13[1] * math.log(W) + b13[2] * math.log(R) + b13[3] * WlineR
    b56 = COEFFICIENTS["L56"]
    g56 = b56[0] + b56[1] * math.log(W) + b56[2] * math.sqrt(R) + b56[3] * WlineR
    return {{
        "L13_nH": math.exp(g13),
        "L56_pH": g56 * g56,
        "model_name": "simple_rounded_2dp_all_data_fit",
    }}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("W", type=float)
    parser.add_argument("R", type=float)
    parser.add_argument("WlineR", type=float)
    args = parser.parse_args()
    print(predict_L13_L56(args.W, args.R, args.WlineR))
'''
    (OUTPUT_DIR / "predict_L13_L56_simple_rounded_2dp.py").write_text(direct_text, encoding="utf-8")

    anchor_text = f'''from __future__ import annotations

import math
import sys
from pathlib import Path

DEPS = Path(r".\\HFSS\\For_Paper\\ForModelling\\_codex_pydeps_lfit")
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np
import pandas as pd


DEFAULT_DATA_SOURCE = Path(r"{DATA_SOURCE}")
ROUND_DECIMALS = {COEFF_DECIMALS}


def _mark_default_anchors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_default_anchor"] = (
        out["W"].isin([out["W"].min(), out["W"].max()])
        & out["R"].isin([out["R"].min(), out["R"].max()])
        & out["WlineR"].isin([out["WlineR"].min(), out["WlineR"].max()])
    )
    return out


def _x_l13(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), np.log(df["W"]), np.log(df["R"]), df["WlineR"]])


def _x_l56(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), np.log(df["W"]), np.sqrt(df["R"]), df["WlineR"]])


def fit_coefficients_from_anchor_points(anchor_df: pd.DataFrame, round_decimals: int = ROUND_DECIMALS) -> dict:
    required = {{"W", "R", "WlineR", "L13"}}
    missing = required - set(anchor_df.columns)
    if missing:
        raise ValueError(f"Anchor data missing required columns: {{sorted(missing)}}")
    l56_col = "L56_fit_adjusted" if "L56_fit_adjusted" in anchor_df.columns else "L56"
    if l56_col not in anchor_df.columns:
        raise ValueError("Anchor data must contain L56 or L56_fit_adjusted.")
    beta13, *_ = np.linalg.lstsq(_x_l13(anchor_df), np.log(anchor_df["L13"].to_numpy(float)), rcond=None)
    beta56, *_ = np.linalg.lstsq(_x_l56(anchor_df), np.sqrt(anchor_df[l56_col].to_numpy(float)), rcond=None)
    return {{
        "L13": np.round(beta13, round_decimals).astype(float).tolist(),
        "L56": np.round(beta56, round_decimals).astype(float).tolist(),
        "round_decimals": round_decimals,
        "l56_source_column": l56_col,
    }}


def load_default_8_anchor_points(data_source=DEFAULT_DATA_SOURCE) -> pd.DataFrame:
    df = pd.read_csv(data_source)
    df = _mark_default_anchors(df)
    return df[df["is_default_anchor"]].copy().reset_index(drop=True)


def predict_L13_L56(W, R, WlineR, coefficients: dict):
    W = float(W)
    R = float(R)
    WlineR = float(WlineR)
    b13 = coefficients["L13"]
    g13 = b13[0] + b13[1] * math.log(W) + b13[2] * math.log(R) + b13[3] * WlineR
    b56 = coefficients["L56"]
    g56 = b56[0] + b56[1] * math.log(W) + b56[2] * math.sqrt(R) + b56[3] * WlineR
    return {{
        "L13_nH": math.exp(g13),
        "L56_pH": g56 * g56,
        "model_name": "simple_rounded_2dp_from_anchor_points",
    }}


def fit_default_8_anchor_coefficients(data_source=DEFAULT_DATA_SOURCE) -> dict:
    return fit_coefficients_from_anchor_points(load_default_8_anchor_points(data_source))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("W", type=float, nargs="?")
    parser.add_argument("R", type=float, nargs="?")
    parser.add_argument("WlineR", type=float, nargs="?")
    parser.add_argument("--anchor-csv", default=str(DEFAULT_DATA_SOURCE))
    args = parser.parse_args()
    anchors = load_default_8_anchor_points(args.anchor_csv)
    coeffs = fit_coefficients_from_anchor_points(anchors)
    print("coefficients:", coeffs)
    if args.W is not None and args.R is not None and args.WlineR is not None:
        print("prediction:", predict_L13_L56(args.W, args.R, args.WlineR, coeffs))
'''
    (OUTPUT_DIR / "predict_from_anchor_points_simple_rounded_2dp.py").write_text(anchor_text, encoding="utf-8")


def write_report(df: pd.DataFrame, summary: pd.DataFrame, coefs: pd.DataFrame) -> None:
    lines = [
        "# Simple Rounded Fit From fit_data_20260511",
        "",
        f"Data source: `{DATA_SOURCE}`",
        "",
        f"L13 target: `L13`; L56 target: `{df['L56_fit_ref_source'].iloc[0]}`.",
        f"Coefficients are rounded to at most {COEFF_DECIMALS} decimal places and predictions in the main plots use rounded coefficients.",
        "",
        "## Formulas",
        "",
    ]
    for target in ["L13", "L56"]:
        row = summary[(summary["fit_scope"] == "all_data") & (summary["target"] == target) & (summary["coeff_kind"] == "rounded_2dp")].iloc[0]
        lines.append(f"- {target}: `{row['formula']}`")
    lines += [
        "",
        "## Error Summary With Rounded Coefficients",
        "",
        "| scope | target | RMSE | max error | R2 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in summary[summary["coeff_kind"] == "rounded_2dp"].iterrows():
        lines.append(f"| {row['fit_scope']} | {row['target']} | {row['RMSE']:.6g} | {row['max_abs_error']:.6g} | {row['R2']:.6g} |")
    lines += [
        "",
        "Default anchor points are the eight outer corners: `W={90,120}`, `R={1.1,2.0}`, `WlineR={0.15,0.30}`.",
    ]
    (OUTPUT_DIR / "executive_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = mark_default_anchors(load_data())
    write_audit(df)
    summary, coefs = fit_all_models(df)
    write_predictions(df)

    for target in ["L13", "L56"]:
        beta = rounded(fit_beta(df, target, use_anchor_only=False))
        write_2d_html(df, target, beta, mark_anchors=False)
        write_3d_html(df, target, beta, mark_anchors=False)
        write_2d_html(df, target, beta, mark_anchors=True)
        write_3d_html(df, target, beta, mark_anchors=True)
        write_anchor_marked_png(df, target, beta)

    write_prediction_scripts(summary, coefs)
    write_report(df, summary, coefs)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Data source: {DATA_SOURCE}")
    print("Rounded coefficient summary:")
    cols = ["fit_scope", "target", "coeff_kind", "RMSE", "max_abs_error", "R2", "formula"]
    print(summary[summary["coeff_kind"] == "rounded_2dp"][cols].to_string(index=False))


if __name__ == "__main__":
    main()

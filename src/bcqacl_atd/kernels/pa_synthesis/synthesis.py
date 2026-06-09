"""Grid-search synthesis routines."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import skrf as rf

from .config import PaSynthesisConfig, geometry_candidates
from .data_loaders import LoadPullData
from .network_utils import build_full_pa_network, s21_db, transformer_omn_input_impedances
from .objectives import gain_window_loss, omn_impedance_loss
from .predictors import TransformerPredictor
from .report import write_csv, write_json


def _candidate_error_row(w_um: float, r: float, exc: Exception) -> dict[str, Any]:
    return {
        "W_um": float(w_um),
        "R": float(r),
        "loss": float("inf"),
        "status": "error",
        "error": str(exc),
    }


def _uniform_pair_indices(n: int, limit: int | None) -> list[tuple[int, int]]:
    total = n * n
    if limit is None or limit >= total:
        return [(idx // n, idx % n) for idx in range(total)]
    if limit <= 0:
        return []

    flat_indices = sorted({int(round(value)) for value in np.linspace(0, total - 1, int(limit))})
    probe = 0
    while len(flat_indices) < limit and probe < total:
        if probe not in flat_indices:
            flat_indices.append(probe)
        probe += 1
    return [(idx // n, idx % n) for idx in sorted(flat_indices[:limit])]


def run_omn_synthesis(
    cfg: PaSynthesisConfig,
    loadpull: LoadPullData,
    transformer_predictor: TransformerPredictor,
    *,
    max_candidates: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out_dir = Path(cfg.paths.output_dir)
    candidates = geometry_candidates(cfg.grid)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    rows: list[dict[str, Any]] = []
    last_progress = time.monotonic()
    for idx, (w_um, r) in enumerate(candidates, start=1):
        try:
            tf6 = transformer_predictor.build(w_um, r, name=f"omn_candidate_{idx}")
            z_in1, z_in2, z_avg = transformer_omn_input_impedances(
                tf6,
                load_ohm=float(cfg.transformer.z0),
            )
            loss_result = omn_impedance_loss(z_avg, loadpull.zopt_single, cfg.objectives)
            row: dict[str, Any] = {
                "W_um": float(w_um),
                "R": float(r),
                "loss": loss_result.loss,
                "status": "ok",
                **loss_result.details,
            }
            for fghz, zin1, zin2, zavg in zip(loadpull.freq_ghz, z_in1, z_in2, z_avg):
                tag = f"{fghz:g}GHz".replace(".", "p")
                row[f"zin_in1_re_{tag}"] = float(zin1.real)
                row[f"zin_in1_im_{tag}"] = float(zin1.imag)
                row[f"zin_in2_re_{tag}"] = float(zin2.real)
                row[f"zin_in2_im_{tag}"] = float(zin2.imag)
                row[f"zin_avg_re_{tag}"] = float(zavg.real)
                row[f"zin_avg_im_{tag}"] = float(zavg.imag)
            rows.append(row)
        except Exception as exc:
            if not cfg.search.continue_on_candidate_error:
                raise
            rows.append(_candidate_error_row(w_um, r, exc))

        if cfg.search.progress_interval and (
            idx % cfg.search.progress_interval == 0 or time.monotonic() - last_progress > 60.0
        ):
            print(f"[OMN] evaluated {idx}/{len(candidates)} candidates")
            last_progress = time.monotonic()

    ok_rows = [row for row in rows if row.get("status") == "ok" and np.isfinite(row["loss"])]
    if not ok_rows:
        raise RuntimeError("OMN synthesis produced no valid candidate.")
    ok_rows.sort(key=lambda row: row["loss"])
    best = dict(ok_rows[0])
    best["candidate"] = transformer_predictor.describe_candidate(best["W_um"], best["R"])

    write_csv(out_dir / "omn_search_results.csv", rows)
    write_json(out_dir / "omn_best_solution.json", best)
    return best, rows


def run_full_pa_synthesis(
    cfg: PaSynthesisConfig,
    loadpull: LoadPullData,
    transformer_predictor: TransformerPredictor,
    driver_s4p: rf.Network,
    final_s4p: rf.Network,
    omn_best: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out_dir = Path(cfg.paths.output_dir)
    grid = geometry_candidates(cfg.grid)
    pair_indices = _uniform_pair_indices(len(grid), cfg.search.full_pa_max_pairs)
    omn = transformer_predictor.build(float(omn_best["W_um"]), float(omn_best["R"]), name="omn_fixed")

    rows: list[dict[str, Any]] = []
    total = len(pair_indices)
    last_progress = time.monotonic()
    for evaluated, (imn_idx, ismn_idx) in enumerate(pair_indices, start=1):
        w_imn, r_imn = grid[imn_idx]
        w_ismn, r_ismn = grid[ismn_idx]
        try:
            imn = transformer_predictor.build(w_imn, r_imn, name="imn_candidate")
            ismn = transformer_predictor.build(w_ismn, r_ismn, name="ismn_candidate")
            full_pa = build_full_pa_network(
                freq_hz=loadpull.freq_hz,
                driver_s4p=driver_s4p,
                final_s4p=final_s4p,
                imn=imn,
                ismn=ismn,
                omn=omn,
                z0=float(cfg.transformer.z0),
                include_dc_blocks=bool(cfg.transformer.include_dc_blocks),
            )
            gain_db = s21_db(full_pa)
            loss_result = gain_window_loss(gain_db, cfg.objectives)
            row: dict[str, Any] = {
                "W_IMN_um": float(w_imn),
                "R_IMN": float(r_imn),
                "W_ISMN_um": float(w_ismn),
                "R_ISMN": float(r_ismn),
                "W_OMN_um": float(omn_best["W_um"]),
                "R_OMN": float(omn_best["R"]),
                "loss": loss_result.loss,
                "status": "ok",
                **loss_result.details,
            }
            for fghz, gain in zip(loadpull.freq_ghz, gain_db):
                tag = f"{fghz:g}GHz".replace(".", "p")
                row[f"s21_db_{tag}"] = float(gain)
            rows.append(row)
        except Exception as exc:
            if not cfg.search.continue_on_candidate_error:
                raise
            rows.append(
                {
                    "W_IMN_um": float(w_imn),
                    "R_IMN": float(r_imn),
                    "W_ISMN_um": float(w_ismn),
                    "R_ISMN": float(r_ismn),
                    "W_OMN_um": float(omn_best["W_um"]),
                    "R_OMN": float(omn_best["R"]),
                    "loss": float("inf"),
                    "status": "error",
                    "error": str(exc),
                }
            )

        if cfg.search.progress_interval and (
            evaluated % cfg.search.progress_interval == 0 or time.monotonic() - last_progress > 60.0
        ):
            print(f"[FULL] evaluated {evaluated}/{total} IMN/ISMN pairs")
            last_progress = time.monotonic()

    ok_rows = [row for row in rows if row.get("status") == "ok" and np.isfinite(row["loss"])]
    if not ok_rows:
        raise RuntimeError("Full PA synthesis produced no valid candidate.")
    ok_rows.sort(key=lambda row: row["loss"])
    best = dict(ok_rows[0])
    best["OMN"] = {"W_um": float(omn_best["W_um"]), "R": float(omn_best["R"])}
    best["IMN"] = {"W_um": float(best["W_IMN_um"]), "R": float(best["R_IMN"])}
    best["ISMN"] = {"W_um": float(best["W_ISMN_um"]), "R": float(best["R_ISMN"])}

    write_csv(out_dir / "full_pa_search_results.csv", rows)
    write_json(out_dir / "final_best_solution.json", best)
    return best, rows

"""Configurable objective functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ObjectiveConfig


@dataclass
class LossResult:
    loss: float
    details: dict[str, float]


def _weights(npoints: int, cfg: ObjectiveConfig) -> np.ndarray:
    if cfg.frequency_weights is None:
        return np.ones(npoints, dtype=float)
    weights = np.asarray(cfg.frequency_weights, dtype=float)
    if len(weights) != npoints:
        raise ValueError(
            f"frequency_weights has {len(weights)} points, but the response has {npoints}."
        )
    if np.any(weights < 0):
        raise ValueError("frequency_weights must be non-negative.")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("At least one frequency weight must be positive.")
    return weights * (npoints / total)


def omn_impedance_loss(
    zin_single: np.ndarray,
    zopt_single: np.ndarray,
    cfg: ObjectiveConfig,
) -> LossResult:
    err = np.asarray(zin_single, dtype=complex) - np.asarray(zopt_single, dtype=complex)
    norm = max(float(cfg.omn_z_norm_ohm), 1e-12)

    if cfg.omn_error_mode == "real_imag":
        err_norm_sq = (err.real / norm) ** 2 + (err.imag / norm) ** 2
    elif cfg.omn_error_mode == "complex":
        err_norm_sq = np.abs(err / norm) ** 2
    else:
        raise ValueError(f"Unsupported OMN error mode: {cfg.omn_error_mode}")

    weights = _weights(len(err_norm_sq), cfg)
    abs_err = np.abs(err)
    loss = float(np.mean(weights * err_norm_sq))
    return LossResult(
        loss=loss,
        details={
            "mean_abs_z_error_ohm": float(np.mean(abs_err)),
            "max_abs_z_error_ohm": float(np.max(abs_err)),
            "mean_real_error_ohm": float(np.mean(err.real)),
            "mean_imag_error_ohm": float(np.mean(err.imag)),
        },
    )


def gain_window_loss(gain_db: np.ndarray, cfg: ObjectiveConfig) -> LossResult:
    gain = np.asarray(gain_db, dtype=float)
    low_violation = np.maximum(cfg.gain_low_db - gain, 0.0)
    high_violation = np.maximum(gain - cfg.gain_high_db, 0.0)
    violation = low_violation + high_violation

    norm = max(float(cfg.gain_violation_norm_db), 1e-12)
    weights = _weights(len(gain), cfg)
    base_loss = float(np.mean(weights * (violation / norm) ** 2))
    violation_fraction = float(np.mean(violation > 0.0))
    extra_fraction = max(violation_fraction - cfg.allowed_gain_violation_fraction, 0.0)
    loss = base_loss + float(cfg.gain_violation_count_weight) * extra_fraction**2

    return LossResult(
        loss=loss,
        details={
            "gain_min_db": float(np.min(gain)),
            "gain_max_db": float(np.max(gain)),
            "gain_mean_db": float(np.mean(gain)),
            "gain_violation_fraction": violation_fraction,
            "gain_max_violation_db": float(np.max(violation)),
        },
    )

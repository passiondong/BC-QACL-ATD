"""Log-domain trilinear bridge-inductance (L_b) law.

This is a clean, dependency-light port of the 27-anchor calibration used in the
paper (Section II, eq. for ln(L_b/L_ref)).  Given the three normalized geometry
coordinates

    u = 2*(W_TF      - W_lo)   / (W_hi   - W_lo)   - 1
    v = 2*(alpha_LW  - R_lo)   / (R_hi   - R_lo)   - 1
    q = 2*(alpha_wcW - Q_lo)   / (Q_hi   - Q_lo)   - 1

the bridge inductance follows the 8-coefficient law

    ln(L_b / L_ref) = a0 + a_u u + a_v v + a_q q
                         + a_uv uv + a_uq uq + a_vq vq + a_uvq uvq.

The coefficients are fitted once per technology/design-box from the 27 anchor
geometries (a 3x3x3 min/center/max grid).  A first-order (4-coefficient)
variant is also supported for diagnostics.

This module is pure NumPy/pandas and is independent of the EM solver, so it can
be unit-tested and reused on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

try:  # pandas is optional for the array API; required only for the DataFrame helpers
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# Paper design box (default normalization ranges).  Override per technology.
DEFAULT_W_RANGE = (90.0, 120.0)
DEFAULT_R_RANGE = (0.8, 2.0)
DEFAULT_Q_RANGE = (0.15, 0.30)

# 3x3x3 anchor target levels (min, center, max) within the ranges above.
DEFAULT_W_LEVELS = (90.0, 105.0, 120.0)
DEFAULT_R_LEVELS = (0.8, 1.4, 2.0)
DEFAULT_Q_LEVELS = (0.15, 0.225, 0.30)

L_REF_NH = 1.0  # reference inductance, ln(L_b / L_ref); L_ref = 1 nH in the paper

FULL_TERMS = ("1", "u", "v", "q", "uv", "uq", "vq", "uvq")
FIRST_ORDER_TERMS = ("1", "u", "v", "q")


@dataclass(frozen=True)
class NormRange:
    """Normalization range [lo, hi] mapped to [-1, +1]."""

    lo: float
    hi: float

    def normalize(self, value):
        arr = np.asarray(value, dtype=float)
        return 2.0 * (arr - self.lo) / (self.hi - self.lo) - 1.0


def _design_box(w_range, r_range, q_range) -> tuple[NormRange, NormRange, NormRange]:
    return (
        NormRange(*w_range),
        NormRange(*r_range),
        NormRange(*q_range),
    )


def _feature_matrix(u, v, q, *, full: bool) -> np.ndarray:
    u = np.atleast_1d(np.asarray(u, dtype=float))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    q = np.atleast_1d(np.asarray(q, dtype=float))
    ones = np.ones_like(u)
    if full:
        return np.column_stack([ones, u, v, q, u * v, u * q, v * q, u * v * q])
    return np.column_stack([ones, u, v, q])


@dataclass
class LogTrilinearLbLaw:
    """Fitted (or supplied) log-domain trilinear L_b law.

    Parameters
    ----------
    beta:
        Coefficient vector.  Length 8 for the full model, 4 for first-order.
    full:
        Whether ``beta`` is the full 8-coefficient model.
    w_range, r_range, q_range:
        Normalization ranges for W_TF, alpha_L/W, alpha_wc/W.
    l_ref_nH:
        Reference inductance used in ln(L_b / L_ref).
    """

    beta: np.ndarray
    full: bool = True
    w_range: tuple[float, float] = DEFAULT_W_RANGE
    r_range: tuple[float, float] = DEFAULT_R_RANGE
    q_range: tuple[float, float] = DEFAULT_Q_RANGE
    l_ref_nH: float = L_REF_NH

    def __post_init__(self) -> None:
        self.beta = np.asarray(self.beta, dtype=float).ravel()
        expected = 8 if self.full else 4
        if self.beta.size != expected:
            raise ValueError(f"beta must have {expected} entries for full={self.full}, got {self.beta.size}")

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        W: Sequence[float],
        R: Sequence[float],
        WlineR: Sequence[float],
        Lb_nH: Sequence[float],
        *,
        full: bool = True,
        w_range: tuple[float, float] = DEFAULT_W_RANGE,
        r_range: tuple[float, float] = DEFAULT_R_RANGE,
        q_range: tuple[float, float] = DEFAULT_Q_RANGE,
        l_ref_nH: float = L_REF_NH,
    ) -> "LogTrilinearLbLaw":
        """Least-squares fit of ln(L_b / L_ref) on the anchor geometries."""
        wr, rr, qr = _design_box(w_range, r_range, q_range)
        u = wr.normalize(W)
        v = rr.normalize(R)
        q = qr.normalize(WlineR)
        X = _feature_matrix(u, v, q, full=full)
        y = np.log(np.asarray(Lb_nH, dtype=float) / float(l_ref_nH))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return cls(beta=beta, full=full, w_range=w_range, r_range=r_range, q_range=q_range, l_ref_nH=l_ref_nH)

    @classmethod
    def fit_dataframe(cls, df, *, full: bool = True, **kw) -> "LogTrilinearLbLaw":
        """Fit from a DataFrame with columns W, R, WlineR, L_b_nH."""
        if pd is None:  # pragma: no cover
            raise RuntimeError("pandas is required for fit_dataframe")
        return cls.fit(
            df["W"].to_numpy(float),
            df["R"].to_numpy(float),
            df["WlineR"].to_numpy(float),
            df["L_b_nH"].to_numpy(float),
            full=full,
            **kw,
        )

    # -------------------------------------------------------------- predict
    def predict(self, W, R, WlineR) -> np.ndarray:
        """Predict L_b in nH at one or many geometries."""
        wr, rr, qr = _design_box(self.w_range, self.r_range, self.q_range)
        X = _feature_matrix(wr.normalize(W), rr.normalize(R), qr.normalize(WlineR), full=self.full)
        out = self.l_ref_nH * np.exp(X @ self.beta)
        return out if out.size > 1 else float(out[0])

    __call__ = predict

    # ------------------------------------------------------------- metrics
    def r2(self, W, R, WlineR, Lb_nH) -> float:
        y = np.asarray(Lb_nH, dtype=float)
        yhat = np.atleast_1d(np.asarray(self.predict(W, R, WlineR), dtype=float))
        ss_res = float(np.sum((yhat - y) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # ----------------------------------------------------------- reporting
    @property
    def terms(self) -> tuple[str, ...]:
        return FULL_TERMS if self.full else FIRST_ORDER_TERMS

    def formula_text(self) -> str:
        parts = [f"{c:+.6g}*{name}" for c, name in zip(self.beta, self.terms)]
        return "ln(L_b/L_ref) = " + " ".join(parts)

    # ------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return {
            "beta": self.beta.tolist(),
            "terms": list(self.terms),
            "full": self.full,
            "w_range": list(self.w_range),
            "r_range": list(self.r_range),
            "q_range": list(self.q_range),
            "l_ref_nH": self.l_ref_nH,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "LogTrilinearLbLaw":
        return cls(
            beta=np.asarray(payload["beta"], dtype=float),
            full=bool(payload.get("full", len(payload["beta"]) == 8)),
            w_range=tuple(payload.get("w_range", DEFAULT_W_RANGE)),
            r_range=tuple(payload.get("r_range", DEFAULT_R_RANGE)),
            q_range=tuple(payload.get("q_range", DEFAULT_Q_RANGE)),
            l_ref_nH=float(payload.get("l_ref_nH", L_REF_NH)),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "LogTrilinearLbLaw":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _nearest_lower_tie(value: float, available: Iterable[float]) -> float:
    av = np.asarray(sorted(float(v) for v in available), dtype=float)
    dist = np.abs(av - float(value))
    dmin = float(np.min(dist))
    return float(np.min(av[np.isclose(dist, dmin, rtol=0.0, atol=1e-12)]))


def select_anchor_grid(
    available_W: Sequence[float],
    available_R: Sequence[float],
    available_Q: Sequence[float],
    *,
    w_levels: Sequence[float] = DEFAULT_W_LEVELS,
    r_levels: Sequence[float] = DEFAULT_R_LEVELS,
    q_levels: Sequence[float] = DEFAULT_Q_LEVELS,
) -> list[tuple[float, float, float]]:
    """Choose the 27 (W, R, WlineR) anchor geometries.

    For each of the 3x3x3 target levels (min / center / max of each variable),
    snap to the nearest available simulated value; exact ties resolve to the
    lower value -- matching the paper's anchor-replacement rule.  Returns the
    27 *substituted* coordinates a designer must EM-simulate (full transformer
    + half transformer) to calibrate the model.
    """
    anchors: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for tw in w_levels:
        aw = _nearest_lower_tie(tw, available_W)
        for tr in r_levels:
            ar = _nearest_lower_tie(tr, available_R)
            for tq in q_levels:
                aq = _nearest_lower_tie(tq, available_Q)
                key = (round(aw, 9), round(ar, 9), round(aq, 9))
                if key not in seen:
                    seen.add(key)
                    anchors.append((aw, ar, aq))
    return anchors

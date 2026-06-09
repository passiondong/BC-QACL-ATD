"""Fast, dependency-light unit tests for the clean core (no EM kernels)."""

from __future__ import annotations

import numpy as np

from bcqacl_atd.config import Config
from bcqacl_atd.lb_law import LogTrilinearLbLaw, select_anchor_grid


PAPER_BETA = [-0.741, 0.057, 0.382, -0.519, 0.155, -0.022, -0.026, -0.017]


def test_lb_law_recovers_coefficients():
    rng = np.random.default_rng(0)
    W = rng.uniform(90, 120, 80)
    R = rng.uniform(0.8, 2.0, 80)
    Q = rng.uniform(0.15, 0.30, 80)
    u = 2 * (W - 90) / 30 - 1
    v = 2 * (R - 0.8) / 1.2 - 1
    q = 2 * (Q - 0.15) / 0.15 - 1
    ln = (PAPER_BETA[0] + PAPER_BETA[1] * u + PAPER_BETA[2] * v + PAPER_BETA[3] * q
          + PAPER_BETA[4] * u * v + PAPER_BETA[5] * u * q + PAPER_BETA[6] * v * q
          + PAPER_BETA[7] * u * v * q)
    Lb = np.exp(ln)
    law = LogTrilinearLbLaw.fit(W, R, Q, Lb, full=True)
    assert np.allclose(law.beta, PAPER_BETA, atol=1e-9)
    assert law.r2(W, R, Q, Lb) > 1 - 1e-9


def test_lb_law_serialization_roundtrip():
    law = LogTrilinearLbLaw(beta=PAPER_BETA, full=True)
    law2 = LogTrilinearLbLaw.from_dict(law.to_dict())
    assert np.allclose(law.beta, law2.beta)
    assert abs(float(law.predict(111.5, 1.38, 0.25)) - float(law2.predict(111.5, 1.38, 0.25))) < 1e-12


def test_anchor_grid_has_27_unique():
    anchors = select_anchor_grid(
        np.arange(90, 121, 5),
        np.round(np.arange(0.8, 2.01, 0.2), 3),
        np.round(np.arange(0.15, 0.301, 0.025), 3),
    )
    assert len(anchors) == 27
    assert len(set(anchors)) == 27


def test_config_yaml_roundtrip(tmp_path):
    cfg = Config()
    p = tmp_path / "cfg.yaml"
    cfg.save(p)
    cfg2 = Config.load(p)
    assert cfg2.target.band_lo_ghz == cfg.target.band_lo_ghz
    assert cfg2.design_space.w_tf_um.step == cfg.design_space.w_tf_um.step
    assert cfg2.optimizer.lb_model == cfg.optimizer.lb_model

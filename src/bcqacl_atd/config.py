"""Configuration schema for the BC-QACL-ATD flow.

The schema mirrors the six design-flow steps so that a YAML/JSON config (or the
Streamlit wizard) maps one-to-one onto the paper:

    Step 1  TechnologyConfig + TargetConfig + TransistorConfig
    Step 2  DesignSpaceConfig + AnchorConfig + LbLawConfig
    Step 3-5 OptimizerConfig (CMA-ES budget + objective weights)
    Step 6  (EM verification is performed by the user; see docs)

Everything that used to be a hard-coded NUS/Windows path now lives in
PathsConfig / *Config file fields, so the tool is portable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


# --------------------------------------------------------------------------- #
# Step 1: process / technology stack
# --------------------------------------------------------------------------- #
@dataclass
class TechnologyConfig:
    """Process/metal-stack parameters for the SG-CL quasi-TEM solver.

    Defaults reproduce the paper's 130-nm SiGe stack."""

    name: str = "GF_130nm_SiGe"
    eps_r: dict[str, float] = field(default_factory=lambda: {"Air": 1.0, "SiO2": 4.1, "Si": 11.9})
    tan_delta: dict[str, float] = field(default_factory=lambda: {"Air": 0.0, "SiO2": 0.01, "Si": 0.0})
    sigma_S_per_m: dict[str, float] = field(
        default_factory=lambda: {"Metal1": 3.57e8, "Metal2": 5.56e8, "MetalGND": 4.916e8}
    )
    metal_thickness_um: dict[str, float] = field(default_factory=lambda: {"Metal1": 4.0, "Metal2": 3.0})
    use_loss: bool = True
    z0_ohm: float = 50.0
    # SG-CL surrogate calibration constants (process/layout-family; frozen after calibration)
    w_sgcl_over_wc: float = 0.75
    w_slot_over_wc: float = 1.8
    # coupling-length layout constant l_eff = L_TF + W_TF - w_c - C_l (um); C_l from port/opening dims
    lport_um: float = 5.0
    wport_um: float = 10.0
    width_open_um: float = 24.0


# --------------------------------------------------------------------------- #
# Step 1: PA-level performance targets
# --------------------------------------------------------------------------- #
@dataclass
class TargetConfig:
    band_lo_ghz: float = 30.0
    band_hi_ghz: float = 80.0
    gain_lo_db: float = 16.0
    gain_hi_db: float = 20.0
    # peak-search band (allowed peak location), usually a bit wider than the target band
    peak_lo_ghz: float = 25.0
    peak_hi_ghz: float = 90.0
    # total matching length budget: sum_n W_TF,n * alpha_L/W,n  (um) -- floor-plan constraint
    total_length_budget_um: float = 480.0
    # NOTE on Psat: this is a small-signal model, so the Psat target is met indirectly
    # by tracking the load-pull-optimal Z_OPT(f) with the OMN (see TransistorConfig).


# --------------------------------------------------------------------------- #
# Step 1: biased transistor S-parameters + load-pull Z_OPT(f)
# --------------------------------------------------------------------------- #
@dataclass
class TransistorConfig:
    """User-supplied, pre-biased transistor blocks for the inner-loop cascade.

    The user must export single-ended 4-port S-parameters of each stage at a
    bias where the device is unconditionally stable above at least half the
    lowest design-band frequency, so the cascade stays stable in band.

    Z_OPT(f) is the load-pull-optimal load of *one* power-device transistor
    (single-ended).  The OMN objective drives the ideal-balun OMN differential
    input impedance, referred to one side as Z_diff/2, toward this Z_OPT_single.
    """

    driver_s4p: str = "data/transistor/driver_2x12_single_ended_z0_50.s4p"
    power_s4p: str = "data/transistor/power_2x18_single_ended_z0_50.s4p"
    # load-pull table (Excel) with columns freq, Zopt_single_re, Zopt_single_im
    loadpull_xlsx: str = "data/loadpull/loadpull_marker_stats_30_80GHz.xlsx"
    loadpull_sheet: str | None = None


# --------------------------------------------------------------------------- #
# Step 2: design space (subset of the model's first-order validity box)
# --------------------------------------------------------------------------- #
@dataclass
class Range1D:
    lo: float
    hi: float
    step: float


@dataclass
class DesignSpaceConfig:
    """Per-matching-network geometry box; a subset of the validated model box
    W in [90,120] um, alpha_L/W in [0.8,2.0], alpha_wc/W in [0.15,0.30]."""

    n_matching_networks: int = 3  # IMN, ISMN, OMN for a two-stage PA
    w_tf_um: Range1D = field(default_factory=lambda: Range1D(90.0, 120.0, 0.5))
    alpha_lw: Range1D = field(default_factory=lambda: Range1D(0.8, 2.0, 0.01))
    alpha_wcw: Range1D = field(default_factory=lambda: Range1D(0.15, 0.30, 0.005))


# --------------------------------------------------------------------------- #
# Step 2: anchors + L_b law calibration
# --------------------------------------------------------------------------- #
@dataclass
class AnchorConfig:
    """27 EM anchor geometries (3x3x3 min/center/max).  For each anchor the user
    supplies a full-transformer six-port Touchstone and a half-transformer
    four-port Touchstone, used to extract L_b and calibrate the law."""

    anchor_dir: str = "data/em_anchors"
    full_tf_glob: str = "full_*.s6p"
    half_tf_glob: str = "half_*.s4p"
    # explicit override of the fitted L_b law coefficients (skip fitting if given)
    lb_law_json: str | None = None
    fit_full_model: bool = True  # 8-coefficient (True) vs first-order 4-coefficient


# --------------------------------------------------------------------------- #
# Steps 3-5: CMA-ES inner-loop search + objective
# --------------------------------------------------------------------------- #
@dataclass
class OptimizerConfig:
    """CMA-ES budget and objective weights.

    Budget meaning (shown to the user in the wizard):
      * popsize    -- candidates evaluated per generation (lambda)
      * generations-- generations per restart
      * restarts   -- independent CMA-ES restarts (escape local minima)
      * total evaluations ~= popsize * generations * restarts
      * polish_*   -- local coordinate refinement on the best feasible designs
    Larger budget -> better coverage of the ~10^16 combination space, longer run.
    """

    popsize: int = 20
    generations: int = 10
    restarts: int = 6
    sigma0: float = 0.26
    seed: int = 20260608
    # L_b (L12/L34) bridge-inductance source: "anchors8" (paper default) or
    # "full80-log-trilinear"
    lb_model: str = "anchors8"
    allow_extrapolation: bool = False
    # objective weights (unit weights by default; the final design is taken from
    # the (G, B, Z) Pareto front, so weights only steer exploration)
    w_gain: float = 1.0
    w_bandwidth: float = 1.0
    w_impedance: float = 1.0
    # local polish
    polish_top_k: int = 30
    polish_rounds: int = 2
    polish_radius: int = 2


@dataclass
class PathsConfig:
    output_dir: str = "outputs/run"
    cache_dir: str = "cache"


@dataclass
class Config:
    """Top-level BC-QACL-ATD configuration."""

    technology: TechnologyConfig = field(default_factory=TechnologyConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    transistor: TransistorConfig = field(default_factory=TransistorConfig)
    design_space: DesignSpaceConfig = field(default_factory=DesignSpaceConfig)
    anchors: AnchorConfig = field(default_factory=AnchorConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    # frequency grid for the inner loop (GHz)
    freq_start_ghz: float = 1.0
    freq_stop_ghz: float = 110.0
    freq_step_ghz: float = 1.0

    # ---------------------------------------------------------------- I/O
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = self.to_dict()
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:  # pragma: no cover
                raise RuntimeError("PyYAML required to write YAML config")
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        else:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:  # pragma: no cover
                raise RuntimeError("PyYAML required to read YAML config")
            payload = yaml.safe_load(text) or {}
        else:
            payload = json.loads(text)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Config":
        def build(dc_type, value):
            if value is None:
                return dc_type()
            kwargs = {}
            for f in dc_type.__dataclass_fields__.values():  # type: ignore[attr-defined]
                if f.name not in value:
                    continue
                v = value[f.name]
                if f.type in (Range1D, "Range1D") and isinstance(v, Mapping):
                    kwargs[f.name] = Range1D(**v)
                else:
                    kwargs[f.name] = v
            return dc_type(**kwargs)

        # Back-compat: the optimizer's L_b source was formerly named "l13_model".
        opt_payload = payload.get("optimizer")
        if isinstance(opt_payload, Mapping) and "lb_model" not in opt_payload and "l13_model" in opt_payload:
            opt_payload = {**opt_payload, "lb_model": opt_payload["l13_model"]}

        return cls(
            technology=build(TechnologyConfig, payload.get("technology")),
            target=build(TargetConfig, payload.get("target")),
            transistor=build(TransistorConfig, payload.get("transistor")),
            design_space=build(DesignSpaceConfig, payload.get("design_space")),
            anchors=build(AnchorConfig, payload.get("anchors")),
            optimizer=build(OptimizerConfig, opt_payload),
            paths=build(PathsConfig, payload.get("paths")),
            freq_start_ghz=float(payload.get("freq_start_ghz", 1.0)),
            freq_stop_ghz=float(payload.get("freq_stop_ghz", 110.0)),
            freq_step_ghz=float(payload.get("freq_step_ghz", 1.0)),
        )

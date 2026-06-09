"""Configuration objects for the PA synthesis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import copy
import json
from pathlib import Path
from typing import Any


@dataclass
class PathConfig:
    driver_transistor_s4p: str = r".\Data\Z_paras_Transistor\driver_2x12_single_ended_z0_50.s4p"
    final_transistor_s4p: str = r".\Data\Z_paras_Transistor\final_2x18_single_ended_z0_50.s4p"
    loadpull_excel: str = r".\Data\Loadpull_Data_Zopt_Freq\loadpull_marker_stats_30_80GHz.xlsx"
    inductor_corners: str = ""
    output_dir: str = r"outputs\pa_synthesis_pipeline"
    cache_dir: str = r"outputs\pa_synthesis_pipeline\cache"


@dataclass
class FrequencyConfig:
    start_ghz: float = 30.0
    stop_ghz: float = 80.0
    use_loadpull_points: bool = True


@dataclass
class GeometryGridConfig:
    w_min_um: float = 90.0
    w_max_um: float = 120.0
    w_step_um: float = 2.0
    r_min: float = 1.3
    r_max: float = 1.7
    r_step: float = 0.05


@dataclass
class SgDvclConfig:
    generate_missing_s4p: bool = True
    reuse_cached_s4p: bool = True
    subdir: str = "sg_dvcl_s4p"
    m_modes: int | None = None


@dataclass
class TransformerConfig:
    z0: float = 50.0
    write_s6p_cache: bool = False
    subdir: str = "tf6_networks"
    port5_termination: str = "ground"
    port6_termination: str = "open"
    include_dc_blocks: bool = False
    substrates: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "MA": {
                "h_um": 16.72,
                "ep_r": 4.1,
                "mu_r": 1.0,
                "sigma_S_per_m": 3.57e8,
                "t_um": 4.0,
                "tand": 0.01,
            },
            "E1": {
                "h_um": 9.62,
                "ep_r": 4.1,
                "mu_r": 1.0,
                "sigma_S_per_m": 5.56e8,
                "t_um": 3.0,
                "tand": 0.01,
            },
        }
    )


@dataclass
class ObjectiveConfig:
    omn_impedance_mode: str = "single_ended_ports_avg"
    omn_error_mode: str = "complex"
    omn_z_norm_ohm: float = 50.0
    frequency_weights: list[float] | None = None
    gain_low_db: float = 16.0
    gain_high_db: float = 19.0
    gain_violation_norm_db: float = 1.0
    allowed_gain_violation_fraction: float = 0.1
    gain_violation_count_weight: float = 10.0


@dataclass
class SearchConfig:
    full_pa_top_omn_candidates: int = 1
    full_pa_max_pairs: int | None = None
    continue_on_candidate_error: bool = True
    progress_interval: int = 25


@dataclass
class PaSynthesisConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    grid: GeometryGridConfig = field(default_factory=GeometryGridConfig)
    sg_dvcl: SgDvclConfig = field(default_factory=SgDvclConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    objectives: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    search: SearchConfig = field(default_factory=SearchConfig)


def inclusive_float_range(start: float, stop: float, step: float, ndigits: int = 10) -> list[float]:
    """Return an inclusive float range with stable rounding."""
    if step <= 0:
        raise ValueError("step must be positive.")
    values: list[float] = []
    idx = 0
    value = start
    tol = abs(step) * 1e-6
    while value <= stop + tol:
        values.append(round(value, ndigits))
        idx += 1
        value = start + idx * step
    return values


def geometry_candidates(grid: GeometryGridConfig) -> list[tuple[float, float]]:
    return [
        (w, r)
        for w in inclusive_float_range(grid.w_min_um, grid.w_max_um, grid.w_step_um)
        for r in inclusive_float_range(grid.r_min, grid.r_max, grid.r_step)
    ]


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _dataclass_from_dict(cls: type, payload: dict[str, Any]):
    kwargs = {}
    for field_info in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        value = payload.get(field_info.name)
        default_value = getattr(cls(), field_info.name)
        if is_dataclass(default_value) and isinstance(value, dict):
            kwargs[field_info.name] = _dataclass_from_dict(type(default_value), value)
        else:
            kwargs[field_info.name] = value
    return cls(**kwargs)


def config_to_dict(cfg: PaSynthesisConfig) -> dict[str, Any]:
    return asdict(cfg)


def config_from_dict(payload: dict[str, Any]) -> PaSynthesisConfig:
    merged = _deep_update(asdict(PaSynthesisConfig()), payload)
    return _dataclass_from_dict(PaSynthesisConfig, merged)


def load_config(path: str | Path | None = None) -> PaSynthesisConfig:
    """Load JSON/YAML config, falling back to defaults when YAML support is absent."""
    if path is None:
        return PaSynthesisConfig()

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    suffix = cfg_path.suffix.lower()
    if suffix == ".json":
        with cfg_path.open("r", encoding="utf-8") as handle:
            return config_from_dict(json.load(handle))

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception:
            print(
                f"[warn] PyYAML is not installed; ignoring {cfg_path} and using built-in defaults. "
                "Use JSON config or install PyYAML to load YAML."
            )
            return PaSynthesisConfig()
        with cfg_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return config_from_dict(payload)

    raise ValueError(f"Unsupported config suffix: {cfg_path.suffix}")


def save_resolved_config(cfg: PaSynthesisConfig, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(config_to_dict(cfg), handle, indent=2)

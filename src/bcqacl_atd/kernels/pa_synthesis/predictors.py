"""Wrappers around the existing inductor and transformer predictors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import skrf as rf

from .config import PaSynthesisConfig
from .network_utils import align_network


def _safe_tag(value: float, digits: int = 4) -> str:
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _uniform_ghz_grid(freq_hz: np.ndarray) -> tuple[float, float, float]:
    freq_ghz = np.asarray(freq_hz, dtype=float) / 1e9
    if len(freq_ghz) < 2:
        return float(freq_ghz[0]), float(freq_ghz[0]), 1.0
    steps = np.diff(freq_ghz)
    if not np.allclose(steps, steps[0]):
        raise ValueError("TF_Cal_S6P_Predicting.py currently requires a uniform frequency grid.")
    return float(freq_ghz[0]), float(freq_ghz[-1]), float(steps[0])


@dataclass
class InductorPredictor:
    corners_path: Path | None = None

    def __post_init__(self) -> None:
        from bilinear_L_predictor import DEFAULT_L56DELTA_PH, load_corners

        self.corners = None
        self.l56delta_pH = DEFAULT_L56DELTA_PH
        if self.corners_path and self.corners_path.exists():
            self.corners, self.l56delta_pH = load_corners(self.corners_path)

    def predict(self, w_um: float, r: float) -> dict[str, float]:
        from bilinear_L_predictor import predict_l_params

        return predict_l_params(w_um, r, self.corners, self.l56delta_pH)


class SgDvclProvider:
    """Generate or reuse Cal_0423 SG-DVCL S4P files for each W/R candidate."""

    def __init__(self, cfg: PaSynthesisConfig, freq_hz: np.ndarray) -> None:
        self.cfg = cfg
        self.freq_hz = tuple(float(v) for v in freq_hz)
        self.cache_dir = Path(cfg.paths.cache_dir) / cfg.sg_dvcl.subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def geometry(self, w_um: float, r: float) -> dict[str, float]:
        from VCL_length_calculator import calculate_compensation_msl

        return calculate_compensation_msl(w_um, r)

    def path_for(self, w_um: float, r: float) -> Path:
        return self.cache_dir / f"sg_dvcl_W{_safe_tag(w_um)}_R{_safe_tag(r)}.s4p"

    def get_s4p(self, w_um: float, r: float) -> Path:
        out_path = self.path_for(w_um, r)
        if self.cfg.sg_dvcl.reuse_cached_s4p and out_path.exists():
            return out_path
        if not self.cfg.sg_dvcl.generate_missing_s4p:
            raise FileNotFoundError(
                f"Missing SG-DVCL S4P for W={w_um:g}, R={r:g}: {out_path}. "
                "Set sg_dvcl.generate_missing_s4p=true to generate it with Cal_0423.py."
            )

        geom = self.geometry(w_um, r)
        import Cal_0423 as cal

        params_kwargs = {
            "W_line": 0.25 * float(w_um),
            "line_length_um": float(geom["real_coupling_length1_um"]),
            "freq_list_Hz": self.freq_hz,
            "output_dir": str(self.cache_dir),
            "export_touchstone_filename": out_path.name,
            "quiet": True,
            "write_summary_json_enabled": False,
            "generate_geometry_plots": False,
            "generate_potential_plots": False,
            "print_geometry_table": False,
        }
        if self.cfg.sg_dvcl.m_modes is not None:
            params_kwargs["M_modes"] = int(self.cfg.sg_dvcl.m_modes)
        params = cal.Config(**params_kwargs)
        cal.ensure_output_dir(params.output_dir)
        bundle = cal.prepare_sweep_bundle(params)
        cal.export_sweep_s4p_from_modal_records(
            params,
            bundle.modal_records,
            params.line_length_um * cal.UM_TO_M,
        )
        if not out_path.exists():
            raise FileNotFoundError(f"Cal_0423.py did not create expected S4P: {out_path}")
        return out_path


class TransformerPredictor:
    """Build the compensated six-port transformer network for a W/R candidate."""

    def __init__(
        self,
        cfg: PaSynthesisConfig,
        freq_hz: np.ndarray,
        inductor_predictor: InductorPredictor,
        sg_dvcl_provider: SgDvclProvider,
    ) -> None:
        self.cfg = cfg
        self.freq_hz = np.asarray(freq_hz, dtype=float)
        self.inductor_predictor = inductor_predictor
        self.sg_dvcl_provider = sg_dvcl_provider
        self.cache: dict[tuple[float, float], rf.Network] = {}
        self.s6p_cache_dir = Path(cfg.paths.cache_dir) / cfg.transformer.subdir
        self.s6p_cache_dir.mkdir(parents=True, exist_ok=True)

    def build(self, w_um: float, r: float, *, name: str = "tf6") -> rf.Network:
        key = (round(float(w_um), 6), round(float(r), 6))
        if key in self.cache:
            out = self.cache[key].copy()
            out.name = name
            return out

        import TF_Cal_S6P_Predicting as tf6

        f_start_ghz, f_stop_ghz, f_step_ghz = _uniform_ghz_grid(self.freq_hz)
        s4p = self.sg_dvcl_provider.get_s4p(w_um, r)
        geom = self.sg_dvcl_provider.geometry(w_um, r)
        inductors = self.inductor_predictor.predict(w_um, r)

        local_cfg = dict(tf6.CFG)
        local_cfg.update(
            {
                "s4p_top": str(s4p),
                "s4p_bot": str(s4p),
                "f_start_ghz": f_start_ghz,
                "f_stop_ghz": f_stop_ghz,
                "f_step_ghz": f_step_ghz,
                "z0": float(self.cfg.transformer.z0),
                "L13_nH": float(inductors["L13_nH"]),
                "L24_nH": float(inductors["L24_nH"]),
                "L56_pH": float(inductors["L56_pH"]),
                "L56V_extra_pH": float(inductors["L56delta_pH"]),
                "Wline_um": float(inductors.get("W_MSL_um", geom["W_MSL_um"])),
                "Lline_um": float(inductors.get("L_MSL_um", geom["L_MSL_um"])),
                "out_dir": str(self.s6p_cache_dir),
                "out_name": f"tf6_W{_safe_tag(w_um)}_R{_safe_tag(r)}",
                "MA": dict(self.cfg.transformer.substrates["MA"]),
                "E1": dict(self.cfg.transformer.substrates["E1"]),
            }
        )

        ntw = tf6.build_tf_6port(local_cfg)
        ntw = align_network(ntw, self.freq_hz, name=f"tf6_W{_safe_tag(w_um)}_R{_safe_tag(r)}")
        if self.cfg.transformer.write_s6p_cache:
            tf6.write_s6p(ntw, local_cfg)

        self.cache[key] = ntw.copy()
        out = ntw.copy()
        out.name = name
        return out

    def describe_candidate(self, w_um: float, r: float) -> dict[str, object]:
        geom = self.sg_dvcl_provider.geometry(w_um, r)
        inductors = self.inductor_predictor.predict(w_um, r)
        return {
            "W_um": float(w_um),
            "R": float(r),
            "sg_dvcl_s4p": str(self.sg_dvcl_provider.path_for(w_um, r)),
            "geometry": geom,
            "inductors": inductors,
        }

    def write_candidate_manifest(self, w_um: float, r: float, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(self.describe_candidate(w_um, r), handle, indent=2)

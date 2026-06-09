from __future__ import annotations

import argparse
from bisect import bisect_right
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


EPS0 = 8.854187817e-12
MU0 = 4.0e-7 * np.pi
UM_TO_M = 1e-6
M_TO_UM = 1e6
MATERIAL_EPS_R = {"Air": 1.0, "SiO2": 4.1, "Si": 11.9}
CONDUCTOR_VOLTAGE_KEYS = {"Metal1": 1, "Metal2": 2, "MetalGND": 0}
CHARGE_EDGE_BASE_SAMPLES = 401
CHARGE_EDGE_DENSE_SAMPLES = 801
CHARGE_EDGE_CLUSTER_POWER = 0.60
EDGE_GRID_CACHE: Dict[Tuple[int, float], np.ndarray] = {}


@dataclass
class Config:
    """Top-level solver configuration.

    Unit convention:
    - User-facing geometry inputs stay in micrometers for readability.
    - build_geometry() converts everything to meters before assembly.
    - All field solve / energy / charge / C' calculations are done in SI units.

    Geometry/line-length convention:
    - L is the cross-section half-width, not the coupled-line length.
    - line_length_um is the physical transmission-line length used for sNp export.

    Touchstone port convention:
    - Internal modal 4-port order before export permutation is
      [MA_A, E1_A, MA_B, E1_B].
    - With the default touchstone_port_perm=(1, 2, 3, 0), the exported
      4-port order is [E1_A, MA_B, E1_B, MA_A].

    Dirichlet boundary projection mode:
    - outer_dirichlet_projection_mode = "x_seg" projects each top/bottom
      boundary row only over the local cell span [x1, x2].
    - outer_dirichlet_projection_mode = "x_full" projects each such row over
      the full width [0, 2L], matching Cal_0313.py's current top/bottom
      outer-boundary treatment.

    Optional coarse-region local refinement:
    - Geometry starts from either 5 coarse x regions (defect enabled) or
      3 coarse x regions (closed-GND case).
    - When enable_targeted_region_subdivision is true, any currently present
      coarse region may be subdivided uniformly along x according to
      targeted_region_subdivision_counts.
    - In practice this means:
      * defect enabled: coarse regions 1/2/3/4/5 may be refined
      * closed GND: coarse regions 1/2/3 may be refined

    Optional defect-edge local refinement:
    - When enable_defect_edge_refinement is true and W_GND_defect > 0, extra
      x breakpoints are inserted around x_defect_left / x_defect_right.
    - The spacing unit is chosen by defect_edge_refinement_scale:
      * "quarter_line" -> delta = W_line / 4
      * "eighth_line"  -> delta = W_line / 8
    - defect_edge_refinement_steps_each_side = N inserts up to N extra
      breakpoints on each side of each defect edge, clipped to the local
      coarse-region span.
    - This refinement is combined with the uniform coarse-region subdivision
      above by taking the union of all breakpoints.

    Optional slab-selection control for x refinement:
    - When enable_selective_slab_x_refinement is false, every slab uses the
      same x-refinement rules.
    - When it is true, only slabs listed in x_refinement_slab_names receive
      the targeted x subdivision / defect-edge refinement; other slabs fall
      back to the coarse x partition only.

    Optional targeted slab y refinement:
    - When enable_targeted_slab_y_refinement is true, selected base slabs may
      be subdivided uniformly along y.
    - targeted_slab_y_split_counts maps base slab names to split counts.
    - Supported split counts are 1..5, where 1 means "do not split".
    - Refined sub-slabs keep the original base slab identity for material and
      conductor assignment, but become separate rows in the assembled geometry.

    Optional HFSS matrix override:
    - use_hfss_cprime_override switches the final C' used for L/C/S export
      from the internally extracted matrix to the provided HFSS matrix.
    - use_hfss_cair_override does the same for C_air.
    - The internally extracted matrices are still computed and printed for
      comparison; only the final matrices used downstream are switched.

    Optional RLGC loss model:
    - use_loss = False keeps the current lossless LC flow.
    - use_loss = True adds the simple Cal_0313-style loss model:
      * G'(f) = omega * tan_delta_eff * C'
      * R'(f) from conductor surface resistance and effective width
    - tan_delta_by_material lets each dielectric material carry its own loss.
    - tan_delta_eff_override may be used to bypass the material-based
      aggregation and force a single effective tan(delta).
    """

    L: float = 400.0  # um
    W_line: float = 22.5  # um W90
    # W_line: float = 25  # um W100
    # W_line: float = 25.25  # um W101
    # W_line: float = 26.5  # um W106
    # W_line: float = 27.5  # um W110
    # W_line: float = 27.75  # um W111
    # W_line: float = 30  # um W120
    line_length_um: float = 160.1
    # line_length_um: float = 214  # W111R1.47
    # line_length_um: float = 182.55  # W106R1.26
    # line_length_um: float = 213.8  # W101R1.65

    # 耦合线长度 [m] W90R1.3 160.1
    # 耦合线长度 [m] W90R1.4 169.1
    # 耦合线长度 [m] W90R1.5 178.1
    # 耦合线长度 [m] W90R1.6 187.1
    # 耦合线长度 [m] W90R1.7 196.1

    # 耦合线长度 [m] W100R1.3 176.8
    # 耦合线长度 [m] W100R1.4 186.8
    # 耦合线长度 [m] W100R1.5 196.8
    # 耦合线长度 [m] W100R1.6 206.8
    # 耦合线长度 [m] W100R1.7 216.8

    # 耦合线长度 [m]W110R1.3 193.5
    # 耦合线长度 [m]W110R1.4 204.5
    # 耦合线长度 [m]W110R1.5 215.5
    # 耦合线长度 [m]W110R1.6 226.5
    # 耦合线长度 [m]W110R1.7 237.5

    # 耦合线长度 [m]W120R1.3 210.1
    # 耦合线长度 [m]W120R1.4 222.1
    # 耦合线长度 [m]W120R1.5 234.1
    # 耦合线长度 [m]W120R1.6 246.1
    # 耦合线长度 [m]W120R1.7 258.1

    W_GND_defect: Optional[float] = None  # um; 0 closes the GND defect, None falls back to 1.3 * W_line

    quadrature_order: int = 12  # Gauss-Legendre order for conductor-edge charge integration
    plot_resolution: int = 220  # grid size used only for sampled potential plots
    outer_dirichlet_projection_mode: str = "x_seg"  # "x_seg" = per-cell local span; "x_full" = full [0, 2L] span
    enable_targeted_region_subdivision: bool = 1  # 缺陷上方区域划分
    targeted_region_subdivision_counts: Dict[int, int] = field(
        default_factory=lambda: {1: 1, 2: 2, 3: 1, 4: 2, 5: 1}
    )
    enable_defect_edge_refinement: bool = 0  # 向缺陷两边拓展子域？
    defect_edge_refinement_scale: str = "eighth_line"  # "quarter_line" or "eighth_line"
    defect_edge_refinement_steps_each_side: int = 1  # 外扩几列？
    enable_selective_slab_x_refinement: bool = 1  # 选层细分
    x_refinement_slab_names: Tuple[str, ...] = (
        "slab3",
        "slab1",
    )
    M_modes: int = 3
    enable_targeted_slab_y_refinement: bool = 1
    targeted_slab_y_split_counts: Dict[str, int] = field(
        default_factory=lambda: {
            "slab_b3": 1,
            "slab_b2": 1,
            "slab_b1": 1,
            "slab0": 1,
            "slab1": 1,
            "slab2": 3,
            "slab3": 1,
            "slab4": 3,
            "slab5": 1,
        }
    )

    use_hfss_cprime_override: bool = 0
    use_hfss_cair_override: bool = 0
    Cprime_hfss: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (386e-12, -338e-12),
        (-381e-12, 559e-12),
    )
    Cair_hfss: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (77e-12, -67e-12),
        (-78e-12, 119e-12),
    )
    freq_list_Hz: Tuple[float, ...] = field(
        default_factory=lambda: tuple(np.linspace(1e9, 200e9, 20))
    )
    z0_ref: float = 50.0
    export_touchstone_filename: str = "cal04_line.s4p"
    touchstone_port_perm: Tuple[int, int, int, int] = (1, 2, 3, 0)
    touchstone_port_labels: Tuple[str, str, str, str] = ("E1_A", "MA_B", "E1_B", "MA_A")
    use_loss: bool = 1
    metal_sigma_by_conductor_S_per_m: Dict[str, float] = field(
        default_factory=lambda: {
            "Metal1": 3.57e8,
            "Metal2": 5.56e8,
            "MetalGND": 4.916e8,
        }
    )
    metal_roughness_um: float = 0.0
    tan_delta_by_material: Dict[str, float] = field(
        default_factory=lambda: {"Air": 0.0, "SiO2": 0.01, "Si": 0.0}
    )
    tan_delta_eff_override: Optional[float] = None
    print_geometry_table: bool = False
    generate_geometry_plots: bool = False
    generate_potential_plots: bool = False
    write_summary_json_enabled: bool = False
    quiet: bool = False

    # Cair_hfss: Tuple[Tuple[float, float], Tuple[float, float]] = (
    #     (82.25e-12, -64.5e-12),
    #     (-64.5e-12, 90.08e-12),
    # )
    output_dir: str = "cal04_outputs"
    y_levels: Tuple[float, ...] = (-100.0, 0.0, 3.32, 3.92, 13.54, 16.54, 20.64, 24.64, 28.94, 200.0)  # um
    slab_names: Tuple[str, ...] = (
        "slab_b3",
        "slab_b2",
        "slab_b1",
        "slab0",
        "slab1",
        "slab2",
        "slab3",
        "slab4",
        "slab5",
    )

    def __post_init__(self) -> None:
        if self.W_GND_defect is None:
            self.W_GND_defect = 1.3 * self.W_line
        if self.outer_dirichlet_projection_mode not in ("x_seg", "x_full"):
            raise ValueError(
                "outer_dirichlet_projection_mode must be either 'x_seg' or 'x_full'."
            )
        if self.defect_edge_refinement_scale not in ("quarter_line", "eighth_line"):
            raise ValueError(
                "defect_edge_refinement_scale must be 'quarter_line' or 'eighth_line'."
            )
        if int(self.defect_edge_refinement_steps_each_side) < 0:
            raise ValueError("defect_edge_refinement_steps_each_side must be >= 0.")
        invalid_slab_names = [name for name in self.x_refinement_slab_names if name not in self.slab_names]
        if invalid_slab_names:
            raise ValueError(
                f"x_refinement_slab_names contains unknown slabs: {invalid_slab_names}"
            )
        invalid_y_slab_names = [name for name in self.targeted_slab_y_split_counts if name not in self.slab_names]
        if invalid_y_slab_names:
            raise ValueError(
                f"targeted_slab_y_split_counts contains unknown slabs: {invalid_y_slab_names}"
            )
        for slab_name, count in self.targeted_slab_y_split_counts.items():
            if int(count) < 1 or int(count) > 5:
                raise ValueError(
                    f"Y split count for {slab_name} must be within 1..5."
                )
        for coarse_region_id, count in self.targeted_region_subdivision_counts.items():
            if coarse_region_id not in (1, 2, 3, 4, 5):
                raise ValueError(
                    "targeted_region_subdivision_counts may only specify coarse regions 1/2/3/4/5."
                )
            if int(count) < 1:
                raise ValueError(
                    f"Subdivision count for coarse region {coarse_region_id} must be >= 1."
                )
        self.freq_list_Hz = tuple(float(v) for v in self.freq_list_Hz)
        if len(self.freq_list_Hz) == 0:
            raise ValueError("freq_list_Hz must contain at least one frequency point.")
        if tuple(sorted(self.touchstone_port_perm)) != (0, 1, 2, 3):
            raise ValueError(
                "touchstone_port_perm must be a permutation of (0, 1, 2, 3)."
            )
        if len(self.touchstone_port_labels) != 4:
            raise ValueError("touchstone_port_labels must contain exactly 4 labels.")
        self.Cprime_hfss = tuple(tuple(float(v) for v in row) for row in self.Cprime_hfss)
        self.Cair_hfss = tuple(tuple(float(v) for v in row) for row in self.Cair_hfss)
        invalid_sigma_conductors = [
            name
            for name in self.metal_sigma_by_conductor_S_per_m
            if name not in CONDUCTOR_VOLTAGE_KEYS
        ]
        if invalid_sigma_conductors:
            raise ValueError(
                "metal_sigma_by_conductor_S_per_m contains unknown conductors: "
                f"{invalid_sigma_conductors}"
            )
        for conductor_name, sigma in self.metal_sigma_by_conductor_S_per_m.items():
            if float(sigma) <= 0.0:
                raise ValueError(
                    f"metal_sigma_by_conductor_S_per_m[{conductor_name!r}] must be positive."
                )
        self.metal_sigma_by_conductor_S_per_m = {
            str(name): float(value)
            for name, value in self.metal_sigma_by_conductor_S_per_m.items()
        }
        if self.metal_roughness_um < 0.0:
            raise ValueError("metal_roughness_um must be non-negative.")
        invalid_td_materials = [
            name for name in self.tan_delta_by_material if name not in MATERIAL_EPS_R
        ]
        if invalid_td_materials:
            raise ValueError(
                f"tan_delta_by_material contains unknown materials: {invalid_td_materials}"
            )
        for material_name, tan_delta in self.tan_delta_by_material.items():
            if float(tan_delta) < 0.0:
                raise ValueError(
                    f"tan_delta_by_material[{material_name!r}] must be non-negative."
                )
        self.tan_delta_by_material = {
            str(name): float(value) for name, value in self.tan_delta_by_material.items()
        }
        if self.tan_delta_eff_override is not None:
            self.tan_delta_eff_override = float(self.tan_delta_eff_override)
            if self.tan_delta_eff_override < 0.0:
                raise ValueError("tan_delta_eff_override must be non-negative.")


def matrix2_from_config(matrix_like: Sequence[Sequence[float]], name: str) -> np.ndarray:
    """Convert a config-provided 2x2 matrix to a validated NumPy array."""

    arr = np.asarray(matrix_like, dtype=float)
    if arr.shape != (2, 2):
        raise ValueError(f"{name} must be a 2x2 matrix, got shape {arr.shape}.")
    return arr


def resolve_effective_matrix(
    computed: np.ndarray,
    hfss_override: Sequence[Sequence[float]],
    use_override: bool,
    matrix_name: str,
) -> Tuple[np.ndarray, str, np.ndarray]:
    """Return the matrix used downstream, its source label, and the HFSS matrix."""

    hfss_matrix = matrix2_from_config(hfss_override, matrix_name)
    if use_override:
        return hfss_matrix.copy(), "hfss_override", hfss_matrix
    return np.asarray(computed, dtype=float).copy(), "code", hfss_matrix


@dataclass
class Cell:
    """Rectangular cell in the 9x5 table.

    All geometric coordinates stored here are in meters.
    """

    cell_id: int
    slab_index: int
    slab_name: str
    base_slab_index: int
    base_slab_name: str
    region_id: int
    coarse_region_id: int
    x1: float
    x2: float
    y1: float
    y2: float
    material_name: str = ""
    eps_r: float = 1.0
    eps_abs: float = EPS0
    is_conductor: bool = False
    conductor_name: Optional[str] = None
    conductor_id: Optional[int] = None
    y_ref: float = 0.0
    basis_size: int = 0


@dataclass
class BasisConfig:
    """Global separated-variable basis configuration."""

    L: float
    M_modes: int
    km_array: np.ndarray


@dataclass
class CaseResult:
    """Results for one electrostatic excitation case."""

    case_name: str
    excitation: Dict[int, float]
    x: np.ndarray
    lambda_vec: np.ndarray
    dof_slices: Dict[int, slice]
    constraint_stats: Dict[str, int]
    charges: Dict[str, float]
    edge_breakdown: Dict[str, Dict[str, float]]
    potential_grid: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    loss_placeholder: float = 0.0


@dataclass(frozen=True)
class ExcitationSpec:
    """One electrostatic excitation to be solved on a prepared geometry."""

    case_name: str
    excitation: Dict[int, float]
    compute_potential_grid: bool = True


@dataclass(frozen=True)
class SlabLocatorEntry:
    """Fast lookup helper for one y-slab of cells."""

    y1: float
    y2: float
    x_starts: Tuple[float, ...]
    cells: Tuple[Cell, ...]


@dataclass(frozen=True)
class SpatialLocator:
    """Fast rectangular cell locator using y-slab and x-interval bisection."""

    y_starts: Tuple[float, ...]
    slabs: Tuple[SlabLocatorEntry, ...]


@dataclass(frozen=True)
class ConstraintRhsTerm:
    """One excitation-dependent RHS contribution inside a cached constraint system."""

    row_index: int
    conductor_id: int
    scale: float


@dataclass(frozen=True)
class ConstraintTemplate:
    """Excitation-invariant constraint matrix plus compact RHS metadata."""

    C_mat: np.ndarray
    rhs_terms: Tuple[ConstraintRhsTerm, ...]
    stats: Dict[str, int]


@dataclass(frozen=True)
class ReducedConstraintSystem:
    """Compressed constraint system used for faster cold-start KKT solves."""

    nonzero_row_mask: np.ndarray
    q_transpose: np.ndarray
    reduced_C: np.ndarray


@dataclass(frozen=True)
class ChargeFactorGroup:
    """Pre-integrated charge factors for one dielectric cell on one contour edge."""

    cell_id: int
    factor_vec: np.ndarray


@dataclass(frozen=True)
class ChargeBaseFactorGroup:
    """Geometry-only contour factors before multiplying by local permittivity."""

    cell_id: int
    base_factor_vec: np.ndarray


@dataclass(frozen=True)
class ChargeEdgeBasePlan:
    """Reusable geometry-only contour-integration plan for one conductor edge."""

    conductor_name: str
    conductor_id: int
    edge_name: str
    factor_groups: Tuple[ChargeBaseFactorGroup, ...]


@dataclass(frozen=True)
class ChargeEdgePlan:
    """Reusable contour-integration plan for one conductor edge."""

    conductor_name: str
    conductor_id: int
    edge_name: str
    factor_groups: Tuple[ChargeFactorGroup, ...]


@dataclass(frozen=True)
class PreparedStaticSystem:
    """Geometry-specific cached data for the electrostatic solve."""

    cells: Tuple[Cell, ...]
    basis_cfg: BasisConfig
    M_big: np.ndarray
    dof_slices: Dict[int, slice]
    constraint_template: ConstraintTemplate
    reduced_constraint_system: ReducedConstraintSystem
    charge_plans: Tuple[ChargeEdgePlan, ...]


@dataclass(frozen=True)
class PreparedSweepBundle:
    """Width-dependent results reused across multiple physical line lengths."""

    cells: Tuple[Cell, ...]
    basis_cfg: BasisConfig
    case_A: CaseResult
    case_B: CaseResult
    Cprime_computed: np.ndarray
    Cprime_final_used: np.ndarray
    Cprime_hfss: np.ndarray
    Cprime_source: str
    Cair_computed: np.ndarray
    Cair_final_used: np.ndarray
    Cair_hfss: np.ndarray
    Cair_source: str
    Lprime: np.ndarray
    modal_records: Tuple[Dict[str, object], ...]
    tan_delta_eff: float
    conductor_geom: Dict[int, Dict[str, float]]


PREPARED_SWEEP_CACHE: Dict[Tuple[object, ...], PreparedSweepBundle] = {}


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def serialize_complex_array(arr: np.ndarray) -> List[Dict[str, float]]:
    """Serialize a 1D complex array into JSON-friendly real/imag dicts."""

    flat = np.asarray(arr, dtype=complex).ravel()
    return [{"real": float(np.real(v)), "imag": float(np.imag(v))} for v in flat]


def _freeze_for_cache(value):
    """Convert nested config values into hashable tuples for cache keys."""

    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_for_cache(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_for_cache(v) for v in value)
    if isinstance(value, np.ndarray):
        return tuple(_freeze_for_cache(v) for v in value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def prepared_bundle_cache_key(params: Config) -> Tuple[object, ...]:
    """Return the cache key for width-dependent preparation reused across lengths."""

    excluded = {
        "line_length_um",
        "output_dir",
        "export_touchstone_filename",
        "touchstone_port_perm",
        "touchstone_port_labels",
        "z0_ref",
        "print_geometry_table",
        "generate_geometry_plots",
        "generate_potential_plots",
        "write_summary_json_enabled",
        "quiet",
    }
    payload = asdict(params)
    return tuple(
        sorted(
            (name, _freeze_for_cache(value))
            for name, value in payload.items()
            if name not in excluded
        )
    )


def validate_geometry(cfg: Config) -> Dict[str, float]:
    """Validate the x partition and the fixed 9-slab y partition.

    Inputs in cfg are in micrometers; returned breakpoints are in meters.
    """

    x_left = 0.0
    x_right = 2.0 * cfg.L * UM_TO_M
    if cfg.W_GND_defect < 0:
        raise ValueError("W_GND_defect must be non-negative.")
    x_line_left = (cfg.L - cfg.W_line / 2.0) * UM_TO_M
    x_line_right = (cfg.L + cfg.W_line / 2.0) * UM_TO_M
    has_gnd_defect = cfg.W_GND_defect > 0.0
    if has_gnd_defect:
        if not (cfg.W_GND_defect > cfg.W_line):
            raise ValueError("W_GND_defect must be greater than W_line when the defect is enabled.")
        x_defect_left = (cfg.L - cfg.W_GND_defect / 2.0) * UM_TO_M
        x_defect_right = (cfg.L + cfg.W_GND_defect / 2.0) * UM_TO_M
        x_breaks = [x_left, x_defect_left, x_line_left, x_line_right, x_defect_right, x_right]
    else:
        x_defect_left = x_line_left
        x_defect_right = x_line_right
        x_breaks = [x_left, x_line_left, x_line_right, x_right]
    y_breaks = [float(v) * UM_TO_M for v in cfg.y_levels]
    if any(b <= a for a, b in zip(x_breaks[:-1], x_breaks[1:])):
        raise ValueError(f"x breaks must be strictly increasing, got {x_breaks}")
    if any(b <= a for a, b in zip(y_breaks[:-1], y_breaks[1:])):
        raise ValueError(f"y breaks must be strictly increasing, got {y_breaks}")
    if len(cfg.slab_names) != len(y_breaks) - 1:
        raise ValueError("slab_names length must equal len(y_levels)-1.")
    return {
        "x_left": x_left,
        "x_right": x_right,
        "x_defect_left": x_defect_left,
        "x_line_left": x_line_left,
        "x_line_right": x_line_right,
        "x_defect_right": x_defect_right,
        "x_breaks": x_breaks,
        "region_count": len(x_breaks) - 1,
        "has_gnd_defect": has_gnd_defect,
    }


def defect_edge_refinement_step_m(params: Config) -> float:
    """Return the defect-edge refinement spacing in meters."""

    if params.defect_edge_refinement_scale == "quarter_line":
        return float(params.W_line) * UM_TO_M / 4.0
    return float(params.W_line) * UM_TO_M / 8.0


def slab_uses_x_refinement(slab_name: str, params: Config) -> bool:
    """Return whether one slab participates in optional x refinement."""

    if not params.enable_selective_slab_x_refinement:
        return True
    return slab_name in set(params.x_refinement_slab_names)


def slab_y_split_count(base_slab_name: str, params: Config) -> int:
    """Return the y subdivision count for one base slab."""

    if not params.enable_targeted_slab_y_refinement:
        return 1
    return int(params.targeted_slab_y_split_counts.get(base_slab_name, 1))


def local_x_breaks_for_coarse_region(
    coarse_region_id: int,
    xa: float,
    xb: float,
    geo: Dict[str, float],
    params: Config,
) -> List[float]:
    """Build the sorted local x-break list for one coarse region."""

    local_breaks = {float(xa), float(xb)}

    if params.enable_targeted_region_subdivision:
        subdiv_count = int(params.targeted_region_subdivision_counts.get(coarse_region_id, 1))
        if subdiv_count > 1:
            for val in np.linspace(xa, xb, subdiv_count + 1)[1:-1]:
                local_breaks.add(float(val))

    if params.enable_defect_edge_refinement and bool(geo["has_gnd_defect"]):
        step = defect_edge_refinement_step_m(params)
        x_defect_left = float(geo["x_defect_left"])
        x_defect_right = float(geo["x_defect_right"])
        step_count = int(params.defect_edge_refinement_steps_each_side)
        for n in range(1, step_count + 1):
            offsets: List[float] = []
            if coarse_region_id == 1:
                offsets.append(x_defect_left - n * step)
            elif coarse_region_id == 2:
                offsets.append(x_defect_left + n * step)
            elif coarse_region_id == 4:
                offsets.append(x_defect_right - n * step)
            elif coarse_region_id == 5:
                offsets.append(x_defect_right + n * step)
            for x_val in offsets:
                if xa < x_val < xb:
                    local_breaks.add(float(x_val))

    return sorted(local_breaks)


def build_geometry(params: Config) -> List[Cell]:
    """Generate the x-by-y cell table in meters.

    When W_GND_defect > 0, the x partition has 5 regions:
    left / defect shoulder / line / defect shoulder / right.
    When W_GND_defect == 0, the GND defect is closed and the x partition
    collapses to 3 regions split only by the signal-line width.
    If enable_targeted_region_subdivision is true, every currently present
    coarse region may be further split uniformly along x according to
    targeted_region_subdivision_counts.
    If enable_defect_edge_refinement is true in the defect case, extra
    non-uniform breakpoints are inserted around the left/right defect edges.
    """

    geo = validate_geometry(params)
    x_breaks = list(geo["x_breaks"])

    cells: List[Cell] = []
    cell_id = 0
    actual_slab_index = 0
    for base_slab_index, base_slab_name in enumerate(params.slab_names):
        x_segments: List[Tuple[int, float, float]] = []
        use_refinement = slab_uses_x_refinement(base_slab_name, params)
        for coarse_region_id in range(1, len(x_breaks)):
            xa = float(x_breaks[coarse_region_id - 1])
            xb = float(x_breaks[coarse_region_id])
            if use_refinement:
                fine_breaks = local_x_breaks_for_coarse_region(coarse_region_id, xa, xb, geo, params)
            else:
                fine_breaks = [xa, xb]
            for local_idx in range(len(fine_breaks) - 1):
                x_segments.append(
                    (
                        coarse_region_id,
                        float(fine_breaks[local_idx]),
                        float(fine_breaks[local_idx + 1]),
                    )
                )
        y1_base = float(params.y_levels[base_slab_index]) * UM_TO_M
        y2_base = float(params.y_levels[base_slab_index + 1]) * UM_TO_M
        y_split_count = slab_y_split_count(base_slab_name, params)
        y_breaks_local = np.linspace(y1_base, y2_base, y_split_count + 1)
        for y_local_idx in range(y_split_count):
            y1 = float(y_breaks_local[y_local_idx])
            y2 = float(y_breaks_local[y_local_idx + 1])
            if y_split_count == 1:
                slab_name = base_slab_name
            else:
                slab_name = f"{base_slab_name}_ys{y_local_idx + 1}of{y_split_count}"
            for region_id, (coarse_region_id, x1, x2) in enumerate(x_segments, start=1):
                cells.append(
                    Cell(
                        cell_id=cell_id,
                        slab_index=actual_slab_index,
                        slab_name=slab_name,
                        base_slab_index=base_slab_index,
                        base_slab_name=base_slab_name,
                        region_id=region_id,
                        coarse_region_id=coarse_region_id,
                        x1=x1,
                        x2=x2,
                        y1=y1,
                        y2=y2,
                        y_ref=0.5 * (y1 + y2),
                    )
                )
                cell_id += 1
            actual_slab_index += 1
    return cells


def assign_materials(cells: Sequence[Cell], params: Config) -> None:
    """Assign slab-wise material metadata."""

    for cell in cells:
        if cell.base_slab_name == "slab_b3":
            material = "Si"
        elif cell.base_slab_name == "slab5":
            material = "Air"
        else:
            material = "SiO2"
        cell.material_name = material
        cell.eps_r = MATERIAL_EPS_R[material]
        cell.eps_abs = EPS0 * cell.eps_r


def assign_conductors(cells: Sequence[Cell], params: Config) -> None:
    """Assign conductor occupancy on top of the material map."""

    geo = validate_geometry(params)
    x_line_left = float(geo["x_line_left"])
    x_line_right = float(geo["x_line_right"])
    x_defect_left = float(geo["x_defect_left"])
    x_defect_right = float(geo["x_defect_right"])
    has_gnd_defect = bool(geo["has_gnd_defect"])
    tol = 1e-18

    for cell in cells:
        cell.is_conductor = False
        cell.conductor_name = None
        cell.conductor_id = None

        within_line_span = cell.x1 >= x_line_left - tol and cell.x2 <= x_line_right + tol
        left_of_defect = cell.x2 <= x_defect_left + tol
        right_of_defect = cell.x1 >= x_defect_right - tol

        if cell.base_slab_name == "slab_b1" and (
            (not has_gnd_defect) or left_of_defect or right_of_defect
        ):
            cell.is_conductor = True
            cell.conductor_name = "MetalGND"
            cell.conductor_id = CONDUCTOR_VOLTAGE_KEYS["MetalGND"]
        elif cell.base_slab_name == "slab1" and within_line_span:
            cell.is_conductor = True
            cell.conductor_name = "Metal2"
            cell.conductor_id = CONDUCTOR_VOLTAGE_KEYS["Metal2"]
        elif cell.base_slab_name == "slab3" and within_line_span:
            cell.is_conductor = True
            cell.conductor_name = "Metal1"
            cell.conductor_id = CONDUCTOR_VOLTAGE_KEYS["Metal1"]


def validate_conductor_layout(cells: Sequence[Cell]) -> None:
    """Check that conductor rectangles do not geometrically overlap."""

    conductor_cells = [cell for cell in cells if cell.is_conductor]
    for i, cell_a in enumerate(conductor_cells):
        for cell_b in conductor_cells[i + 1 :]:
            xa = max(cell_a.x1, cell_b.x1)
            xb = min(cell_a.x2, cell_b.x2)
            ya = max(cell_a.y1, cell_b.y1)
            yb = min(cell_a.y2, cell_b.y2)
            if (xb - xa) > 1e-12 and (yb - ya) > 1e-12:
                raise ValueError(
                    f"Conductors overlap: {cell_a.conductor_name} cell {cell_a.cell_id} "
                    f"and {cell_b.conductor_name} cell {cell_b.cell_id}"
                )


def build_basis_for_cell(cell: Cell, M_modes: int) -> None:
    """Attach local basis metadata to one dielectric cell."""

    if cell.is_conductor:
        cell.basis_size = 0
    else:
        cell.basis_size = 2 * M_modes
        cell.y_ref = 0.5 * (cell.y1 + cell.y2)


def build_cell_lookup(cells: Sequence[Cell]) -> Dict[Tuple[str, int], Cell]:
    return {(cell.slab_name, cell.region_id): cell for cell in cells}


def get_region_ids(cells: Sequence[Cell]) -> List[int]:
    """Return the shared region ids used by all slabs."""

    return sorted({int(cell.region_id) for cell in cells})


def group_cells_by_slab(cells: Sequence[Cell]) -> Dict[str, List[Cell]]:
    """Return slab-name -> cells sorted from left to right."""

    slab_map: Dict[str, List[Cell]] = {}
    for cell in cells:
        slab_map.setdefault(cell.slab_name, []).append(cell)
    for slab_name in slab_map:
        slab_map[slab_name].sort(key=lambda c: (c.x1, c.x2, c.cell_id))
    return slab_map


def iter_horizontal_overlap_pairs(
    lower_cells: Sequence[Cell],
    upper_cells: Sequence[Cell],
    tol: float = 1e-18,
) -> List[Tuple[Cell, Cell, Tuple[float, float]]]:
    """Return all lower/upper cell pairs with positive x overlap."""

    pairs: List[Tuple[Cell, Cell, Tuple[float, float]]] = []
    i = 0
    j = 0
    lower_sorted = sorted(lower_cells, key=lambda c: (c.x1, c.x2, c.cell_id))
    upper_sorted = sorted(upper_cells, key=lambda c: (c.x1, c.x2, c.cell_id))
    while i < len(lower_sorted) and j < len(upper_sorted):
        cell_low = lower_sorted[i]
        cell_up = upper_sorted[j]
        xa = max(cell_low.x1, cell_up.x1)
        xb = min(cell_low.x2, cell_up.x2)
        if xb - xa > tol:
            pairs.append((cell_low, cell_up, (float(xa), float(xb))))
        if cell_low.x2 <= cell_up.x2 + tol:
            i += 1
        else:
            j += 1
    return pairs


def summarize_geometry(cells: Sequence[Cell]) -> None:
    """Print the cell table in micrometers for readability."""

    header = (
        f"{'id':>3} {'slab':<8} {'r':>2} {'cr':>2} {'x1':>10} {'x2':>10} "
        f"{'y1':>10} {'y2':>10} {'material':<6} {'conductor':<10}"
    )
    print(header)
    print("-" * len(header))
    for cell in cells:
        conductor = cell.conductor_name if cell.conductor_name else "-"
        print(
            f"{cell.cell_id:>3d} {cell.slab_name:<8} {cell.region_id:>2d} {cell.coarse_region_id:>2d} "
            f"{cell.x1 * M_TO_UM:>10.3f} {cell.x2 * M_TO_UM:>10.3f} "
            f"{cell.y1 * M_TO_UM:>10.3f} {cell.y2 * M_TO_UM:>10.3f} "
            f"{cell.material_name:<6} {conductor:<10}"
        )


def build_spatial_locator(cells: Sequence[Cell]) -> SpatialLocator:
    """Prebuild a fast lookup structure for point-to-cell queries."""

    slab_entries: List[SlabLocatorEntry] = []
    slab_keys = sorted(
        {(cell.slab_index, cell.slab_name, float(cell.y1), float(cell.y2)) for cell in cells},
        key=lambda item: (item[2], item[3], item[0], item[1]),
    )
    for slab_index, slab_name, y1, y2 in slab_keys:
        slab_cells = tuple(
            sorted(
                [cell for cell in cells if cell.slab_index == slab_index and cell.slab_name == slab_name],
                key=lambda cell: (cell.x1, cell.x2, cell.cell_id),
            )
        )
        slab_entries.append(
            SlabLocatorEntry(
                y1=float(y1),
                y2=float(y2),
                x_starts=tuple(float(cell.x1) for cell in slab_cells),
                cells=slab_cells,
            )
        )
    return SpatialLocator(
        y_starts=tuple(entry.y1 for entry in slab_entries),
        slabs=tuple(slab_entries),
    )


def compute_km(M_modes: int, L: float) -> np.ndarray:
    """Separated-variable wave numbers."""

    m = np.arange(1, M_modes + 1, dtype=float)
    return m * np.pi / (2.0 * L)


def _int_cosh(a: float, ua: float, ub: float) -> float:
    return (np.sinh(a * ub) - np.sinh(a * ua)) / a


def _int_sinh(a: float, ua: float, ub: float) -> float:
    return (np.cosh(a * ub) - np.cosh(a * ua)) / a


def _y_int_cosh2(k: float, z0: float, z1: float) -> float:
    return (np.sinh(2.0 * k * z1) - np.sinh(2.0 * k * z0)) / (4.0 * k) + 0.5 * (z1 - z0)


def _y_int_sinh2(k: float, z0: float, z1: float) -> float:
    return (np.sinh(2.0 * k * z1) - np.sinh(2.0 * k * z0)) / (4.0 * k) - 0.5 * (z1 - z0)


def _y_int_sinhcosh(k: float, z0: float, z1: float) -> float:
    return (np.cosh(2.0 * k * z1) - np.cosh(2.0 * k * z0)) / (4.0 * k)


@lru_cache(maxsize=None)
def Ix_sin_sin(km: float, kn: float, xa: float, xb: float, tol: float = 1e-14) -> float:
    """Integral of sin(km x) sin(kn x) over [xa, xb]."""

    if abs(km - kn) < tol:
        return 0.5 * (xb - xa) - (np.sin(2.0 * km * xb) - np.sin(2.0 * km * xa)) / (4.0 * km)
    amb = km - kn
    apb = km + kn
    return 0.5 * (
        (np.sin(amb * xb) - np.sin(amb * xa)) / amb
        - (np.sin(apb * xb) - np.sin(apb * xa)) / apb
    )


@lru_cache(maxsize=None)
def Ix_cos_cos(km: float, kn: float, xa: float, xb: float, tol: float = 1e-14) -> float:
    """Integral of cos(km x) cos(kn x) over [xa, xb]."""

    if abs(km - kn) < tol:
        return 0.5 * (xb - xa) + (np.sin(2.0 * km * xb) - np.sin(2.0 * km * xa)) / (4.0 * km)
    amb = km - kn
    apb = km + kn
    return 0.5 * (
        (np.sin(amb * xb) - np.sin(amb * xa)) / amb
        + (np.sin(apb * xb) - np.sin(apb * xa)) / apb
    )


@lru_cache(maxsize=None)
def _Y_block_components(
    km: float,
    kn: float,
    ua: float,
    ub: float,
    tol: float = 1e-14,
) -> Tuple[float, float, float, float]:
    """Analytical y-integral blocks, cached because the same spans repeat heavily."""

    if abs(km - kn) < tol:
        common = _y_int_sinhcosh(km, ua, ub)
        return _y_int_cosh2(km, ua, ub), common, common, _y_int_sinh2(km, ua, ub)
    kp = km + kn
    kmn = km - kn
    y_ab = 0.5 * (_int_sinh(kp, ua, ub) + _int_sinh(kmn, ua, ub))
    return (
        0.5 * (_int_cosh(kp, ua, ub) + _int_cosh(kmn, ua, ub)),
        y_ab,
        y_ab,
        0.5 * (_int_cosh(kp, ua, ub) - _int_cosh(kmn, ua, ub)),
    )


def Y_blocks(km: float, kn: float, ua: float, ub: float, tol: float = 1e-14) -> Dict[str, float]:
    """Analytical y integrals for cosh/sinh blocks."""

    y_aa, y_ab, y_ba, y_bb = _Y_block_components(km, kn, ua, ub, tol)
    return {
        "Y_AA": y_aa,
        "Y_AB": y_ab,
        "Y_BA": y_ba,
        "Y_BB": y_bb,
    }


def Z_blocks_from_Y(km: float, kn: float, y_blocks: Dict[str, float]) -> Dict[str, float]:
    """Derivative-derived y blocks."""

    kmkn = km * kn
    return {
        "Z_AA": kmkn * y_blocks["Y_BB"],
        "Z_AB": kmkn * y_blocks["Y_BA"],
        "Z_BA": kmkn * y_blocks["Y_AB"],
        "Z_BB": kmkn * y_blocks["Y_AA"],
    }


def build_Mr_for_cell(cell: Cell, basis_cfg: BasisConfig) -> Optional[np.ndarray]:
    """Analytical local energy matrix for one dielectric cell.

    This matches Cal_0313.py:
    - x-derivative term: ix_cc * km * kn * Y
    - y-derivative term: ix_ss * Z
    with all lengths in meters.
    """

    if cell.is_conductor:
        return None
    M = basis_cfg.M_modes
    Mr = np.zeros((2 * M, 2 * M), dtype=float)
    z0 = cell.y1 - cell.y_ref
    z1 = cell.y2 - cell.y_ref
    for m in range(M):
        km = float(basis_cfg.km_array[m])
        for n in range(M):
            kn = float(basis_cfg.km_array[n])
            ix_ss = Ix_sin_sin(km, kn, cell.x1, cell.x2)
            ix_cc = Ix_cos_cos(km, kn, cell.x1, cell.x2)
            y_aa, y_ab, y_ba, y_bb = _Y_block_components(km, kn, z0, z1)
            iA, iB = m, m + M
            jA, jB = n, n + M
            factor = cell.eps_abs
            kmkn = km * kn
            Mr[iA, jA] += factor * (ix_cc * kmkn * y_aa + ix_ss * (kmkn * y_bb))
            Mr[iA, jB] += factor * (ix_cc * kmkn * y_ab + ix_ss * (kmkn * y_ba))
            Mr[iB, jA] += factor * (ix_cc * kmkn * y_ba + ix_ss * (kmkn * y_ab))
            Mr[iB, jB] += factor * (ix_cc * kmkn * y_bb + ix_ss * (kmkn * y_aa))
    return Mr


def assemble_energy_matrix(cells: Sequence[Cell], basis_cfg: BasisConfig) -> Tuple[np.ndarray, Dict[int, slice]]:
    """Assemble the global block-diagonal Ritz energy matrix."""

    dof_slices: Dict[int, slice] = {}
    cursor = 0
    for cell in cells:
        if cell.is_conductor:
            continue
        dof_slices[cell.cell_id] = slice(cursor, cursor + cell.basis_size)
        cursor += cell.basis_size
    M_big = np.zeros((cursor, cursor), dtype=float)
    for cell in cells:
        if cell.is_conductor:
            continue
        sl = dof_slices[cell.cell_id]
        Mr = build_Mr_for_cell(cell, basis_cfg)
        M_big[sl, sl] = Mr
    return M_big, dof_slices


def row_contrib_phi(cell: Cell, y_iface: float, x_seg: Tuple[float, float], basis_cfg: BasisConfig, p_mode: int) -> np.ndarray:
    """Projected phi contribution on a horizontal interface."""

    xa, xb = x_seg
    mode_count = basis_cfg.M_modes
    km_array = basis_cfg.km_array
    coeffs = np.zeros(2 * mode_count, dtype=float)
    kp = km_array[p_mode]
    y_shift = y_iface - cell.y_ref
    for m in range(mode_count):
        km = km_array[m]
        ipm = Ix_sin_sin(km, kp, xa, xb)
        coeffs[m] = ipm * np.cosh(km * y_shift)
        coeffs[m + mode_count] = ipm * np.sinh(km * y_shift)
    return coeffs


def row_contrib_dphi_dy(
    cell: Cell,
    y_iface: float,
    x_seg: Tuple[float, float],
    basis_cfg: BasisConfig,
    p_mode: int,
    eps_scale: float,
) -> np.ndarray:
    """Projected eps * dphi/dy contribution on a horizontal interface."""

    xa, xb = x_seg
    mode_count = basis_cfg.M_modes
    km_array = basis_cfg.km_array
    coeffs = np.zeros(2 * mode_count, dtype=float)
    kp = km_array[p_mode]
    y_shift = y_iface - cell.y_ref
    for m in range(mode_count):
        km = km_array[m]
        ipm = Ix_sin_sin(km, kp, xa, xb)
        coeffs[m] = eps_scale * ipm * km * np.sinh(km * y_shift)
        coeffs[m + mode_count] = eps_scale * ipm * km * np.cosh(km * y_shift)
    return coeffs


def row_contrib_phi_sidewall(
    cell: Cell,
    x_iface: float,
    y_seg: Tuple[float, float],
    basis_cfg: BasisConfig,
    p_mode: int,
    test_family: str = "cosh",
) -> np.ndarray:
    """Projected phi contribution on a vertical interface."""

    ya, yb = y_seg
    ua = ya - cell.y_ref
    ub = yb - cell.y_ref
    mode_count = basis_cfg.M_modes
    km_array = basis_cfg.km_array
    coeffs = np.zeros(2 * mode_count, dtype=float)
    kp = km_array[p_mode]
    for m in range(mode_count):
        km = km_array[m]
        y_aa, y_ab, y_ba, y_bb = _Y_block_components(km, kp, ua, ub)
        x_factor = np.sin(km * x_iface)
        if test_family == "cosh":
            coeffs[m] = x_factor * y_aa
            coeffs[m + mode_count] = x_factor * y_ba
        elif test_family == "sinh":
            coeffs[m] = x_factor * y_ab
            coeffs[m + mode_count] = x_factor * y_bb
        else:
            raise ValueError(f"Unsupported test_family={test_family}")
    return coeffs


def row_contrib_dphi_dx_sidewall(
    cell: Cell,
    x_iface: float,
    y_seg: Tuple[float, float],
    basis_cfg: BasisConfig,
    p_mode: int,
    eps_scale: float,
    test_family: str = "cosh",
) -> np.ndarray:
    """Projected eps * dphi/dx contribution on a vertical interface."""

    ya, yb = y_seg
    ua = ya - cell.y_ref
    ub = yb - cell.y_ref
    mode_count = basis_cfg.M_modes
    km_array = basis_cfg.km_array
    coeffs = np.zeros(2 * mode_count, dtype=float)
    kp = km_array[p_mode]
    for m in range(mode_count):
        km = km_array[m]
        y_aa, y_ab, y_ba, y_bb = _Y_block_components(km, kp, ua, ub)
        x_factor = km * np.cos(km * x_iface)
        if test_family == "cosh":
            coeffs[m] = eps_scale * x_factor * y_aa
            coeffs[m + mode_count] = eps_scale * x_factor * y_ba
        elif test_family == "sinh":
            coeffs[m] = eps_scale * x_factor * y_ab
            coeffs[m + mode_count] = eps_scale * x_factor * y_bb
        else:
            raise ValueError(f"Unsupported test_family={test_family}")
    return coeffs


def add_row(
    row: np.ndarray,
    rhs: float,
    rows: List[np.ndarray],
    rhs_list: List[float],
    stats: Dict[str, int],
    key: str,
) -> None:
    rows.append(row)
    rhs_list.append(rhs)
    stats[key] = stats.get(key, 0) + 1


def build_constraint_template(
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    dof_slices: Dict[int, slice],
    params: Config,
) -> ConstraintTemplate:
    """Build the excitation-invariant constraint matrix once per geometry."""

    slab_cell_map = group_cells_by_slab(cells)
    ndof = max((sl.stop for sl in dof_slices.values()), default=0)
    rows: List[np.ndarray] = []
    rhs_terms: List[ConstraintRhsTerm] = []
    stats: Dict[str, int] = {
        "horizontal_phi_continuity": 0,
        "horizontal_displacement_continuity": 0,
        "metal_dirichlet_horizontal": 0,
        "vertical_phi_continuity": 0,
        "vertical_displacement_continuity": 0,
        "outer_boundary_dirichlet": 0,
        "outer_boundary_x_strong_by_basis": 2,
    }
    slab_names = [name for _, name in sorted({(c.slab_index, c.slab_name) for c in cells})]
    x_full = (0.0, 2.0 * basis_cfg.L)

    def dirichlet_x_projection(cell: Cell) -> Tuple[float, float]:
        if params.outer_dirichlet_projection_mode == "x_full":
            return x_full
        return (float(cell.x1), float(cell.x2))

    def add_template_row(
        row: np.ndarray,
        key: str,
        conductor_id: Optional[int] = None,
        rhs_scale: float = 0.0,
    ) -> None:
        row_index = len(rows)
        rows.append(row)
        stats[key] = stats.get(key, 0) + 1
        if conductor_id is not None and rhs_scale != 0.0:
            rhs_terms.append(
                ConstraintRhsTerm(
                    row_index=row_index,
                    conductor_id=int(conductor_id),
                    scale=float(rhs_scale),
                )
            )

    for slab_idx in range(len(slab_names) - 1):
        slab_low = slab_names[slab_idx]
        slab_up = slab_names[slab_idx + 1]
        lower_cells = slab_cell_map[slab_low]
        upper_cells = slab_cell_map[slab_up]
        for cell_low, cell_up, x_seg in iter_horizontal_overlap_pairs(lower_cells, upper_cells):
            y_iface = cell_low.y2
            if (not cell_low.is_conductor) and (not cell_up.is_conductor):
                sl_low = dof_slices[cell_low.cell_id]
                sl_up = dof_slices[cell_up.cell_id]
                for p_mode in range(basis_cfg.M_modes):
                    row = np.zeros(ndof, dtype=float)
                    row[sl_low] += row_contrib_phi(cell_low, y_iface, x_seg, basis_cfg, p_mode)
                    row[sl_up] -= row_contrib_phi(cell_up, y_iface, x_seg, basis_cfg, p_mode)
                    add_template_row(row, "horizontal_phi_continuity")

                    row = np.zeros(ndof, dtype=float)
                    row[sl_low] += row_contrib_dphi_dy(cell_low, y_iface, x_seg, basis_cfg, p_mode, cell_low.eps_abs)
                    row[sl_up] -= row_contrib_dphi_dy(cell_up, y_iface, x_seg, basis_cfg, p_mode, cell_up.eps_abs)
                    add_template_row(row, "horizontal_displacement_continuity")
            elif (not cell_low.is_conductor) and cell_up.is_conductor:
                sl_low = dof_slices[cell_low.cell_id]
                conductor_id = int(cell_up.conductor_id)
                for p_mode in range(basis_cfg.M_modes):
                    row = np.zeros(ndof, dtype=float)
                    row[sl_low] += row_contrib_phi(cell_low, y_iface, x_seg, basis_cfg, p_mode)
                    kp = basis_cfg.km_array[p_mode]
                    rhs_scale = (-np.cos(kp * x_seg[1]) + np.cos(kp * x_seg[0])) / kp
                    add_template_row(row, "metal_dirichlet_horizontal", conductor_id, rhs_scale)
            elif cell_low.is_conductor and (not cell_up.is_conductor):
                sl_up = dof_slices[cell_up.cell_id]
                conductor_id = int(cell_low.conductor_id)
                for p_mode in range(basis_cfg.M_modes):
                    row = np.zeros(ndof, dtype=float)
                    row[sl_up] += row_contrib_phi(cell_up, y_iface, x_seg, basis_cfg, p_mode)
                    kp = basis_cfg.km_array[p_mode]
                    rhs_scale = (-np.cos(kp * x_seg[1]) + np.cos(kp * x_seg[0])) / kp
                    add_template_row(row, "metal_dirichlet_horizontal", conductor_id, rhs_scale)

    for slab_name in slab_names:
        slab_cells = slab_cell_map[slab_name]
        for cell_left, cell_right in zip(slab_cells[:-1], slab_cells[1:]):
            x_iface = cell_left.x2
            y_seg = (cell_left.y1, cell_left.y2)
            if (not cell_left.is_conductor) and (not cell_right.is_conductor):
                sl_left = dof_slices[cell_left.cell_id]
                sl_right = dof_slices[cell_right.cell_id]
                for p_mode in range(basis_cfg.M_modes):
                    for family in ("cosh", "sinh"):
                        row = np.zeros(ndof, dtype=float)
                        row[sl_left] += row_contrib_phi_sidewall(cell_left, x_iface, y_seg, basis_cfg, p_mode, family)
                        row[sl_right] -= row_contrib_phi_sidewall(cell_right, x_iface, y_seg, basis_cfg, p_mode, family)
                        add_template_row(row, "vertical_phi_continuity")

                        row = np.zeros(ndof, dtype=float)
                        row[sl_left] += row_contrib_dphi_dx_sidewall(
                            cell_left, x_iface, y_seg, basis_cfg, p_mode, cell_left.eps_abs, family
                        )
                        row[sl_right] -= row_contrib_dphi_dx_sidewall(
                            cell_right, x_iface, y_seg, basis_cfg, p_mode, cell_right.eps_abs, family
                        )
                        add_template_row(row, "vertical_displacement_continuity")

    bottom_slab = slab_names[0]
    top_slab = slab_names[-1]
    for cell_bottom in slab_cell_map[bottom_slab]:
        if not cell_bottom.is_conductor:
            sl = dof_slices[cell_bottom.cell_id]
            x_seg_bottom = dirichlet_x_projection(cell_bottom)
            for p_mode in range(basis_cfg.M_modes):
                row = np.zeros(ndof, dtype=float)
                row[sl] += row_contrib_phi(cell_bottom, cell_bottom.y1, x_seg_bottom, basis_cfg, p_mode)
                add_template_row(row, "outer_boundary_dirichlet")
    for cell_top in slab_cell_map[top_slab]:
        if not cell_top.is_conductor:
            sl = dof_slices[cell_top.cell_id]
            x_seg_top = dirichlet_x_projection(cell_top)
            for p_mode in range(basis_cfg.M_modes):
                row = np.zeros(ndof, dtype=float)
                row[sl] += row_contrib_phi(cell_top, cell_top.y2, x_seg_top, basis_cfg, p_mode)
                add_template_row(row, "outer_boundary_dirichlet")

    C_mat = np.vstack(rows) if rows else np.zeros((0, ndof), dtype=float)
    return ConstraintTemplate(
        C_mat=C_mat,
        rhs_terms=tuple(rhs_terms),
        stats=stats,
    )


def build_constraint_rhs(
    template: ConstraintTemplate,
    excitation_case: Dict[int, float],
) -> np.ndarray:
    """Build one RHS vector from a cached constraint template."""

    d_vec = np.zeros(template.C_mat.shape[0], dtype=float)
    for term in template.rhs_terms:
        d_vec[term.row_index] += float(excitation_case.get(term.conductor_id, 0.0)) * term.scale
    return d_vec


def assemble_constraints(
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    dof_slices: Dict[int, slice],
    excitation_case: Dict[int, float],
    params: Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Assemble all KKT constraints Cx=d."""

    template = build_constraint_template(cells, basis_cfg, dof_slices, params)
    d_vec = build_constraint_rhs(template, excitation_case)
    return template.C_mat, d_vec, dict(template.stats)


def prepare_reduced_constraint_system(C: np.ndarray) -> ReducedConstraintSystem:
    """Compress Cx=d into Q^T d = Rx to reduce the KKT size for cold-start runs."""

    if C.ndim != 2:
        raise ValueError("Constraint matrix must be two-dimensional.")
    row_norms = np.linalg.norm(C, axis=1) if C.size else np.zeros(C.shape[0], dtype=float)
    nonzero_row_mask = row_norms > 1e-14
    reduced_rows = int(np.count_nonzero(nonzero_row_mask))
    if reduced_rows == 0:
        return ReducedConstraintSystem(
            nonzero_row_mask=nonzero_row_mask,
            q_transpose=np.zeros((0, 0), dtype=float),
            reduced_C=np.zeros((0, C.shape[1]), dtype=float),
        )
    C_nonzero = np.asarray(C[nonzero_row_mask], dtype=float)
    Q, R = np.linalg.qr(C_nonzero, mode="reduced")
    return ReducedConstraintSystem(
        nonzero_row_mask=nonzero_row_mask,
        q_transpose=np.asarray(Q.T, dtype=float),
        reduced_C=np.asarray(R, dtype=float),
    )


def reduce_constraint_rhs(
    reduced_system: ReducedConstraintSystem,
    d_matrix: np.ndarray,
) -> np.ndarray:
    """Project one or more RHS columns into the reduced constraint coordinates."""

    d_arr = np.asarray(d_matrix, dtype=float)
    if d_arr.ndim == 1:
        d_arr = d_arr[:, None]
    if reduced_system.nonzero_row_mask.size:
        d_arr = d_arr[reduced_system.nonzero_row_mask]
    if reduced_system.q_transpose.size == 0:
        return np.zeros((0, d_arr.shape[1]), dtype=float)
    return reduced_system.q_transpose @ d_arr


def _scipy_lstsq_fallback(KKT: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Lazy SciPy fallback kept only for unexpected singular solve cases."""

    from scipy.linalg import lstsq as scipy_lstsq

    sol, *_ = scipy_lstsq(
        KKT,
        rhs,
        lapack_driver="gelsy",
        check_finite=False,
        overwrite_a=True,
        overwrite_b=True,
    )
    return sol


def solve_kkt(
    M: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    reduced_system: Optional[ReducedConstraintSystem] = None,
    warn: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the saddle-point KKT system [M C^T; C 0] [x;lambda] = [0; d]."""

    n = M.shape[0]
    if C.shape[1] != n:
        raise ValueError("Constraint matrix width does not match energy matrix size.")
    if C.shape[0] != d.shape[0]:
        raise ValueError("Constraint RHS length does not match row count.")

    if reduced_system is None:
        reduced_system = prepare_reduced_constraint_system(C)
    C_reduced = reduced_system.reduced_C
    d_reduced = reduce_constraint_rhs(reduced_system, d)[:, 0]
    m_rows = C_reduced.shape[0]
    KKT = np.zeros((n + m_rows, n + m_rows), dtype=float, order="F")
    KKT[:n, :n] = M
    KKT[:n, n:] = C_reduced.T
    KKT[n:, :n] = C_reduced
    rhs = np.empty(n + m_rows, dtype=float, order="F")
    rhs[:n] = 0.0
    rhs[n:] = d_reduced
    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        sol = _scipy_lstsq_fallback(KKT, rhs)
    x = sol[:n]
    lambda_vec = sol[n:]

    con_res = np.linalg.norm(C @ x - d) if C.size else 0.0
    eq_res = np.linalg.norm(M @ x + C_reduced.T @ lambda_vec)
    if warn and (con_res > 1e-6 or eq_res > 1e-6):
        print(f"[warn] KKT residuals: ||Cx-d||={con_res:.3e}, ||Mx+C^T lambda||={eq_res:.3e}")
    return x, lambda_vec


def solve_kkt_multi(
    M: np.ndarray,
    C: np.ndarray,
    d_matrix: np.ndarray,
    reduced_system: Optional[ReducedConstraintSystem] = None,
    warn: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the KKT system for one or more RHS columns in a single factorization."""

    d_arr = np.asarray(d_matrix, dtype=float)
    squeeze_output = d_arr.ndim == 1
    if squeeze_output:
        d_arr = d_arr[:, None]

    n = M.shape[0]
    if C.shape[1] != n:
        raise ValueError("Constraint matrix width does not match energy matrix size.")
    if C.shape[0] != d_arr.shape[0]:
        raise ValueError("Constraint RHS row count does not match constraint matrix row count.")

    if reduced_system is None:
        reduced_system = prepare_reduced_constraint_system(C)
    C_reduced = reduced_system.reduced_C
    d_reduced = reduce_constraint_rhs(reduced_system, d_arr)

    m_rows = C_reduced.shape[0]
    KKT = np.zeros((n + m_rows, n + m_rows), dtype=float, order="F")
    KKT[:n, :n] = M
    KKT[:n, n:] = C_reduced.T
    KKT[n:, :n] = C_reduced
    rhs = np.empty((n + m_rows, d_reduced.shape[1]), dtype=float, order="F")
    rhs[:n, :] = 0.0
    rhs[n:, :] = d_reduced
    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        sol = _scipy_lstsq_fallback(KKT, rhs)

    x = sol[:n]
    lambda_mat = sol[n:]

    if warn:
        con_residuals = np.linalg.norm(C @ x - d_arr, axis=0) if C.size else np.zeros(d_arr.shape[1], dtype=float)
        eq_residuals = np.linalg.norm(M @ x + C_reduced.T @ lambda_mat, axis=0)
        for idx, (con_res, eq_res) in enumerate(zip(con_residuals, eq_residuals)):
            if con_res > 1e-6 or eq_res > 1e-6:
                print(
                    f"[warn] KKT residuals rhs[{idx}]: "
                    f"||Cx-d||={con_res:.3e}, ||Mx+C^T lambda||={eq_res:.3e}"
                )

    if squeeze_output:
        return x[:, 0], lambda_mat[:, 0]
    return x, lambda_mat


def _locate_cell_in_slab(entry: SlabLocatorEntry, x: float, y: float, tol: float) -> Optional[Cell]:
    """Locate a point inside one slab entry."""

    if not ((entry.y1 - tol) <= y <= (entry.y2 + tol)):
        return None
    idx = bisect_right(entry.x_starts, x) - 1
    candidate_indices = [idx, idx + 1, idx - 1]
    seen: set[int] = set()
    for candidate_idx in candidate_indices:
        if candidate_idx in seen or candidate_idx < 0 or candidate_idx >= len(entry.cells):
            continue
        seen.add(candidate_idx)
        cell = entry.cells[candidate_idx]
        if (cell.x1 - tol) <= x <= (cell.x2 + tol) and (cell.y1 - tol) <= y <= (cell.y2 + tol):
            return cell
    return None


def locate_cell(cells: Sequence[Cell], x: float, y: float, locator: Optional[SpatialLocator] = None) -> Optional[Cell]:
    """Return the cell containing (x, y), preferring interior points."""

    tol = 1e-12
    if locator is not None and locator.slabs:
        slab_idx = bisect_right(locator.y_starts, y) - 1
        candidate_slab_indices = [slab_idx, slab_idx + 1, slab_idx - 1]
        seen: set[int] = set()
        for candidate_slab_idx in candidate_slab_indices:
            if candidate_slab_idx in seen or candidate_slab_idx < 0 or candidate_slab_idx >= len(locator.slabs):
                continue
            seen.add(candidate_slab_idx)
            hit = _locate_cell_in_slab(locator.slabs[candidate_slab_idx], x, y, tol)
            if hit is not None:
                return hit
    for cell in cells:
        x_ok = (cell.x1 - tol) <= x <= (cell.x2 + tol)
        y_ok = (cell.y1 - tol) <= y <= (cell.y2 + tol)
        if x_ok and y_ok:
            return cell
    return None


def make_phi_function(
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    x_big: np.ndarray,
    dof_slices: Dict[int, slice],
    excitation_case: Dict[int, float],
):
    """Return callable phi(x,y), grad_phi(x,y) using the solved coefficients."""

    locator = build_spatial_locator(cells)

    def phi_xy(x: float, y: float) -> float:
        cell = locate_cell(cells, x, y, locator)
        if cell is None:
            return 0.0
        if cell.is_conductor:
            return excitation_case.get(int(cell.conductor_id), 0.0)
        sl = dof_slices[cell.cell_id]
        coeffs = x_big[sl]
        total = 0.0
        for m in range(basis_cfg.M_modes):
            km = basis_cfg.km_array[m]
            A = coeffs[m]
            B = coeffs[m + basis_cfg.M_modes]
            y_part = A * np.cosh(km * (y - cell.y_ref)) + B * np.sinh(km * (y - cell.y_ref))
            total += y_part * np.sin(km * x)
        return float(total)

    def grad_phi_xy(x: float, y: float) -> Tuple[float, float]:
        cell = locate_cell(cells, x, y, locator)
        if cell is None or cell.is_conductor:
            return 0.0, 0.0
        sl = dof_slices[cell.cell_id]
        coeffs = x_big[sl]
        dphix = 0.0
        dphiy = 0.0
        for m in range(basis_cfg.M_modes):
            km = basis_cfg.km_array[m]
            A = coeffs[m]
            B = coeffs[m + basis_cfg.M_modes]
            y_part = A * np.cosh(km * (y - cell.y_ref)) + B * np.sinh(km * (y - cell.y_ref))
            y_part_dy = A * km * np.sinh(km * (y - cell.y_ref)) + B * km * np.cosh(km * (y - cell.y_ref))
            dphix += y_part * km * np.cos(km * x)
            dphiy += y_part_dy * np.sin(km * x)
        return float(dphix), float(dphiy)

    return phi_xy, grad_phi_xy


def make_grad_eps_sampler(
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    x_big: np.ndarray,
    dof_slices: Dict[int, slice],
):
    """Return a fast dielectric-side sampler for grad(phi) and local eps."""

    dielectric_cells = tuple(cell for cell in cells if not cell.is_conductor)
    locator = build_spatial_locator(dielectric_cells)
    km_array = basis_cfg.km_array
    mode_count = basis_cfg.M_modes
    coeff_cache: Dict[int, np.ndarray] = {}

    def sample_grad_eps(x: float, y: float) -> Tuple[float, float, float]:
        cell = locate_cell(dielectric_cells, x, y, locator)
        if cell is None:
            return 0.0, 0.0, float(EPS0)
        coeffs = coeff_cache.get(cell.cell_id)
        if coeffs is None:
            coeffs = np.asarray(x_big[dof_slices[cell.cell_id]], dtype=float)
            coeff_cache[cell.cell_id] = coeffs
        A = coeffs[:mode_count]
        B = coeffs[mode_count:]
        y_shift = float(y - cell.y_ref)
        km_y = km_array * y_shift
        sinh_terms = np.sinh(km_y)
        cosh_terms = np.cosh(km_y)
        sin_terms = np.sin(km_array * float(x))
        cos_terms = np.cos(km_array * float(x))
        y_part = A * cosh_terms + B * sinh_terms
        y_part_dy = A * km_array * sinh_terms + B * km_array * cosh_terms
        dphix = float(np.dot(y_part * km_array, cos_terms))
        dphiy = float(np.dot(y_part_dy, sin_terms))
        return dphix, dphiy, float(cell.eps_abs)

    return sample_grad_eps


def evaluate_potential(
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    x_big: np.ndarray,
    dof_slices: Dict[int, slice],
    excitation_case: Dict[int, float],
    plot_resolution: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the potential on a regular grid for plotting."""

    x_min = min(cell.x1 for cell in cells)
    x_max = max(cell.x2 for cell in cells)
    y_min = min(cell.y1 for cell in cells)
    y_max = max(cell.y2 for cell in cells)
    grid_x = np.linspace(x_min, x_max, plot_resolution)
    grid_y = np.linspace(y_min, y_max, plot_resolution)
    phi_xy, _ = make_phi_function(cells, basis_cfg, x_big, dof_slices, excitation_case)
    phi_grid = np.zeros((plot_resolution, plot_resolution), dtype=float)
    for iy, yy in enumerate(grid_y):
        for ix, xx in enumerate(grid_x):
            phi_grid[iy, ix] = phi_xy(xx, yy)
    return grid_x, grid_y, phi_grid


def _gauss_integrate_1d(fun, a: float, b: float, order: int) -> float:
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(order)
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    x = mid + half * nodes
    vals = np.asarray([fun(xx) for xx in x], dtype=float)
    return float(half * np.dot(weights, vals))


def _make_clustered_edge_grid(sample_count: int, cluster_power: float = CHARGE_EDGE_CLUSTER_POWER) -> np.ndarray:
    """Cluster line-integral samples toward the edge endpoints, like Cal_0313."""

    cache_key = (int(sample_count), float(cluster_power))
    cached = EDGE_GRID_CACHE.get(cache_key)
    if cached is not None:
        return cached
    sample_count = max(int(sample_count), 3)
    u = np.linspace(0.0, 1.0, sample_count)
    left = 0.5 * np.power(2.0 * u, cluster_power)
    right = 1.0 - 0.5 * np.power(2.0 * (1.0 - u), cluster_power)
    grid = np.where(u <= 0.5, left, right)
    grid[0] = 0.0
    grid[-1] = 1.0
    EDGE_GRID_CACHE[cache_key] = grid
    return grid


def get_adjacent_cell(cells: Sequence[Cell], cell: Cell, edge: str) -> Optional[Cell]:
    """Return adjacent cell across one rectangle edge."""

    slab_cell_map = group_cells_by_slab(cells)
    slab_names = [name for _, name in sorted({(c.slab_index, c.slab_name) for c in cells})]
    slab_cells = slab_cell_map[cell.slab_name]
    tol = 1e-18
    if edge == "top":
        if cell.slab_index >= len(slab_names) - 1:
            return None
        candidates = slab_cell_map[slab_names[cell.slab_index + 1]]
        for cand in candidates:
            if min(cell.x2, cand.x2) > max(cell.x1, cand.x1) + tol:
                return cand
        return None
    if edge == "bottom":
        if cell.slab_index <= 0:
            return None
        candidates = slab_cell_map[slab_names[cell.slab_index - 1]]
        for cand in candidates:
            if min(cell.x2, cand.x2) > max(cell.x1, cand.x1) + tol:
                return cand
        return None
    if edge == "left":
        idx = slab_cells.index(cell)
        if idx <= 0:
            return None
        return slab_cells[idx - 1]
    if edge == "right":
        idx = slab_cells.index(cell)
        if idx >= len(slab_cells) - 1:
            return None
        return slab_cells[idx + 1]
    raise ValueError(f"Unknown edge {edge}")


def build_charge_base_plan(
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
) -> Tuple[ChargeEdgeBasePlan, ...]:
    """Precompute geometry-only contour factors reused across dielectric assignments."""

    dielectric_cells = tuple(cell for cell in cells if not cell.is_conductor)
    locator = build_spatial_locator(dielectric_cells)
    km_array = np.asarray(basis_cfg.km_array, dtype=float)
    domain_x_min = min(float(cell.x1) for cell in cells)
    domain_x_max = max(float(cell.x2) for cell in cells)
    domain_y_min = min(float(cell.y1) for cell in cells)
    domain_y_max = max(float(cell.y2) for cell in cells)

    def intervals_overlap(a1: float, a2: float, b1: float, b2: float) -> bool:
        return min(float(a2), float(b2)) > max(float(a1), float(b1))

    def summarize_edge(
        x_mid: np.ndarray,
        y_mid: np.ndarray,
        seg_lens: np.ndarray,
        nx: float,
        ny: float,
    ) -> Tuple[ChargeBaseFactorGroup, ...]:
        grouped_samples: Dict[int, List[object]] = {}
        for xb, yb, seg_len in zip(x_mid, y_mid, seg_lens):
            cell = locate_cell(dielectric_cells, float(xb), float(yb), locator)
            if cell is None:
                continue
            bucket = grouped_samples.get(cell.cell_id)
            if bucket is None:
                bucket = [cell, [], [], []]
                grouped_samples[cell.cell_id] = bucket
            bucket[1].append(float(xb))
            bucket[2].append(float(yb))
            bucket[3].append(float(seg_len))

        grouped: Dict[int, np.ndarray] = {}
        for cell_id, (cell, xs, ys, lens) in grouped_samples.items():
            xs_arr = np.asarray(xs, dtype=float)
            ys_arr = np.asarray(ys, dtype=float)
            lens_arr = np.asarray(lens, dtype=float)
            xk = xs_arr[:, None] * km_array[None, :]
            yk = (ys_arr - float(cell.y_ref))[:, None] * km_array[None, :]
            sin_terms = np.sin(xk)
            cos_terms = np.cos(xk)
            sinh_terms = np.sinh(yk)
            cosh_terms = np.cosh(yk)
            factor_scale = (-lens_arr)[:, None]
            factor_a = np.sum(
                factor_scale
                * (
                    nx * km_array[None, :] * cosh_terms * cos_terms
                    + ny * km_array[None, :] * sinh_terms * sin_terms
                ),
                axis=0,
            )
            factor_b = np.sum(
                factor_scale
                * (
                    nx * km_array[None, :] * sinh_terms * cos_terms
                    + ny * km_array[None, :] * cosh_terms * sin_terms
                ),
                axis=0,
            )
            grouped[int(cell_id)] = np.concatenate([factor_a, factor_b])
        return tuple(
            ChargeBaseFactorGroup(cell_id=int(cell_id), base_factor_vec=base_factor_vec)
            for cell_id, base_factor_vec in sorted(grouped.items())
        )

    plans: List[ChargeEdgeBasePlan] = []
    for cell in cells:
        if not cell.is_conductor:
            continue

        conductor_name = cell.conductor_name or f"metal_{cell.conductor_id}"
        conductor_id = int(cell.conductor_id)
        x1, x2 = float(cell.x1), float(cell.x2)
        y1, y2 = float(cell.y1), float(cell.y2)
        width = max(x2 - x1, 1e-12)
        height = max(y2 - y1, 1e-12)
        sample_offset = max(2.5e-4 * max(width, height), 1e-18)

        clear_left = max(x1 - domain_x_min, sample_offset)
        clear_right = max(domain_x_max - x2, sample_offset)
        clear_bottom = max(y1 - domain_y_min, sample_offset)
        clear_top = max(domain_y_max - y2, sample_offset)

        for other in cells:
            if not other.is_conductor or int(other.conductor_id) == conductor_id:
                continue
            ox1, ox2 = float(other.x1), float(other.x2)
            oy1, oy2 = float(other.y1), float(other.y2)
            if ox2 <= x1 and intervals_overlap(y1, y2, oy1, oy2):
                clear_left = min(clear_left, max(x1 - ox2, sample_offset))
            if ox1 >= x2 and intervals_overlap(y1, y2, oy1, oy2):
                clear_right = min(clear_right, max(ox1 - x2, sample_offset))
            if oy2 <= y1 and intervals_overlap(x1, x2, ox1, ox2):
                clear_bottom = min(clear_bottom, max(y1 - oy2, sample_offset))
            if oy1 >= y2 and intervals_overlap(x1, x2, ox1, ox2):
                clear_top = min(clear_top, max(oy1 - y2, sample_offset))

        margin_x = min(max(0.05 * width, 8.0 * sample_offset), 0.4 * min(clear_left, clear_right))
        margin_y = min(max(0.50 * height, 8.0 * sample_offset), 0.4 * min(clear_bottom, clear_top))
        contour_x1 = x1 - margin_x
        contour_x2 = x2 + margin_x
        contour_y1 = y1 - margin_y
        contour_y2 = y2 + margin_y

        edge_specs = (
            ("bottom", CHARGE_EDGE_BASE_SAMPLES, 0.0, -1.0),
            ("top", CHARGE_EDGE_DENSE_SAMPLES, 0.0, +1.0),
            ("left", CHARGE_EDGE_DENSE_SAMPLES, -1.0, 0.0),
            ("right", CHARGE_EDGE_DENSE_SAMPLES, +1.0, 0.0),
        )
        for edge_name, sample_count, nx, ny in edge_specs:
            ts = _make_clustered_edge_grid(sample_count)
            t1 = ts[:-1]
            t2 = ts[1:]
            tm = 0.5 * (t1 + t2)
            if edge_name in ("bottom", "top"):
                xa = contour_x1
                xb = contour_x2
                y_const = contour_y1 if edge_name == "bottom" else contour_y2
                x1s = xa + (xb - xa) * t1
                x2s = xa + (xb - xa) * t2
                x_mid = xa + (xb - xa) * tm
                y1s = np.full_like(x1s, y_const)
                y2s = np.full_like(x2s, y_const)
                y_mid = np.full_like(x_mid, y_const)
            else:
                ya = contour_y1
                yb = contour_y2
                x_const = contour_x1 if edge_name == "left" else contour_x2
                y1s = ya + (yb - ya) * t1
                y2s = ya + (yb - ya) * t2
                y_mid = ya + (yb - ya) * tm
                x1s = np.full_like(y1s, x_const)
                x2s = np.full_like(y2s, x_const)
                x_mid = np.full_like(y_mid, x_const)
            seg_lens = np.sqrt((x2s - x1s) ** 2 + (y2s - y1s) ** 2)
            plans.append(
                ChargeEdgeBasePlan(
                    conductor_name=conductor_name,
                    conductor_id=conductor_id,
                    edge_name=edge_name,
                    factor_groups=summarize_edge(x_mid, y_mid, seg_lens, nx, ny),
                )
            )
    return tuple(plans)


def build_charge_integration_plan(
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    charge_base_plans: Optional[Sequence[ChargeEdgeBasePlan]] = None,
) -> Tuple[ChargeEdgePlan, ...]:
    """Precompute contour charge-integration factors for one fixed geometry."""

    base_plans = tuple(charge_base_plans) if charge_base_plans is not None else build_charge_base_plan(cells, basis_cfg)
    eps_by_cell_id = {int(cell.cell_id): float(cell.eps_abs) for cell in cells if not cell.is_conductor}
    plans: List[ChargeEdgePlan] = []
    for base_plan in base_plans:
        plans.append(
            ChargeEdgePlan(
                conductor_name=base_plan.conductor_name,
                conductor_id=base_plan.conductor_id,
                edge_name=base_plan.edge_name,
                factor_groups=tuple(
                    ChargeFactorGroup(
                        cell_id=group.cell_id,
                        factor_vec=float(eps_by_cell_id[group.cell_id]) * group.base_factor_vec,
                    )
                    for group in base_plan.factor_groups
                ),
            )
        )
    return tuple(plans)


def compute_conductor_charges_from_plan(
    charge_plans: Sequence[ChargeEdgePlan],
    x_big: np.ndarray,
    dof_slices: Dict[int, slice],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Fast charge evaluation using pre-integrated contour factors."""

    charges: Dict[str, float] = {"Q1": 0.0, "Q2": 0.0, "Qgnd": 0.0}
    breakdown: Dict[str, Dict[str, float]] = {}
    conductor_totals: Dict[int, float] = {}
    coeff_cache: Dict[int, np.ndarray] = {}

    for plan in charge_plans:
        edge_q = 0.0
        for group in plan.factor_groups:
            coeffs = coeff_cache.get(group.cell_id)
            if coeffs is None:
                coeffs = np.asarray(x_big[dof_slices[group.cell_id]], dtype=float)
                coeff_cache[group.cell_id] = coeffs
            edge_q += float(np.dot(group.factor_vec, coeffs))
        edge_info = breakdown.setdefault(
            plan.conductor_name,
            {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0, "total": 0.0},
        )
        edge_info[plan.edge_name] += edge_q
        edge_info["total"] += edge_q
        conductor_totals[plan.conductor_id] = conductor_totals.get(plan.conductor_id, 0.0) + edge_q

    charges["Q1"] = conductor_totals.get(CONDUCTOR_VOLTAGE_KEYS["Metal1"], 0.0)
    charges["Q2"] = conductor_totals.get(CONDUCTOR_VOLTAGE_KEYS["Metal2"], 0.0)
    charges["Qgnd"] = conductor_totals.get(CONDUCTOR_VOLTAGE_KEYS["MetalGND"], 0.0)
    return charges, breakdown


def compute_line_charge_per_conductor(
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    x_big: np.ndarray,
    dof_slices: Dict[int, slice],
    excitation_case: Dict[int, float],
    quadrature_order: int,
) -> Tuple[Dict[int, float], Dict[str, Dict[str, float]]]:
    """Compute conductor charges by dielectric-side displacement flux.

    Sign convention used here:
    - n_metal is the outward normal from metal into the adjacent dielectric.
    - D = eps * E with E = -grad(phi).
    - We use Q = ∮ (D · n_metal) dl.
    This convention gives positive diagonal C' entries for positively excited conductors.
    """

    _, grad_phi_xy = make_phi_function(cells, basis_cfg, x_big, dof_slices, excitation_case)
    q_by_conductor: Dict[int, float] = {}
    breakdown: Dict[str, Dict[str, float]] = {}
    x_span = max(cell.x2 for cell in cells) - min(cell.x1 for cell in cells)
    y_span = max(cell.y2 for cell in cells) - min(cell.y1 for cell in cells)
    offset = 1e-6 * max(x_span, y_span)

    for cell in cells:
        if not cell.is_conductor:
            continue
        conductor_key = cell.conductor_name or f"metal_{cell.conductor_id}"
        breakdown.setdefault(conductor_key, {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0, "total": 0.0})
        q_total_cell = 0.0
        for edge in ("top", "bottom", "left", "right"):
            neighbor = get_adjacent_cell(cells, cell, edge)
            if neighbor is None or neighbor.is_conductor:
                continue
            eps_edge = neighbor.eps_abs
            if edge == "top":
                a, b = cell.x1, cell.x2
                y_edge = cell.y2
                nvec = (0.0, 1.0)  # outward from metal into dielectric
                def integrand(xx: float) -> float:
                    dphix, dphiy = grad_phi_xy(xx, y_edge + offset)
                    Ex, Ey = -dphix, -dphiy
                    return eps_edge * (Ex * nvec[0] + Ey * nvec[1])
            elif edge == "bottom":
                a, b = cell.x1, cell.x2
                y_edge = cell.y1
                nvec = (0.0, -1.0)  # outward from metal into dielectric
                def integrand(xx: float) -> float:
                    dphix, dphiy = grad_phi_xy(xx, y_edge - offset)
                    Ex, Ey = -dphix, -dphiy
                    return eps_edge * (Ex * nvec[0] + Ey * nvec[1])
            elif edge == "left":
                a, b = cell.y1, cell.y2
                x_edge = cell.x1
                nvec = (-1.0, 0.0)  # outward from metal into dielectric
                def integrand(yy: float) -> float:
                    dphix, dphiy = grad_phi_xy(x_edge - offset, yy)
                    Ex, Ey = -dphix, -dphiy
                    return eps_edge * (Ex * nvec[0] + Ey * nvec[1])
            else:
                a, b = cell.y1, cell.y2
                x_edge = cell.x2
                nvec = (1.0, 0.0)  # outward from metal into dielectric
                def integrand(yy: float) -> float:
                    dphix, dphiy = grad_phi_xy(x_edge + offset, yy)
                    Ex, Ey = -dphix, -dphiy
                    return eps_edge * (Ex * nvec[0] + Ey * nvec[1])
            q_edge = _gauss_integrate_1d(integrand, a, b, quadrature_order)
            breakdown[conductor_key][edge] += q_edge
            q_total_cell += q_edge

        breakdown[conductor_key]["total"] += q_total_cell
        q_by_conductor[int(cell.conductor_id)] = q_by_conductor.get(int(cell.conductor_id), 0.0) + q_total_cell

    if "MetalGND" in breakdown:
        breakdown["MetalGND"]["total"] = (
            breakdown["MetalGND"]["top"]
            + breakdown["MetalGND"]["bottom"]
            + breakdown["MetalGND"]["left"]
            + breakdown["MetalGND"]["right"]
        )
    return q_by_conductor, breakdown


def compute_line_charge_per_conductor_cal0313_style(
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    x_big: np.ndarray,
    dof_slices: Dict[int, slice],
    excitation_case: Dict[int, float],
) -> Tuple[Dict[int, float], Dict[str, Dict[str, float]]]:
    """Contour-based charge extraction matching the current Cal_0313 idea."""

    sample_grad_eps = make_grad_eps_sampler(cells, basis_cfg, x_big, dof_slices)
    q_by_conductor: Dict[int, float] = {}
    breakdown: Dict[str, Dict[str, float]] = {}
    domain_x_min = min(float(cell.x1) for cell in cells)
    domain_x_max = max(float(cell.x2) for cell in cells)
    domain_y_min = min(float(cell.y1) for cell in cells)
    domain_y_max = max(float(cell.y2) for cell in cells)

    def intervals_overlap(a1: float, a2: float, b1: float, b2: float) -> bool:
        return min(float(a2), float(b2)) > max(float(a1), float(b1))

    def integrate_contour_edge(x_func, y_func, nx: float, ny: float, sample_count: int) -> float:
        ts = _make_clustered_edge_grid(sample_count)
        t1 = ts[:-1]
        t2 = ts[1:]
        tm = 0.5 * (t1 + t2)
        x1s = np.asarray([float(x_func(val)) for val in t1], dtype=float)
        y1s = np.asarray([float(y_func(val)) for val in t1], dtype=float)
        x2s = np.asarray([float(x_func(val)) for val in t2], dtype=float)
        y2s = np.asarray([float(y_func(val)) for val in t2], dtype=float)
        xbs = np.asarray([float(x_func(val)) for val in tm], dtype=float)
        ybs = np.asarray([float(y_func(val)) for val in tm], dtype=float)
        seg_lens = np.sqrt((x2s - x1s) ** 2 + (y2s - y1s) ** 2)

        q_accum = 0.0
        for xb, yb, seg_len in zip(xbs, ybs, seg_lens):
            dphix, dphiy, eps_local = sample_grad_eps(float(xb), float(yb))
            Ex = -dphix
            Ey = -dphiy
            En = Ex * nx + Ey * ny
            sigma = eps_local * En
            q_accum += sigma * float(seg_len)
        return float(q_accum)

    for cell in cells:
        if not cell.is_conductor:
            continue

        conductor_key = cell.conductor_name or f"metal_{cell.conductor_id}"
        breakdown.setdefault(
            conductor_key,
            {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0, "total": 0.0},
        )

        x1, x2 = float(cell.x1), float(cell.x2)
        y1, y2 = float(cell.y1), float(cell.y2)
        width = max(x2 - x1, 1e-12)
        height = max(y2 - y1, 1e-12)
        sample_offset = max(2.5e-4 * max(width, height), 1e-18)

        clear_left = max(x1 - domain_x_min, sample_offset)
        clear_right = max(domain_x_max - x2, sample_offset)
        clear_bottom = max(y1 - domain_y_min, sample_offset)
        clear_top = max(domain_y_max - y2, sample_offset)

        for other in cells:
            if not other.is_conductor or int(other.conductor_id) == int(cell.conductor_id):
                continue
            ox1, ox2 = float(other.x1), float(other.x2)
            oy1, oy2 = float(other.y1), float(other.y2)
            if ox2 <= x1 and intervals_overlap(y1, y2, oy1, oy2):
                clear_left = min(clear_left, max(x1 - ox2, sample_offset))
            if ox1 >= x2 and intervals_overlap(y1, y2, oy1, oy2):
                clear_right = min(clear_right, max(ox1 - x2, sample_offset))
            if oy2 <= y1 and intervals_overlap(x1, x2, ox1, ox2):
                clear_bottom = min(clear_bottom, max(y1 - oy2, sample_offset))
            if oy1 >= y2 and intervals_overlap(x1, x2, ox1, ox2):
                clear_top = min(clear_top, max(oy1 - y2, sample_offset))

        margin_x = min(max(0.05 * width, 8.0 * sample_offset), 0.4 * min(clear_left, clear_right))
        margin_y = min(max(0.50 * height, 8.0 * sample_offset), 0.4 * min(clear_bottom, clear_top))
        contour_x1 = x1 - margin_x
        contour_x2 = x2 + margin_x
        contour_y1 = y1 - margin_y
        contour_y2 = y2 + margin_y

        def xb(t: float) -> float:
            return contour_x1 + (contour_x2 - contour_x1) * t

        def yb(_: float) -> float:
            return contour_y1

        def xt(t: float) -> float:
            return contour_x1 + (contour_x2 - contour_x1) * t

        def yt(_: float) -> float:
            return contour_y2

        def xl(_: float) -> float:
            return contour_x1

        def yl(t: float) -> float:
            return contour_y1 + (contour_y2 - contour_y1) * t

        def xr(_: float) -> float:
            return contour_x2

        def yr(t: float) -> float:
            return contour_y1 + (contour_y2 - contour_y1) * t

        q_bottom = integrate_contour_edge(xb, yb, nx=0.0, ny=-1.0, sample_count=CHARGE_EDGE_BASE_SAMPLES)
        q_top = integrate_contour_edge(xt, yt, nx=0.0, ny=+1.0, sample_count=CHARGE_EDGE_DENSE_SAMPLES)
        q_left = integrate_contour_edge(xl, yl, nx=-1.0, ny=0.0, sample_count=CHARGE_EDGE_DENSE_SAMPLES)
        q_right = integrate_contour_edge(xr, yr, nx=+1.0, ny=0.0, sample_count=CHARGE_EDGE_DENSE_SAMPLES)

        breakdown[conductor_key]["bottom"] += q_bottom
        breakdown[conductor_key]["top"] += q_top
        breakdown[conductor_key]["left"] += q_left
        breakdown[conductor_key]["right"] += q_right
        breakdown[conductor_key]["total"] += q_bottom + q_top + q_left + q_right

        cid = int(cell.conductor_id)
        q_by_conductor[cid] = q_by_conductor.get(cid, 0.0) + q_bottom + q_top + q_left + q_right

    return q_by_conductor, breakdown


def compute_conductor_charges(
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    x_big: np.ndarray,
    dof_slices: Dict[int, slice],
    excitation_case: Dict[int, float],
    quadrature_order: int,
    charge_plans: Optional[Sequence[ChargeEdgePlan]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """User-facing wrapper for conductor charges."""

    if charge_plans is not None:
        return compute_conductor_charges_from_plan(charge_plans, x_big, dof_slices)

    q_by_id, breakdown = compute_line_charge_per_conductor_cal0313_style(
        cells, basis_cfg, x_big, dof_slices, excitation_case
    )
    charges = {
        "Q1": q_by_id.get(CONDUCTOR_VOLTAGE_KEYS["Metal1"], 0.0),
        "Q2": q_by_id.get(CONDUCTOR_VOLTAGE_KEYS["Metal2"], 0.0),
        "Qgnd": q_by_id.get(CONDUCTOR_VOLTAGE_KEYS["MetalGND"], 0.0),
    }
    return charges, breakdown


def run_case(
    case_name: str,
    excitation: Dict[int, float],
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    params: Config,
    compute_potential_grid: bool = True,
    prepared_system: Optional[PreparedStaticSystem] = None,
) -> CaseResult:
    """Run one electrostatic excitation from geometry through charges."""

    if prepared_system is None:
        M_big, dof_slices = assemble_energy_matrix(cells, basis_cfg)
        C_mat, d_vec, constraint_stats = assemble_constraints(cells, basis_cfg, dof_slices, excitation, params)
        charge_plans = None
    else:
        M_big = prepared_system.M_big
        dof_slices = prepared_system.dof_slices
        C_mat = prepared_system.constraint_template.C_mat
        d_vec = build_constraint_rhs(prepared_system.constraint_template, excitation)
        constraint_stats = dict(prepared_system.constraint_template.stats)
        charge_plans = prepared_system.charge_plans
    reduced_constraint_system = prepared_system.reduced_constraint_system if prepared_system is not None else None
    x, lambda_vec = solve_kkt(
        M_big,
        C_mat,
        d_vec,
        reduced_system=reduced_constraint_system,
        warn=not params.quiet,
    )
    if compute_potential_grid:
        grid_x, grid_y, potential_grid = evaluate_potential(
            cells, basis_cfg, x, dof_slices, excitation, params.plot_resolution
        )
    else:
        grid_x = np.zeros(0, dtype=float)
        grid_y = np.zeros(0, dtype=float)
        potential_grid = np.zeros((0, 0), dtype=float)
    charges, breakdown = compute_conductor_charges(
        cells, basis_cfg, x, dof_slices, excitation, params.quadrature_order, charge_plans
    )
    return CaseResult(
        case_name=case_name,
        excitation=excitation,
        x=x,
        lambda_vec=lambda_vec,
        dof_slices=dof_slices,
        constraint_stats=constraint_stats,
        charges=charges,
        edge_breakdown=breakdown,
        potential_grid=potential_grid,
        grid_x=grid_x,
        grid_y=grid_y,
    )


def run_excitation_batch(
    excitation_specs: Sequence[ExcitationSpec],
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    params: Config,
    prepared_system: Optional[PreparedStaticSystem] = None,
) -> List[CaseResult]:
    """Solve multiple excitations on the same geometry while reusing the matrix factorization."""

    if not excitation_specs:
        return []

    if prepared_system is None:
        M_big, dof_slices = assemble_energy_matrix(cells, basis_cfg)
        C_ref: Optional[np.ndarray] = None
        d_columns: List[np.ndarray] = []
        stats_by_case: List[Dict[str, int]] = []

        for spec in excitation_specs:
            C_mat, d_vec, constraint_stats = assemble_constraints(cells, basis_cfg, dof_slices, spec.excitation, params)
            if C_ref is None:
                C_ref = C_mat
            else:
                if C_mat.shape != C_ref.shape or not np.array_equal(C_mat, C_ref):
                    max_diff = float(np.max(np.abs(C_mat - C_ref))) if C_mat.size and C_ref.size else 0.0
                    raise RuntimeError(
                        f"Constraint matrix changed across excitations on the same geometry. max_diff={max_diff:.3e}"
                    )
            d_columns.append(d_vec)
            stats_by_case.append(constraint_stats)

        assert C_ref is not None
        charge_plans = None
        d_matrix = np.column_stack(d_columns)
        x_matrix, lambda_matrix = solve_kkt_multi(M_big, C_ref, d_matrix, warn=not params.quiet)
    else:
        M_big = prepared_system.M_big
        dof_slices = prepared_system.dof_slices
        C_ref = prepared_system.constraint_template.C_mat
        stats_by_case = [dict(prepared_system.constraint_template.stats) for _ in excitation_specs]
        d_matrix = np.column_stack(
            [build_constraint_rhs(prepared_system.constraint_template, spec.excitation) for spec in excitation_specs]
        )
        charge_plans = prepared_system.charge_plans
        x_matrix, lambda_matrix = solve_kkt_multi(
            M_big,
            C_ref,
            d_matrix,
            reduced_system=prepared_system.reduced_constraint_system,
            warn=not params.quiet,
        )

    results: List[CaseResult] = []
    for idx, spec in enumerate(excitation_specs):
        x = x_matrix[:, idx]
        lambda_vec = lambda_matrix[:, idx]
        if spec.compute_potential_grid:
            grid_x, grid_y, potential_grid = evaluate_potential(
                cells, basis_cfg, x, dof_slices, spec.excitation, params.plot_resolution
            )
        else:
            grid_x = np.zeros(0, dtype=float)
            grid_y = np.zeros(0, dtype=float)
            potential_grid = np.zeros((0, 0), dtype=float)
        charges, breakdown = compute_conductor_charges(
            cells, basis_cfg, x, dof_slices, spec.excitation, params.quadrature_order, charge_plans
        )
        results.append(
            CaseResult(
                case_name=spec.case_name,
                excitation=spec.excitation,
                x=x,
                lambda_vec=lambda_vec,
                dof_slices=dof_slices,
                constraint_stats=stats_by_case[idx],
                charges=charges,
                edge_breakdown=breakdown,
                potential_grid=potential_grid,
                grid_x=grid_x,
                grid_y=grid_y,
            )
        )
    return results


def extract_Cprime_matrix(
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    params: Config,
    compute_potential_grids: bool = True,
    prepared_system: Optional[PreparedStaticSystem] = None,
) -> Tuple[np.ndarray, CaseResult, CaseResult]:
    """Solve the two canonical excitations and build the 2x2 per-unit-length C'."""

    case_A, case_B = run_excitation_batch(
        [
            ExcitationSpec("Case A", {0: 0.0, 1: 1.0, 2: 0.0}, compute_potential_grid=compute_potential_grids),
            ExcitationSpec("Case B", {0: 0.0, 1: 0.0, 2: 1.0}, compute_potential_grid=compute_potential_grids),
        ],
        cells,
        basis_cfg,
        params,
        prepared_system=prepared_system,
    )
    Cprime = np.array(
        [
            [case_A.charges["Q1"], case_B.charges["Q1"]],
            [case_A.charges["Q2"], case_B.charges["Q2"]],
        ],
        dtype=float,
    )
    return Cprime, case_A, case_B


def build_air_cells_from_cells(cells: Sequence[Cell]) -> List[Cell]:
    """Clone the geometry/conductor map and replace every dielectric by air."""

    air_cells: List[Cell] = []
    for cell in cells:
        cloned = Cell(**asdict(cell))
        if not cloned.is_conductor:
            cloned.material_name = "Air"
            cloned.eps_r = MATERIAL_EPS_R["Air"]
            cloned.eps_abs = EPS0
        air_cells.append(cloned)
    return air_cells


def prepare_static_system(
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    params: Config,
    charge_base_plans: Optional[Sequence[ChargeEdgeBasePlan]] = None,
) -> PreparedStaticSystem:
    """Build reusable geometry-specific solve data for one dielectric assignment."""

    frozen_cells = tuple(cells)
    M_big, dof_slices = assemble_energy_matrix(frozen_cells, basis_cfg)
    constraint_template = build_constraint_template(frozen_cells, basis_cfg, dof_slices, params)
    reduced_constraint_system = prepare_reduced_constraint_system(constraint_template.C_mat)
    charge_plans = build_charge_integration_plan(frozen_cells, basis_cfg, charge_base_plans)
    return PreparedStaticSystem(
        cells=frozen_cells,
        basis_cfg=basis_cfg,
        M_big=M_big,
        dof_slices=dof_slices,
        constraint_template=constraint_template,
        reduced_constraint_system=reduced_constraint_system,
        charge_plans=charge_plans,
    )


def extract_Cair_matrix(
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    params: Config,
    prepared_system: Optional[PreparedStaticSystem] = None,
) -> Tuple[np.ndarray, CaseResult, CaseResult]:
    """Extract the all-air capacitance matrix with the same conductor geometry."""

    if prepared_system is not None:
        return extract_Cprime_matrix(
            prepared_system.cells,
            prepared_system.basis_cfg,
            params,
            compute_potential_grids=False,
            prepared_system=prepared_system,
        )
    air_cells = build_air_cells_from_cells(cells)
    return extract_Cprime_matrix(air_cells, basis_cfg, params, compute_potential_grids=False)


def compute_Lprime_from_Cair(Cair: np.ndarray) -> np.ndarray:
    """Quasi-TEM relation L' = mu0 * eps0 * inv(C_air)."""

    return MU0 * EPS0 * np.linalg.inv(Cair)


def clone_case_result_with_potential(
    case_result: CaseResult,
    cells: Sequence[Cell],
    basis_cfg: BasisConfig,
    params: Config,
) -> CaseResult:
    """Attach a sampled potential grid to an existing solved excitation."""

    if case_result.potential_grid.size:
        return case_result
    grid_x, grid_y, potential_grid = evaluate_potential(
        cells,
        basis_cfg,
        case_result.x,
        case_result.dof_slices,
        case_result.excitation,
        params.plot_resolution,
    )
    return CaseResult(
        case_name=case_result.case_name,
        excitation=dict(case_result.excitation),
        x=case_result.x,
        lambda_vec=case_result.lambda_vec,
        dof_slices=case_result.dof_slices,
        constraint_stats=dict(case_result.constraint_stats),
        charges=dict(case_result.charges),
        edge_breakdown={name: dict(values) for name, values in case_result.edge_breakdown.items()},
        potential_grid=potential_grid,
        grid_x=grid_x,
        grid_y=grid_y,
        loss_placeholder=case_result.loss_placeholder,
    )


def extract_conductor_geom_from_cells(cells: Sequence[Cell]) -> Dict[int, Dict[str, float]]:
    """Extract simple conductor geometry descriptors from the active cell table.

    The result mirrors the simple loss model used in Cal_0313.py:
    - W: overall conductor width in meters
    - t: overall conductor thickness in meters
    """

    geom: Dict[int, Dict[str, float]] = {}
    for cid in (CONDUCTOR_VOLTAGE_KEYS["Metal1"], CONDUCTOR_VOLTAGE_KEYS["Metal2"]):
        conductor_cells = [cell for cell in cells if cell.is_conductor and int(cell.conductor_id) == cid]
        if not conductor_cells:
            raise ValueError(f"Conductor id {cid} is missing from the geometry.")
        x1 = min(float(cell.x1) for cell in conductor_cells)
        x2 = max(float(cell.x2) for cell in conductor_cells)
        y1 = min(float(cell.y1) for cell in conductor_cells)
        y2 = max(float(cell.y2) for cell in conductor_cells)
        geom[cid] = {
            "W": max(x2 - x1, 1e-18),
            "t": max(y2 - y1, 1e-18),
        }
    return geom


def compute_global_tan_delta_eff(cells: Sequence[Cell], params: Config) -> float:
    """Return the effective dielectric loss tangent used by the simple G' model.

    We keep this intentionally simple and stable:
    - If tan_delta_eff_override is given, use it directly.
    - Otherwise area-weight the configured material tan(delta) values over all
      dielectric cells in the current geometry.
    """

    if params.tan_delta_eff_override is not None:
        return float(params.tan_delta_eff_override)

    weighted_sum = 0.0
    total_area = 0.0
    for cell in cells:
        if cell.is_conductor:
            continue
        area = max((cell.x2 - cell.x1) * (cell.y2 - cell.y1), 0.0)
        tan_delta = float(params.tan_delta_by_material.get(cell.material_name, 0.0))
        weighted_sum += area * tan_delta
        total_area += area
    if total_area <= 0.0:
        return 0.0
    return float(weighted_sum / total_area)


def compute_Gprime_simple(Cprime: np.ndarray, freq_hz: float, tan_delta_eff: float) -> np.ndarray:
    """Simple dielectric-loss model G'(f) = omega * tan(delta)_eff * C'."""

    omega = 2.0 * np.pi * freq_hz
    return omega * float(tan_delta_eff) * np.asarray(Cprime, dtype=float)


def compute_Rprime_simple(
    cells: Sequence[Cell],
    params: Config,
    freq_hz: float,
) -> np.ndarray:
    """Simple conductor-loss model based on surface resistance.

    This follows the same spirit as Cal_0313.py:
    - surface resistance Rs = sqrt(pi f mu / sigma)
    - effective width Weff = W + t / pi
    - R' is returned as a diagonal 2x2 matrix for Metal1/Metal2
    """

    geom = extract_conductor_geom_from_cells(cells)
    roughness_m = float(params.metal_roughness_um) * UM_TO_M

    rdiag: List[float] = []
    for conductor_name in ("Metal1", "Metal2"):
        cid = CONDUCTOR_VOLTAGE_KEYS[conductor_name]
        W = max(float(geom[cid]["W"]), 1e-18)
        t = max(float(geom[cid]["t"]), 1e-18)
        sigma = float(params.metal_sigma_by_conductor_S_per_m[conductor_name])
        omega = 2.0 * np.pi * freq_hz
        skin_depth = np.sqrt(2.0 / max(omega * MU0 * sigma, 1e-30))
        Rs = np.sqrt(np.pi * freq_hz * MU0 / sigma)
        if roughness_m > 0.0:
            ksr = 1.0 + (2.0 / np.pi) * np.arctan(1.4 * (roughness_m / max(skin_depth, 1e-18)) ** 2)
        else:
            ksr = 1.0
        Weff = W + t / np.pi
        rdiag.append(float(Rs * ksr / max(Weff, 1e-18)))
    return np.diag(np.asarray(rdiag, dtype=float))


def compute_Rprime_simple_from_geom(
    conductor_geom: Dict[int, Dict[str, float]],
    params: Config,
    freq_hz: float,
) -> np.ndarray:
    """Frequency-dependent simple conductor loss using precomputed conductor geometry."""

    roughness_m = float(params.metal_roughness_um) * UM_TO_M

    rdiag: List[float] = []
    for conductor_name in ("Metal1", "Metal2"):
        cid = CONDUCTOR_VOLTAGE_KEYS[conductor_name]
        W = max(float(conductor_geom[cid]["W"]), 1e-18)
        t = max(float(conductor_geom[cid]["t"]), 1e-18)
        sigma = float(params.metal_sigma_by_conductor_S_per_m[conductor_name])
        omega = 2.0 * np.pi * freq_hz
        skin_depth = np.sqrt(2.0 / max(omega * MU0 * sigma, 1e-30))
        Rs = np.sqrt(np.pi * freq_hz * MU0 / sigma)
        if roughness_m > 0.0:
            ksr = 1.0 + (2.0 / np.pi) * np.arctan(1.4 * (roughness_m / max(skin_depth, 1e-18)) ** 2)
        else:
            ksr = 1.0
        Weff = W + t / np.pi
        rdiag.append(float(Rs * ksr / max(Weff, 1e-18)))
    return np.diag(np.asarray(rdiag, dtype=float))


def modal_from_LC(Lprime: np.ndarray, Cprime: np.ndarray, freq_hz: float):
    """Lossless LC modal solve mirroring Cal_0313.py."""

    omega = 2.0 * np.pi * freq_hz
    Zp = 1j * omega * Lprime
    Yp = 1j * omega * Cprime
    A = Zp @ Yp
    eigvals, T = np.linalg.eig(A)
    gamma = np.sqrt(eigvals.astype(complex))
    for k in range(len(gamma)):
        if np.real(gamma[k]) < 0:
            gamma[k] = -gamma[k]
        if np.imag(gamma[k]) < 0:
            gamma[k] = -gamma[k]
    idx = np.argsort(np.imag(gamma))
    gamma = gamma[idx]
    T = T[:, idx]
    U = np.zeros_like(T, dtype=complex)
    for k in range(T.shape[1]):
        U[:, k] = (Yp @ T[:, k]) / gamma[k]
    Uinv = np.linalg.inv(U)
    Z0_modes = np.diag(Uinv @ T)
    return gamma, T, U, Zp, Yp, Z0_modes


def modal_from_RLGC(
    Rprime: np.ndarray,
    Lprime: np.ndarray,
    Gprime: np.ndarray,
    Cprime: np.ndarray,
    freq_hz: float,
):
    """Lossy RLGC modal solve mirroring the current Cal_0313.py path."""

    omega = 2.0 * np.pi * freq_hz
    Zp = np.asarray(Rprime, dtype=complex) + 1j * omega * np.asarray(Lprime, dtype=complex)
    Yp = np.asarray(Gprime, dtype=complex) + 1j * omega * np.asarray(Cprime, dtype=complex)
    A = Zp @ Yp
    eigvals, T = np.linalg.eig(A)
    gamma = np.sqrt(eigvals.astype(complex))
    for k in range(len(gamma)):
        if np.real(gamma[k]) < 0:
            gamma[k] = -gamma[k]
        if np.imag(gamma[k]) < 0:
            gamma[k] = -gamma[k]
    idx = np.argsort(np.imag(gamma))
    gamma = gamma[idx]
    T = T[:, idx]
    U = np.zeros_like(T, dtype=complex)
    for k in range(T.shape[1]):
        U[:, k] = (Yp @ T[:, k]) / gamma[k]
    Uinv = np.linalg.inv(U)
    Z0_modes = np.diag(Uinv @ T)
    return gamma, T, U, Zp, Yp, Z0_modes


def solve_modes_for_frequency(
    params: Config,
    cells: Sequence[Cell],
    Cprime: np.ndarray,
    Lprime: np.ndarray,
    freq_hz: float,
) -> Dict[str, object]:
    """Solve the modal system for one frequency using either LC or RLGC."""

    freq_hz = float(freq_hz)
    if params.use_loss:
        tan_delta_eff = compute_global_tan_delta_eff(cells, params)
        Gprime = compute_Gprime_simple(Cprime, freq_hz, tan_delta_eff)
        Rprime = compute_Rprime_simple(cells, params, freq_hz)
        gamma, T, U, Zp, Yp, Z0_modes = modal_from_RLGC(
            Rprime, Lprime, Gprime, Cprime, freq_hz
        )
    else:
        tan_delta_eff = 0.0
        Rprime = np.zeros((2, 2), dtype=float)
        Gprime = np.zeros((2, 2), dtype=float)
        gamma, T, U, Zp, Yp, Z0_modes = modal_from_LC(Lprime, Cprime, freq_hz)
    return {
        "freq_hz": freq_hz,
        "tan_delta_eff": float(tan_delta_eff),
        "Rprime": Rprime,
        "Gprime": Gprime,
        "gamma": gamma,
        "T": T,
        "U": U,
        "Zp": Zp,
        "Yp": Yp,
        "Z0_modes": Z0_modes,
    }


def solve_modes_for_frequency_precomputed(
    params: Config,
    Cprime: np.ndarray,
    Lprime: np.ndarray,
    freq_hz: float,
    tan_delta_eff: float,
    conductor_geom: Dict[int, Dict[str, float]],
) -> Dict[str, object]:
    """Solve the modal system using precomputed loss-model invariants."""

    freq_hz = float(freq_hz)
    if params.use_loss:
        Gprime = compute_Gprime_simple(Cprime, freq_hz, tan_delta_eff)
        Rprime = compute_Rprime_simple_from_geom(conductor_geom, params, freq_hz)
        gamma, T, U, Zp, Yp, Z0_modes = modal_from_RLGC(
            Rprime, Lprime, Gprime, Cprime, freq_hz
        )
    else:
        Rprime = np.zeros((2, 2), dtype=float)
        Gprime = np.zeros((2, 2), dtype=float)
        gamma, T, U, Zp, Yp, Z0_modes = modal_from_LC(Lprime, Cprime, freq_hz)
    return {
        "freq_hz": freq_hz,
        "tan_delta_eff": float(tan_delta_eff),
        "Rprime": Rprime,
        "Gprime": Gprime,
        "gamma": gamma,
        "T": T,
        "U": U,
        "Zp": Zp,
        "Yp": Yp,
        "Z0_modes": Z0_modes,
    }


def build_4port_ZS_from_modal(
    T: np.ndarray,
    U: np.ndarray,
    gamma_modes: np.ndarray,
    length_m: float,
    z0_ref: float,
    port_perm: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build 4-port Z and S from modal voltage/current transforms.

    Internal block order before export permutation is [MA_A, E1_A, MA_B, E1_B].
    Equivalently, using the older geometric names, this is [UL, LL, UR, LR]
    because Metal1/MA is the upper conductor and Metal2/E1 is the lower
    conductor.
    Exported order follows Config.touchstone_port_perm / touchstone_port_labels.
    """

    T = np.asarray(T, dtype=complex)
    U = np.asarray(U, dtype=complex)
    gamma_modes = np.asarray(gamma_modes, dtype=complex)
    Gl = gamma_modes * length_m
    coth_diag = np.zeros(len(gamma_modes), dtype=complex)
    csch_diag = np.zeros(len(gamma_modes), dtype=complex)
    eps_gl = 1e-15
    for k, x in enumerate(Gl):
        if abs(x) < eps_gl:
            coth_diag[k] = 1.0 / (x + eps_gl) + x / 3.0
            csch_diag[k] = 1.0 / (x + eps_gl) - x / 6.0
        else:
            coth_diag[k] = np.cosh(x) / np.sinh(x)
            csch_diag[k] = 1.0 / np.sinh(x)
    Uinv = np.linalg.inv(U)
    COTH = np.diag(coth_diag)
    CSCH = np.diag(csch_diag)
    Z11 = T @ COTH @ Uinv
    Z12 = T @ CSCH @ Uinv
    Z21 = Z12
    Z22 = Z11
    Z4 = np.block([[Z11, Z12], [Z21, Z22]])
    port_perm = tuple(int(v) for v in port_perm)
    Z4 = Z4[np.ix_(port_perm, port_perm)]
    I4 = np.eye(4, dtype=complex)
    S4 = (Z4 - z0_ref * I4) @ np.linalg.inv(Z4 + z0_ref * I4)
    return Z4, S4


def write_touchstone_s4p(
    filename: str,
    freq_list_Hz: Sequence[float],
    S4_list: Sequence[np.ndarray],
    z0_ref: float,
    port_labels: Sequence[str],
) -> None:
    """Write a standard Touchstone .s4p file in GHz / S / MA format."""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# GHz S MA R {z0_ref:.6f}\n")
        f.write("! Cal_04.py exported 4-port coupled-line data\n")
        for i, name in enumerate(port_labels, start=1):
            f.write(f"! Port[{i}] = {name}\n")
        for freq_hz, S4 in zip(freq_list_Hz, S4_list):
            freq_ghz = float(freq_hz) * 1e-9
            S4 = np.asarray(S4, dtype=complex)
            f.write(f"{freq_ghz:<30g}")
            for i in range(4):
                for j in range(4):
                    mag = abs(S4[i, j])
                    ang = np.degrees(np.angle(S4[i, j]))
                    f.write(f" {mag:.15g} {ang:.15g}")
            f.write("\n")


def build_modal_records_for_bundle(
    params: Config,
    cells: Sequence[Cell],
    Cprime_final_used: np.ndarray,
    Lprime: np.ndarray,
) -> Tuple[Tuple[Dict[str, object], ...], float, Dict[int, Dict[str, float]]]:
    """Precompute width-dependent modal data reused across multiple lengths."""

    tan_delta_eff = compute_global_tan_delta_eff(cells, params) if params.use_loss else 0.0
    conductor_geom = extract_conductor_geom_from_cells(cells)
    modal_records: List[Dict[str, object]] = []
    for freq_hz in params.freq_list_Hz:
        modal_records.append(
            solve_modes_for_frequency_precomputed(
                params,
                Cprime_final_used,
                Lprime,
                float(freq_hz),
                tan_delta_eff,
                conductor_geom,
            )
        )
    return tuple(modal_records), float(tan_delta_eff), conductor_geom


def export_sweep_s4p_from_modal_records(
    params: Config,
    modal_records: Sequence[Dict[str, object]],
    length_m: float,
) -> Dict[str, object]:
    """Length-dependent Touchstone export from cached modal records."""

    Z4_list: List[np.ndarray] = []
    S4_list: List[np.ndarray] = []
    for modal_data in modal_records:
        Z4, S4 = build_4port_ZS_from_modal(
            np.asarray(modal_data["T"], dtype=complex),
            np.asarray(modal_data["U"], dtype=complex),
            np.asarray(modal_data["gamma"], dtype=complex),
            length_m,
            params.z0_ref,
            params.touchstone_port_perm,
        )
        Z4_list.append(Z4)
        S4_list.append(S4)
    touchstone_path = os.path.join(params.output_dir, params.export_touchstone_filename)
    write_touchstone_s4p(
        touchstone_path,
        params.freq_list_Hz,
        S4_list,
        params.z0_ref,
        params.touchstone_port_labels,
    )
    return {
        "modal_records": list(modal_records),
        "Z4_list": Z4_list,
        "S4_list": S4_list,
        "touchstone_path": touchstone_path,
    }


def export_single_freq_s4p(
    params: Config,
    cells: Sequence[Cell],
    Cprime: np.ndarray,
    Cair: np.ndarray,
    Lprime: np.ndarray,
    length_m: float,
) -> Dict[str, object]:
    """Single-frequency modal solve and .s4p export."""

    freq_hz = float(params.freq_list_Hz[0])
    modal_data = solve_modes_for_frequency(params, cells, Cprime, Lprime, freq_hz)
    Z4, S4 = build_4port_ZS_from_modal(
        modal_data["T"],
        modal_data["U"],
        modal_data["gamma"],
        length_m,
        params.z0_ref,
        params.touchstone_port_perm,
    )
    touchstone_path = os.path.join(params.output_dir, params.export_touchstone_filename)
    write_touchstone_s4p(
        touchstone_path,
        [freq_hz],
        [S4],
        params.z0_ref,
        params.touchstone_port_labels,
    )
    return {
        "freq_hz": freq_hz,
        "tan_delta_eff": modal_data["tan_delta_eff"],
        "Rprime": modal_data["Rprime"],
        "Gprime": modal_data["Gprime"],
        "gamma": modal_data["gamma"],
        "Z0_modes": modal_data["Z0_modes"],
        "Zp": modal_data["Zp"],
        "Yp": modal_data["Yp"],
        "Z4": Z4,
        "S4": S4,
        "touchstone_path": touchstone_path,
    }


def export_sweep_s4p(
    params: Config,
    cells: Sequence[Cell],
    Cprime: np.ndarray,
    Cair: np.ndarray,
    Lprime: np.ndarray,
    length_m: float,
) -> Dict[str, object]:
    """Frequency sweep modal solve and .s4p export."""

    gamma_list: List[np.ndarray] = []
    Z0_list: List[np.ndarray] = []
    Z4_list: List[np.ndarray] = []
    S4_list: List[np.ndarray] = []
    modal_records: List[Dict[str, object]] = []
    for freq_hz in params.freq_list_Hz:
        modal_data = solve_modes_for_frequency(params, cells, Cprime, Lprime, float(freq_hz))
        Z4, S4 = build_4port_ZS_from_modal(
            modal_data["T"],
            modal_data["U"],
            modal_data["gamma"],
            length_m,
            params.z0_ref,
            params.touchstone_port_perm,
        )
        gamma_list.append(modal_data["gamma"])
        Z0_list.append(modal_data["Z0_modes"])
        Z4_list.append(Z4)
        S4_list.append(S4)
        modal_records.append(
            {
                "freq_hz": float(freq_hz),
                "tan_delta_eff": modal_data["tan_delta_eff"],
                "Rprime": modal_data["Rprime"],
                "Gprime": modal_data["Gprime"],
                "gamma": modal_data["gamma"],
                "Z0_modes": modal_data["Z0_modes"],
                "Zp": modal_data["Zp"],
                "Yp": modal_data["Yp"],
            }
        )
    touchstone_path = os.path.join(params.output_dir, params.export_touchstone_filename)
    write_touchstone_s4p(
        touchstone_path,
        params.freq_list_Hz,
        S4_list,
        params.z0_ref,
        params.touchstone_port_labels,
    )
    return {
        "modal_records": modal_records,
        "Z4_list": Z4_list,
        "S4_list": S4_list,
        "touchstone_path": touchstone_path,
    }


def prepare_sweep_bundle(params: Config) -> PreparedSweepBundle:
    """Build and cache the width-dependent electrostatic and modal results."""

    cache_key = prepared_bundle_cache_key(params)
    cached = PREPARED_SWEEP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cells = build_geometry(params)
    assign_materials(cells, params)
    assign_conductors(cells, params)
    validate_conductor_layout(cells)
    for cell in cells:
        build_basis_for_cell(cell, params.M_modes)

    basis_cfg = BasisConfig(
        L=params.L * UM_TO_M,
        M_modes=params.M_modes,
        km_array=compute_km(params.M_modes, params.L * UM_TO_M),
    )

    charge_base_plans = build_charge_base_plan(cells, basis_cfg)
    prepared_system = prepare_static_system(cells, basis_cfg, params, charge_base_plans=charge_base_plans)
    Cprime_computed, case_A, case_B = extract_Cprime_matrix(
        prepared_system.cells,
        prepared_system.basis_cfg,
        params,
        compute_potential_grids=False,
        prepared_system=prepared_system,
    )

    air_cells = build_air_cells_from_cells(cells)
    air_prepared_system = prepare_static_system(air_cells, basis_cfg, params, charge_base_plans=charge_base_plans)
    Cair_computed, _, _ = extract_Cair_matrix(
        air_prepared_system.cells,
        air_prepared_system.basis_cfg,
        params,
        prepared_system=air_prepared_system,
    )

    Cprime_final_used, cprime_source, Cprime_hfss = resolve_effective_matrix(
        Cprime_computed,
        params.Cprime_hfss,
        params.use_hfss_cprime_override,
        "Cprime_hfss",
    )
    Cair_final_used, cair_source, Cair_hfss = resolve_effective_matrix(
        Cair_computed,
        params.Cair_hfss,
        params.use_hfss_cair_override,
        "Cair_hfss",
    )
    Lprime = compute_Lprime_from_Cair(Cair_final_used)
    modal_records, tan_delta_eff, conductor_geom = build_modal_records_for_bundle(
        params,
        prepared_system.cells,
        Cprime_final_used,
        Lprime,
    )

    bundle = PreparedSweepBundle(
        cells=prepared_system.cells,
        basis_cfg=prepared_system.basis_cfg,
        case_A=case_A,
        case_B=case_B,
        Cprime_computed=Cprime_computed,
        Cprime_final_used=Cprime_final_used,
        Cprime_hfss=Cprime_hfss,
        Cprime_source=cprime_source,
        Cair_computed=Cair_computed,
        Cair_final_used=Cair_final_used,
        Cair_hfss=Cair_hfss,
        Cair_source=cair_source,
        Lprime=Lprime,
        modal_records=modal_records,
        tan_delta_eff=tan_delta_eff,
        conductor_geom=conductor_geom,
    )
    PREPARED_SWEEP_CACHE[cache_key] = bundle
    return bundle


def plot_geometry_material_map(cells: Sequence[Cell], params: Config, path: str) -> None:
    """Save a material map image.

    Internal geometry is in meters; plotting converts back to micrometers.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    colors = {"Air": "#cfe8ff", "SiO2": "#f7d48b", "Si": "#8dc28d"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for cell in cells:
        ax.add_patch(
            Rectangle(
                (cell.x1, cell.y1),
                (cell.x2 - cell.x1),
                (cell.y2 - cell.y1),
                facecolor=colors[cell.material_name],
                edgecolor="k",
                linewidth=0.7,
            )
        )
    for patch in ax.patches:
        patch.set_x(patch.get_x() * M_TO_UM)
        patch.set_y(patch.get_y() * M_TO_UM)
        patch.set_width(patch.get_width() * M_TO_UM)
        patch.set_height(patch.get_height() * M_TO_UM)
    for material, color in colors.items():
        ax.scatter([], [], c=color, label=material)
    ax.set_title("Material Map")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_xlim(0.0, 2.0 * params.L)
    ax.set_ylim(params.y_levels[0], params.y_levels[-1])
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_geometry_conductor_map(cells: Sequence[Cell], params: Config, path: str) -> None:
    """Save a conductor map image, plotted in micrometers."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    colors = {"None": "#f0f0f0", "Metal1": "#d62728", "Metal2": "#1f77b4", "MetalGND": "#444444"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for cell in cells:
        key = cell.conductor_name if cell.conductor_name else "None"
        ax.add_patch(
            Rectangle(
                (cell.x1, cell.y1),
                (cell.x2 - cell.x1),
                (cell.y2 - cell.y1),
                facecolor=colors[key],
                edgecolor="k",
                linewidth=0.7,
            )
        )
    for patch in ax.patches:
        patch.set_x(patch.get_x() * M_TO_UM)
        patch.set_y(patch.get_y() * M_TO_UM)
        patch.set_width(patch.get_width() * M_TO_UM)
        patch.set_height(patch.get_height() * M_TO_UM)
    for label, color in colors.items():
        ax.scatter([], [], c=color, label=label)
    ax.set_title("Conductor Map")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_xlim(0.0, 2.0 * params.L)
    ax.set_ylim(params.y_levels[0], params.y_levels[-1])
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_potential(case_result: CaseResult, cells: Sequence[Cell], params: Config, path: str) -> None:
    """Save one potential contour plot in micrometers."""

    if case_result.potential_grid.size == 0:
        raise ValueError("Potential grid is empty; enable generate_potential_plots to create it.")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 5))
    X, Y = np.meshgrid(case_result.grid_x * M_TO_UM, case_result.grid_y * M_TO_UM)
    contour = ax.contourf(X, Y, case_result.potential_grid, levels=80, cmap="coolwarm")
    for cell in cells:
        if cell.is_conductor:
            ax.add_patch(
                Rectangle(
                    (cell.x1, cell.y1),
                    (cell.x2 - cell.x1),
                    (cell.y2 - cell.y1),
                    facecolor="none",
                    edgecolor="k",
                    linewidth=1.0,
                )
            )
    for patch in ax.patches:
        patch.set_x(patch.get_x() * M_TO_UM)
        patch.set_y(patch.get_y() * M_TO_UM)
        patch.set_width(patch.get_width() * M_TO_UM)
        patch.set_height(patch.get_height() * M_TO_UM)
    ax.set_title(f"Potential {case_result.case_name}")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_xlim(0.0, 2.0 * params.L)
    ax.set_ylim(params.y_levels[0], params.y_levels[-1])
    fig.colorbar(contour, ax=ax, label="phi (V)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_case_summary(case_result: CaseResult, quiet: bool = False) -> None:
    """Print charges and edge contributions for one excitation."""

    if quiet:
        return
    print(f"\n[{case_result.case_name}] excitation = {case_result.excitation}")
    print(
        f"Q1 = {case_result.charges['Q1']:.6e} C/m, "
        f"Q2 = {case_result.charges['Q2']:.6e} C/m, "
        f"Qgnd = {case_result.charges['Qgnd']:.6e} C/m"
    )
    for conductor_name, edge_info in case_result.edge_breakdown.items():
        print(
            f"  {conductor_name}: "
            f"top={edge_info['top']:.6e}, bottom={edge_info['bottom']:.6e}, "
            f"left={edge_info['left']:.6e}, right={edge_info['right']:.6e}, "
            f"total={edge_info['total']:.6e}"
        )
    print(f"  constraint_stats = {case_result.constraint_stats}")


def write_summary_json(
    path: str,
    params: Config,
    cells: Sequence[Cell],
    case_A: CaseResult,
    case_B: CaseResult,
    Cprime_computed: np.ndarray,
    Cprime_final_used: np.ndarray,
    Cprime_source: str,
    Cprime_hfss: np.ndarray,
    Cair_computed: Optional[np.ndarray],
    Cair_final_used: Optional[np.ndarray],
    Cair_source: str,
    Cair_hfss: np.ndarray,
    Lprime: Optional[np.ndarray],
    charge_conservation: Dict[str, float],
    symmetry_error_computed: float,
    symmetry_error_final_used: float,
    sparam_summary: Optional[Dict[str, object]] = None,
) -> None:
    """Persist a compact run summary for debugging."""

    import json

    payload = {
        "params": asdict(params),
        "cells": [asdict(cell) for cell in cells],
        "case_A": {
            "charges": case_A.charges,
            "edge_breakdown": case_A.edge_breakdown,
            "constraint_stats": case_A.constraint_stats,
        },
        "case_B": {
            "charges": case_B.charges,
            "edge_breakdown": case_B.edge_breakdown,
            "constraint_stats": case_B.constraint_stats,
        },
        "Cprime_computed": Cprime_computed.tolist(),
        "Cprime_final_used": Cprime_final_used.tolist(),
        "Cprime_source": Cprime_source,
        "Cprime_hfss": Cprime_hfss.tolist(),
        "Cair_computed": None if Cair_computed is None else Cair_computed.tolist(),
        "Cair_final_used": None if Cair_final_used is None else Cair_final_used.tolist(),
        "Cair_source": Cair_source,
        "Cair_hfss": Cair_hfss.tolist(),
        "Lprime": None if Lprime is None else Lprime.tolist(),
        "charge_conservation": charge_conservation,
        "symmetry_error_computed": float(symmetry_error_computed),
        "symmetry_error_final_used": float(symmetry_error_final_used),
    }
    if sparam_summary is not None:
        payload["sparam_summary"] = sparam_summary
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Cal_0408 coupled-line solver and export an s4p file."
    )
    parser.add_argument(
        "--w-line",
        type=float,
        default=None,
        help="Override Config.W_line in micrometers.",
    )
    parser.add_argument(
        "--line-length-um",
        type=float,
        default=None,
        help="Override Config.line_length_um in micrometers.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override Config.output_dir.",
    )
    parser.add_argument(
        "--touchstone-file",
        type=str,
        default=None,
        help="Override Config.export_touchstone_filename.",
    )
    parser.add_argument(
        "--m-modes",
        type=int,
        default=None,
        help="Override Config.M_modes.",
    )
    parser.add_argument(
        "--slab0",
        type=int,
        default=None,
        help="Override targeted_slab_y_split_counts['slab0'] within 1..5.",
    )
    parser.add_argument(
        "--slab1",
        type=int,
        default=None,
        help="Override targeted_slab_y_split_counts['slab1'] within 1..5.",
    )
    parser.add_argument(
        "--slab2",
        type=int,
        default=None,
        help="Override targeted_slab_y_split_counts['slab2'] within 1..5.",
    )
    parser.add_argument(
        "--slab3",
        type=int,
        default=None,
        help="Override targeted_slab_y_split_counts['slab3'] within 1..5.",
    )
    parser.add_argument(
        "--slab4",
        type=int,
        default=None,
        help="Override targeted_slab_y_split_counts['slab4'] within 1..5.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential console output for faster batch runs.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    overrides: Dict[str, object] = {}
    if args.w_line is not None:
        overrides["W_line"] = float(args.w_line)
    if args.line_length_um is not None:
        overrides["line_length_um"] = float(args.line_length_um)
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.touchstone_file is not None:
        overrides["export_touchstone_filename"] = args.touchstone_file
    if args.m_modes is not None:
        overrides["M_modes"] = int(args.m_modes)
    slab_overrides = {
        "slab0": args.slab0,
        "slab1": args.slab1,
        "slab2": args.slab2,
        "slab3": args.slab3,
        "slab4": args.slab4,
    }
    if any(value is not None for value in slab_overrides.values()):
        split_counts = {
            "slab_b3": 1,
            "slab_b2": 1,
            "slab_b1": 1,
            "slab0": 1,
            "slab1": 2,
            "slab2": 2,
            "slab3": 5,
            "slab4": 2,
            "slab5": 1,
        }
        for slab_name, value in slab_overrides.items():
            if value is not None:
                split_counts[slab_name] = int(value)
        overrides["enable_targeted_slab_y_refinement"] = True
        overrides["targeted_slab_y_split_counts"] = split_counts
    if args.quiet:
        overrides["quiet"] = True
    return Config(**overrides)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entry point for the full static C' extraction flow."""

    args = build_arg_parser().parse_args(argv)
    params = config_from_args(args)
    ensure_output_dir(params.output_dir)

    def vprint(*args, **kwargs) -> None:
        if not params.quiet:
            print(*args, **kwargs)

    vprint("Geometry parameters:")
    vprint(
        f"  L = {params.L:.3f} um, W_line = {params.W_line:.3f} um, "
        f"W_GND_defect = {params.W_GND_defect:.3f} um, "
        f"M_modes = {params.M_modes}, quadrature_order = {params.quadrature_order}"
    )
    vprint(
        "  Dirichlet boundary projection mode = "
        f"{params.outer_dirichlet_projection_mode}"
    )
    vprint(
        f"  Frequency sweep = {len(params.freq_list_Hz)} points, "
        f"{params.freq_list_Hz[0]:.6e} Hz to {params.freq_list_Hz[-1]:.6e} Hz"
    )
    vprint(
        f"  use_loss = {params.use_loss}, metal_sigma_by_conductor = {params.metal_sigma_by_conductor_S_per_m}, "
        f"metal_roughness = {params.metal_roughness_um:.6f} um"
    )
    vprint(
        f"  tan_delta_by_material = {params.tan_delta_by_material}, "
        f"tan_delta_eff_override = {params.tan_delta_eff_override}"
    )
    vprint(
        "  speed flags = "
        f"print_geometry_table={params.print_geometry_table}, "
        f"generate_geometry_plots={params.generate_geometry_plots}, "
        f"generate_potential_plots={params.generate_potential_plots}, "
        f"write_summary_json_enabled={params.write_summary_json_enabled}"
    )
    vprint(
        "Touchstone port order: "
        f"perm={list(params.touchstone_port_perm)}, "
        f"labels={list(params.touchstone_port_labels)}"
    )

    bundle = prepare_sweep_bundle(params)
    cells = bundle.cells
    basis_cfg = bundle.basis_cfg

    if params.print_geometry_table:
        summarize_geometry(cells)

    material_map_path = os.path.join(params.output_dir, "geometry_material_map.png")
    conductor_map_path = os.path.join(params.output_dir, "geometry_conductor_map.png")
    potential_A_path = os.path.join(params.output_dir, "potential_case_A.png")
    potential_B_path = os.path.join(params.output_dir, "potential_case_B.png")
    summary_json_path = os.path.join(params.output_dir, "cal04_summary.json")

    if params.generate_geometry_plots:
        plot_geometry_material_map(cells, params, material_map_path)
        plot_geometry_conductor_map(cells, params, conductor_map_path)

    case_A = bundle.case_A
    case_B = bundle.case_B
    if params.generate_potential_plots:
        case_A = clone_case_result_with_potential(case_A, cells, basis_cfg, params)
        case_B = clone_case_result_with_potential(case_B, cells, basis_cfg, params)
    Cprime_computed = bundle.Cprime_computed
    Cair_computed = bundle.Cair_computed
    Cprime_final_used = bundle.Cprime_final_used
    Cair_final_used = bundle.Cair_final_used
    Cprime_hfss = bundle.Cprime_hfss
    Cair_hfss = bundle.Cair_hfss
    cprime_source = bundle.Cprime_source
    cair_source = bundle.Cair_source
    Lprime = bundle.Lprime
    if params.generate_potential_plots:
        plot_potential(case_A, cells, params, potential_A_path)
        plot_potential(case_B, cells, params, potential_B_path)

    print_case_summary(case_A, quiet=params.quiet)
    print_case_summary(case_B, quiet=params.quiet)

    charge_conservation = {
        "Case A": case_A.charges["Q1"] + case_A.charges["Q2"] + case_A.charges["Qgnd"],
        "Case B": case_B.charges["Q1"] + case_B.charges["Q2"] + case_B.charges["Qgnd"],
    }
    symmetry_error_computed = float(
        np.linalg.norm(Cprime_computed - Cprime_computed.T) / max(np.linalg.norm(Cprime_computed), 1e-30)
    )
    symmetry_error_final_used = float(
        np.linalg.norm(Cprime_final_used - Cprime_final_used.T) / max(np.linalg.norm(Cprime_final_used), 1e-30)
    )

    vprint("\nC' matrix, code-computed (F/m):")
    vprint(Cprime_computed)
    vprint(f"C' source used = {cprime_source}")
    if cprime_source == "hfss_override":
        vprint("C' HFSS override (F/m):")
        vprint(Cprime_hfss)
    vprint("C' final used (F/m):")
    vprint(Cprime_final_used)
    vprint("\nC_air matrix, code-computed (F/m):")
    vprint(Cair_computed)
    vprint(f"C_air source used = {cair_source}")
    if cair_source == "hfss_override":
        vprint("C_air HFSS override (F/m):")
        vprint(Cair_hfss)
    vprint("C_air final used (F/m):")
    vprint(Cair_final_used)
    vprint("\nL' matrix (H/m):")
    vprint(Lprime)
    vprint(f"Charge conservation error Case A = {charge_conservation['Case A']:.6e} C/m")
    vprint(f"Charge conservation error Case B = {charge_conservation['Case B']:.6e} C/m")
    vprint(f"C' symmetry error, code-computed = {symmetry_error_computed:.6e}")
    vprint(f"C' symmetry error, final used = {symmetry_error_final_used:.6e}")
    diag_ok = bool(np.all(np.diag(Cprime_final_used) > 0.0))
    offdiag_ok = bool(Cprime_final_used[0, 1] < 0.0 and Cprime_final_used[1, 0] < 0.0)
    vprint(f"C' diagonal positive check = {diag_ok}")
    vprint(f"C' mutual entries negative check = {offdiag_ok}")
    max_charge_scale = max(
        abs(case_A.charges["Q1"]),
        abs(case_A.charges["Q2"]),
        abs(case_B.charges["Q1"]),
        abs(case_B.charges["Q2"]),
        1e-30,
    )
    if (
        abs(charge_conservation["Case A"]) / max_charge_scale > 5e-2
        or abs(charge_conservation["Case B"]) / max_charge_scale > 5e-2
    ):
        vprint("[warn] Charge conservation relative error is larger than 5e-2.")
    if symmetry_error_final_used > 1e-2:
        vprint("[warn] C' reciprocity/symmetry error is larger than 1e-2.")
    if not diag_ok:
        vprint("[warn] C' has a non-positive diagonal entry.")
    if not offdiag_ok:
        vprint("[warn] C' has a non-negative mutual entry.")

    length_m = params.line_length_um * UM_TO_M
    if len(params.freq_list_Hz) == 1:
        modal_data = bundle.modal_records[0]
        Z4, S4 = build_4port_ZS_from_modal(
            np.asarray(modal_data["T"], dtype=complex),
            np.asarray(modal_data["U"], dtype=complex),
            np.asarray(modal_data["gamma"], dtype=complex),
            length_m,
            params.z0_ref,
            params.touchstone_port_perm,
        )
        touchstone_path = os.path.join(params.output_dir, params.export_touchstone_filename)
        write_touchstone_s4p(
            touchstone_path,
            [float(modal_data["freq_hz"])],
            [S4],
            params.z0_ref,
            params.touchstone_port_labels,
        )
        sparam_out = {
            "freq_hz": float(modal_data["freq_hz"]),
            "tan_delta_eff": float(modal_data["tan_delta_eff"]),
            "Rprime": modal_data["Rprime"],
            "Gprime": modal_data["Gprime"],
            "gamma": modal_data["gamma"],
            "Z0_modes": modal_data["Z0_modes"],
            "Zp": modal_data["Zp"],
            "Yp": modal_data["Yp"],
            "Z4": Z4,
            "S4": S4,
            "touchstone_path": touchstone_path,
        }
        modal_records = [
            {
                "freq_hz": float(sparam_out["freq_hz"]),
                "tan_delta_eff": float(sparam_out["tan_delta_eff"]),
                "Rprime": np.asarray(sparam_out["Rprime"], dtype=float).tolist(),
                "Gprime": np.asarray(sparam_out["Gprime"], dtype=float).tolist(),
                "gamma": serialize_complex_array(sparam_out["gamma"]),
                "Z0_modes": serialize_complex_array(sparam_out["Z0_modes"]),
            }
        ]
    else:
        sparam_out = export_sweep_s4p_from_modal_records(
            params,
            bundle.modal_records,
            length_m,
        )
        modal_records = [
            {
                "freq_hz": float(rec["freq_hz"]),
                "tan_delta_eff": float(rec["tan_delta_eff"]),
                "Rprime": np.asarray(rec["Rprime"], dtype=float).tolist(),
                "Gprime": np.asarray(rec["Gprime"], dtype=float).tolist(),
                "gamma": serialize_complex_array(rec["gamma"]),
                "Z0_modes": serialize_complex_array(rec["Z0_modes"]),
            }
            for rec in sparam_out["modal_records"]
        ]

    vprint(f"\nPhysical line length = {params.line_length_um:.3f} um ({length_m:.6e} m)")
    for rec in modal_records:
        vprint(f"f = {rec['freq_hz']:.6e} Hz")
        vprint(f"  tan_delta_eff = {rec['tan_delta_eff']:.6e}")
        vprint(f"  Rprime = {rec['Rprime']}")
        vprint(f"  Gprime = {rec['Gprime']}")
        vprint(f"  gamma = {rec['gamma']}")
        vprint(f"  Z0_modes = {rec['Z0_modes']}")

    if params.write_summary_json_enabled:
        write_summary_json(
            summary_json_path,
            params,
            cells,
            case_A,
            case_B,
            Cprime_computed,
            Cprime_final_used,
            cprime_source,
            Cprime_hfss,
            Cair_computed,
            Cair_final_used,
            cair_source,
            Cair_hfss,
            Lprime,
            charge_conservation,
            symmetry_error_computed,
            symmetry_error_final_used,
            sparam_summary={
                "touchstone_path": sparam_out["touchstone_path"],
                "touchstone_port_perm": list(params.touchstone_port_perm),
                "touchstone_port_labels": list(params.touchstone_port_labels),
                "Cprime_source_used": cprime_source,
                "Cair_source_used": cair_source,
                "modal_records": modal_records,
            },
        )
    vprint("\nSaved outputs:")
    if params.generate_geometry_plots:
        vprint(f"  {material_map_path}")
        vprint(f"  {conductor_map_path}")
    if params.generate_potential_plots:
        vprint(f"  {potential_A_path}")
        vprint(f"  {potential_B_path}")
    if params.write_summary_json_enabled:
        vprint(f"  {summary_json_path}")
    vprint(f"  {sparam_out['touchstone_path']}")


if __name__ == "__main__":
    main()

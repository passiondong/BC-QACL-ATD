"""End-to-end BC-QACL-ATD synthesis flow.

This module is the clean entry point that drives the *exact* paper synthesis
(the vendored CMA-ES Pareto optimizer) from a :class:`bcqacl_atd.config.Config`.
It maps the config onto the optimizer's argument namespace, stages the
user-supplied transistor blocks under the filenames the optimizer expects, runs
the search, and returns the output directory containing the ranked candidates,
the ``(G, B, Z)`` Pareto front, the CMA-ES trace, and the build metadata.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import kernels  # noqa: F401  -- sys.path shim for the vendored optimizer
from .config import Config

# Filenames the vendored optimizer expects inside its --transistor-dir.
_DRIVER_NAME = "driver_2x12_single_ended_z0_50.s4p"
_POWER_NAME = "final_2x18_single_ended_z0_50.s4p"


def _optimizer_module():
    """Return the vendored exact-flow CMA-ES Pareto optimizer module."""
    import optimize_predicted_pa_global_relative_3db_len480_pareto_cal0529_speed as opt  # type: ignore
    return opt


def build_namespace(
    cfg: Config,
    *,
    transistor_dir: Path,
    output_dir: Path,
    no_plots: bool = False,
) -> argparse.Namespace:
    """Translate a Config into the vendored optimizer's argparse Namespace.

    The geometry grids and the (reporting-only) requested box are fixed inside
    the optimizer at the full validated model box; ``cfg.design_space`` is used
    by the wizard for display/validation and to bound new designs within it.
    """
    opt = cfg.optimizer
    tgt = cfg.target
    total_evals = int(opt.popsize) * int(opt.generations) * int(opt.restarts)
    return argparse.Namespace(
        transistor_dir=Path(transistor_dir),
        loadpull_xlsx=Path(cfg.transistor.loadpull_xlsx),
        output_dir=Path(output_dir),
        reuse_dir=[],
        # No external seeding by default (clean, self-contained run).
        previous_top_csv=Path(output_dir) / "_no_previous_seed.csv",
        previous_seed_count=0,
        requested_random_seed_count=0,
        random_seed_count=max(50, int(opt.popsize) * 4),
        l13_model=str(opt.lb_model),  # kernel namespace key stays l13_model; this is the paper's L_b (L12/L34)
        allow_extrapolation=bool(opt.allow_extrapolation),
        line_length_scale=None,
        gnd_width_factor=None,
        target_lo_ghz=float(tgt.band_lo_ghz),
        target_hi_ghz=float(tgt.band_hi_ghz),
        fallback_lo_ghz=float(tgt.band_lo_ghz) + 3.0,
        fallback_hi_ghz=float(tgt.band_hi_ghz) - 3.0,
        peak_lo_ghz=float(tgt.peak_lo_ghz),
        peak_hi_ghz=float(tgt.peak_hi_ghz),
        gain_min_db=float(tgt.gain_lo_db),
        gain_max_db=float(tgt.gain_hi_db),
        length_limit_um=float(tgt.total_length_budget_um),
        max_evals=total_evals,
        restarts=int(opt.restarts),
        popsize=int(opt.popsize),
        sigma=float(opt.sigma0),
        seed=int(opt.seed),
        polish_rounds=int(opt.polish_rounds),
        polish_top_k=int(opt.polish_top_k),
        polish_radius=int(opt.polish_radius),
        top_n=20,
        no_plots=bool(no_plots),
    )


def _stage_transistors(cfg: Config, staging_dir: Path) -> Path:
    """Copy the user's transistor S-parameters under the expected filenames."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    driver = Path(cfg.transistor.driver_s4p)
    power = Path(cfg.transistor.power_s4p)
    for src in (driver, power):
        if not src.exists():
            raise FileNotFoundError(f"Transistor S-parameter file not found: {src}")
    shutil.copyfile(driver, staging_dir / _DRIVER_NAME)
    shutil.copyfile(power, staging_dir / _POWER_NAME)
    return staging_dir


def run_synthesis(cfg: Config, *, dry_run: bool = False, no_plots: bool = False) -> Path | argparse.Namespace:
    """Run the full BC-QACL-ATD synthesis for ``cfg``.

    With ``dry_run=True`` the resolved optimizer Namespace is returned without
    executing the search (no data files are read) -- useful for validating the
    wiring/config. Otherwise the search runs and the output directory is
    returned.
    """
    output_dir = Path(cfg.paths.output_dir)
    staging_dir = output_dir / "_transistor_staging"

    lb_json = getattr(cfg.anchors, "lb_law_json", None)
    custom_lb = bool(lb_json) and Path(lb_json).exists()

    if dry_run:
        # Use the loadpull/transistor parent as a nominal transistor_dir for display.
        ns = build_namespace(cfg, transistor_dir=Path(cfg.transistor.driver_s4p).parent,
                             output_dir=output_dir, no_plots=no_plots)
        ns.custom_lb_law = lb_json if custom_lb else None
        return ns

    # Cross-technology re-calibration: if a fitted L_b law is supplied, inject it
    # in place of the embedded law before the search runs.
    if custom_lb:
        from .lb_law import LogTrilinearLbLaw
        from . import recalibrate
        recalibrate.install_custom_lb_law(LogTrilinearLbLaw.load_json(lb_json))

    transistor_dir = _stage_transistors(cfg, staging_dir)
    ns = build_namespace(cfg, transistor_dir=transistor_dir, output_dir=output_dir, no_plots=no_plots)
    opt = _optimizer_module()
    return Path(opt.run(ns))

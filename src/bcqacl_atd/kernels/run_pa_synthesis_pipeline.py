#!/usr/bin/env python3
"""Run the ML/predicted-model assisted PA matching synthesis pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from pa_synthesis.config import (
    PaSynthesisConfig,
    config_to_dict,
    load_config,
    save_resolved_config,
)
from pa_synthesis.data_loaders import load_loadpull_zopt, load_transistor_s4p
from pa_synthesis.predictors import InductorPredictor, SgDvclProvider, TransformerPredictor
from pa_synthesis.report import write_json
from pa_synthesis.synthesis import run_full_pa_synthesis, run_omn_synthesis


def export_best_gain_response(
    final_best: dict,
    freq_ghz,
    cfg: PaSynthesisConfig,
    out_dir: Path,
) -> Path:
    """Plot and export the best full-PA S21 response."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gains = []
    for freq in freq_ghz:
        tag = f"{freq:g}GHz".replace(".", "p")
        gains.append(float(final_best[f"s21_db_{tag}"]))

    csv_path = out_dir / "best_gain_response.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("freq_ghz,s21_db\n")
        for freq, gain in zip(freq_ghz, gains):
            handle.write(f"{float(freq):.12g},{gain:.12g}\n")

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    ax.plot(freq_ghz, gains, marker="o", linewidth=2.0, label="Best PA S21")
    ax.axhspan(
        cfg.objectives.gain_low_db,
        cfg.objectives.gain_high_db,
        color="#80b1d3",
        alpha=0.22,
        label=f"Target {cfg.objectives.gain_low_db:g}-{cfg.objectives.gain_high_db:g} dB",
    )
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S21 (dB)")
    ax.set_title("Best Synthesized PA Gain Response")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / "best_gain_response.png"
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PA matching-network synthesis pipeline.")
    parser.add_argument("--config", default=None, help="Optional JSON/YAML config file.")
    parser.add_argument(
        "--stage",
        choices=["all", "omn", "full"],
        default="all",
        help="Pipeline stage to run. 'full' reuses omn_best_solution.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load config/data and print candidate counts only.")
    parser.add_argument("--max-omn-candidates", type=int, default=None, help="Debug limit for OMN candidates.")
    parser.add_argument("--max-full-pairs", type=int, default=None, help="Debug limit for IMN/ISMN pairs.")
    parser.add_argument(
        "--use-default-inductors",
        action="store_true",
        help="Use the built-in bilinear_L_predictor.py corner table instead of loading configured corners.",
    )
    parser.add_argument(
        "--no-generate-s4p",
        action="store_true",
        help="Do not call Cal_0423.py for missing SG-DVCL S4P files.",
    )
    return parser.parse_args()


def apply_cli_overrides(cfg: PaSynthesisConfig, args: argparse.Namespace) -> PaSynthesisConfig:
    if args.max_full_pairs is not None:
        cfg.search.full_pa_max_pairs = args.max_full_pairs
    if args.use_default_inductors:
        cfg.paths.inductor_corners = ""
    if args.no_generate_s4p:
        cfg.sg_dvcl.generate_missing_s4p = False
    return cfg


def candidate_counts(cfg: PaSynthesisConfig) -> tuple[int, int]:
    from pa_synthesis.config import geometry_candidates

    n = len(geometry_candidates(cfg.grid))
    full = n * n
    if cfg.search.full_pa_max_pairs is not None:
        full = min(full, cfg.search.full_pa_max_pairs)
    return n, full


def main() -> int:
    t0 = time.perf_counter()
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, out_dir / "resolved_pa_synthesis_config.json")

    loadpull = load_loadpull_zopt(
        cfg.paths.loadpull_excel,
        f_start_ghz=cfg.frequency.start_ghz,
        f_stop_ghz=cfg.frequency.stop_ghz,
    )
    omn_count, full_count = candidate_counts(cfg)
    print(f"Loaded load-pull targets: {len(loadpull.freq_ghz)} points")
    print(f"OMN candidates: {omn_count}")
    print(f"IMN/ISMN pairs: {full_count}")

    if args.dry_run:
        print(json.dumps(config_to_dict(cfg), indent=2))
        return 0

    inductor_corners = Path(cfg.paths.inductor_corners) if cfg.paths.inductor_corners else None
    inductor_predictor = InductorPredictor(inductor_corners)
    sg_provider = SgDvclProvider(cfg, loadpull.freq_hz)
    transformer_predictor = TransformerPredictor(cfg, loadpull.freq_hz, inductor_predictor, sg_provider)

    omn_best = None
    if args.stage in {"all", "omn"}:
        omn_best, _ = run_omn_synthesis(
            cfg,
            loadpull,
            transformer_predictor,
            max_candidates=args.max_omn_candidates,
        )
        print(f"Best OMN: W={omn_best['W_um']}, R={omn_best['R']}, loss={omn_best['loss']:.6g}")

    if args.stage in {"all", "full"}:
        if omn_best is None:
            omn_path = out_dir / "omn_best_solution.json"
            if not omn_path.exists():
                raise FileNotFoundError(f"Missing OMN result for --stage full: {omn_path}")
            with omn_path.open("r", encoding="utf-8") as handle:
                omn_best = json.load(handle)

        driver = load_transistor_s4p(
            cfg.paths.driver_transistor_s4p,
            target_freq_hz=loadpull.freq_hz,
            z0=float(cfg.transformer.z0),
            name="driver_2x12",
        )
        final = load_transistor_s4p(
            cfg.paths.final_transistor_s4p,
            target_freq_hz=loadpull.freq_hz,
            z0=float(cfg.transformer.z0),
            name="final_2x18",
        )
        final_best, _ = run_full_pa_synthesis(
            cfg,
            loadpull,
            transformer_predictor,
            driver_s4p=driver,
            final_s4p=final,
            omn_best=omn_best,
        )
        plot_path = export_best_gain_response(final_best, loadpull.freq_ghz, cfg, out_dir)
        print(
            "Best PA: "
            f"OMN(W={final_best['OMN']['W_um']}, R={final_best['OMN']['R']}), "
            f"IMN(W={final_best['IMN']['W_um']}, R={final_best['IMN']['R']}), "
            f"ISMN(W={final_best['ISMN']['W_um']}, R={final_best['ISMN']['R']}), "
            f"loss={final_best['loss']:.6g}"
        )
        print(f"Best gain response plot: {plot_path}")
    else:
        write_json(out_dir / "final_best_solution.json", {"OMN": omn_best, "note": "Only OMN stage was run."})

    elapsed_s = time.perf_counter() - t0
    write_json(
        out_dir / "run_summary.json",
        {
            "elapsed_seconds": elapsed_s,
            "elapsed_minutes": elapsed_s / 60.0,
            "stage": args.stage,
            "used_default_bilinear_corners": bool(args.use_default_inductors or not cfg.paths.inductor_corners),
            "omn_candidates": omn_count,
            "full_pa_pairs": full_count,
            "max_omn_candidates_cli": args.max_omn_candidates,
            "max_full_pairs_cli": args.max_full_pairs,
            "output_dir": str(out_dir),
        },
    )
    print(f"Total elapsed time: {elapsed_s:.3f} s ({elapsed_s / 60.0:.3f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

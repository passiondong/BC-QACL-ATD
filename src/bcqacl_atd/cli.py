"""Command-line interface for BC-QACL-ATD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import Config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bcqacl-atd",
        description="Bridge-Compensated Quasi-Analytical Coupled-Line Automated Transformer Design.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Write a starter config YAML.")
    p_init.add_argument("--out", "-o", default="bcqacl_config.yaml")

    p_run = sub.add_parser("run", help="Run the synthesis flow from a config.")
    p_run.add_argument("--config", "-c", required=True, help="Path to the YAML/JSON config.")
    p_run.add_argument("--dry-run", action="store_true", help="Resolve and print the optimizer arguments without running.")
    p_run.add_argument("--no-plots", action="store_true", help="Skip Pareto/S-parameter figure generation.")

    p_cal = sub.add_parser("calibrate-lb", help="Fit the L_b law from your own 27 anchor EM files (new technology).")
    p_cal.add_argument("--anchor-dir", required=True, help="Folder with full_*.s6p and half_*.s4p anchors.")
    p_cal.add_argument("--out", "-o", default="lb_law.json", help="Output JSON for the fitted law.")
    p_cal.add_argument("--full-tf-glob", default="full_*.s6p")
    p_cal.add_argument("--half-tf-glob", default="half_*.s4p")
    p_cal.add_argument("--first-order", action="store_true", help="Fit the 4-coefficient (first-order) law instead of the full 8.")
    p_cal.add_argument("--band-lo-ghz", type=float, default=11.0)
    p_cal.add_argument("--band-hi-ghz", type=float, default=107.0)

    args = parser.parse_args(argv)

    if args.cmd == "init":
        Config().save(args.out)
        print(f"Wrote starter config to {args.out}")
        return 0

    if args.cmd == "run":
        from . import flow  # deferred import (pulls vendored kernels only when needed)

        cfg = Config.load(args.config)
        result = flow.run_synthesis(cfg, dry_run=args.dry_run, no_plots=args.no_plots)
        if args.dry_run:
            print("Resolved optimizer namespace (dry run, nothing executed):")
            print(json.dumps({k: str(v) for k, v in vars(result).items()}, indent=2))
        else:
            print(f"Synthesis complete. Output directory:\n  {result}")
        return 0

    if args.cmd == "calibrate-lb":
        from . import recalibrate

        law, table = recalibrate.recalibrate_lb_law(
            args.anchor_dir,
            full_glob=args.full_tf_glob,
            half_glob=args.half_tf_glob,
            full_model=not args.first_order,
            band_ghz=(args.band_lo_ghz, args.band_hi_ghz),
            out_json=args.out,
            verbose=True,
        )
        print(f"\nFitted L_b law from {len(table)} anchors -> {args.out}")
        print(f"Reference it from your config:\n  anchors:\n    lb_law_json: {args.out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

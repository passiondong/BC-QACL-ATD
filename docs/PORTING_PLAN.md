# Porting plan: research code → open-source BC-QACL-ATD

This document is the build roadmap for turning the research tree
`Code/V-MCLIN_Cal/` into this clean, reproducible, publishable package. It
records the dependency analysis, the source→module mapping, the de-pathing
work, and the staged checklist.

## 1. Dependency closure (paper-reproducing flow)

The CMA-ES Pareto entry point
`optimize_predicted_pa_global_relative_3db_len480_pareto_cal0529_speed.py`
(plus `Cal_0529_speed`, the `L_b` validator, `run_pa_synthesis_pipeline`,
`six_port_prediction_model_v3`) has a **43-module / ~22.3k-LOC** import closure.
Most of it is legacy variants pulled in for a few shared helpers.

### Architecture decision — **hybrid**
- **Vendor verbatim** (only de-path) the irreplaceable numeric kernels.
- **Re-implement cleanly** the orchestration (config/objective/CMA-ES/Pareto/flow/UI).
- **Drop** the legacy `cal0521`/`cal0524` optimizer + `tf_analysis_pipeline_cli_*`
  variants that exist only because the entry point imported older siblings.

## 2. Source → module mapping

### VENDOR → `src/bcqacl_atd/kernels/` (verbatim + de-path only)
| Source | Role |
|---|---|
| `Cal_0423.py` | SG-CL quasi-TEM cross-section solver (Poisson FEM, modal LC/RLGC) |
| `Cal_0509.py`, `Cal_0520.py` | config builder + frequency/geometry plumbing over Cal_0423 |
| `Cal_0529_v2.py` | half-transformer→SG-CL geometry rule (0.75/1.8, `l_eff`), baseline params |
| `Cal_0529_speed.py` | in-memory cache wrapper (the paper solver entry) |
| `TF_Cal_S6P_Predicting_0504.py` | scikit-rf six-port assembly (2×SG-CL + bridges L13/L24/L56) |
| `run_three_tf_v3_pa_cascade.py`, `tf_analysis_pipeline_cli_0529_v2.py` | six-port build + cascade helpers used by the cal0529 flow |
| `bilinear_L_predictor.py`, `VCL_length_calculator.py`, `fit_simple_rounded_from_fit_data_20260511.py` | inductance/MSL helpers used by the assembler |

### RE-IMPLEMENT (clean) → `src/bcqacl_atd/`
| New module | Replaces / distills | Status |
|---|---|---|
| `config.py` | scattered constants, `pa_synthesis/config.py`, CLI args | ✅ done |
| `lb_law.py` | `validate_l13_27anchor_multilinear.py` (log-trilinear fit/predict + anchors) | ✅ done + tested |
| `data_io.py` | `pa_synthesis/data_loaders.py` (transistor s4p, load-pull xlsx, anchor Touchstone) | ⬜ |
| `model.py` | `six_port_prediction_model_v3.py` (geometry → six-port via kernels + `L_b`) | ⬜ |
| `cascade.py` | `pa_synthesis/network_utils.py` (cascade, tap grounding/open, balun, `Z_diff/2`) | ⬜ |
| `objectives.py` | optimizer `relative_3db_metrics` + `pa_synthesis/objectives.py` (G, B, Z, scalar U) | ⬜ |
| `cmaes.py` | optimizer `cma_es` (self-contained) | ⬜ |
| `pareto.py` | optimizer `nondominated_mask`, `coordinate_polish` | ⬜ |
| `flow.py` | the 6-step orchestration (`GlobalLengthParetoEvaluator`, run loop) | ⬜ |
| `cli.py` | the optimizer `main()` / argparse | ⬜ |
| `app/wizard.py` | new Streamlit step-wizard | ⬜ |

### DROP (legacy, not needed for the clean cal0529 flow)
`optimize_predicted_pa_{cascade_cmaes, global_anchor_objective_cal0521_v2,
global_center_band[_cal0521], local_*cal0521*}`, `tf_analysis_pipeline_cli{,_0520,
_0521,_0521_v2,_v3}`, `Cal_0504/0510/0511v3/0521/0521_v2/0524_speed`,
`run_pa_synthesis_pipeline_0504`, `six_port_prediction_model{,_v3}` (kept only as
reference for `model.py`), `sgdvcl_length_calculator`, `TF_Cal_S6P_Predicting`
(superseded by `_0504`). Re-add individually only if a vendored kernel imports it.

## 3. De-pathing (hardcoded → config) — DONE

The vendored kernels originally carried author-local **absolute path defaults**
(machine-specific output and data roots, including an institutional OneDrive
path). Before publication these were **scrubbed to relative placeholders**
(see the pre-release scrub). They were only ever *defaults*: at run time every
data/output location comes from `Config` via the optimizer Namespace that
`flow.build_namespace` constructs, so no kernel default is used by the clean
flow. The design-box / requested-box / grid constants remain in the vendored
optimizer and are driven by `design_space`; tap terminations (L24 open / L56
short) are kernel constants.

Pre-publication hygiene: no absolute paths, author name, institution, or
machine-specific roots remain in the source tree.

## 4. Build stages (checklist)

- [x] Repo skeleton, LICENSE (MIT), README, pyproject, requirements, .gitignore
- [x] `config.py` schema (tested: YAML roundtrip)
- [x] `lb_law.py` log-trilinear law + 27-anchor selection (tested: R²=1.0 recovers paper β)
- [x] Vendor kernels into `kernels/` (22 files + `pa_synthesis`) + `sys.path` shim (tested: all import; SG-CL solver runs in-memory in ~0.7 s)
- [x] `model.py` (geometry → six-port) — tested end-to-end (paper L_b coeffs → valid 6-port in ~2 s; L24 open / L56 short)
- [x] **Vendor the reproduction closure** (39 files + `pa_synthesis`): the exact
      CMA-ES Pareto optimizer (cal0529) imports cleanly under the shim,
      with callable `run()`/`parse_args()`. *(Decision change vs §1: to guarantee
      bit-faithful reproduction, the orchestration — cascade, G/B/Z objectives,
      CMA-ES, Pareto, polish — is vendored rather than re-implemented. The repo
      stays clean at the **interface** layer below; the exact flow is driven from it.)*
- [x] `flow.py` (Config → exact optimizer Namespace → `run()`, stages transistor
      files) + `cli.py` — **dry-run verified** (config maps to
      the exact knobs: target 30–80, gain 16–20, 1200 evals, seed/polish/L13-model)
- [x] `app/wizard.py` (Streamlit 6-step UI driving `flow.run_synthesis`) — syntax-verified (launch needs `pip install streamlit`)
- [x] `data_io.py` helpers + synthetic demo data + `docs/data.md`
- [x] **End-to-end smoke PASSED** — full synthesis on synthetic data in ~57 s (17 candidates, 12 hard-feasible, 4-point Pareto front, all CSVs written)
- [x] `configs/demo_synthetic.yaml` + `configs/paper_repro.yaml`
- [x] `tests/test_core.py` (lb_law / config / anchors) + `CITATION.cff`
- [x] **Cross-technology L_b re-calibration** (`recalibrate.py`: extract L_b per anchor →
      fit law → `install_custom_lb_law`; CLI `calibrate-lb`; `anchors.lb_law_json` flow hook;
      wizard Step-3 button) — verified: 27 synthetic anchors recover the law at R²=1.0
- [x] GitHub Actions CI (`.github/workflows/ci.yml`, pytest on 3.10–3.12) + `GETTING_STARTED.md`
- [x] `CONTRIBUTING.md`, `GETTING_STARTED.md`, `CITATION.cff`, `.gitignore`/`.gitattributes`, PyPI metadata (classifiers, scripts) — repo `git init` + initial commits done
- [ ] Deferred by decision: ship synthetic demo data (clearly labeled in `examples/README.md`); add a **publishable real transformer-EM anchor set** later (transistor/load-pull stay user-supplied — NDA). `python -m build` + `twine upload` to PyPI when ready (steps in CONTRIBUTING).

## 5. Calculator

The published artifact is the `Cal_0529_speed` kernels plus the clean flow above
(config / CLI / Streamlit wizard), reproducing the paper.

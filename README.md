# BC-QACL-ATD

**Bridge-Compensated Quasi-Analytical Coupled-Line — Automated Transformer Design**

An EM-free inner-loop synthesis flow for millimeter-wave, transformer-coupled
power-amplifier (PA) matching networks. It accompanies the paper:

> Q. Dong *et al.*, "Bridge-Compensated Quasi-Analytical Coupled-Line Model for
> Automated Transformer Design: A 34.5–78.5 GHz SiGe Power Amplifier
> Demonstration," *IEEE Trans. Microw. Theory Techn.* (under review).

The tool maps an on-chip transformer's geometry `(W_TF, α_L/W, α_wc/W)` directly
to its multiport scattering parameters using a physics-based, bridge-compensated
quasi-analytical coupled-line (BC-QACL) model, then uses that model as the fast,
EM-free evaluator inside a CMA-ES search that co-optimizes a PA's input,
interstage, and output matching networks against PA-level gain, bandwidth, and
load-pull (`Z_OPT`) objectives. A full electromagnetic (EM) sweep is needed only
to (a) calibrate the model from 27 anchor geometries once per technology and
(b) sign off the final selected design.

> **Status:** active build. The portable core (config schema, log-trilinear
> `L_b` law, anchor selection) is implemented and tested. The vendored SG-CL
> field solver / six-port assembler and the orchestration + Streamlit GUI are
> being wired in — see [`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md).

---

## The design flow (mirrors the paper)

The package, CLI, and Streamlit wizard all follow the same six steps:

| Step | What you provide / what happens | Config section |
|---|---|---|
| **1. Specifications** | Process/metal stack; PA targets (band, gain window, length budget); pre-biased transistor S-parameters and the single-transistor load-pull `Z_OPT(f)`. Stability of each biased stage must hold above at least half the lowest in-band frequency. | `technology`, `target`, `transistor` |
| **2a. Design space** | The transformer width range (primary), plus aspect-ratio `α_L/W` and line-width-ratio `α_wc/W` ranges — a subset of the model's validated box. | `design_space` |
| **2b. Anchors + calibration** | EM-simulate the **27** anchor transformers (3×3×3 min/center/max) + their half-transformers; the tool extracts `L_b`, fits the log-trilinear law, and assembles the scikit-rf six-port prediction model. | `anchors` |
| **3–5. Search** | CMA-ES drives the EM-free six-port model inside the PA cascade (primary/lower tap grounded, secondary/upper tap open), scores each candidate on gain `G`, 3-dB bandwidth overlap `B`, and OMN `Z_in→Z_OPT` RMS `Z`, and archives a ranked list. | `optimizer` |
| **Pareto select** | When the budget is spent, the `(G, B, Z)` Pareto front is extracted; you pick the design that matches your priority. | — |
| **6. EM verify + remediate** | EM-verify the selected geometry. If it meets spec → tape out. If not → relax specs / adjust the active core / change architecture, then re-run. | — |

### How the power target is handled (small-signal model)

The BC-QACL model is small-signal, so a saturated-power (`P_sat`) target is met
**indirectly**: the output matching network (OMN) is driven, under an ideal-balun
excitation, so that its differential input impedance — referred to one side as
`Z_diff/2` — tracks the load-pull-optimal `Z_OPT` of a single power-device
transistor. You therefore supply `Z_OPT(f)` (single-ended) instead of a `P_sat`
number.

---

## Install

```powershell
git clone https://github.com/passiondong/BC-QACL-ATD.git bcqacl-atd
cd bcqacl-atd
py -m venv .venv                    # Windows launcher (python3 on macOS/Linux)
Set-ExecutionPolicy -Scope Process -Bypass -Force   # allow venv activation this session
.\.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[gui]"   # deps + Streamlit wizard + the bcqacl-atd command
```

Python ≥ 3.10. After the prompt shows `(.venv)`, the `streamlit` and `bcqacl-atd`
commands work. **No venv?** Run everything through the launcher instead:
`py -m pip install -r requirements.txt`, then
`py -m streamlit run src/bcqacl_atd/app/wizard.py`, and
`$env:PYTHONPATH="src"; py -m bcqacl_atd.cli run --config configs/example_pa_30_80GHz.yaml`.

## Usage

**Streamlit wizard (recommended):**
```bash
streamlit run src/bcqacl_atd/app/wizard.py
```
A step-by-step UI walks you through Steps 1–6, validates your data, runs the
calibration and CMA-ES search, shows the Pareto front, and exports the chosen
design + an EM-verification checklist.

**CLI / config-driven:**
```bash
bcqacl-atd run --config configs/example_pa_30_80GHz.yaml
```

**Library:**
```python
from bcqacl_atd import LogTrilinearLbLaw, select_anchor_grid
from bcqacl_atd.config import Config
cfg = Config.load("configs/example_pa_30_80GHz.yaml")
```

## Data you must supply

Large/EM/measurement data is **not** shipped (see `.gitignore`). Place your own:

- `data/transistor/driver_*.s4p`, `data/transistor/power_*.s4p` — single-ended
  4-port S-parameters of each biased stage (Z0 = 50 Ω).
- `data/loadpull/*.xlsx` — load-pull table with `freq`, `Zopt_single_re`,
  `Zopt_single_im` columns.
- `data/em_anchors/full_*.s6p` and `data/em_anchors/half_*.s4p` — the 27 anchor
  full-transformer six-ports and half-transformer four-ports.

See [`docs/data.md`](docs/data.md) for the exact naming/format conventions.

## Reproducing the paper

The repo ships the **calibrated `L_b` coefficients** and the design-box / CMA-ES
settings used in the paper. With your (or the provided) anchor EM data and the
two transistor blocks, `bcqacl-atd run --config configs/paper_repro.yaml`
reproduces the selected geometry
`IMN (111.5 µm, 1.38, 0.250) / ISMN (103.0 µm, 1.28, 0.235) / OMN (101.0 µm, 1.78, 0.245)`
and the `(G, B, Z)` Pareto front.

## Re-calibrating L_b for a new technology

The embedded L_b law is calibrated for the paper's 130-nm SiGe stack. For a **new
technology / design box**, re-fit the log-trilinear law from your own 27 anchor
EM files (`full_*.s6p` + `half_*.s4p`, 3×3×3 min/center/max — pick them with
`bcqacl_atd.lb_law.select_anchor_grid`):

```bash
bcqacl-atd calibrate-lb --anchor-dir data/em_anchors --out lb_law.json
```
This extracts L_b per anchor (assembler-fit of the half-transformer + bridge to
your full-transformer EM), fits the law, and writes `lb_law.json`. Point your
config at it and the synthesis uses it automatically:

```yaml
anchors:
  lb_law_json: lb_law.json
```
(also available as a button in Step 3 of the wizard, or via
`bcqacl_atd.recalibrate.recalibrate_lb_law(...)` / `install_custom_lb_law(...)`).
Also update `technology` (stack ε_r/σ/thickness, SG-CL 0.75/1.8, C_l) for the new
process.

## Architecture (hybrid: vendored kernels + clean layer)

To stay both *reproducible* and *publishable*, the numerically exact research
code — the SG-CL quasi-TEM field solver, the six-port assembler, **and** the
CMA-ES Pareto search that produced the paper's results — is **vendored verbatim**
under `src/bcqacl_atd/kernels/` (data/output paths are supplied from your Config,
not hard-coded). The **clean, documented, config-driven layer** you actually use
(`config`, `lb_law`, `model`, `flow`, `cli`, `app`) sits on top and drives that
exact flow. This guarantees bit-faithful reproduction while keeping the public
interface small and readable. See [`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md).

## License

MIT — see [`LICENSE`](LICENSE).

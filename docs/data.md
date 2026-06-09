# Input data formats

The repo ships **no** proprietary EM/measurement data. Supply your own (paths
are set in the config / Step 2 of the wizard). To try the tool with **synthetic**
data instead, run:

```python
from bcqacl_atd.data_io import make_demo_dataset
make_demo_dataset("examples/demo_data")
```
(or click *Generate synthetic demo data* in the wizard) and use
`configs/demo_synthetic.yaml`.

## 1. Biased transistor S-parameters (required)

- **Format:** Touchstone `.s4p`, single-ended **4-port**, normalized to 50 Ω.
- **Port order:** `[in+, in-, out+, out-]` (differential in / differential out).
- **One file per stage:** the driver stage and the power stage.
- **Bias/stability:** export at the operating bias, with each stage
  unconditionally stable above **at least half the lowest in-band frequency**,
  so the cascade stays stable across the target band.
- Config keys: `transistor.driver_s4p`, `transistor.power_s4p`.

## 2. Load-pull Z_OPT(f) (required)

- **Format:** Excel `.xlsx` with (case-insensitive) columns containing
  `freq` (GHz), `Zopt_single_re` (Ω), `Zopt_single_im` (Ω).
- This is the load-pull-optimal load of **one** power-device transistor
  (single-ended). The OMN objective drives the ideal-balun OMN differential
  input impedance, referred to one side as `Z_diff/2`, toward this `Z_OPT_single`.
- Config key: `transistor.loadpull_xlsx`.

## 3. Anchor EM data (only for `lb_model: full80-log-trilinear`)

The default `lb_model: anchors8` uses the embedded calibrated L_b law and needs
**no** anchor files. To re-calibrate the log-trilinear L_b law for a new
technology/box, supply the **27 anchor geometries** (3×3×3 = min/center/max of
each of `W_TF`, `α_L/W`, `α_wc/W`; use
`bcqacl_atd.lb_law.select_anchor_grid` to choose them on your grid):

- `data/em_anchors/full_*.s6p` — the full-transformer **six-port** of each anchor
  (port order `[in1, in2, out1, out2, E1TAP, MATAP]`).
- `data/em_anchors/half_*.s4p` — the corresponding half-transformer **four-port**
  (port order `[E1_A, MA_B, E1_B, MA_A]`).
- Config keys: `anchors.anchor_dir`, `anchors.full_tf_glob`, `anchors.half_tf_glob`.

`L_b` is extracted per anchor (the half-transformer + a single bridge inductance,
assembled and fit to your full-transformer EM over the band) and the
log-trilinear law is fitted automatically by:

```bash
bcqacl-atd calibrate-lb --anchor-dir data/em_anchors --out lb_law.json
```
or `bcqacl_atd.recalibrate.recalibrate_lb_law(...)`. Reference the result via
`anchors.lb_law_json: lb_law.json`; the synthesis then uses your custom law
instead of the embedded one.

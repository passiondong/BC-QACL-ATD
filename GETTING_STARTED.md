# Getting started (5 minutes)

## Install
```powershell
git clone https://github.com/passiondong/BC-QACL-ATD.git bcqacl-atd
cd bcqacl-atd
py -m venv .venv                               # Windows launcher (python3 on macOS/Linux)
Set-ExecutionPolicy -Scope Process -Bypass -Force   # allow venv activation this session
.\.venv\Scripts\Activate.ps1                   # macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[gui]"
```
After the prompt shows `(.venv)`, the `streamlit` and `bcqacl-atd` commands work.
**Don't want a venv?** Use the launcher instead: `py -m pip install -r requirements.txt`,
then `py -m streamlit run src/bcqacl_atd/app/wizard.py`, and
`$env:PYTHONPATH="src"; py -m bcqacl_atd.cli run --config configs/demo_synthetic.yaml`.

## Try it on synthetic data (no proprietary files needed)
```bash
python -c "from bcqacl_atd.data_io import make_demo_dataset; make_demo_dataset('examples/demo_data')"
bcqacl-atd run -c configs/demo_synthetic.yaml
```
You'll get a `(G, B, Z)` Pareto front and ranked-candidate CSVs in
`outputs/demo_synthetic/` in ~1 minute.

## Use the GUI wizard
```bash
streamlit run src/bcqacl_atd/app/wizard.py
```
Walk the 6 steps: *Specifications → Transistor data → Design space + anchors →
Optimizer budget → Run + select → EM verify*. Click **Generate synthetic demo
data** in the sidebar to try it instantly.

## Run your own design
1. Put your data where the config points (see [`docs/data.md`](docs/data.md)):
   - `transistor.driver_s4p`, `transistor.power_s4p` — single-ended 4-port (50 Ω),
     stable above ≥ half the lowest in-band frequency.
   - `transistor.loadpull_xlsx` — single-transistor `Z_OPT(f)` (`freq`,
     `Zopt_single_re`, `Zopt_single_im`).
2. Edit a config (start from `configs/example_pa_30_80GHz.yaml`) and run:
   ```bash
   bcqacl-atd run -c my_config.yaml
   ```
3. EM-verify the selected geometry; tape out or remediate (Step 6).

## Reproduce the paper
`bcqacl-atd run -c configs/paper_repro.yaml` (supply the two transistor blocks +
load-pull). Uses the embedded calibrated L_b law, the paper's design box, and the
paper's CMA-ES settings (seed 20260608).

## New technology? Re-calibrate L_b
```bash
bcqacl-atd calibrate-lb --anchor-dir data/em_anchors --out lb_law.json
```
then set `anchors.lb_law_json: lb_law.json` in your config. See
[README](README.md#re-calibrating-l_b-for-a-new-technology).

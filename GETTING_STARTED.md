# Getting started (5 minutes)

> On Windows, run Python through the `py` launcher (as shown below). On
> macOS/Linux use `python3`. **No virtual environment or PowerShell policy
> change is required.**

## Install
```powershell
git clone https://github.com/passiondong/BC-QACL-ATD.git bcqacl-atd
cd bcqacl-atd
py -m pip install -e ".[gui]"
```
Python ≥ 3.10. (If the editable install errors, use `py -m pip install -r requirements.txt`.)

## See the GUI wizard (recommended)
```powershell
py -m streamlit run src/bcqacl_atd/app/wizard.py
```
This opens `http://localhost:8501` in your browser. The server keeps running in
the terminal — that's normal; press **Ctrl+C** there to stop it. In the wizard,
click **Generate synthetic demo data** in the sidebar, then walk Steps 1–6 and
press **Run** on Step 5.

## Or run from the command line (synthetic demo, ~1 min)
```powershell
py -c "from bcqacl_atd.data_io import make_demo_dataset; make_demo_dataset('examples/demo_data')"
py -m bcqacl_atd.cli run --config configs/demo_synthetic.yaml
```
Results — the `(G, B, Z)` Pareto front and ranked-candidate CSVs — land in
`outputs/demo_synthetic/`.

## Run your own design
1. Put your data where the config points (see [`docs/data.md`](docs/data.md)):
   - `transistor.driver_s4p`, `transistor.power_s4p` — single-ended 4-port (50 Ω),
     stable above ≥ half the lowest in-band frequency.
   - `transistor.loadpull_xlsx` — single-transistor `Z_OPT(f)` (`freq`,
     `Zopt_single_re`, `Zopt_single_im`).
2. Edit a config (start from `configs/example_pa_30_80GHz.yaml`) and run:
   ```powershell
   py -m bcqacl_atd.cli run --config my_config.yaml
   ```
3. EM-verify the selected geometry; tape out or remediate (Step 6).

## Reproduce the paper
```powershell
py -m bcqacl_atd.cli run --config configs/paper_repro.yaml
```
(Supply the two transistor blocks + load-pull.) Uses the embedded calibrated L_b
law, the paper's design box, and the paper's CMA-ES settings.

## New technology? Re-calibrate L_b
```powershell
py -m bcqacl_atd.cli calibrate-lb --anchor-dir data/em_anchors --out lb_law.json
```
then set `anchors.lb_law_json: lb_law.json` in your config. See
[README](README.md#re-calibrating-l_b-for-a-new-technology).

> Installed into a venv and activated it? Then you can drop the `py -m` prefix and
> just call `streamlit …` / `bcqacl-atd …`.

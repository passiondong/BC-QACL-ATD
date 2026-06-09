"""BC-QACL-ATD Streamlit step-wizard.

Run with:
    streamlit run src/bcqacl_atd/app/wizard.py

A 6-step UI that mirrors the paper's design flow and drives the exact synthesis
in :func:`bcqacl_atd.flow.run_synthesis`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run src/bcqacl_atd/app/wizard.py` without installing the package.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from bcqacl_atd.config import Config  # noqa: E402
from bcqacl_atd import data_io  # noqa: E402

st.set_page_config(page_title="BC-QACL-ATD", layout="wide")

STEPS = [
    "1 · Specifications",
    "2 · Transistor data",
    "3 · Design space + anchors",
    "4 · Optimizer budget",
    "5 · Run + select",
    "6 · EM verify + remediate",
]

if "cfg" not in st.session_state:
    st.session_state.cfg = Config()
cfg: Config = st.session_state.cfg

# --------------------------------------------------------------------------- #
st.sidebar.title("BC-QACL-ATD")
st.sidebar.caption("Bridge-Compensated Quasi-Analytical Coupled-Line Automated Transformer Design")
step = st.sidebar.radio("Flow step", STEPS, index=0)

if st.sidebar.button("Generate synthetic demo data"):
    paths = data_io.make_demo_dataset("examples/demo_data")
    cfg.transistor.driver_s4p = str(paths["driver_s4p"])
    cfg.transistor.power_s4p = str(paths["power_s4p"])
    cfg.transistor.loadpull_xlsx = str(paths["loadpull_xlsx"])
    st.sidebar.success("Demo data generated and wired into the config.")

with st.sidebar.expander("Save / load config"):
    if st.button("Save config to bcqacl_config.yaml"):
        cfg.save("bcqacl_config.yaml")
        st.success("Saved bcqacl_config.yaml")
    up = st.file_uploader("Load a YAML config", type=["yaml", "yml"], key="cfgup")
    if up is not None:
        tmp = Path("uploaded_config.yaml")
        tmp.write_bytes(up.getvalue())
        st.session_state.cfg = Config.load(tmp)
        st.rerun()


# --------------------------------------------------------------------------- #
def step1() -> None:
    st.header("Step 1 · Specifications")
    st.subheader("Process / metal stack")
    c1, c2, c3 = st.columns(3)
    cfg.technology.name = c1.text_input("Technology name", cfg.technology.name)
    cfg.technology.z0_ohm = c2.number_input("Reference Z0 (Ω)", value=float(cfg.technology.z0_ohm))
    cfg.technology.use_loss = c3.toggle("Include losses", value=cfg.technology.use_loss)
    st.caption(
        "Conductor σ (S/m): "
        + ", ".join(f"{k}={v:.3g}" for k, v in cfg.technology.sigma_S_per_m.items())
        + f" · SG-CL widths 0.75/1.8 · C_l from L_port={cfg.technology.lport_um}, "
        f"W_port={cfg.technology.wport_um}, W_open={cfg.technology.width_open_um} µm "
        "(edit via the YAML config for full control)."
    )

    st.subheader("PA-level targets")
    c1, c2, c3, c4 = st.columns(4)
    cfg.target.band_lo_ghz = c1.number_input("Band low (GHz)", value=float(cfg.target.band_lo_ghz))
    cfg.target.band_hi_ghz = c2.number_input("Band high (GHz)", value=float(cfg.target.band_hi_ghz))
    cfg.target.gain_lo_db = c3.number_input("Gain window low (dB)", value=float(cfg.target.gain_lo_db))
    cfg.target.gain_hi_db = c4.number_input("Gain window high (dB)", value=float(cfg.target.gain_hi_db))
    c1, c2, c3 = st.columns(3)
    cfg.target.peak_lo_ghz = c1.number_input("Peak-search low (GHz)", value=float(cfg.target.peak_lo_ghz))
    cfg.target.peak_hi_ghz = c2.number_input("Peak-search high (GHz)", value=float(cfg.target.peak_hi_ghz))
    cfg.target.total_length_budget_um = c3.number_input(
        "Total matching-length budget (µm)", value=float(cfg.target.total_length_budget_um)
    )
    st.info(
        "This is a **small-signal** model: the P_sat target is met indirectly by "
        "tracking the single-transistor load-pull Z_OPT(f) with the OMN "
        "(supply Z_OPT in Step 2)."
    )


def step2() -> None:
    st.header("Step 2 · Biased transistor data")
    st.markdown(
        "Provide single-ended **4-port** S-parameters for each biased stage "
        "(stable above ≥ half the lowest in-band frequency) and the "
        "single-transistor **load-pull Z_OPT(f)** Excel table."
    )
    cfg.transistor.driver_s4p = st.text_input("Driver-stage S4P path", cfg.transistor.driver_s4p)
    cfg.transistor.power_s4p = st.text_input("Power-stage S4P path", cfg.transistor.power_s4p)
    cfg.transistor.loadpull_xlsx = st.text_input("Load-pull Z_OPT Excel path", cfg.transistor.loadpull_xlsx)
    rep = data_io.validate_inputs(cfg)
    (st.success if rep["ok"] else st.warning)(
        "All inputs found." if rep["ok"] else "Missing inputs:\n- " + "\n- ".join(rep["issues"])
    )


def step3() -> None:
    st.header("Step 3 · Design space + anchors")
    st.subheader("Geometry box (subset of the validated model box)")
    for label, rng in [
        ("Transformer width W_TF (µm)", cfg.design_space.w_tf_um),
        ("Aspect ratio α_L/W", cfg.design_space.alpha_lw),
        ("Line-width ratio α_wc/W", cfg.design_space.alpha_wcw),
    ]:
        c1, c2, c3 = st.columns(3)
        rng.lo = c1.number_input(f"{label} — min", value=float(rng.lo), format="%.4f")
        rng.hi = c2.number_input(f"{label} — max", value=float(rng.hi), format="%.4f")
        rng.step = c3.number_input(f"{label} — step", value=float(rng.step), format="%.4f")
    st.subheader("L_b (bridge inductance) model")
    cfg.optimizer.l13_model = st.selectbox(
        "L13 / L_b source", ["anchors8", "full80-log-trilinear"],
        index=0 if cfg.optimizer.l13_model == "anchors8" else 1,
        help="'anchors8' uses the embedded calibrated law (no extra files). "
        "'full80-log-trilinear' fits the 8-coefficient law from your 27 anchor EM files.",
    )
    if cfg.optimizer.l13_model == "full80-log-trilinear":
        cfg.anchors.anchor_dir = st.text_input("Anchor EM directory (27 full + 27 half)", cfg.anchors.anchor_dir)
        st.caption("Anchor grid = 3×3×3 min/center/max of (W_TF, α_L/W, α_wc/W); see docs/data.md.")

    st.divider()
    st.subheader("Re-calibrate L_b for a NEW technology (optional)")
    st.caption(
        "Fit the log-trilinear L_b law from **your own** 27 anchor EM files "
        "(full_*.s6p + half_*.s4p, named like W90_R0p8_WlineR0p15). The fitted law "
        "then overrides the embedded one for the whole run. For the paper's "
        "technology you do NOT need this."
    )
    rc1, rc2 = st.columns(2)
    adir = rc1.text_input("Anchor directory", cfg.anchors.anchor_dir, key="rc_dir")
    out_json = rc2.text_input("Save fitted law to", cfg.anchors.lb_law_json or "lb_law.json", key="rc_out")
    full_model = st.toggle("Full 8-coefficient law (else first-order)", value=cfg.anchors.fit_full_model, key="rc_full")
    if st.button("Re-calibrate L_b from anchors"):
        from bcqacl_atd import recalibrate
        with st.spinner("Fitting L_b from your anchors (assembler-based, EM-free)…"):
            law, table = recalibrate.recalibrate_lb_law(adir, full_model=full_model, out_json=out_json, verbose=False)
        cfg.anchors.lb_law_json = out_json
        cfg.anchors.fit_full_model = full_model
        st.success(f"Fitted from {len(table)} anchors → {out_json}. This run will use the custom law.")
        st.code(law.formula_text())
        st.dataframe(table, use_container_width=True)
    if cfg.anchors.lb_law_json:
        st.info(f"Custom re-calibrated L_b law in use: `{cfg.anchors.lb_law_json}`")


def step4() -> None:
    st.header("Step 4 · Optimizer budget")
    o = cfg.optimizer
    c1, c2, c3 = st.columns(3)
    o.popsize = int(c1.number_input("Population λ", value=int(o.popsize), min_value=4))
    o.generations = int(c2.number_input("Generations / restart", value=int(o.generations), min_value=1))
    o.restarts = int(c3.number_input("Restarts", value=int(o.restarts), min_value=1))
    c1, c2, c3 = st.columns(3)
    o.sigma0 = float(c1.number_input("Initial step σ0", value=float(o.sigma0)))
    o.seed = int(c2.number_input("Seed", value=int(o.seed)))
    o.polish_rounds = int(c3.number_input("Polish rounds", value=int(o.polish_rounds), min_value=0))
    st.metric("Approx. CMA-ES evaluations", o.popsize * o.generations * o.restarts)
    st.caption("Weights w_G/w_B/w_Z only steer exploration; the final design is chosen from the (G,B,Z) Pareto front.")


def step5() -> None:
    st.header("Step 5 · Run synthesis + select from the Pareto front")
    cfg.paths.output_dir = st.text_input("Output directory", cfg.paths.output_dir)
    rep = data_io.validate_inputs(cfg)
    if not rep["ok"]:
        st.error("Resolve Step-2 inputs first:\n- " + "\n- ".join(rep["issues"]))
        return
    st.caption(f"L13 model: **{cfg.optimizer.l13_model}**")
    if st.button("▶ Run synthesis", type="primary"):
        from bcqacl_atd import flow
        with st.spinner("Running CMA-ES inner-loop search (EM-free)… this can take minutes."):
            out = flow.run_synthesis(cfg, no_plots=False)
        st.session_state.last_out = str(out)
        st.success(f"Done. Outputs in {out}")
    out = st.session_state.get("last_out")
    if out:
        front_csv = Path(out) / "global_len480_cal0529_pareto_front_equal_priority.csv"
        if front_csv.exists():
            df = pd.read_csv(front_csv)
            st.subheader(f"(G, B, Z) Pareto front — {len(df)} points")
            st.dataframe(df, use_container_width=True)
            for fig in sorted(Path(out).glob("*pareto*.png")):
                st.image(str(fig), caption=fig.name)


def step6() -> None:
    st.header("Step 6 · EM verification + remediation")
    st.markdown(
        """
Take the selected geometry from Step 5 and **verify by full-wave EM**:

1. Lay out the three matching transformers at the selected `(W_TF, α_L/W, α_wc/W)`.
2. EM-simulate each, cascade with the biased transistor blocks, and re-check
   gain `G`, 3-dB bandwidth `B`, and `Z_in,OMN → Z_OPT`.
3. **If it meets spec → tape out.**
4. **If not → remediate, then re-run:**
   - relax the Step-1 specifications;
   - adjust the active core (stage count, transistor periphery, or bias) while
     keeping the transformer-coupled architecture;
   - or switch architecture (e.g. LC matching / distributed amplifier).

The inner-loop model is a fast *screener*; full-wave EM remains the sign-off.
"""
    )


{
    STEPS[0]: step1, STEPS[1]: step2, STEPS[2]: step3,
    STEPS[3]: step4, STEPS[4]: step5, STEPS[5]: step6,
}[step]()

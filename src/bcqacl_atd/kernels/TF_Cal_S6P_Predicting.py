import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import skrf as rf
from skrf.circuit import Circuit
from skrf.media import MLine, DefinedGammaZ0


# ============================================================
# User settings
# ============================================================
CFG = {
    "s4p_top": r".\cal04_outputs\cal04_line.s4p",
    "s4p_bot": r".\cal04_outputs\cal04_line.s4p",
    "f_start_ghz": 1.0,
    "f_stop_ghz": 200.0,
    "f_step_ghz": 0.1,
    "z0": 50.0,
    # W90R1.3 compensation/parasitic values.
    "L13_nH": 1.37,
    "L24_nH": 0.14,
    "L56_pH": 0.013,
    "L56V_extra_pH": 5.0,
    "Wline_um": 22.5,
    "Lline_um": 4.16625,
    # Logical S4P port order fallback, used only when Touchstone port labels
    # are missing. Equivalent user-facing names are also accepted:
    #   E1_A  == in1
    #   MA_B  == out1
    #   E1_B  == E1TAP
    #   MA_A  == MATAP
    "s4p_port_order": ["E1_A", "MA_B", "E1_B", "MA_A"],
    "out_prefix": "TF_Cal_S6P_Predicting",
    # MSUB MA
    "MA": {
        "h_um": 16.72,
        "ep_r": 4.1,
        "mu_r": 1.0,
        "sigma_S_per_m": 3.57e8,
        "t_um": 4.0,
        "tand": 0.01,
    },
    # MSUB E1
    "E1": {
        "h_um": 9.62,
        "ep_r": 4.1,
        "mu_r": 1.0,
        "sigma_S_per_m": 5.56e8,
        "t_um": 3.0,
        "tand": 0.01,
    },
}


# ============================================================
# Helpers
# ============================================================
def make_frequency(f_start_ghz: float, f_stop_ghz: float, f_step_ghz: float) -> rf.Frequency:
    npoints = int(round((f_stop_ghz - f_start_ghz) / f_step_ghz)) + 1
    return rf.Frequency(f_start_ghz, f_stop_ghz, npoints, unit="GHz")



def align_network_to_freq(ntw: rf.Network, target_freq: rf.Frequency) -> rf.Network:
    port_names = list(ntw.port_names or [])
    same_grid = (
        len(ntw.f) == len(target_freq.f)
        and np.allclose(ntw.f, target_freq.f)
    )
    if same_grid:
        return ntw
    out = ntw.interpolate(target_freq)
    out.name = ntw.name
    if port_names:
        try:
            out.port_names = port_names
        except Exception:
            pass
    return out



def db20(x: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(x), 1e-30))



def gamma_to_zin(gamma: np.ndarray, z0: float) -> np.ndarray:
    denom = 1.0 - gamma
    denom = np.where(np.abs(denom) < 1e-15, 1e-15 + 0j, denom)
    return z0 * (1.0 + gamma) / denom



def make_mline(freq: rf.Frequency, sub_cfg: dict, w_um: float, z0_port: float) -> MLine:
    return MLine(
        frequency=freq,
        w=w_um * 1e-6,
        h=sub_cfg["h_um"] * 1e-6,
        t=sub_cfg["t_um"] * 1e-6,
        ep_r=sub_cfg["ep_r"],
        mu_r=sub_cfg.get("mu_r", 1.0),
        tand=sub_cfg["tand"],
        rho=1.0 / sub_cfg["sigma_S_per_m"],
        z0_port=z0_port,
    )



def line_from_mline(media: MLine, length_um: float, name: str) -> rf.Network:
    return media.line(d=length_um * 1e-6, unit="m", name=name)



def ideal_series_inductor(media: DefinedGammaZ0, L_henry: float, name: str) -> rf.Network:
    ntw = media.inductor(L_henry)
    ntw.name = name
    return ntw



def copy_named(ntw: rf.Network, name: str) -> rf.Network:
    out = ntw.copy()
    out.name = name
    return out


def _norm_port_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("-", "_")


S4P_PORT_ALIASES = {
    "E1_A": {"e1_a", "in1", "ll"},
    "MA_B": {"ma_b", "out1", "ur"},
    "MA_A": {"ma_a", "matap", "ul"},
    "E1_B": {"e1_b", "e1tap", "lr"},
}

EXPECTED_S4P_LOGICAL_ORDER = ["E1_A", "MA_B", "E1_B", "MA_A"]
EXPECTED_S4P_ORDER_TEXT = "1=E1_A/in1, 2=MA_B/out1, 3=E1_B/E1TAP, 4=MA_A/MATAP"


def canonical_s4p_port_name(name: str) -> str | None:
    """Return the canonical S4P logical name for supported aliases."""
    norm = _norm_port_name(name)
    for logical, aliases in S4P_PORT_ALIASES.items():
        if norm == _norm_port_name(logical) or norm in aliases:
            return logical
    return None


def warn_if_s4p_port_order_unexpected(
    port_labels: list[str],
    *,
    source: str,
) -> list[str]:
    """Warn when visible S4P labels do not follow the expected physical order."""
    if not port_labels:
        warnings.warn(
            f"{source}: S4P has no visible port names; expected {EXPECTED_S4P_ORDER_TEXT}. "
            "Falling back to cfg['s4p_port_order'].",
            RuntimeWarning,
            stacklevel=3,
        )
        return []

    canonical_order = [canonical_s4p_port_name(name) for name in port_labels]
    if any(name is None for name in canonical_order):
        warnings.warn(
            f"{source}: S4P port names contain unrecognized labels {port_labels!r}; "
            f"expected {EXPECTED_S4P_ORDER_TEXT}. "
            "Unrecognized positions will need cfg['s4p_port_order'] fallback.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif canonical_order != EXPECTED_S4P_LOGICAL_ORDER:
        warnings.warn(
            f"{source}: S4P port order is {port_labels!r} -> {canonical_order!r}; "
            f"expected {EXPECTED_S4P_ORDER_TEXT}. The assembler will use labels to map ports, "
            "but the file order should be checked.",
            RuntimeWarning,
            stacklevel=3,
        )
    return [name if name is not None else "" for name in canonical_order]


def infer_s4p_port_indices(ntw: rf.Network, cfg: dict | None = None) -> dict[str, int]:
    """Infer logical S4P conductor-end indices from Touchstone port names.

    The current Cal_0423.py output labels the four ports as
    ``E1_A, MA_B, E1_B, MA_A``. Other equivalent files may label the same
    physical ports as ``in1, out1, E1TAP, MATAP`` or quadrant aliases
    ``LL, UR, LR, UL``. If no names are present, ``cfg["s4p_port_order"]``
    is used as the fallback logical order and aliases are accepted there too.
    """
    cfg = cfg or {}
    names = list(ntw.port_names or [])
    canonical_order = warn_if_s4p_port_order_unexpected(names, source=ntw.name or "<s4p>")
    mapping: dict[str, int] = {}
    for idx, logical in enumerate(canonical_order):
        if logical:
            mapping[logical] = idx

    if len(mapping) == 4:
        return mapping

    fallback = list(cfg.get("s4p_port_order", ["E1_A", "MA_B", "E1_B", "MA_A"]))
    if len(fallback) != ntw.nports:
        raise ValueError(f"Invalid fallback s4p_port_order for {ntw.name}: {fallback}")
    warn_if_s4p_port_order_unexpected(fallback, source=f"{ntw.name or '<s4p>'} cfg['s4p_port_order']")
    for idx, logical in enumerate(fallback):
        canonical = canonical_s4p_port_name(str(logical))
        if canonical is None:
            canonical = str(logical)
        mapping.setdefault(canonical, idx)

    missing = sorted(set(S4P_PORT_ALIASES) - set(mapping))
    if missing:
        raise ValueError(
            f"Could not infer S4P port mapping for {ntw.name}; "
            f"port_names={names!r}, missing={missing}"
        )
    return mapping


# ============================================================
# Main circuit
# ============================================================
def build_circuit(cfg: dict) -> rf.Network:
    freq = make_frequency(cfg["f_start_ghz"], cfg["f_stop_ghz"], cfg["f_step_ghz"])
    z0 = cfg["z0"]
    ideal = DefinedGammaZ0(frequency=freq, z0=z0)

    # External 6-port order:
    #   1 in1   -> original left-balun + node, top branch
    #   2 in2   -> original left-balun - node, bottom branch
    #   3 out1  -> original right-balun + node, top branch
    #   4 out2  -> original right-balun - node, bottom branch
    #   5 E1TAP -> midpoint between the E1 compensation microstrip lines
    #   6 MATAP -> midpoint between the MA compensation microstrip lines
    port_in1 = Circuit.Port(freq, name="in1", z0=z0)
    port_in2 = Circuit.Port(freq, name="in2", z0=z0)
    port_out1 = Circuit.Port(freq, name="out1", z0=z0)
    port_out2 = Circuit.Port(freq, name="out2", z0=z0)
    port_e1tap = Circuit.Port(freq, name="E1TAP", z0=z0)
    port_matap = Circuit.Port(freq, name="MATAP", z0=z0)

    # Top and bottom S4P blocks
    s4p_top = rf.Network(cfg["s4p_top"])
    s4p_top.name = "S4P_TOP"
    s4p_top = align_network_to_freq(s4p_top, freq)

    s4p_bot = rf.Network(cfg["s4p_bot"])
    s4p_bot.name = "S4P_BOT"
    s4p_bot = align_network_to_freq(s4p_bot, freq)

    top_ports = infer_s4p_port_indices(s4p_top, cfg)
    bot_ports = infer_s4p_port_indices(s4p_bot, cfg)

    # Microstrip lines between E1TAP nodes and between MATAP nodes
    mline_e1 = make_mline(freq, cfg["E1"], cfg["Wline_um"], z0)
    mline_ma = make_mline(freq, cfg["MA"], cfg["Wline_um"], z0)

    TL35 = line_from_mline(mline_e1, cfg["Lline_um"], "TL35")
    TL36 = line_from_mline(mline_e1, cfg["Lline_um"], "TL36")
    TL37 = line_from_mline(mline_ma, cfg["Lline_um"], "TL37")
    TL38 = line_from_mline(mline_ma, cfg["Lline_um"], "TL38")

    # Added inductors on the left/right outer differential branches
    # L56V = L56 + 5, using the same interpretation as in the earlier scripts:
    #   L13 in nH, L56 / L56V in pH.
    L56V_pH = cfg["L56_pH"] + cfg["L56V_extra_pH"]
    L56V_base = ideal_series_inductor(ideal, L56V_pH * 1e-12, "L56V_base")
    L13_base = ideal_series_inductor(ideal, cfg["L13_nH"] * 1e-9, "L13_base")
    L24_base = ideal_series_inductor(ideal, cfg["L24_nH"] * 1e-9, "L24_base")

    L67 = copy_named(L56V_base, "L67")
    L66 = copy_named(L56V_base, "L66")
    L89 = copy_named(L56V_base, "L89")
    L88 = copy_named(L56V_base, "L88")
    L68 = copy_named(L13_base, "L68")
    L73 = copy_named(L13_base, "L73")
    L24_TOP_IN_E1 = copy_named(L24_base, "L24_TOP_IN_E1")
    L24_TOP_OUT_MA = copy_named(L24_base, "L24_TOP_OUT_MA")
    L24_BOT_IN_E1 = copy_named(L24_base, "L24_BOT_IN_E1")
    L24_BOT_OUT_MA = copy_named(L24_base, "L24_BOT_OUT_MA")

    # Additional L24 shunt/bridge inductors inside each S4P block:
    #   TOP: in1(index 0) <-> E1TAP(index 2), out1(index 1) <-> MATAP(index 3)
    #   BOT: in1(index 0) <-> E1TAP(index 2), out1(index 1) <-> MATAP(index 3)

    # Topology after removing the ideal baluns:
    #   Left side:
    #       in1 -- L67 -- top S4P in1
    #       in2 -- L66 -- bottom S4P in1
    #       in1 -- L68 -- in2
    #   Right side:
    #       top S4P out1 -- L89 -- out1
    #       bottom S4P out1 -- L88 -- out2
    #       out1 -- L73 -- out2
    #   Middle branches remain:
    #       top E1TAP -- TL35 -- E1TAP -- TL36 -- bottom E1TAP
    #       top MATAP -- TL37 -- MATAP -- TL38 -- bottom MATAP
    cnx = [
        # Left external input nodes at the former left-balun differential ports
        [(port_in1, 0), (L67, 0), (L68, 0)],
        [(port_in2, 0), (L66, 0), (L68, 1)],

        # Left side series access into the two S4P in1 ports
        [(L67, 1), (s4p_top, top_ports["E1_A"]), (L24_TOP_IN_E1, 0)],
        [(L66, 1), (s4p_bot, bot_ports["E1_A"]), (L24_BOT_IN_E1, 0)],

        # Right external output nodes at the former right-balun differential ports
        [(L89, 1), (port_out1, 0), (L73, 0)],
        [(L88, 1), (port_out2, 0), (L73, 1)],

        # Right side series access from the two S4P out1 ports
        [(s4p_top, top_ports["MA_B"]), (L89, 0), (L24_TOP_OUT_MA, 0)],
        [(s4p_bot, bot_ports["MA_B"]), (L88, 0), (L24_BOT_OUT_MA, 0)],

        # E1TAP branch with an external center-tap port at the midpoint
        [(s4p_top, top_ports["E1_B"]), (TL35, 0), (L24_TOP_IN_E1, 1)],
        [(TL35, 1), (TL36, 0), (port_e1tap, 0)],
        [(TL36, 1), (s4p_bot, bot_ports["E1_B"]), (L24_BOT_IN_E1, 1)],

        # MATAP branch with an external center-tap port at the midpoint
        [(s4p_top, top_ports["MA_A"]), (TL37, 0), (L24_TOP_OUT_MA, 1)],
        [(TL37, 1), (TL38, 0), (port_matap, 0)],
        [(TL38, 1), (s4p_bot, bot_ports["MA_A"]), (L24_BOT_OUT_MA, 1)],
    ]

    cir = Circuit(cnx)
    ntw = cir.network
    ntw.name = "TF_Cal_S6P_Predicting"
    ntw.port_names = ["in1", "in2", "out1", "out2", "E1TAP", "MATAP"]
    return ntw


def build_tf_6port(cfg: dict) -> rf.Network:
    return build_circuit(cfg)


def write_s6p(ntw: rf.Network, cfg: dict) -> Path:
    out_dir = Path(cfg.get("out_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = cfg.get("out_name") or cfg.get("out_prefix", "TF_Cal_S6P_Predicting")
    out_prefix = out_dir / str(out_name)
    ntw.write_touchstone(str(out_prefix))
    return out_prefix.with_suffix(".s6p")


# ============================================================
# Plot and export
# ============================================================
def plot_results(ntw: rf.Network, cfg: dict) -> None:
    f_ghz = ntw.f / 1e9
    z0 = cfg["z0"]
    port_names = ["in1", "in2", "out1", "out2", "E1TAP", "MATAP"]

    ntw.write_touchstone(cfg["out_prefix"])

    plt.figure(figsize=(10, 6))
    for idx, name in enumerate(port_names):
        plt.plot(f_ghz, db20(ntw.s[:, idx, idx]), label=f"S{name},{name}")
    plt.plot(f_ghz, db20(ntw.s[:, 2, 0]), "--", label="Sout1,in1")
    plt.plot(f_ghz, db20(ntw.s[:, 3, 1]), "--", label="Sout2,in2")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Selected S-parameters of 6-port Dual S4P + MLIN Branches")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(f"{cfg['out_prefix']}_sparams.png", dpi=220)

    plt.figure(figsize=(10, 6))
    for idx, name in enumerate(port_names):
        zin = gamma_to_zin(ntw.s[:, idx, idx], z0)
        plt.plot(f_ghz, np.real(zin), label=f"Re(Z{name})")
        plt.plot(f_ghz, np.imag(zin), "--", label=f"Im(Z{name})")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Impedance (Ohm)")
    plt.title("Single-ended Port Impedances")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{cfg['out_prefix']}_zin.png", dpi=220)

    plt.show()



def main() -> None:
    ntw = build_circuit(CFG)
    print(ntw)
    plot_results(ntw, CFG)


if __name__ == "__main__":
    main()

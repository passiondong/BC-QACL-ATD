"""Network connection and RF calculation helpers."""

from __future__ import annotations

import numpy as np
import skrf as rf
from skrf.circuit import Circuit

from .data_loaders import make_frequency


def align_network(ntw: rf.Network, freq_hz: np.ndarray, name: str | None = None) -> rf.Network:
    target = make_frequency(np.asarray(freq_hz, dtype=float))
    port_names = list(ntw.port_names or [])
    same_grid = len(ntw.f) == len(target.f) and np.allclose(ntw.f, target.f)
    out = ntw.copy() if same_grid else ntw.interpolate(target)
    if name:
        out.name = name
    if port_names and len(port_names) == out.nports:
        try:
            out.port_names = port_names
        except Exception:
            pass
    return out


def copy_named(ntw: rf.Network, name: str) -> rf.Network:
    out = ntw.copy()
    out.name = name
    return out


def ideal_dc_block(freq: rf.Frequency, z0: float, name: str) -> rf.Network:
    """Return an ideal AC-through DC-block 2-port for S-parameter analysis."""
    s = np.zeros((len(freq.f), 2, 2), dtype=complex)
    s[:, 0, 1] = 1.0
    s[:, 1, 0] = 1.0
    return rf.Network(frequency=freq, s=s, z0=z0, name=name)


def _termination_admittance(z_ohm: complex | float | str) -> complex | None:
    if isinstance(z_ohm, str):
        z_text = z_ohm.lower()
        if z_text in {"open", "inf", "infinite"}:
            return 0.0 + 0.0j
        if z_text in {"short", "ground", "gnd"}:
            return None
        return 1.0 / complex(float(z_ohm), 0.0)
    z = complex(z_ohm)
    if abs(z) == 0:
        return None
    if not np.isfinite(z):
        return 0.0 + 0.0j
    return 1.0 / z


def reduce_y_with_terminations(
    ntw: rf.Network,
    keep_ports: list[int],
    terminations: dict[int, complex | float | str],
) -> np.ndarray:
    """Return reduced Y matrices after grounding/opening/loading other ports.

    A termination of 0 ohm, ``"ground"``, or ``"short"`` fixes that port voltage
    at ground. ``np.inf`` or ``"open"`` leaves the port open. Finite impedances
    are shunt loads to ground.
    """
    nports = ntw.nports
    keep = list(keep_ports)
    if len(set(keep)) != len(keep):
        raise ValueError("keep_ports contains duplicates.")
    missing = sorted(set(range(nports)) - set(keep) - set(terminations))
    if missing:
        raise ValueError(f"Missing terminations for ports: {missing}")

    solve_ports: list[int] = []
    y_loads: list[complex] = []
    for port in range(nports):
        if port in keep:
            continue
        y_load = _termination_admittance(terminations[port])
        if y_load is None:
            continue
        solve_ports.append(port)
        y_loads.append(y_load)

    y = ntw.y
    y_reduced = np.zeros((len(ntw.f), len(keep), len(keep)), dtype=complex)
    for idx in range(len(ntw.f)):
        ymat = y[idx]
        ykk = ymat[np.ix_(keep, keep)]
        if not solve_ports:
            y_reduced[idx] = ykk
            continue

        ykt = ymat[np.ix_(keep, solve_ports)]
        ytk = ymat[np.ix_(solve_ports, keep)]
        ytt = ymat[np.ix_(solve_ports, solve_ports)] + np.diag(y_loads)
        y_reduced[idx] = ykk - ykt @ np.linalg.solve(ytt, ytk)
    return y_reduced


def differential_impedance_from_y(y2: np.ndarray) -> np.ndarray:
    """Return differential impedance for a two-node port pair.

    The excitation is ideal odd mode: V1=+0.5 V, V2=-0.5 V. The differential
    current is Idiff=(I1-I2)/2, so Zdiff=Vdiff/Idiff.
    """
    if y2.shape[1:] != (2, 2):
        raise ValueError(f"Expected Y shape (nfreq, 2, 2), got {y2.shape}.")
    v = np.asarray([0.5, -0.5], dtype=complex)
    currents = np.einsum("fij,j->fi", y2, v)
    idiff = (currents[:, 0] - currents[:, 1]) / 2.0
    return 1.0 / idiff


def z_parameters_from_y(y: np.ndarray) -> np.ndarray:
    if y.ndim != 3 or y.shape[1] != y.shape[2]:
        raise ValueError(f"Expected Y shape (nfreq, nports, nports), got {y.shape}.")
    z = np.zeros_like(y, dtype=complex)
    for idx in range(y.shape[0]):
        z[idx] = np.linalg.inv(y[idx])
    return z


def transformer_omn_input_impedances(
    tf6: rf.Network,
    *,
    load_ohm: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute single-ended OMN input impedances at in1 and in2.

    Terminations follow the agreed OMN use:
    in1/in2 are observed, out1 sees the 50-ohm load, out2 and E1TAP are
    grounded, and MATAP is open. The returned average is used by the first
    version objective against ``Zopt_single_Re/Im``.
    """
    y_pair = reduce_y_with_terminations(
        tf6,
        keep_ports=[0, 1],
        terminations={2: load_ohm, 3: 0.0, 4: 0.0, 5: np.inf},
    )
    z_pair = z_parameters_from_y(y_pair)
    z_in1 = z_pair[:, 0, 0]
    z_in2 = z_pair[:, 1, 1]
    return z_in1, z_in2, 0.5 * (z_in1 + z_in2)


def transformer_single_input_impedance(
    tf6: rf.Network,
    *,
    load_ohm: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute OMN input impedance seen by an ideal balun at ports 1/2.

    Port convention is zero-based internally:
    - ports 0/1 are the differential transistor side.
    - port 2 sees the 50-ohm load.
    - port 3 and port 4 are grounded.
    - port 5 is open.

    Returns ``(z_diff, z_single)``, where ``z_single = z_diff / 2``.
    """
    y_pair = reduce_y_with_terminations(
        tf6,
        keep_ports=[0, 1],
        terminations={2: load_ohm, 3: 0.0, 4: 0.0, 5: np.inf},
    )
    z_diff = differential_impedance_from_y(y_pair)
    return z_diff, z_diff / 2.0


def build_full_pa_network(
    *,
    freq_hz: np.ndarray,
    driver_s4p: rf.Network,
    final_s4p: rf.Network,
    imn: rf.Network,
    ismn: rf.Network,
    omn: rf.Network,
    z0: float = 50.0,
    include_dc_blocks: bool = False,
) -> rf.Network:
    """Build the two-port small-signal PA network with scikit-rf Circuit.

    Six-port transformer convention:
    0/1 = in1/in2, 2/3 = out1/out2, 4 = E1TAP, 5 = MATAP.
    Default connections do not insert DC-blocks; they can be enabled for
    comparison with earlier ADS-style experiments.
    """
    freq = make_frequency(np.asarray(freq_hz, dtype=float))

    driver = align_network(driver_s4p, freq_hz, "driver_s4p")
    final = align_network(final_s4p, freq_hz, "final_s4p")
    imn = align_network(copy_named(imn, "imn_tf6"), freq_hz)
    ismn = align_network(copy_named(ismn, "ismn_tf6"), freq_hz)
    omn = align_network(copy_named(omn, "omn_tf6"), freq_hz)

    p_in = Circuit.Port(freq, name="pa_input_50ohm", z0=z0)
    p_out = Circuit.Port(freq, name="pa_output_50ohm", z0=z0)

    def gnd(name: str) -> rf.Network:
        return Circuit.Ground(freq, name=name, z0=z0)

    def opn(name: str) -> rf.Network:
        return Circuit.Open(freq, name=name, z0=z0)

    def dcb(name: str) -> rf.Network:
        return ideal_dc_block(freq, z0, name)

    cnx: list[list[tuple[rf.Network, int]]] = []

    def connect(a: tuple[rf.Network, int], b: tuple[rf.Network, int], dcblock_name: str | None = None) -> None:
        if include_dc_blocks and dcblock_name:
            block = dcb(dcblock_name)
            cnx.append([a, (block, 0)])
            cnx.append([(block, 1), b])
        else:
            cnx.append([a, b])

    connect((p_in, 0), (imn, 2), "dcblock_imn_out1")
    cnx.extend(
        [
            [(gnd("imn_out2_ground"), 0), (imn, 3)],
            [(gnd("imn_e1tap_ground"), 0), (imn, 4)],
            [(opn("imn_matap_open"), 0), (imn, 5)],
        ]
    )
    connect((imn, 0), (driver, 0), "dcblock_imn_in1")
    connect((imn, 1), (driver, 1), "dcblock_imn_in2")

    connect((driver, 2), (ismn, 0), "dcblock_ismn_in1")
    connect((driver, 3), (ismn, 1), "dcblock_ismn_in2")
    connect((ismn, 2), (final, 0), "dcblock_ismn_out1")
    connect((ismn, 3), (final, 1), "dcblock_ismn_out2")
    cnx.extend(
        [
            [(gnd("ismn_e1tap_ground"), 0), (ismn, 4)],
            [(opn("ismn_matap_open"), 0), (ismn, 5)],
        ]
    )

    connect((final, 2), (omn, 0), "dcblock_omn_in1")
    connect((final, 3), (omn, 1), "dcblock_omn_in2")
    connect((omn, 2), (p_out, 0), "dcblock_omn_out1")
    cnx.extend(
        [
            [(gnd("omn_out2_ground"), 0), (omn, 3)],
            [(gnd("omn_e1tap_ground"), 0), (omn, 4)],
            [(opn("omn_matap_open"), 0), (omn, 5)],
        ]
    )
    ntw = Circuit(cnx).network
    ntw.name = "full_pa_2port"
    return ntw


def s21_db(ntw2: rf.Network) -> np.ndarray:
    if ntw2.nports != 2:
        raise ValueError(f"Expected a 2-port network, got {ntw2.nports} ports.")
    return 20.0 * np.log10(np.maximum(np.abs(ntw2.s[:, 1, 0]), 1e-30))

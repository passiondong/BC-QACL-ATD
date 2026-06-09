"""Data loading helpers for transistor SNP files and load-pull targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import skrf as rf


_Z_HEADER_RE = re.compile(r"^freq\s+Z\((\d+),(\d+)\)\s*$", re.IGNORECASE)
_COMPLEX_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*"
    r"([+-])\s*j\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*$"
)


@dataclass
class LoadPullData:
    frame: pd.DataFrame
    freq_ghz: np.ndarray
    freq_hz: np.ndarray
    zopt_single: np.ndarray


def make_frequency(freq_hz: np.ndarray) -> rf.Frequency:
    return rf.Frequency.from_f(np.asarray(freq_hz, dtype=float), unit="hz")


def _with_port_names(ntw: rf.Network, names: list[str] | None) -> rf.Network:
    if names and len(names) == ntw.nports:
        try:
            ntw.port_names = list(names)
        except Exception:
            pass
    return ntw


def _interpolate_preserve_names(ntw: rf.Network, target_freq_hz: np.ndarray) -> rf.Network:
    port_names = list(ntw.port_names or [])
    out = ntw.interpolate(make_frequency(np.asarray(target_freq_hz, dtype=float)))
    out.name = ntw.name
    return _with_port_names(out, port_names)


def _parse_ads_complex(text: str) -> complex:
    match = _COMPLEX_RE.match(text)
    if not match:
        normalized = text.replace(" ", "").replace("j", "j")
        try:
            return complex(normalized)
        except Exception as exc:
            raise ValueError(f"Cannot parse complex value: {text!r}") from exc
    real = float(match.group(1))
    sign = 1.0 if match.group(2) == "+" else -1.0
    imag = sign * float(match.group(3))
    return complex(real, imag)


def _split_freq_and_complex(line: str) -> tuple[float, complex]:
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Expected '<freq> <complex>', got: {line!r}")
    return float(parts[0]), _parse_ads_complex(parts[1])


def load_z_parameter_txt(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the local block-form Z(i,j) text file.

    Returns
    -------
    freq_hz:
        1D frequency array in Hz.
    z:
        Complex array with shape ``(nfreq, nports, nports)`` in ohms.
    """
    path = Path(path)
    blocks: dict[tuple[int, int], list[tuple[float, complex]]] = {}
    current_key: tuple[int, int] | None = None

    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            header = _Z_HEADER_RE.match(line)
            if header:
                current_key = (int(header.group(1)), int(header.group(2)))
                blocks[current_key] = []
                continue
            if current_key is None:
                raise ValueError(f"Data row encountered before a Z(i,j) header in {path}.")
            blocks[current_key].append(_split_freq_and_complex(line))

    if not blocks:
        raise ValueError(f"No Z(i,j) blocks found in {path}.")

    nports = max(max(i, j) for i, j in blocks)
    first_key = next(iter(blocks))
    freq_hz = np.asarray([row[0] for row in blocks[first_key]], dtype=float)
    z = np.zeros((len(freq_hz), nports, nports), dtype=complex)

    for (i, j), rows in blocks.items():
        block_freq = np.asarray([row[0] for row in rows], dtype=float)
        if len(block_freq) != len(freq_hz) or not np.allclose(block_freq, freq_hz):
            raise ValueError(f"Frequency grid mismatch in Z({i},{j}) of {path}.")
        z[:, i - 1, j - 1] = np.asarray([row[1] for row in rows], dtype=complex)

    return freq_hz, z


def load_z4p_network(
    path: str | Path,
    *,
    target_freq_hz: np.ndarray | None = None,
    z0: float = 50.0,
    name: str | None = None,
) -> rf.Network:
    freq_hz, z = load_z_parameter_txt(path)
    ntw = rf.Network(frequency=make_frequency(freq_hz), z=z, z0=z0, name=name or Path(path).stem)
    try:
        ntw.port_names = ["in_plus", "in_minus", "out_plus", "out_minus"]
    except Exception:
        pass

    if target_freq_hz is not None:
        ntw = _interpolate_preserve_names(ntw, np.asarray(target_freq_hz, dtype=float))
        ntw.name = name or Path(path).stem
    return ntw


def load_snp_network(
    path: str | Path,
    *,
    target_freq_hz: np.ndarray | None = None,
    z0: float | None = None,
    name: str | None = None,
    expected_nports: int | None = None,
    default_port_names: list[str] | None = None,
) -> rf.Network:
    """Load a Touchstone SNP file and optionally interpolate it.

    Port names embedded in the Touchstone comments are preserved. If the file
    has no names, ``default_port_names`` is applied when its length matches.
    """
    path = Path(path)
    ntw = rf.Network(str(path))
    if expected_nports is not None and ntw.nports != expected_nports:
        raise ValueError(f"{path} has {ntw.nports} ports; expected {expected_nports}.")
    ntw.name = name or path.stem
    if not ntw.port_names and default_port_names:
        _with_port_names(ntw, default_port_names)
    if z0 is not None:
        try:
            ntw.renormalize(float(z0))
        except Exception:
            pass
    if target_freq_hz is not None:
        ntw = _interpolate_preserve_names(ntw, np.asarray(target_freq_hz, dtype=float))
        ntw.name = name or path.stem
    return ntw


def load_transistor_s4p(
    path: str | Path,
    *,
    target_freq_hz: np.ndarray | None = None,
    z0: float = 50.0,
    name: str | None = None,
) -> rf.Network:
    return load_snp_network(
        path,
        target_freq_hz=target_freq_hz,
        z0=z0,
        name=name,
        expected_nports=4,
        default_port_names=["in_plus", "in_minus", "out_plus", "out_minus"],
    )


def _find_column(columns: list[str], *needles: str) -> str:
    normalized = [(col, str(col).lower()) for col in columns]
    for col, lowered in normalized:
        if all(needle.lower() in lowered for needle in needles):
            return col
    raise KeyError(f"Could not find column containing all tokens: {needles}")


def _select_sheet(path: Path, sheet_name: str | int | None) -> str | int:
    if sheet_name is not None:
        return sheet_name
    xls = pd.ExcelFile(path)
    return "LoadPull_Data" if "LoadPull_Data" in xls.sheet_names else 0


def _read_excel_auto_header(path: Path, sheet_name: str | int | None) -> pd.DataFrame:
    selected_sheet = _select_sheet(path, sheet_name)
    raw = pd.read_excel(path, sheet_name=selected_sheet, header=None)
    header_row = None
    for idx, row in raw.iterrows():
        cells = [str(value).strip().lower() for value in row.tolist() if pd.notna(value)]
        if any("freq" in cell for cell in cells):
            header_row = int(idx)
            break
    if header_row is None:
        raise ValueError(f"Could not locate a frequency header row in {path}.")

    df = raw.iloc[header_row + 1 :].copy()
    df.columns = [str(value).strip() for value in raw.iloc[header_row].tolist()]
    return df


def load_loadpull_zopt(
    path: str | Path,
    *,
    sheet_name: str | int | None = None,
    f_start_ghz: float = 30.0,
    f_stop_ghz: float = 80.0,
) -> LoadPullData:
    path = Path(path)
    df = _read_excel_auto_header(path, sheet_name)
    df = df.dropna(how="all")
    columns = list(df.columns)
    freq_col = _find_column(columns, "freq")
    zre_col = _find_column(columns, "zopt_single_re")
    zim_col = _find_column(columns, "zopt_single_im")

    out = df[[freq_col, zre_col, zim_col]].copy()
    out.columns = ["freq_ghz", "zopt_single_re_ohm", "zopt_single_im_ohm"]
    out = out.dropna()
    out["freq_ghz"] = out["freq_ghz"].astype(float)
    out["zopt_single_re_ohm"] = out["zopt_single_re_ohm"].astype(float)
    out["zopt_single_im_ohm"] = out["zopt_single_im_ohm"].astype(float)
    out = out[(out["freq_ghz"] >= f_start_ghz) & (out["freq_ghz"] <= f_stop_ghz)]
    out = out.sort_values("freq_ghz").reset_index(drop=True)

    freq_ghz = out["freq_ghz"].to_numpy(dtype=float)
    zopt = out["zopt_single_re_ohm"].to_numpy(dtype=float) + 1j * out[
        "zopt_single_im_ohm"
    ].to_numpy(dtype=float)
    return LoadPullData(
        frame=out,
        freq_ghz=freq_ghz,
        freq_hz=freq_ghz * 1e9,
        zopt_single=zopt,
    )

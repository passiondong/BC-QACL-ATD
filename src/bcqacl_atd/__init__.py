"""BC-QACL-ATD: Bridge-Compensated Quasi-Analytical Coupled-Line Automated
Transformer Design.

An EM-free inner-loop synthesis flow for millimeter-wave transformer-coupled
power amplifiers, accompanying the TMTT paper

    "Bridge-Compensated Quasi-Analytical Coupled-Line Model for Automated
     Transformer Design: A 34.5-78.5 GHz SiGe Power Amplifier Demonstration."

The package exposes the six steps of the design flow as a clean, config-driven
API (see :mod:`bcqacl_atd.flow`):

    1. Design specifications  -> technology + PA-level targets + transistor data
    2. Design space + anchors -> 3x3x3 EM calibration, log-trilinear L_b law,
                                 scikit-rf six-port assembly
    3-5. CMA-ES inner-loop search against gain / bandwidth / Z_OPT objectives
    6. EM verification + remediation

The numerically exact SG-CL field solver and six-port assembler are vendored
verbatim under :mod:`bcqacl_atd.kernels` (only de-pathed); everything above
them is a clean re-implementation so results remain reproducible.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .lb_law import LogTrilinearLbLaw, NormRange, select_anchor_grid

__all__ = ["LogTrilinearLbLaw", "NormRange", "select_anchor_grid", "__version__"]

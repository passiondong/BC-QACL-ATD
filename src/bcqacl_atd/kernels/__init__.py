"""Vendored numerical kernels for BC-QACL-ATD.

These modules are copied **verbatim** from the research tree
(``Code/V-MCLIN_Cal``) and are the numerically exact SG-CL quasi-TEM field
solver and scikit-rf six-port assembler used to produce the paper's results.
They are intentionally *not* refactored, so that reproduction is bit-faithful;
the clean, documented, config-driven API lives in the parent package
(:mod:`bcqacl_atd`).

The original files import each other with top-level names (``import Cal_0423``,
``from pa_synthesis...``).  To keep them verbatim, this package prepends its own
directory to ``sys.path`` on import, so those statements resolve to the vendored
copies.  Hard-coded data/output paths inside these kernels are overridden at run
time via :mod:`bcqacl_atd.kernels._paths` (populated from the user's Config);
they are not used by the in-memory solve path.

Provenance: the vendored revisions correspond to the ``Cal_0529_speed``
lineage. Do not edit these by hand except to de-hardcode paths; improvements
belong in the clean layer.
"""

from __future__ import annotations

import os
import sys

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _KERNEL_DIR not in sys.path:
    # Prepend so the vendored verbatim ``import Cal_0423`` etc. resolve here.
    sys.path.insert(0, _KERNEL_DIR)


def kernel_dir() -> str:
    """Absolute path of the vendored-kernel directory."""
    return _KERNEL_DIR

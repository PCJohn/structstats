"""structstats -- fast single-pass gradient structure-tensor statistics.

Mirrors tensorstats: construct once for a fixed shape, call compute() per frame.
The C++ core returns raw integer accumulators [Sxx, Syy, Sxy, count] per cell per
pyramid level; derive energy / orientation / coherence / cornerness with the
`features` module (eigenvalues are nonlinear and do not pool across scales, so
the library returns the sums, which do).

    import structstats as ss
    from structstats import features as F

    sc = ss.StructComputer(shape=(256, 256), grid=[(5, 5), (4, 4)], global_=True)
    r = sc.compute(luma_u8)          # dict: grid_0, grid_1, global
    energy = F.energy(r["grid_0"])   # (32, 32)
    theta  = F.orientation(r["grid_0"])
"""

import numpy as np

from . import structstats_core as _core
from . import features

__all__ = ["StructComputer", "features"]
__version__ = "0.1.0"


def _parse_grid(grid):
    """-> list of [exp_y, exp_x]. Accepts k, (ky,kx), or a list of those."""
    if isinstance(grid, int):
        return [[grid, grid]]
    g = list(grid)
    specs = g if isinstance(g[0], (tuple, list)) else [g]
    return [
        [int(s), int(s)] if isinstance(s, int) else [int(s[0]), int(s[1])]
        for s in specs
    ]


def _parse_stride(stride):
    if stride is None:
        return [1, 1]
    if isinstance(stride, int):
        return [stride, stride]
    return [int(stride[0]), int(stride[1])]


class StructComputer:
    """Stateful structure-tensor computer for a fixed (H, W) uint8 luma input.

    grid     : k, a (ky, kx) tuple (2^k cells/axis), or a list of those for a
               dyadic pyramid. Levels must be sorted finest->coarsest and nested
               (each coarser grid divides the finer), and the finest must divide
               the image shape -- the coarser levels are then exact sums of finer
               cells, built by aggregating the finest grid rather than re-scanning
               the image, so the image is read exactly once whatever the depth.
    stride   : None/int/(sy, sx). Subsamples which pixels accumulate; gradient
               stencil and cell boundaries stay at full resolution.
    global_  : also return a single all-pixels tensor under key "global".
    """

    def __init__(self, shape, grid, stride=None, global_=False):
        if len(shape) != 2:
            raise ValueError("shape must be 2-D (H, W) single-channel luma")
        self._shape = (int(shape[0]), int(shape[1]))
        self._grid = _parse_grid(grid)
        self._global = global_
        self._keys = [f"grid_{i}" for i in range(len(self._grid))]
        self._impl = _core._StructComputerImpl()
        self._impl.set_config(
            list(self._shape), self._grid, _parse_stride(stride), global_
        )

    def compute(self, img):
        if img.shape != self._shape:
            raise ValueError(f"shape mismatch: expected {self._shape}, got {img.shape}")
        if img.dtype != np.uint8:
            raise TypeError("input must be uint8 single-channel luma")
        lv = self._impl.compute(np.ascontiguousarray(img))
        out = {k: lv[i].copy() for i, k in enumerate(self._keys)}
        if self._global:
            out["global"] = lv[len(self._keys)].copy()
        return out

# structstats

Fast single-pass **gradient structure-tensor** statistics for images. A companion
to [`tensorstats`](https://github.com/PCJohn/tensorstats): where tensorstats
summarizes the per-cell *intensity distribution* (mean/var/skew/kurtosis),
structstats summarizes the per-cell *gradient structure* (edge energy, dominant
orientation, coherence, cornerness) -- the part of the signal moments are blind to.

It is built for the same regime: one uint8 luma plane in, raw integer accumulators
out, global + per-patch + multi-scale in a single pass, ~sub-ms on a thumbnail.

## Idea

Per pixel the 3x3 Sobel responses `Gx, Gy` are exact integers, so the three
products `Gx^2, Gy^2, Gx*Gy` and their per-cell sums are exact integers. Those
three sums *are* the cell's structure tensor `J = [[Sxx, Sxy], [Sxy, Syy]]`. The
hot loop is pure integer with no floating point. Every pixel scatters only into
the **finest** grid; coarser dyadic levels are exact sums of finer cells, so they
are built by cascade-aggregating the small finest grid rather than re-scanning the
image -- the image is read **exactly once** whatever the pyramid depth (a 4-level
pyramid costs essentially the same as the finest level alone). Sobel is applied
separably via per-row caches so neighbouring pixels reuse the vertical pass.

Everything interesting is a closed form on the summed 2x2 -- computed by the
caller, not per pixel:

| feature | from the sums |
| --- | --- |
| edge energy | `Sxx + Syy` (trace) |
| orientation | `0.5 * atan2(2 Sxy, Sxx - Syy)` |
| coherence | `sqrt((Sxx-Syy)^2 + 4 Sxy^2) / (Sxx+Syy)` |
| cornerness (Shi-Tomasi) | `lambda_min` |
| Harris | `det - k * trace^2` |

The library returns the **raw sums** because eigenvalues are nonlinear and do not
pool across scales, but the sums do (a coarse cell's sums are its children's sums).
Same "return raw, caller derives" boundary tensorstats holds.

## Use

```python
import structstats as ss
from structstats import features as F

sc = ss.StructComputer(shape=(256, 256), grid=[(5, 5), (4, 4)], global_=True)
r = sc.compute(luma_u8)            # dict: grid_0 (32x32x4), grid_1 (16x16x4), global
energy = F.energy(r["grid_0"])     # (32, 32)
theta  = F.orientation(r["grid_0"])
coh    = F.coherence(r["grid_0"])
```

`compute` returns `(cells_y, cells_x, 4)` int64 arrays, last axis
`[Sxx, Syy, Sxy, count]`. Replicate border; cell boundaries at full resolution
(stride subsamples which pixels accumulate, not the boundaries).

## Tests

```
pip install -e ".[dev]"
pytest -s        # -s prints the quality (stride) and latency numbers
```

Three tiers: bit-exact integer equality vs a numpy reference (plus multi-scale and
global invariants); synthetic semantic checks (flat / edge / corner / noise /
oriented line through the derived features); and a quality+latency tier (stride
sweeps scored by energy correlation and circular orientation error, an OpenCV
correlation sanity check, and reported per-call latency).
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

sc = ss.StructComputer(shape=(256, 256), grid=[(5, 5), (4, 4)])

# net-ready float32 maps, derived in C++ in the same single pass (channel order
# = ss.FEATURES). This is the hot path -- no Python per-cell math.
f = sc.features(luma_u8)         # dict: grid_0 (32,32,5), grid_1 (16,16,5), global (5,)
# ss.FEATURES == ("energy", "coherence", "ori_cos", "ori_sin", "cornerness")

# or the raw integer sums, if you want to derive your own quantities
r = sc.compute(luma_u8)          # dict: grid_0 (32,32,4) int64 [Sxx,Syy,Sxy,count], ...
```

`features()` returns five float32 channels per cell, all from the three sums:
`energy = trace/n`, `coherence = |l1-l2|/trace in [0,1]`, the continuous
double-angle orientation vector `ori_cos = (Sxx-Syy)/trace`,
`ori_sin = 2 Sxy/trace` (magnitude = coherence, no `atan2`, no wraparound), and
`cornerness = lambda_min/n`. Orientation is the continuous double-angle vector
(magnitude = coherence, no `atan2`, no wraparound); recover the angle when needed
as `0.5*atan2(ori_sin, ori_cos)`. Anything else is a one-liner on the raw sums
from `compute()` (e.g. Harris = `det - k*trace^2`).

**Zero-copy input.** `compute`/`features` accept a strided 2-D `uint8` view, so
framegate can pass `hsv[:, :, 2]` (one interleaved channel) directly -- it is read
in place, with no copy of the frame or the channel.

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
oriented line, plus 90-deg rotation equivariance) through the derived features); and a quality+latency tier (stride
sweeps scored by energy correlation and circular orientation error, an OpenCV
correlation sanity check, and reported per-call latency).

## Notes for efficient use

- **Construct once, reuse.** All buffers are allocated in the constructor for a
  fixed `(H, W)`; `compute`/`features` do no heap allocation. Keep one
  `StructComputer` per frame size and call it per frame -- do not rebuild it.
- **Pass the channel view directly.** `hsv[:, :, 2]` (or any strided 2-D `uint8`
  view) is read in place. Do *not* `np.ascontiguousarray`/`.copy()` it first --
  that reintroduces the ~20 us per-frame channel copy this avoids.
- **Use `features()` for the model path.** It returns net-ready float32 maps
  derived in C++ in the same pass (~10-15 us for the whole pyramid). Deriving the
  same quantities in Python costs ~10x more. Use `compute()` only when you want
  the raw sums to derive something custom.
- **Stride is the main latency lever.** stride 1 -> 2 is ~3x faster for a small,
  robust quality loss (orientation essentially unchanged). stride >= 4 aliases raw
  *energy* on sharp edges; prefilter (resize with `INTER_AREA`) before striding
  that hard. `coherence` and the orientation vector stay robust under stride.
- **Resize with `INTER_AREA`, not `INTER_NEAREST`.** Gradients are sensitive to
  nearest-neighbour aliasing in a way moments are not; area resampling gives
  cleaner orientation.
- **Pyramid depth is nearly free.** The image is read once regardless of how many
  levels; a 4-level pyramid costs ~the same as the finest level alone. Levels must
  be nested finest->coarsest with the finest dividing `(H, W)`.
- **Channel scales differ.** `coherence`, `ori_cos`, `ori_sin` are bounded
  ([0,1] / [-1,1]); `energy` and `cornerness` are unbounded per-pixel means -- apply
  a `log1p` or normalization to those two before a network.
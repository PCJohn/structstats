"""Tests for structstats, layered in three tiers (see README):

1. Exactness   -- bit-exact integer equality vs a numpy reference using the
                  identical Sobel kernel, plus structural invariants
                  (multi-scale consistency, global = grand total).
2. Semantics   -- synthetic images with known answers (flat / edge / corner /
                  noise / oriented line) checked through the derived features,
                  so we test that a channel *means* what we claim, not merely
                  that two implementations agree.
3. Quality/latency -- knob sweeps (stride) scored against the stride-1 output
                  with vector-aware metrics (energy correlation, circular
                  orientation error), an OpenCV sanity correlation, and a
                  reported per-call latency.
"""

import math
import time

import numpy as np
import pytest

import structstats as ss
from structstats import features as F

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------
# numpy reference (identical integer Sobel, replicate border, floor cells)
# --------------------------------------------------------------------------
def sobel_int(img):
    p = np.pad(img.astype(np.int64), 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (
        p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2]
    )
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (
        p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]
    )
    return gx, gy


def ref_levels(img, grid, stride):
    h, w = img.shape
    gx, gy = sobel_int(img)
    sxx, syy, sxy = gx * gx, gy * gy, gx * gy
    sy, sx = stride
    rows, cols = np.arange(0, h, sy), np.arange(0, w, sx)
    out = []
    for ky, kx in grid:
        nr, nc = 1 << ky, 1 << kx
        rc = (np.arange(h) * nr) // h
        cc = (np.arange(w) * nc) // w
        rr, ccp = np.meshgrid(rc[rows], cc[cols], indexing="ij")
        idx = (rr * nc + ccp).ravel()
        flat = np.zeros((nr * nc, 4), np.int64)
        sel = np.ix_(rows, cols)
        np.add.at(flat[:, 0], idx, sxx[sel].ravel())
        np.add.at(flat[:, 1], idx, syy[sel].ravel())
        np.add.at(flat[:, 2], idx, sxy[sel].ravel())
        np.add.at(flat[:, 3], idx, np.ones(idx.size, np.int64))
        out.append(flat.reshape(nr, nc, 4))
    return out


# --------------------------------------------------------------------------
# synthetic image builders (uint8)
# --------------------------------------------------------------------------
def flat(v=128, n=256):
    return np.full((n, n), v, np.uint8)


def hedge(n=256):  # horizontal edge -> vertical gradient
    a = np.zeros((n, n), np.uint8)
    a[n // 2 :] = 255
    return a


def vedge(n=256):
    return hedge(n).T.copy()


def diag(n=256):  # lower triangle white -> 45-deg edge
    a = np.zeros((n, n), np.uint8)
    yy, xx = np.mgrid[0:n, 0:n]
    a[yy > xx] = 255
    return a


def checker(n=256, c=16):
    yy, xx = np.mgrid[0:n, 0:n]
    return np.where(((yy // c) + (xx // c)) % 2 == 0, 0, 255).astype(np.uint8)


def noise(n=256):
    return rng.integers(0, 256, (n, n), dtype=np.uint8)


def textured(n=256):  # mixed content for correlation tests
    a = (127 + 80 * np.sin(np.mgrid[0:n, 0:n][1] / 18.0)).astype(np.uint8)
    a[40:80, 40:200] = 255
    a[120:200, 120:140] = 0
    return np.clip(a + rng.integers(-8, 8, (n, n)), 0, 255).astype(np.uint8)


GRIDS = [[(5, 5)], [(5, 5), (4, 4), (3, 3), (2, 2)], [(0, 0)], [(6, 6)]]
STRIDES = [(1, 1), (2, 2), (4, 4)]


# ============================== Tier 1: exactness =========================
@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("img", [flat(), hedge(), vedge(), diag(), checker(), noise()])
def test_exact_vs_reference(img, grid, stride):
    sc = ss.StructComputer(img.shape, grid=grid, stride=stride)
    r = sc.compute(img)
    ref = ref_levels(img, grid, stride)
    for i, t in enumerate(ref):
        assert np.array_equal(r[f"grid_{i}"], t), f"level {i} mismatch"


def test_flat_is_zero():
    sc = ss.StructComputer((256, 256), grid=[(5, 5)])
    t = sc.compute(flat())["grid_0"]
    assert t[..., :3].sum() == 0  # no gradient anywhere
    assert (t[..., 3] > 0).all()  # counts populated


def test_multiscale_consistency():
    # coarse cell sums == block-sum of finer cells (dyadic, divisible thumb)
    grid = [(5, 5), (4, 4), (3, 3)]
    sc = ss.StructComputer((256, 256), grid=grid)
    r = sc.compute(textured())
    for i in range(len(grid) - 1):
        fine, coarse = r[f"grid_{i}"], r[f"grid_{i + 1}"]
        gy, gx = coarse.shape[:2]
        fy, fx = fine.shape[0] // gy, fine.shape[1] // gx
        pooled = fine.reshape(gy, fy, gx, fx, 4).sum(axis=(1, 3))
        assert np.array_equal(pooled, coarse)


def test_global_equals_total():
    sc = ss.StructComputer((256, 256), grid=[(5, 5)], global_=True)
    r = sc.compute(textured())
    assert np.array_equal(r["global"], r["grid_0"].reshape(-1, 4).sum(0))


# ============================== Tier 2: semantics =========================
def _interior(m):  # drop border cells (clamped border distorts them)
    return m[1:-1, 1:-1]


def test_horizontal_edge():
    t = ss.StructComputer((256, 256), grid=[(5, 5)]).compute(hedge())["grid_0"]
    ori = F.orientation(t)
    coh = F.coherence(t)
    edge = _interior(t[..., 1]) > 0  # cells straddling the edge: strong Syy
    assert edge.any()
    # gradient is vertical -> |orientation| ~ pi/2
    assert np.allclose(np.abs(ori[1:-1, 1:-1][edge]), math.pi / 2, atol=0.05)
    assert (coh[1:-1, 1:-1][edge] > 0.9).all()


def test_vertical_edge():
    t = ss.StructComputer((256, 256), grid=[(5, 5)]).compute(vedge())["grid_0"]
    ori = F.orientation(t)
    coh = F.coherence(t)
    edge = _interior(t[..., 0]) > 0
    assert edge.any()
    assert np.allclose(ori[1:-1, 1:-1][edge], 0.0, atol=0.05)  # horizontal gradient
    assert (coh[1:-1, 1:-1][edge] > 0.9).all()


def test_diagonal_edge():
    t = ss.StructComputer((256, 256), grid=[(5, 5)]).compute(diag())["grid_0"]
    ori = F.orientation(t)
    energy = t[..., 0] + t[..., 1]
    on = _interior(energy) > np.percentile(energy, 90)
    vals = np.abs(ori[1:-1, 1:-1][on])
    assert np.allclose(vals, math.pi / 4, atol=0.1)


def test_corner_vs_edge():
    # checkerboard corners: both eigenvalues large -> low coherence, high min-eig
    t = ss.StructComputer((256, 256), grid=[(5, 5)]).compute(checker())["grid_0"]
    lmin = F.cornerness(t)
    coh = F.coherence(t)
    he = ss.StructComputer((256, 256), grid=[(5, 5)]).compute(hedge())["grid_0"]
    assert _interior(lmin).max() > 0
    # a straight edge has near-zero min eigenvalue; corners do not
    assert _interior(F.cornerness(he)).max() < 0.05 * _interior(lmin).max()
    assert _interior(coh).mean() < 0.9  # corners reduce coherence


def test_noise_is_isotropic():
    t = ss.StructComputer((256, 256), grid=[(4, 4)]).compute(noise())["grid_0"]
    assert _interior(F.coherence(t)).mean() < 0.3  # no dominant orientation
    assert _interior(t[..., 0] + t[..., 1]).mean() > 0  # but high energy


# ============================== Tier 3: quality + latency =================
def _circ_orient_err(a, b):  # orientation is mod pi
    d = 2 * (a - b)
    return np.abs(np.arctan2(np.sin(d), np.cos(d))) / 2.0


@pytest.mark.parametrize("stride", [(2, 2), (4, 4)])
def test_stride_quality(stride):
    # noise-free so this measures stride degradation, not noise decorrelation
    # (raw per-cell energy is noise-sensitive at small cells; orientation is not)
    n = 256
    img = (127 + 80 * np.sin(np.mgrid[0:n, 0:n][1] / 12.0)).astype(np.uint8)
    img[40:90, 40:210] = 255
    img[130:210, 110:140] = 0
    g = [(5, 5)]
    t1 = ss.StructComputer(img.shape, grid=g).compute(img)["grid_0"]
    ts = ss.StructComputer(img.shape, grid=g, stride=stride).compute(img)["grid_0"]
    e1 = (t1[..., 0] + t1[..., 1]).ravel().astype(float)
    es = (ts[..., 0] + ts[..., 1]).ravel().astype(float)
    corr = np.corrcoef(e1, es)[0, 1]
    strong = e1 > np.percentile(e1, 75)  # orientation only meaningful where energy is
    oerr = _circ_orient_err(
        F.orientation(t1).ravel()[strong], F.orientation(ts).ravel()[strong]
    )
    med_deg = math.degrees(np.median(oerr))
    print(
        f"\n  stride {stride}: energy corr={corr:.3f}  median orient err={med_deg:.2f} deg"
    )
    # orientation is the robust channel and gates tightly; raw energy aliases on
    # sharp edges under aggressive stride (would need INTER_AREA prefiltering),
    # so its floor loosens with stride -- documented, not silently tolerated.
    corr_min = 0.85 if stride == (2, 2) else 0.3
    assert corr > corr_min
    assert med_deg < 3.0


def test_opencv_sanity():
    cv2 = pytest.importorskip("cv2")
    img = textured()
    g = 32
    t = ss.StructComputer(img.shape, grid=[(5, 5)]).compute(img)["grid_0"]
    # cv2 per-pixel structure-tensor components, box-pooled to the same grid
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    comp = np.stack([gx * gx, gy * gy, gx * gy], -1)
    pooled = comp.reshape(g, img.shape[0] // g, g, img.shape[1] // g, 3).sum((1, 3))
    ours = t[..., :3].astype(float)
    for k in range(3):
        c = np.corrcoef(
            _interior(ours[..., k]).ravel(), _interior(pooled[..., k]).ravel()
        )[0, 1]
        assert c > 0.99, f"component {k} corr={c:.3f}"


def test_latency(capsys):
    img = textured()
    pyr = [(5, 5), (4, 4), (3, 3), (2, 2)]
    sc = ss.StructComputer(img.shape, grid=pyr, global_=True)
    for _ in range(20):
        sc.compute(img)  # warmup
    best = []
    for _ in range(50):
        t0 = time.perf_counter()
        sc.compute(img)
        best.append(time.perf_counter() - t0)
    best.sort()
    p50 = best[len(best) // 2] * 1e3
    with capsys.disabled():
        print(
            f"\n  latency 256x256 4-level pyramid: p50={p50:.3f} ms, min={best[0] * 1e3:.3f} ms"
        )
    assert p50 < 5.0  # generous; real target is sub-ms

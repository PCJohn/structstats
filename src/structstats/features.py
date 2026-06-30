"""Derive structural features from the raw structure-tensor sums returned by
StructComputer. Input `t` is (..., 4) = [Sxx, Syy, Sxy, count]; every function is
vectorized over the leading (cell) axes and pure numpy.

The per-cell tensor is J = [[Sxx, Sxy], [Sxy, Syy]] with eigenvalues
    lambda = (Sxx+Syy)/2 +/- sqrt(((Sxx-Syy)/2)^2 + Sxy^2).
lambda1 >= lambda2 >= 0. flat: both ~0; edge: lambda1 >> lambda2; corner: both large.
"""

import numpy as np

_EPS = 1e-12


def _parts(t):
    t = t.astype(np.float64)
    return t[..., 0], t[..., 1], t[..., 2], t[..., 3]


def energy(t, normalize=False):
    """Edge energy = trace = Sxx + Syy. With normalize, per-pixel (/ count)."""
    sxx, syy, _, n = _parts(t)
    e = sxx + syy
    return e / np.maximum(n, 1.0) if normalize else e


def _eig(t):
    sxx, syy, sxy, _ = _parts(t)
    tr = sxx + syy
    disc = np.sqrt(((sxx - syy) * 0.5) ** 2 + sxy * sxy)
    half = tr * 0.5
    return half + disc, half - disc  # lambda1, lambda2


def coherence(t):
    """(lambda1 - lambda2) / (lambda1 + lambda2) in [0, 1]: 1 = a single dominant
    orientation (clean edge), 0 = isotropic (flat / noise / corner)."""
    sxx, syy, sxy, _ = _parts(t)
    tr = sxx + syy
    disc = np.sqrt((sxx - syy) ** 2 + 4.0 * sxy * sxy)
    return np.where(tr > _EPS, disc / (tr + _EPS), 0.0)


def orientation(t):
    """Dominant gradient orientation in (-pi/2, pi/2], mod pi: 0 = horizontal
    gradient (vertical edge), +/-pi/2 = vertical gradient (horizontal edge)."""
    sxx, syy, sxy, _ = _parts(t)
    return 0.5 * np.arctan2(2.0 * sxy, sxx - syy)


def cornerness(t):
    """Shi-Tomasi cornerness = lambda2 (the smaller eigenvalue): large only when
    both eigenvalues are large, i.e. a true corner / well-conditioned feature."""
    return _eig(t)[1]


def harris(t, k=0.04):
    """Harris response = det(J) - k * trace(J)^2."""
    sxx, syy, sxy, _ = _parts(t)
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    return det - k * tr * tr

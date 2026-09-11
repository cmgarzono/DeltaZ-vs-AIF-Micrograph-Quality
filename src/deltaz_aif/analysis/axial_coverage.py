r"""
S1: Analytical axial coverage model.

Specimen-independent, renderer-independent. Depends only on the elemental
focal response

    T(f, u) = exp[-(u / w_z(f))^2]

where w_z(f) is the characteristic axial evidence width of the Gaussian
focal response (NOT the DOF, NOT the axial FWHM), and the
internal/analytical normalized axial spacing

    s_z(f) = Delta_z / w_z(f)

For a uniform axial grid, the mean coverage over a uniformly-distributed
axial position within one sampling cell has the closed form

    C_bar(s_z) = sqrt(pi) / s_z * erf(s_z / 2)

with marginal sensitivity

    S(s_z) = -d C_bar / d s_z
           = [ sqrt(pi) * erf(s_z / 2) - s_z * exp(-s_z^2 / 4) ] / s_z^2

This module implements C_bar and S in closed form (float64), plus a
numerical integration of C_bar directly from the elemental response T(f, u),
and the two asymptotic reference forms:

    fine-sampling  (s_z -> 0):   1 - C_bar(s_z) ~ s_z^2 / 12
    coarse-sampling (s_z -> inf): C_bar(s_z)     ~ sqrt(pi) / s_z

No specimen geometry, no photographic rendering, no through-focus stack,
no fusion, no noise. s_z is retained as the internal/analytical variable
used to derive C_bar and S.

---------------------------------------------------------------------------
Physical reparametrization in delta
---------------------------------------------------------------------------
C_bar(s_z), S(s_z), and both asymptotics above are the analytical basis.

The project-wide through-focus model defines DOF as the axial FWHM of
the contrast response at f_ref, i.e. half-response at |d| = 0.5 DOF. For
the Gaussian elemental response used here, exp[-(u/w_z)^2] = 0.5 at
|u| = sqrt(ln 2) * w_z, so the axial FWHM is

    DOF = 2 * sqrt(ln 2) * w_z          =>   w_z = DOF / (2 * sqrt(ln 2))

w_z is therefore called the "characteristic axial evidence width" here --
it is explicitly NOT the DOF and NOT the axial FWHM.

The project's principal physical spacing variable is

    delta = Delta_z / DOF

Combining the two relations above:

    s_z = Delta_z / w_z = Delta_z * 2*sqrt(ln 2) / DOF = alpha * delta

    alpha = 2 * sqrt(ln 2)  ~= 1.665109222

so:

    s_z = alpha * delta          delta = s_z / alpha

The physical half-DOF landmark delta = 0.5 therefore corresponds to
s_z = alpha * 0.5 = sqrt(ln 2) ~= 0.832554611 -- NOT s_z = 0.5. This
distinction matters because C_bar/S are defined in the internal s_z
coordinate; delta is the physically reported quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate
from scipy.special import erf

SQRT_PI = float(np.sqrt(np.pi))

# ---------------------------------------------------------------------------
# Physical normalization constants (frozen, matches the project-wide
# through-focus model: DOF = axial FWHM of the Gaussian contrast
# response, half-response at |d| = 0.5 DOF).
# ---------------------------------------------------------------------------

# alpha = 2*sqrt(ln 2): converts between the internal/analytical variable
# s_z = Delta_z/w_z and the physical variable delta = Delta_z/DOF, via
# DOF = alpha * w_z  (equivalently  w_z = DOF/alpha).
ALPHA = float(2.0 * np.sqrt(np.log(2.0)))  # ~= 1.665109222

# Physical half-DOF landmark, delta = 0.5, expressed in s_z coordinates.
# NOTE: this is sqrt(ln 2) ~= 0.832554611, NOT 0.5 -- s_z = 0.5 is NOT the
# half-DOF landmark (see test_half_dof_landmark_not_at_s_z_half).
HALF_DOF_DELTA = 0.5
HALF_DOF_S_Z = float(np.sqrt(np.log(2.0)))  # = ALPHA * HALF_DOF_DELTA ~= 0.832554611

# Compatibility alias: retained so external code that imports DOF_LANDMARK
# continues to import. Do NOT use for new figures/labels -- it does NOT correspond
# to the physical half-DOF landmark. Use HALF_DOF_S_Z / HALF_DOF_DELTA.
DOF_LANDMARK = 0.5


# ---------------------------------------------------------------------------
# Closed-form analytical model (frozen)
# ---------------------------------------------------------------------------


def C_bar(s_z: np.ndarray | float) -> np.ndarray | float:
    """Mean analytical axial coverage C_bar(s_z) = sqrt(pi)/s_z * erf(s_z/2).

    Removable singularity at s_z -> 0 is handled by the exact limit
    C_bar(0) = 1 (evidence is fully available when spacing collapses to
    zero), evaluated via a Taylor-safe branch to avoid 0/0 in float64.
    """
    s = np.asarray(s_z, dtype=np.float64)
    out = np.empty_like(s, dtype=np.float64)

    tiny = s < 1e-8
    safe_s = np.where(tiny, 1.0, s)  # placeholder to avoid divide-by-zero
    out = SQRT_PI / safe_s * erf(safe_s / 2.0)
    out = np.where(tiny, 1.0, out)

    result = out if out.shape else float(out)
    return result


def S_analytical(s_z: np.ndarray | float) -> np.ndarray | float:
    """Closed-form marginal sensitivity S(s_z) = -dC_bar/ds_z.

    S(s_z) = [ sqrt(pi)*erf(s_z/2) - s_z*exp(-s_z^2/4) ] / s_z^2

    Limit at s_z -> 0 is S(0) = 0 (the coverage curve is flat at s_z=0,
    where C_bar attains its maximum value of 1); handled via a
    small-s_z safe branch.
    """
    s = np.asarray(s_z, dtype=np.float64)
    tiny = s < 1e-8
    safe_s = np.where(tiny, 1.0, s)

    numerator = SQRT_PI * erf(safe_s / 2.0) - safe_s * np.exp(-(safe_s**2) / 4.0)
    out = numerator / safe_s**2
    out = np.where(tiny, 0.0, out)

    result = out if out.shape else float(out)
    return result


def coverage_deficit(s_z: np.ndarray | float) -> np.ndarray | float:
    """D_C(s_z) = 1 - C_bar(s_z)."""
    return 1.0 - np.asarray(C_bar(s_z), dtype=np.float64)


# ---------------------------------------------------------------------------
# Asymptotic reference forms
# ---------------------------------------------------------------------------


def fine_asymptotic_deficit(s_z: np.ndarray | float) -> np.ndarray | float:
    """Fine-sampling asymptotic form: s_z^2 / 12, valid for s_z << 1."""
    s = np.asarray(s_z, dtype=np.float64)
    return s**2 / 12.0


def coarse_asymptotic_coverage(s_z: np.ndarray | float) -> np.ndarray | float:
    """Coarse-sampling asymptotic form: sqrt(pi)/s_z, valid for s_z >> 1."""
    s = np.asarray(s_z, dtype=np.float64)
    return SQRT_PI / s


# ---------------------------------------------------------------------------
# Numerical derivative (independent check of S_analytical)
# ---------------------------------------------------------------------------


def S_numerical_derivative(s_z: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Central-difference numerical derivative -dC_bar/ds_z at each s_z.

    Uses a per-point central difference; near s_z=0 (s_z < h) falls back
    to a forward difference to stay inside the domain s_z >= 0.
    """
    s = np.asarray(s_z, dtype=np.float64)
    out = np.empty_like(s)
    for i, sv in enumerate(s):
        if sv - h < 0:
            # forward difference, still first-order accurate near 0
            c0 = float(C_bar(sv))
            c1 = float(C_bar(sv + h))
            out[i] = -(c1 - c0) / h
        else:
            c_minus = float(C_bar(sv - h))
            c_plus = float(C_bar(sv + h))
            out[i] = -(c_plus - c_minus) / (2.0 * h)
    return out


# ---------------------------------------------------------------------------
# Direct numerical integration of C_bar from T
# ---------------------------------------------------------------------------


def C_bar_numerical_integration(s_z: float) -> float:
    """Direct numerical integration of C_bar(s_z)
    of the elemental focal response T(f,u) = exp[-(u/w_z)^2], normalized so
    w_z = 1 (s_z is Delta_z/w_z, so setting w_z=1 makes Delta_z = s_z).

    Procedure:
      1. Uniform axial cell of width s_z, centered at focal plane index 0,
         with neighboring focal planes at offsets of +/- s_z, +/- 2*s_z, ...
      2. For a height h drawn uniformly within the cell h in [-s_z/2, s_z/2],
         the coverage is C_Z(h) = max_k exp(-(z_k - h)^2), z_k = k*s_z.
      3. Average C_Z(h) over h in [-s_z/2, s_z/2] via adaptive quadrature.

    Because exp(-u^2) decays fast, only k in {-1, 0, 1} materially
    contribute for h restricted to the central cell; a few extra planes
    are included for margin at large s_z.
    """
    s = float(s_z)
    if s < 1e-8:
        return 1.0

    # Number of neighboring planes to include on each side: enough that
    # exp(-(k*s - s/2)^2) is negligible beyond the window.
    k_max = max(3, int(np.ceil(8.0 / max(s, 1e-3))) + 2)
    plane_offsets = np.arange(-k_max, k_max + 1) * s

    def C_Z(h: float) -> float:
        u = plane_offsets - h
        return float(np.max(np.exp(-(u**2))))

    value, _abserr = integrate.quad(
        C_Z, -s / 2.0, s / 2.0, limit=200, epsabs=1e-12, epsrel=1e-10
    )
    return value / s


@dataclass(frozen=True)
class ValidationRow:
    s_z: float
    analytical: float
    numerical: float
    abs_error: float
    rel_error: float


def validate_against_numerical_integration(s_z_values: np.ndarray) -> list[ValidationRow]:
    rows = []
    for s in s_z_values:
        s = float(s)
        a = float(C_bar(s))
        n = C_bar_numerical_integration(s)
        abs_err = abs(a - n)
        rel_err = abs_err / a if a > 0 else float("nan")
        rows.append(ValidationRow(s, a, n, abs_err, rel_err))
    return rows


# ---------------------------------------------------------------------------
# Maximum-sensitivity localization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityMaximum:
    s_z_max: float
    S_max: float
    C_bar_at_max: float
    coverage_deficit_at_max: float


def locate_sensitivity_maximum(s_min: float = 0.001, s_max: float = 8.0, n: int = 2_000_001) -> SensitivityMaximum:
    """Locate argmax S(s_z) over a dense grid, then refine with a local
    golden-section-style parabolic polish for sub-grid precision.
    """
    grid = np.linspace(s_min, s_max, n)
    S_vals = S_analytical(grid)
    i_max = int(np.argmax(S_vals))

    # Parabolic refinement using the three points around the grid maximum.
    if 0 < i_max < len(grid) - 1:
        x0, x1, x2 = grid[i_max - 1], grid[i_max], grid[i_max + 1]
        y0, y1, y2 = S_vals[i_max - 1], S_vals[i_max], S_vals[i_max + 1]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if denom != 0:
            a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
            b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
            if a != 0:
                s_refined = -b / (2 * a)
                if x0 <= s_refined <= x2:
                    s_star = s_refined
                else:
                    s_star = float(grid[i_max])
            else:
                s_star = float(grid[i_max])
        else:
            s_star = float(grid[i_max])
    else:
        s_star = float(grid[i_max])

    S_star = float(S_analytical(s_star))
    C_star = float(C_bar(s_star))
    D_star = float(coverage_deficit(s_star))
    return SensitivityMaximum(s_star, S_star, C_star, D_star)


# ---------------------------------------------------------------------------
# S1-N: delta-domain reparametrization (physical variable delta = Delta_z/DOF)
#
# These are thin, exact reparametrizations of C_bar/S via s_z = alpha*delta.
# They do not redefine or refit anything -- C_bar(s_z) and S(s_z) above are
# untouched and remain the sole source of truth.
# ---------------------------------------------------------------------------


def delta_to_s_z(delta: np.ndarray | float) -> np.ndarray | float:
    """s_z = alpha * delta."""
    d = np.asarray(delta, dtype=np.float64)
    out = ALPHA * d
    return out if out.shape else float(out)


def s_z_to_delta(s_z: np.ndarray | float) -> np.ndarray | float:
    """delta = s_z / alpha."""
    s = np.asarray(s_z, dtype=np.float64)
    out = s / ALPHA
    return out if out.shape else float(out)


def C_bar_delta(delta: np.ndarray | float) -> np.ndarray | float:
    """Physical-domain coverage: C_bar_delta(delta) = C_bar(alpha*delta).

    Exact reparametrization of the frozen C_bar(s_z); no new fit, no
    change to the underlying model.
    """
    return C_bar(delta_to_s_z(delta))


def S_delta(delta: np.ndarray | float) -> np.ndarray | float:
    """Physical-domain marginal sensitivity: S_delta(delta) = -dC_bar_delta/ddelta.

    By the chain rule, with s_z = alpha*delta:

        S_delta(delta) = -dC_bar_delta/ddelta
                        = -dC_bar/ds_z * ds_z/ddelta
                        = alpha * S(alpha*delta)

    Do not confuse S(s_z) (sensitivity per unit s_z) with S_delta(delta)
    (sensitivity per unit delta) -- they differ by the constant factor
    alpha and are labeled separately throughout.
    """
    d = np.asarray(delta, dtype=np.float64)
    s = ALPHA * d
    out = ALPHA * np.asarray(S_analytical(s), dtype=np.float64)
    return out if out.shape else float(out)


@dataclass(frozen=True)
class SensitivityMaximumDelta:
    s_z_max: float
    delta_max: float
    S_max_s_z: float  # S(s_z) evaluated at s_z_max -- sensitivity per unit s_z
    S_max_delta: float  # S_delta(delta) evaluated at delta_max -- sensitivity per unit delta
    C_bar_at_max: float
    coverage_deficit_at_max: float


def locate_sensitivity_maximum_delta(
    s_min: float = 0.001, s_max: float = 8.0, n: int = 2_000_001
) -> SensitivityMaximumDelta:
    """Locate argmax S(s_z) on the (unchanged) s_z grid, then translate the
    location to delta = s_z_max/alpha and report both S(s_z_max) and the
    physical-domain S_delta(delta_max) = alpha*S(s_z_max) at that same point.

    The maximum in s_z-coordinates is identical to locate_sensitivity_maximum
    (s_z,max ~= 1.935714); only the reporting is extended to the delta
    domain.
    """
    base = locate_sensitivity_maximum(s_min, s_max, n)
    delta_max = base.s_z_max / ALPHA
    S_max_delta = ALPHA * base.S_max
    return SensitivityMaximumDelta(
        s_z_max=base.s_z_max,
        delta_max=delta_max,
        S_max_s_z=base.S_max,
        S_max_delta=S_max_delta,
        C_bar_at_max=base.C_bar_at_max,
        coverage_deficit_at_max=base.coverage_deficit_at_max,
    )

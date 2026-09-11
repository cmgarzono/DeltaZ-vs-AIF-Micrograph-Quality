r"""
Analytical spectral mechanism for axial evidence.

Scientific question: how does spatial-frequency content modify the axial
evidence available when the effective axial evidence width depends on
spatial frequency?

This is a purely ANALYTICAL model. It does not use the H/P photographic
textures, the physical height field, photographic conditions/oracles,
topography, focus fusion, or noise.

It reuses the frozen axial-coverage closed forms C_bar(s_z), S(s_z)
(imported, never reimplemented) and adds a single new ingredient: four
analytical LOGNORMAL spatial-frequency distributions P_r(f), r in
{LF, COARSE, MIXED, FINE}.

---------------------------------------------------------------------------
Physical variables (frozen)
---------------------------------------------------------------------------
    delta   = Delta_z / DOF                      (principal physical spacing)
    alpha   = 2*sqrt(ln 2)                        (DOF = alpha * w_z)
    s       = alpha * delta                       (S1's internal s_z)
    f_ref   = 0.032 cycles/pixel                  (through-focus reference)
    eta     = f / f_ref

---------------------------------------------------------------------------
Spectral family (frozen)
---------------------------------------------------------------------------
Four regimes at geometric centers f_g,r = eta_g,r * f_ref, eta_g in
{0.5, 1.0, 2.0, 4.0} (exact x2 progression):

    LF      f_g = 0.016 cycles/pixel
    COARSE  f_g = 0.032 cycles/pixel
    MIXED   f_g = 0.064 cycles/pixel
    FINE    f_g = 0.128 cycles/pixel

Lognormal-in-frequency family, common shape parameter for all regimes:

    P_r(f) = 1/(f*sigma_ln*sqrt(2*pi)) * exp{-[ln(f/f_g,r)]^2/(2*sigma_ln^2)}

    sigma_ln = ln(sqrt(2)) ~= 0.3465735903

Substituting u = ln(f): P_r(f) df = N(u; mu_r, sigma_ln) du exactly, i.e.
the lognormal-in-f family is a Gaussian-in-log-frequency family with a
common width. All numerical integration below is therefore carried out on
a uniform grid in u = ln(f) (Simpson's rule), which makes the integrand a
plain Gaussian and gives fast, well-behaved convergence.

---------------------------------------------------------------------------
Numerical domain (frozen, documented)
---------------------------------------------------------------------------
    0 < f <= F_MAX = 0.5 cycles/pixel      (1-D Nyquist)
    F_MIN = 1e-6 cycles/pixel              (numerical floor, not physical)

F_MIN is chosen only to keep u = ln(f) finite; at F_MIN the closest
regime (LF, mu = ln(0.016) ~= -4.135) is already ~28 sigma_ln away, so its
tail contribution there is entirely negligible (~1e-170) and F_MIN carries
no physical meaning as a cutoff. Because F_MAX = 0.5 truncates the upper
tail of FINE (and, marginally, MIXED) at a few sigma_ln, each P_r is
renormalized numerically over (F_MIN, F_MAX] so that its integral is
exactly 1 there.

---------------------------------------------------------------------------
Frequency-dependent axial evidence width (frozen)
---------------------------------------------------------------------------
    w_z(f) = w0 * f_ref / f     <=>     s(f, delta) = alpha*delta*f/f_ref

    E_r^FD(delta) = integral P_r(f) * C_bar(alpha*delta*f/f_ref) df
    M_r^FD(delta) = -dE_r^FD/ddelta
                   = integral P_r(f) * alpha*(f/f_ref) * S(alpha*delta*f/f_ref) df

Constant-width control (f-independent w_z = w0, i.e. s(f,delta)=alpha*delta):

    E_r^CW(delta) = integral P_r(f) * C_bar(alpha*delta) df = C_bar(alpha*delta)
    M_r^CW(delta) = integral P_r(f) * alpha*S(alpha*delta) df = alpha*S(alpha*delta)

which must collapse onto S1 exactly because P_r integrates to 1 and the
integrand no longer depends on f.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson

from deltaz_aif.analysis.axial_coverage import ALPHA, C_bar, S_analytical

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

F_REF = 0.032  # cycles/pixel, through-focus reference frequency (frozen)
SIGMA_LN = float(np.log(np.sqrt(2.0)))  # ~= 0.3465735903, common lognormal width

REGIMES: tuple[str, ...] = ("LF", "COARSE", "MIXED", "FINE")
ETA_G = {"LF": 0.5, "COARSE": 1.0, "MIXED": 2.0, "FINE": 4.0}
F_G = {r: ETA_G[r] * F_REF for r in REGIMES}
# LF=0.016, COARSE=0.032, MIXED=0.064, FINE=0.128

# Numerical integration domain (frozen, documented above).
F_MIN = 1.0e-6
F_MAX = 0.5
N_U = 2_001  # uniform grid in u=ln(f); Gaussian-in-u integrand converges to float64
# precision well before this resolution.

U_MIN = float(np.log(F_MIN))
U_MAX = float(np.log(F_MAX))


def _normal_pdf(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((u - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


@dataclass(frozen=True)
class SpectralGrid:
    """Precomputed log-frequency grid and normalized regime densities."""

    u: np.ndarray  # shape (N_U,), uniform grid in u=ln(f)
    f: np.ndarray  # shape (N_U,), = exp(u)
    Q: dict[str, np.ndarray]  # regime -> normalized density in u-space, integrates to 1
    Z: dict[str, float]  # regime -> raw truncated mass before renormalization


def build_spectral_grid(n_u: int = N_U) -> SpectralGrid:
    """Build the shared log-frequency quadrature grid and the four regime
    densities, each renormalized so that integral over (F_MIN, F_MAX] is 1.
    """
    u = np.linspace(U_MIN, U_MAX, n_u)
    f = np.exp(u)
    Q: dict[str, np.ndarray] = {}
    Z: dict[str, float] = {}
    for r in REGIMES:
        mu_r = float(np.log(F_G[r]))
        q_raw = _normal_pdf(u, mu_r, SIGMA_LN)
        z_r = float(simpson(q_raw, x=u))
        Q[r] = q_raw / z_r
        Z[r] = z_r
    return SpectralGrid(u=u, f=f, Q=Q, Z=Z)


def P_r(f: np.ndarray | float, regime: str, grid: SpectralGrid | None = None) -> np.ndarray | float:
    """Normalized lognormal spectral density P_r(f) (density in f, per the
    frozen formula), evaluated pointwise. Uses the same renormalization
    constant Z_r as build_spectral_grid so P_r integrates to 1 over
    (F_MIN, F_MAX].
    """
    if grid is None:
        grid = build_spectral_grid()
    farr = np.asarray(f, dtype=np.float64)
    mu_r = float(np.log(F_G[regime]))
    safe_f = np.where(farr > 0, farr, np.nan)
    raw = np.exp(-0.5 * (np.log(safe_f / F_G[regime]) / SIGMA_LN) ** 2) / (
        safe_f * SIGMA_LN * np.sqrt(2.0 * np.pi)
    )
    out = raw / grid.Z[regime]
    out = np.where(farr > 0, out, 0.0)
    return out if out.shape else float(out)


# ---------------------------------------------------------------------------
# Spectral descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralDescriptors:
    regime: str
    f_g: float
    centroid: float
    median_frequency: float
    rms_frequency: float
    bandwidth: float
    integral_normalization: float
    energy_fraction_above_0p25: float
    energy_fraction_above_0p40: float


def compute_descriptors(grid: SpectralGrid) -> dict[str, SpectralDescriptors]:
    out: dict[str, SpectralDescriptors] = {}
    u, f = grid.u, grid.f
    for r in REGIMES:
        Q = grid.Q[r]
        norm = float(simpson(Q, x=u))
        centroid = float(simpson(f * Q, x=u))
        rms = float(np.sqrt(simpson((f**2) * Q, x=u)))
        bandwidth = float(np.sqrt(simpson(((f - centroid) ** 2) * Q, x=u)))

        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (Q[1:] + Q[:-1]) * np.diff(u))))
        cdf = cdf / cdf[-1]  # guard against residual quadrature drift
        median_u = float(np.interp(0.5, cdf, u))
        median_f = float(np.exp(median_u))

        def energy_above(f_thresh: float) -> float:
            u_thresh = float(np.log(f_thresh))
            cdf_thresh = float(np.interp(u_thresh, u, cdf))
            return 1.0 - cdf_thresh

        out[r] = SpectralDescriptors(
            regime=r,
            f_g=F_G[r],
            centroid=centroid,
            median_frequency=median_f,
            rms_frequency=rms,
            bandwidth=bandwidth,
            integral_normalization=norm,
            energy_fraction_above_0p25=energy_above(0.25),
            energy_fraction_above_0p40=energy_above(0.40),
        )
    return out


# ---------------------------------------------------------------------------
# Constant-width control and frequency-dependent model
# ---------------------------------------------------------------------------


def _batched_integral(
    delta: np.ndarray,
    grid: SpectralGrid,
    regime: str,
    s_of_f: bool,
    quantity: str,
    batch_size: int = 400,
) -> np.ndarray:
    """Core vectorized quadrature used by both CW and FD models.

    s_of_f=True  -> frequency-dependent model: s(f,delta) = alpha*delta*f/f_ref
    s_of_f=False -> constant-width control:    s(f,delta) = alpha*delta (no f dependence)

    quantity in {"E", "M"}.
    """
    delta = np.atleast_1d(np.asarray(delta, dtype=np.float64))
    u, f = grid.u, grid.f
    Q = grid.Q[regime]
    out = np.empty_like(delta)

    for start in range(0, len(delta), batch_size):
        end = min(start + batch_size, len(delta))
        d_chunk = delta[start:end][:, None]  # (B,1)
        if s_of_f:
            s_mat = ALPHA * d_chunk * (f[None, :] / F_REF)  # (B, N_U)
        else:
            s_mat = ALPHA * d_chunk * np.ones_like(f)[None, :]

        if quantity == "E":
            integrand = Q[None, :] * C_bar(s_mat)
        elif quantity == "M":
            if s_of_f:
                weight = ALPHA * (f[None, :] / F_REF)
            else:
                weight = ALPHA * np.ones_like(f)[None, :]
            integrand = Q[None, :] * weight * S_analytical(s_mat)
        else:
            raise ValueError(f"unknown quantity {quantity!r}")

        out[start:end] = simpson(integrand, x=u, axis=1)

    return out


def E_CW(delta: np.ndarray, regime: str, grid: SpectralGrid) -> np.ndarray:
    return _batched_integral(delta, grid, regime, s_of_f=False, quantity="E")


def M_CW(delta: np.ndarray, regime: str, grid: SpectralGrid) -> np.ndarray:
    return _batched_integral(delta, grid, regime, s_of_f=False, quantity="M")


def E_FD(delta: np.ndarray, regime: str, grid: SpectralGrid) -> np.ndarray:
    return _batched_integral(delta, grid, regime, s_of_f=True, quantity="E")


def M_FD(delta: np.ndarray, regime: str, grid: SpectralGrid) -> np.ndarray:
    return _batched_integral(delta, grid, regime, s_of_f=True, quantity="M")


def M_FD_numerical_derivative(delta: np.ndarray, regime: str, grid: SpectralGrid, h: float = 1e-4) -> np.ndarray:
    """Numerical derivative of -dE_r^FD/ddelta via central difference."""
    delta = np.atleast_1d(np.asarray(delta, dtype=np.float64))
    e_plus = E_FD(delta + h, regime, grid)
    e_minus = E_FD(delta - h, regime, grid)
    return -(e_plus - e_minus) / (2.0 * h)


# ---------------------------------------------------------------------------
# Result-set helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityMaxFD:
    regime: str
    delta_max: float
    M_max: float


def locate_fd_sensitivity_maximum_from_grid(regime: str, delta: np.ndarray, m_values: np.ndarray) -> SensitivityMaxFD:
    """Locate argmax of an already-computed M_r^FD(delta) array (the main
    S2 delta grid via grid argmax + local parabolic polish.
    Reuses the main-grid evaluation instead of a separate dense re-scan.
    """
    d = np.asarray(delta, dtype=np.float64)
    m = np.asarray(m_values, dtype=np.float64)
    i = int(np.argmax(m))
    if 0 < i < len(d) - 1:
        x0, x1, x2 = d[i - 1], d[i], d[i + 1]
        y0, y1, y2 = m[i - 1], m[i], m[i + 1]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if denom != 0:
            a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
            b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
            if a != 0:
                d_star = -b / (2 * a)
                if not (x0 <= d_star <= x2):
                    d_star = float(d[i])
            else:
                d_star = float(d[i])
        else:
            d_star = float(d[i])
    else:
        d_star = float(d[i])
    m_star = float(np.interp(d_star, d, m))
    return SensitivityMaxFD(regime=regime, delta_max=d_star, M_max=m_star)

"""Closed-form far-field patterns and diffraction figures of merit."""

from __future__ import annotations

import numpy as np
from scipy.special import j0, j1, jn_zeros

__all__ = [
    "AIRY_FIRST_ZERO",
    "airy_pattern",
    "airy_zero_radius",
    "airy_encircled_energy",
    "single_slit_pattern",
    "double_slit_pattern",
    "grating_pattern",
    "rayleigh_angle",
    "fresnel_number",
    "talbot_distance",
    "gaussian_width",
    "rayleigh_range",
]

#: first zero of J1(x)/x divided by pi: radius of the Airy disk is 1.21967 lambda z / D
AIRY_FIRST_ZERO = float(jn_zeros(1, 1)[0] / np.pi)


def airy_pattern(r, wavelength, diameter, z):
    """Normalised Airy intensity ``[2 J1(v)/v]^2`` with ``v = pi D r / (lambda z)``."""
    v = np.pi * diameter * np.asarray(r, dtype=float) / (wavelength * z)
    out = np.ones_like(v)
    nz = v != 0
    out[nz] = (2 * j1(v[nz]) / v[nz]) ** 2
    return out


def airy_zero_radius(wavelength, diameter, z, order: int = 1):
    """Radius of the ``order``-th dark ring."""
    return float(jn_zeros(1, order)[-1] / np.pi * wavelength * z / diameter)


def airy_encircled_energy(r, wavelength, diameter, z):
    """Fraction of power inside radius ``r``: ``1 - J0(v)^2 - J1(v)^2`` (Rayleigh)."""
    v = np.pi * diameter * np.asarray(r, dtype=float) / (wavelength * z)
    return 1 - j0(v) ** 2 - j1(v) ** 2


def single_slit_pattern(x, wavelength, width, z):
    """Normalised ``sinc^2(a x / (lambda z))`` (``np.sinc`` includes the pi)."""
    return np.sinc(width * np.asarray(x, dtype=float) / (wavelength * z)) ** 2


def double_slit_pattern(x, wavelength, width, separation, z):
    x = np.asarray(x, dtype=float)
    return single_slit_pattern(x, wavelength, width, z) * np.cos(np.pi * separation * x / (wavelength * z)) ** 2


def grating_pattern(x, wavelength, width, period, n_slits, z):
    """N-slit pattern normalised to 1 on axis."""
    x = np.asarray(x, dtype=float)
    beta = np.pi * period * x / (wavelength * z)
    s = np.sin(beta)
    num = np.sin(n_slits * beta)
    with np.errstate(divide="ignore", invalid="ignore"):
        array = np.where(np.abs(s) < 1e-12, 1.0, (num / (n_slits * s)) ** 2)
    return single_slit_pattern(x, wavelength, width, z) * array


def rayleigh_angle(wavelength, diameter):
    """Angular resolution of a circular pupil, ``1.21967 lambda / D`` (radians)."""
    return AIRY_FIRST_ZERO * wavelength / diameter


def fresnel_number(half_width, wavelength, z):
    """``N_F = a^2 / (lambda z)``; ``N_F << 1`` is the Fraunhofer regime."""
    return half_width**2 / (wavelength * z)


def talbot_distance(period, wavelength, paraxial: bool = True):
    """Self-imaging distance of a periodic field.

    Paraxial: ``2 d^2 / lambda``. Exact (first-order harmonic):
    ``lambda / (1 - sqrt(1 - (lambda / d)^2))``.
    """
    if paraxial:
        return 2 * period**2 / wavelength
    s = wavelength / period
    if s >= 1:
        raise ValueError("period must exceed the wavelength")
    return wavelength / (1 - np.sqrt(1 - s**2))


def rayleigh_range(w0, wavelength):
    return np.pi * w0**2 / wavelength


def gaussian_width(w0, wavelength, z):
    return w0 * np.sqrt(1 + (z / rayleigh_range(w0, wavelength)) ** 2)

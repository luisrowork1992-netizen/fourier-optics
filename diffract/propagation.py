"""Numerical scalar propagation with FFTs.

* :func:`angular_spectrum` is exact within scalar theory (no paraxial
  approximation) and keeps the input sampling.
* :func:`fraunhofer` gives the far field, sampled at ``dx2 = lambda z / (n dx)``.
"""

from __future__ import annotations

import numpy as np

from .grid import Grid

__all__ = ["angular_spectrum", "fraunhofer", "intensity", "power", "second_moment_width"]


def _fft(u):
    axes = tuple(range(-u.ndim, 0))
    return np.fft.fftn(np.fft.ifftshift(u, axes=axes), axes=axes)


def _ifft(U):
    axes = tuple(range(-U.ndim, 0))
    return np.fft.fftshift(np.fft.ifftn(U, axes=axes), axes=axes)


def _freqs(grid: Grid):
    f = np.fft.fftfreq(grid.n, grid.dx)
    if grid.ndim == 1:
        return f**2
    FX, FY = np.meshgrid(f, f, indexing="xy")
    return FX**2 + FY**2


def angular_spectrum(u0: np.ndarray, grid: Grid, wavelength: float, z: float) -> np.ndarray:
    """Propagate the field ``u0`` a distance ``z`` (may be negative).

    Transfer function ``H = exp(i k z sqrt(1 - (lambda f)^2))``; evanescent
    components (``lambda f > 1``) decay as ``exp(-k |z| sqrt((lambda f)^2 - 1))``.
    """
    u0 = np.asarray(u0, dtype=complex)
    if u0.shape != (grid.n,) * grid.ndim:
        raise ValueError(f"field shape {u0.shape} does not match grid")
    k = 2 * np.pi / wavelength
    arg = 1 - wavelength**2 * _freqs(grid)
    kz = k * np.sqrt(arg.astype(complex))
    if z < 0:
        # keep evanescent waves decaying in the backward direction too
        kz = np.where(arg >= 0, kz, np.conj(kz))
    H = np.exp(1j * kz * z)
    return _ifft(_fft(u0) * H)


def fraunhofer(u0: np.ndarray, grid: Grid, wavelength: float, z: float):
    """Far-field (Fraunhofer) pattern at distance ``z``.

    Returns ``(u_z, grid_z)`` where ``grid_z`` has spacing ``lambda z / (n dx)``.
    """
    u0 = np.asarray(u0, dtype=complex)
    k = 2 * np.pi / wavelength
    dx2 = wavelength * z / (grid.n * grid.dx)
    g2 = Grid(grid.n, dx2, grid.ndim)
    r2 = g2.r() ** 2
    pref = np.exp(1j * k * z) * np.exp(1j * k * r2 / (2 * z)) / (1j * wavelength * z) ** (grid.ndim / 2)
    U = np.fft.fftshift(_fft(u0), axes=tuple(range(-grid.ndim, 0))) * grid.dx**grid.ndim
    return pref * U, g2


def intensity(u: np.ndarray) -> np.ndarray:
    return np.abs(u) ** 2


def power(u: np.ndarray, grid: Grid) -> float:
    """Integrated intensity, ``sum |u|^2 dx^ndim``."""
    return float(np.sum(intensity(u)) * grid.dx**grid.ndim)


def second_moment_width(u: np.ndarray, grid: Grid) -> float:
    """ISO 11146 beam width along x: ``2 sqrt(<x^2>)`` (equals w for a Gaussian)."""
    I = intensity(u)
    X = grid.coords() if grid.ndim == 1 else grid.coords()[0]
    xc = np.sum(X * I) / np.sum(I)
    return float(2 * np.sqrt(np.sum((X - xc) ** 2 * I) / np.sum(I)))

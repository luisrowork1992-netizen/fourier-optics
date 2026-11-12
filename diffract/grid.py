"""Sampling grids and simple aperture masks.

All lengths share one unit (micrometres in the examples). Arrays are centred:
the sample with index ``n // 2`` sits at x = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Grid",
    "circular_aperture",
    "rectangular_aperture",
    "slit",
    "double_slit",
    "grating",
    "sinusoidal_grating",
    "gaussian_beam",
]


@dataclass(frozen=True)
class Grid:
    """Square ``n x n`` grid (or a 1-D line when ``ndim == 1``) with spacing ``dx``."""

    n: int
    dx: float
    ndim: int = 2

    def __post_init__(self):
        if self.n < 2 or self.dx <= 0:
            raise ValueError("need n >= 2 and dx > 0")
        if self.ndim not in (1, 2):
            raise ValueError("ndim must be 1 or 2")

    @property
    def x(self) -> np.ndarray:
        return (np.arange(self.n) - self.n // 2) * self.dx

    @property
    def width(self) -> float:
        return self.n * self.dx

    def coords(self):
        """``x`` for a 1-D grid, ``(X, Y)`` mesh for a 2-D grid (Y varies along axis 0)."""
        if self.ndim == 1:
            return self.x
        return np.meshgrid(self.x, self.x, indexing="xy")

    def r(self) -> np.ndarray:
        if self.ndim == 1:
            return np.abs(self.x)
        X, Y = self.coords()
        return np.hypot(X, Y)


def circular_aperture(grid: Grid, diameter: float) -> np.ndarray:
    return (grid.r() <= diameter / 2).astype(float)


def rectangular_aperture(grid: Grid, wx: float, wy: float) -> np.ndarray:
    if grid.ndim != 2:
        raise ValueError("rectangular_aperture needs a 2-D grid")
    X, Y = grid.coords()
    return ((np.abs(X) <= wx / 2) & (np.abs(Y) <= wy / 2)).astype(float)


def slit(grid: Grid, width: float) -> np.ndarray:
    """Slit of the given width along x (infinitely long along y on a 2-D grid)."""
    X = grid.coords() if grid.ndim == 1 else grid.coords()[0]
    return (np.abs(X) <= width / 2).astype(float)


def double_slit(grid: Grid, width: float, separation: float) -> np.ndarray:
    X = grid.coords() if grid.ndim == 1 else grid.coords()[0]
    return ((np.abs(X - separation / 2) <= width / 2)
            | (np.abs(X + separation / 2) <= width / 2)).astype(float)


def grating(grid: Grid, period: float, width: float, n_slits: int | None = None) -> np.ndarray:
    """Binary amplitude grating; ``n_slits=None`` fills the whole grid."""
    X = grid.coords() if grid.ndim == 1 else grid.coords()[0]
    phase = np.mod(X + width / 2, period)
    t = (phase < width).astype(float)
    if n_slits is not None:
        half = n_slits * period / 2
        t *= (np.abs(X) < half).astype(float)
    return t


def sinusoidal_grating(grid: Grid, period: float, contrast: float = 1.0) -> np.ndarray:
    """Amplitude transmission ``(1 + m cos(2 pi x / d)) / 2``: only orders 0 and +/-1."""
    X = grid.coords() if grid.ndim == 1 else grid.coords()[0]
    return 0.5 * (1 + contrast * np.cos(2 * np.pi * X / period))


def gaussian_beam(grid: Grid, w0: float) -> np.ndarray:
    """Gaussian waist field ``exp(-r^2 / w0^2)`` (intensity 1/e^2 radius ``w0``)."""
    return np.exp(-(grid.r() ** 2) / w0**2).astype(complex)

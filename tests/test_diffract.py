import numpy as np
import pytest

import diffract as df

WL = 0.5  # um


def _eff_width(mask, grid):
    """Width of a sampled 1-D opening (counts samples, so it matches the FFT exactly)."""
    return mask.sum() * grid.dx


def test_grid_is_centred():
    g = df.Grid(8, 0.5)
    assert g.x[4] == 0.0
    assert g.width == 4.0
    with pytest.raises(ValueError):
        df.Grid(1, 1.0)


def test_airy_constant():
    assert df.AIRY_FIRST_ZERO == pytest.approx(1.21967, abs=1e-5)
    assert df.rayleigh_angle(0.55, 1e5) == pytest.approx(1.21967 * 0.55e-5, rel=1e-5)


def test_airy_encircled_energy_at_first_ring():
    r1 = df.airy_zero_radius(WL, 100.0, 1e5)
    assert float(df.airy_encircled_energy(r1, WL, 100.0, 1e5)) == pytest.approx(0.8378, abs=1e-4)


def test_fraunhofer_circular_aperture_matches_airy():
    g = df.Grid(2048, 0.5)
    mask = df.circular_aperture(g, 100.0)
    d_eff = np.sqrt(4 * mask.sum() * g.dx**2 / np.pi)
    z = 1e5
    u, g2 = df.fraunhofer(mask, g, WL, z)
    I = df.intensity(u)
    row = I[g2.n // 2, g2.n // 2:] / I[g2.n // 2, g2.n // 2]
    r = g2.x[g2.n // 2:]
    keep = r < 3 * df.airy_zero_radius(WL, d_eff, z)
    np.testing.assert_allclose(row[keep], df.airy_pattern(r[keep], WL, d_eff, z), atol=5e-3)


def test_fraunhofer_conserves_power():
    g = df.Grid(256, 1.0)
    mask = df.rectangular_aperture(g, 40.0, 20.0)
    u, g2 = df.fraunhofer(mask, g, WL, 5e4)
    assert df.power(u, g2) == pytest.approx(df.power(mask, g), rel=1e-12)


def test_single_slit_1d():
    g = df.Grid(8192, 0.5, ndim=1)
    mask = df.slit(g, 50.0)
    a = _eff_width(mask, g)
    z = 1e5
    u, g2 = df.fraunhofer(mask, g, WL, z)
    I = df.intensity(u) / df.intensity(u)[g2.n // 2]
    keep = np.abs(g2.x) < 4 * WL * z / a
    np.testing.assert_allclose(I[keep], df.single_slit_pattern(g2.x[keep], WL, a, z), atol=2e-3)


def test_double_slit_1d():
    g = df.Grid(8192, 0.5, ndim=1)
    mask = df.double_slit(g, 20.0, 200.0)
    a = _eff_width(mask, g) / 2
    z = 2e5
    u, g2 = df.fraunhofer(mask, g, WL, z)
    I = df.intensity(u) / df.intensity(u)[g2.n // 2]
    keep = np.abs(g2.x) < 2 * WL * z / a
    np.testing.assert_allclose(I[keep], df.double_slit_pattern(g2.x[keep], WL, a, 200.0, z), atol=3e-3)


def test_grating_principal_maxima():
    period, width, N, z = 20.0, 5.0, 10, 1e5
    x = np.array([m * WL * z / period for m in range(-3, 4)])
    I = df.grating_pattern(x, WL, width, period, N, z)
    np.testing.assert_allclose(I, df.single_slit_pattern(x, WL, width, z), rtol=1e-9)


def test_angular_spectrum_gaussian_spreading():
    g = df.Grid(512, 1.0)
    w0 = 20.0
    u0 = df.gaussian_beam(g, w0)
    for z in (0.0, df.rayleigh_range(w0, WL), 2 * df.rayleigh_range(w0, WL)):
        u = df.angular_spectrum(u0, g, WL, z)
        assert df.second_moment_width(u, g) == pytest.approx(df.gaussian_width(w0, WL, z), rel=2e-3)


def test_angular_spectrum_is_unitary_and_reversible():
    g = df.Grid(256, 1.0)
    rng = np.random.default_rng(0)
    u0 = rng.standard_normal((256, 256)) + 1j * rng.standard_normal((256, 256))
    u = df.angular_spectrum(u0, g, WL, 3000.0)
    assert df.power(u, g) == pytest.approx(df.power(u0, g), rel=1e-12)
    back = df.angular_spectrum(u, g, WL, -3000.0)
    np.testing.assert_allclose(back, u0, atol=1e-10)


def test_evanescent_waves_decay():
    g = df.Grid(256, 0.2)  # sampling finer than lambda/2 -> some evanescent content
    rng = np.random.default_rng(1)
    u0 = rng.standard_normal((256, 256)).astype(complex)
    for z in (5.0, -5.0):
        assert df.power(df.angular_spectrum(u0, g, WL, z), g) < df.power(u0, g)


def test_talbot_self_imaging():
    period = 8.0
    g = df.Grid(4096, 0.25, ndim=1)  # 1024 um = 128 periods, so the grid is exactly periodic
    u0 = df.sinusoidal_grating(g, period)
    zt = df.talbot_distance(period, WL, paraxial=False)
    assert zt == pytest.approx(df.talbot_distance(period, WL), rel=2e-3)
    I0 = df.intensity(u0)
    np.testing.assert_allclose(df.intensity(df.angular_spectrum(u0, g, WL, zt)), I0, atol=1e-10)
    # half a Talbot length: the image is shifted by half a period
    half = df.intensity(df.angular_spectrum(u0, g, WL, zt / 2))
    np.testing.assert_allclose(half, np.roll(I0, int(period / 2 / g.dx)), atol=1e-10)


def test_fresnel_number():
    assert df.fresnel_number(1000.0, 0.5, 2e6) == pytest.approx(1.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        df.angular_spectrum(np.zeros((10, 10)), df.Grid(12, 1.0), WL, 1.0)

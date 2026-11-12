"""diffract: FFT-based scalar diffraction and the analytic patterns to check it against."""

from .analytic import (
    AIRY_FIRST_ZERO,
    airy_encircled_energy,
    airy_pattern,
    airy_zero_radius,
    double_slit_pattern,
    fresnel_number,
    gaussian_width,
    grating_pattern,
    rayleigh_angle,
    rayleigh_range,
    single_slit_pattern,
    talbot_distance,
)
from .grid import (
    Grid,
    circular_aperture,
    double_slit,
    gaussian_beam,
    grating,
    rectangular_aperture,
    sinusoidal_grating,
    slit,
)
from .propagation import angular_spectrum, fraunhofer, intensity, power, second_moment_width

__version__ = "0.1.0"

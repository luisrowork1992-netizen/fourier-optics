# fourier-optics

![tests](https://github.com/luisrowork1992-netizen/fourier-optics/actions/workflows/tests.yml/badge.svg)

`diffract` is a small Python library for **scalar diffraction with FFTs**. It propagates optical fields with the angular-spectrum method, computes Fraunhofer far fields, and compares the results with the closed-form patterns from textbook optics.

## Features

- `Grid`: centred 1-D or 2-D sampling grids
- Aperture masks:
  - circular and rectangular apertures
  - single and double slits
  - binary and sinusoidal gratings
  - Gaussian beams
- `angular_spectrum`: non-paraxial propagation over any distance, including backwards
  - evanescent waves are handled correctly
- `fraunhofer`: far field with the correct phase and scaling; output sampling is λz/(N·dx)
- Analytic references:
  - Airy pattern, dark-ring radii and encircled energy
  - single-slit, double-slit and N-slit patterns
  - Rayleigh angle and Fresnel number
  - Talbot distance (paraxial and exact)
  - Gaussian beam width
- ISO 11146 second-moment beam width

## Install

```bash
git clone https://github.com/luisrowork1992-netizen/fourier-optics.git
cd fourier-optics
pip install -e .
```

## Quick start

```python
import diffract as df

g = df.Grid(2048, 0.5)                      # 1024 um window, lengths in um
pupil = df.circular_aperture(g, 100.0)
u, g_far = df.fraunhofer(pupil, g, wavelength=0.5, z=1e5)
print(df.airy_zero_radius(0.5, 100.0, 1e5)) # ~610 um first dark ring

beam = df.gaussian_beam(df.Grid(512, 1.0), w0=20.0)
u = df.angular_spectrum(beam, df.Grid(512, 1.0), 0.5, z=5000.0)
```

## Notebook

[`notebooks/diffraction_patterns.ipynb`](notebooks/diffraction_patterns.ipynb) covers five examples:

- the Airy pattern compared with [2J₁(v)/v]²
- single- and double-slit fringes
- the near-field to far-field transition as the Fresnel number drops
- a Talbot carpet
- Gaussian-beam spreading compared with w(z)

## Tests

```bash
pip install pytest
pytest
```

The tests check the following:

- FFT far fields match the Airy, single-slit and double-slit formulas
- power is conserved (Parseval)
- angular-spectrum propagation is unitary and reversible
- evanescent components decay
- a Gaussian beam spreads as w₀√(1 + (z/z_R)²)
- a grating self-images exactly at the Talbot distance and shifts by half a period at z_T/2
- the encircled energy inside the first Airy ring is 83.8 %

## Scope and limitations

- This is scalar theory, so it ignores polarisation and is accurate for features larger than the wavelength.
- FFT propagation is periodic. Keep the field well inside the window, or pad it, to avoid wrap-around.

## References

- J. W. Goodman, *Introduction to Fourier Optics*, 4th ed., W. H. Freeman (2017).
- D. G. Voelz, *Computational Fourier Optics: A MATLAB Tutorial*, SPIE Press (2011).
- M. Born and E. Wolf, *Principles of Optics*, 7th ed., Cambridge University Press.

## License

MIT

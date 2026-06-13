import numpy as np

from spectools.fourier.diagnostics import (
    DecayClassification,
    diagnose_fourier_power,
    fourier_modal_power,
    fourier_tail_ratio,
)
from spectools.fourier.transforms import fourier_nodes, fourier_transform_axis


def test_fourier_modal_power_flattens_components():
    coeffs = np.array(
        [
            [[3.0 + 0.0j, 4.0 + 0.0j]],
            [[0.0 + 0.0j, 2.0j]],
        ]
    )

    power = fourier_modal_power(coeffs)

    np.testing.assert_allclose(power, [25.0, 4.0])


def test_fourier_diagnostic_accepts_resolved_sinusoid():
    nodes = fourier_nodes(16)
    values = np.cos(2.0 * nodes)
    coeffs = fourier_transform_axis(values)

    diagnostic = diagnose_fourier_power(
        coeffs,
        tail_modes=2,
        tail_tolerance=1e-12,
        noise_floor_tolerance=0.0,
    )

    assert diagnostic.decay == DecayClassification.SPECTRAL
    assert diagnostic.tail_ratio < 1e-12


def test_fourier_diagnostic_detects_high_mode_tail():
    coeffs = np.zeros(8, dtype=complex)
    coeffs[0] = 1.0
    coeffs[4] = 3.0

    diagnostic = diagnose_fourier_power(
        coeffs,
        tail_modes=1,
        tail_tolerance=1e-8,
    )

    assert diagnostic.decay == DecayClassification.UNDERRESOLVED
    assert diagnostic.tail_ratio > 0.5


def test_fourier_tail_ratio_handles_zero_coefficients():
    coeffs = np.zeros(8, dtype=complex)

    assert fourier_tail_ratio(coeffs) == 0.0

    diagnostic = diagnose_fourier_power(coeffs)

    assert diagnostic.decay == DecayClassification.ZERO

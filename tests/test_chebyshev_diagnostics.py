import numpy as np

from spectools.chebyshev.diagnostics import (
    DecayClassification,
    chebyshev_modal_power,
    chebyshev_tail_ratio,
    diagnose_chebyshev_power,
)


def test_chebyshev_modal_power_flattens_components():
    coeffs = np.array(
        [
            [[3.0, 4.0]],
            [[0.0, 2.0]],
        ]
    )

    power = chebyshev_modal_power(coeffs)

    np.testing.assert_allclose(power, [25.0, 4.0])


def test_chebyshev_power_diagnostic_accepts_small_tail():
    coeffs = np.array([1.0, 1.0e-3, 1.0e-6, 1.0e-12])

    diagnostic = diagnose_chebyshev_power(
        coeffs,
        tail_modes=1,
        tail_tolerance=1e-8,
        noise_floor_tolerance=1e-14,
    )

    assert diagnostic.decay == DecayClassification.SPECTRAL
    assert diagnostic.tail_ratio < 1e-8


def test_chebyshev_power_diagnostic_detects_underresolved_tail():
    coeffs = np.array([1.0, 0.2, 0.4, 2.0])

    diagnostic = diagnose_chebyshev_power(
        coeffs,
        tail_modes=1,
        tail_tolerance=1e-8,
    )

    assert diagnostic.decay == DecayClassification.UNDERRESOLVED
    assert diagnostic.tail_ratio > 0.5


def test_chebyshev_tail_ratio_handles_zero_coefficients():
    coeffs = np.zeros(5)

    assert chebyshev_tail_ratio(coeffs) == 0.0

    diagnostic = diagnose_chebyshev_power(coeffs)

    assert diagnostic.decay == DecayClassification.ZERO
    assert diagnostic.recommended_order == 0

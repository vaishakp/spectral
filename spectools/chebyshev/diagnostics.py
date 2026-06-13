"""Diagnostics for Chebyshev spectral coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class DecayClassification(StrEnum):
    """Coarse coefficient-decay classification."""

    ZERO = "zero"
    SPECTRAL = "spectral"
    ALGEBRAIC = "algebraic"
    STALLED = "stalled"
    NOISY = "noisy"
    UNDERRESOLVED = "underresolved"


@dataclass(frozen=True, slots=True)
class ChebyshevPowerDiagnostic:
    """Summary of Chebyshev modal power and truncation behavior."""

    tail_ratio: float
    decay: DecayClassification
    total_power: float
    tail_power: float
    leading_power: float
    last_power: float
    slope: float
    recommended_order: int | None
    reason: str


def chebyshev_modal_power(coefficients):
    """Return per-mode component power with Chebyshev mode axis first."""

    coeffs = np.asarray(coefficients)
    if coeffs.ndim == 0:
        raise ValueError("coefficients must have at least one mode axis")
    flat = coeffs.reshape(coeffs.shape[0], -1)
    return np.sum(np.abs(flat) ** 2, axis=1)


def chebyshev_tail_ratio(coefficients, tail_modes=3):
    """Return Chebyshev tail power divided by total modal power."""

    power = chebyshev_modal_power(coefficients)
    return chebyshev_tail_ratio_from_power(power, tail_modes=tail_modes)


def chebyshev_tail_ratio_from_power(power, tail_modes=3):
    """Return tail power divided by total power for a modal-power array."""

    power = np.asarray(power, dtype=float)
    if power.ndim != 1:
        raise ValueError("power must be one-dimensional")
    total = float(np.sum(power))
    if total == 0.0:
        return 0.0
    n_tail = max(1, min(int(tail_modes), len(power)))
    return float(np.sum(power[-n_tail:]) / total)


def diagnose_chebyshev_power(
    coefficients,
    *,
    tail_modes=3,
    tail_tolerance=1.0e-8,
    noise_floor_tolerance=1.0e-12,
):
    """Classify Chebyshev coefficient power for adaptive truncation."""

    power = chebyshev_modal_power(coefficients)
    total = float(np.sum(power))
    if total == 0.0:
        return ChebyshevPowerDiagnostic(
            tail_ratio=0.0,
            decay=DecayClassification.ZERO,
            total_power=0.0,
            tail_power=0.0,
            leading_power=0.0,
            last_power=0.0,
            slope=0.0,
            recommended_order=0,
            reason="all Chebyshev coefficients are zero",
        )

    n_tail = max(1, min(int(tail_modes), len(power)))
    tail = float(np.sum(power[-n_tail:]))
    ratio = tail / total
    positive = np.maximum(power, np.finfo(float).tiny)
    log_power = np.log10(positive / max(total, np.finfo(float).tiny))
    mode_index = np.arange(len(power), dtype=float)

    if len(power) >= 3:
        tail_start = max(1, len(power) - max(3, n_tail + 1))
        fit_x = mode_index[tail_start:]
        fit_y = log_power[tail_start:]
        slope = float(np.polyfit(fit_x, fit_y, deg=1)[0])
    else:
        slope = 0.0

    leading = float(power[0])
    last = float(power[-1])
    normalized_last = last / total
    recommended_order = _recommended_order(power, tail_tolerance, n_tail)

    if ratio <= tail_tolerance:
        decay = DecayClassification.SPECTRAL
        reason = "Chebyshev modal tail is below tolerance"
    elif normalized_last <= noise_floor_tolerance:
        decay = DecayClassification.NOISY
        reason = "last Chebyshev modal power is at the configured noise floor"
    elif slope < -1.0:
        decay = DecayClassification.SPECTRAL
        reason = "Chebyshev tail coefficients are decaying rapidly"
    elif slope < -0.1:
        decay = DecayClassification.ALGEBRAIC
        reason = "Chebyshev tail coefficients decay slowly"
    elif len(power) >= 3 and np.argmax(power) >= len(power) - n_tail:
        decay = DecayClassification.UNDERRESOLVED
        reason = "largest Chebyshev modal power is in the tail"
    else:
        decay = DecayClassification.STALLED
        reason = "Chebyshev tail coefficients are not decreasing"

    return ChebyshevPowerDiagnostic(
        tail_ratio=ratio,
        decay=decay,
        total_power=total,
        tail_power=tail,
        leading_power=leading,
        last_power=last,
        slope=slope,
        recommended_order=recommended_order,
        reason=reason,
    )


def _recommended_order(power, tail_tolerance, tail_modes):
    """Return the smallest retained mode count whose tail is within tolerance."""

    power = np.asarray(power, dtype=float)
    total = float(np.sum(power))
    if total == 0.0:
        return 0

    for order in range(max(1, tail_modes), len(power) + 1):
        retained = power[:order]
        if chebyshev_tail_ratio_from_power(
            retained, tail_modes=tail_modes
        ) <= tail_tolerance:
            return order
    return None

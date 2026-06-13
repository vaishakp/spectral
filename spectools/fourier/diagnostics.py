"""Diagnostics for Fourier spectral coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from spectools.fourier.transforms import fourier_mode_numbers


class DecayClassification(StrEnum):
    """Coarse Fourier coefficient-decay classification."""

    ZERO = "zero"
    SPECTRAL = "spectral"
    ALGEBRAIC = "algebraic"
    STALLED = "stalled"
    NOISY = "noisy"
    UNDERRESOLVED = "underresolved"


@dataclass(frozen=True, slots=True)
class FourierPowerDiagnostic:
    """Summary of Fourier modal power and truncation behavior."""

    tail_ratio: float
    decay: DecayClassification
    total_power: float
    tail_power: float
    leading_power: float
    last_power: float
    slope: float
    recommended_order: int | None
    reason: str


def fourier_modal_power(coefficients, axis=0):
    """Return per-mode component power with Fourier mode axis selected."""

    coeffs = np.moveaxis(np.asarray(coefficients), axis, 0)
    if coeffs.ndim == 0:
        raise ValueError("coefficients must have at least one mode axis")
    flat = coeffs.reshape(coeffs.shape[0], -1)
    return np.sum(np.abs(flat) ** 2, axis=1)


def fourier_tail_ratio(coefficients, tail_modes=3, axis=0):
    """Return high-|m| tail power divided by total Fourier modal power."""

    power = fourier_modal_power(coefficients, axis=axis)
    return fourier_tail_ratio_from_power(power, tail_modes=tail_modes)


def fourier_tail_ratio_from_power(power, tail_modes=3):
    """Return high-|m| tail power divided by total modal power."""

    power = np.asarray(power, dtype=float)
    total = float(np.sum(power))
    if total == 0.0:
        return 0.0
    n_tail = max(1, min(int(tail_modes), len(power)))
    tail_indices = _tail_indices(len(power), n_tail)
    return float(np.sum(power[tail_indices]) / total)


def diagnose_fourier_power(
    coefficients,
    *,
    tail_modes=3,
    tail_tolerance=1.0e-8,
    noise_floor_tolerance=1.0e-12,
    axis=0,
):
    """Classify Fourier coefficient power for adaptive truncation."""

    power = fourier_modal_power(coefficients, axis=axis)
    total = float(np.sum(power))
    if total == 0.0:
        return FourierPowerDiagnostic(
            tail_ratio=0.0,
            decay=DecayClassification.ZERO,
            total_power=0.0,
            tail_power=0.0,
            leading_power=0.0,
            last_power=0.0,
            slope=0.0,
            recommended_order=0,
            reason="all Fourier coefficients are zero",
        )

    n_tail = max(1, min(int(tail_modes), len(power)))
    modes = fourier_mode_numbers(len(power))
    sorted_indices = np.argsort(np.abs(modes))
    sorted_power = power[sorted_indices]
    tail_indices = _tail_indices(len(power), n_tail)
    tail = float(np.sum(power[tail_indices]))
    ratio = tail / total
    positive = np.maximum(sorted_power, np.finfo(float).tiny)
    log_power = np.log10(positive / max(total, np.finfo(float).tiny))

    if len(sorted_power) >= 3:
        tail_start = max(1, len(sorted_power) - max(3, n_tail + 1))
        fit_x = np.arange(len(sorted_power), dtype=float)[tail_start:]
        fit_y = log_power[tail_start:]
        slope = float(np.polyfit(fit_x, fit_y, deg=1)[0])
    else:
        slope = 0.0

    leading = float(power[modes == 0][0])
    high_mode_index = int(sorted_indices[-1])
    last = float(power[high_mode_index])
    normalized_last = last / total

    if ratio <= tail_tolerance:
        decay = DecayClassification.SPECTRAL
        reason = "Fourier modal tail is below tolerance"
    elif normalized_last <= noise_floor_tolerance:
        decay = DecayClassification.NOISY
        reason = "highest Fourier modal power is at the configured noise floor"
    elif slope < -1.0:
        decay = DecayClassification.SPECTRAL
        reason = "Fourier tail coefficients are decaying rapidly"
    elif slope < -0.1:
        decay = DecayClassification.ALGEBRAIC
        reason = "Fourier tail coefficients decay slowly"
    elif np.argmax(power) in set(tail_indices.tolist()):
        decay = DecayClassification.UNDERRESOLVED
        reason = "largest Fourier modal power is in the high-|m| tail"
    else:
        decay = DecayClassification.STALLED
        reason = "Fourier tail coefficients are not decreasing"

    return FourierPowerDiagnostic(
        tail_ratio=ratio,
        decay=decay,
        total_power=total,
        tail_power=tail,
        leading_power=leading,
        last_power=last,
        slope=slope,
        recommended_order=None,
        reason=reason,
    )


def _tail_indices(order, tail_modes):
    """Return indices of the largest absolute Fourier modes."""

    modes = fourier_mode_numbers(order)
    sorted_indices = np.argsort(np.abs(modes))
    return sorted_indices[-tail_modes:]

import numpy as np

from spectools.fourier.transforms import (
    fourier_contract_axis,
    fourier_evaluate,
    fourier_mode_numbers,
    fourier_nodes,
    fourier_transform_axis,
    fourier_transform_axes,
)


def test_fourier_nodes_are_endpoint_exclusive_by_default():
    nodes = fourier_nodes(4, a=0.0, b=2.0 * np.pi)

    np.testing.assert_allclose(nodes, [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])


def test_fourier_transform_round_trips_real_periodic_values():
    nodes = fourier_nodes(16)
    values = np.cos(2.0 * nodes) + 0.25 * np.sin(3.0 * nodes)

    coefficients = fourier_transform_axis(values)
    reconstructed = fourier_contract_axis(coefficients)

    np.testing.assert_allclose(reconstructed.real, values, atol=1e-12)
    np.testing.assert_allclose(reconstructed.imag, 0.0, atol=1e-12)


def test_fourier_evaluate_reconstructs_off_grid_values():
    nodes = fourier_nodes(16)
    values = np.cos(2.0 * nodes) + 0.25 * np.sin(3.0 * nodes)
    coefficients = fourier_transform_axis(values)
    query = np.array([0.1, 0.7, 2.3, 5.1])

    evaluated = fourier_evaluate(coefficients, query)

    expected = np.cos(2.0 * query) + 0.25 * np.sin(3.0 * query)
    np.testing.assert_allclose(evaluated.real, expected, atol=1e-12)
    np.testing.assert_allclose(evaluated.imag, 0.0, atol=1e-12)


def test_fourier_transform_axis_preserves_axis_position():
    nodes = fourier_nodes(8)
    values = np.stack([np.cos(nodes), np.sin(2.0 * nodes)], axis=0)

    coefficients = fourier_transform_axis(values, axis=1)
    reconstructed = fourier_contract_axis(coefficients, axis=1)

    assert coefficients.shape == values.shape
    np.testing.assert_allclose(reconstructed.real, values, atol=1e-12)


def test_fourier_transform_axes_transforms_multiple_axes():
    x_nodes = fourier_nodes(8)
    y_nodes = fourier_nodes(10)
    x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    values = np.cos(x_grid) * np.sin(2.0 * y_grid)

    coefficients = fourier_transform_axes(values, axes=(0, 1))
    reconstructed = fourier_contract_axis(
        fourier_contract_axis(coefficients, axis=1),
        axis=0,
    )

    np.testing.assert_allclose(reconstructed.real, values, atol=1e-12)


def test_fourier_mode_numbers_use_numpy_fft_order():
    np.testing.assert_array_equal(fourier_mode_numbers(6), [0, 1, 2, -3, -2, -1])


def test_compute_ifft_round_trips_and_keeps_the_time_step():
    from spectools.fourier.transforms import compute_fft, compute_ifft

    for delta_t in (0.08, 1.0 / 16384):
        n = 1000
        t = np.arange(n) * delta_t
        x = np.exp(-2j * np.pi * 7 * t / (n * delta_t)) + 0.3 * np.cos(2 * np.pi * 3 * t / (n * delta_t))

        freq_axis, utilde = compute_fft(x, delta_t)
        delta_f = freq_axis[1] - freq_axis[0]
        time_axis, x_back = compute_ifft(utilde, delta_f)

        np.testing.assert_allclose(x_back, x, atol=1e-12)
        assert len(time_axis) == n
        np.testing.assert_allclose(np.diff(time_axis), delta_t, rtol=1e-9)
        assert time_axis[0] == 0

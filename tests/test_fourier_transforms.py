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

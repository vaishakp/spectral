import numpy as np

from spectools.chebyshev.basis import (
    ChebyshevBasis,
    chebyshev_batched_transform_axis,
    chebyshev_transform_axes,
    chebyshev_transform_axis,
)
from spectools.chebyshev.chebyshev import ChebyshevSpectral


def test_to_phys_and_to_spec_round_trip():
    basis = ChebyshevBasis(Nfuncs=8)
    x_axis = basis.CollocationPoints()
    coeffs = np.array([1.0, -0.25, 0.5, 0.125, 0.0, 0.01, -0.02, 0.03])

    values = basis.ToPhysMatrix(x_axis) @ coeffs
    recovered = basis.ToSpecMatrix(x_axis) @ values

    np.testing.assert_allclose(recovered, coeffs, atol=1e-12)
    np.testing.assert_allclose(basis.ToPhys(x_axis, coeffs), values, atol=1e-12)


def test_to_phys_matrix_is_cached_for_same_axis():
    basis = ChebyshevBasis(Nfuncs=8)
    x_axis = basis.CollocationPoints()

    first = basis.ToPhysMatrix(x_axis)
    second = basis.ToPhysMatrix(x_axis.copy())

    assert first is second


def test_transform_logical_to_physical_includes_lower_bound():
    grid = ChebyshevSpectral(Nfuncs=8, a=2.0, b=4.0)

    points = grid.TransformLogicalToPhysical(np.array([-1.0, 0.0, 1.0]))

    np.testing.assert_allclose(points, [2.0, 3.0, 4.0])


def test_get_on_axis_uses_grid_domain_not_query_interval():
    grid = ChebyshevSpectral(Nfuncs=8, a=2.0, b=4.0)
    nodes = grid.collocation_points_physical
    values = nodes**2
    query = np.array([2.25, 2.75, 3.5])

    interpolated = grid.GetOnAxis(values, query)

    np.testing.assert_allclose(interpolated, query**2, atol=1e-12)


def test_chebyshev_transform_axis_matches_matrix_multiply():
    basis = ChebyshevBasis(Nfuncs=6)
    x_axis = basis.CollocationPoints()
    matrix = basis.ToSpecMatrix(x_axis)
    values = np.arange(6 * 3, dtype=float).reshape(6, 3)

    transformed = chebyshev_transform_axis(values, matrix, axis=0)

    np.testing.assert_allclose(transformed, matrix @ values)


def test_chebyshev_transform_axis_preserves_axis_position():
    basis = ChebyshevBasis(Nfuncs=5)
    x_axis = basis.CollocationPoints()
    matrix = basis.ToSpecMatrix(x_axis)
    values = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)

    transformed = chebyshev_transform_axis(values, matrix, axis=1)

    expected = np.einsum("mn,dnc->dmc", matrix, values)
    assert transformed.shape == values.shape
    np.testing.assert_allclose(transformed, expected, atol=1e-12)


def test_chebyshev_transform_axes_matches_sequential_manual_transforms():
    basis_x = ChebyshevBasis(Nfuncs=4)
    basis_y = ChebyshevBasis(Nfuncs=5)
    matrix_x = basis_x.ToSpecMatrix(basis_x.CollocationPoints())
    matrix_y = basis_y.ToSpecMatrix(basis_y.CollocationPoints())
    values = np.arange(4 * 5 * 2, dtype=float).reshape(4, 5, 2)

    transformed = chebyshev_transform_axes(
        values,
        matrices=(matrix_x, matrix_y),
        axes=(0, 1),
    )

    expected = np.einsum("ia,jb,abc->ijc", matrix_x, matrix_y, values)
    np.testing.assert_allclose(transformed, expected, atol=1e-12)


def test_chebyshev_batched_transform_axis_supports_shared_matrix():
    basis = ChebyshevBasis(Nfuncs=5)
    matrix = basis.ToSpecMatrix(basis.CollocationPoints())
    values = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)

    transformed = chebyshev_batched_transform_axis(
        values,
        matrix,
        domain_axis=0,
        axis=1,
    )

    expected = np.einsum("mn,dnc->dmc", matrix, values)
    np.testing.assert_allclose(transformed, expected, atol=1e-12)


def test_chebyshev_batched_transform_axis_supports_per_domain_matrices():
    basis = ChebyshevBasis(Nfuncs=5)
    matrix = basis.ToSpecMatrix(basis.CollocationPoints())
    matrices = np.stack([matrix, 2.0 * matrix])
    values = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)

    transformed = chebyshev_batched_transform_axis(
        values,
        matrices,
        domain_axis=0,
        axis=1,
    )

    expected = np.stack(
        [
            np.einsum("mn,nc->mc", matrices[0], values[0]),
            np.einsum("mn,nc->mc", matrices[1], values[1]),
        ]
    )
    np.testing.assert_allclose(transformed, expected, atol=1e-12)

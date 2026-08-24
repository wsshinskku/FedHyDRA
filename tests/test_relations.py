import numpy as np

from fedhydra.methods.relations import (
    RelationEngine,
    calibrate,
    pairwise_jsd,
    pairwise_squared_distance,
)


def test_discrepancies_are_symmetric_finite_and_zero_diagonal() -> None:
    histograms = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
    features = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    for matrix in (pairwise_jsd(histograms), pairwise_squared_distance(features)):
        assert np.all(np.isfinite(matrix))
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 0.0)
        assert np.all(matrix >= 0)


def test_zero_discrepancy_calibration_is_zero() -> None:
    calibrated = calibrate(np.zeros((4, 4)), epsilon=1.0e-12)
    assert np.array_equal(calibrated, np.zeros((4, 4)))


def test_graph_has_exact_directed_topk_and_symmetric_weighted_support() -> None:
    histograms = np.array(
        [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.1, 0.7, 0.2], [0.2, 0.6, 0.2]]
    )
    rff = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 0.9], [0.9, 1.0]])
    engine = RelationEngine(4, neighbors=2, temperature=0.5, scale_epsilon=1.0e-12)
    engine.discrepancies(histograms, rff)
    engine.update_omega(learning_rate=0.1, regularization=0.01)
    state = engine.graph()
    assert state.directed_edges.sum() == 8
    assert np.array_equal(state.adjacency > 0, (state.adjacency > 0).T)
    assert np.allclose(state.adjacency, state.adjacency.T)
    assert np.allclose(np.diag(state.adjacency), 0.0)
    assert 0.0 <= engine.omega <= 1.0


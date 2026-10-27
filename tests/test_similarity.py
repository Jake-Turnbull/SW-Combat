from __future__ import annotations

import numpy as np
import pytest

from simcombat.similarity import build_similarity


def test_spatial_kernel_uses_z_coordinate() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [20.0, 0.0, 0.0]])
    weights, diagnostics = build_similarity(
        3, feature_coords=coordinates, bandwidth_mm=10.0, trust=1.0
    )
    assert weights[0, 1] > weights[0, 2]
    assert diagnostics["bandwidth_mm"] == 10.0


def test_identical_features_have_uniform_structural_rows() -> None:
    coordinates = np.zeros((4, 3))
    weights, _ = build_similarity(4, feature_coords=coordinates)
    assert np.allclose(weights, np.full((4, 4), 0.25))


def test_all_zero_similarity_falls_back_to_uniform_and_positive_rows() -> None:
    zero_similarity = np.zeros((3, 3))
    weights, diagnostics = build_similarity(
        3, custom_similarity=zero_similarity, trust=1.0
    )
    assert np.all(weights.sum(axis=1) > 0)
    assert np.allclose(weights, np.full((3, 3), 1 / 3))
    assert diagnostics["uniform_fallback_rows"].all()


def test_trust_endpoints_and_diagonal_exclusion() -> None:
    similarity = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.1], [0.2, 0.1, 1.0]])
    uniform, _ = build_similarity(3, custom_similarity=similarity, trust=0.0)
    structural, _ = build_similarity(3, custom_similarity=similarity, trust=1.0)
    excluded, _ = build_similarity(
        3, custom_similarity=similarity, trust=1.0, exclude_diagonal=True
    )
    assert np.allclose(uniform, np.full((3, 3), 1 / 3))
    assert np.allclose(structural[0], similarity[0] / similarity[0].sum())
    assert np.allclose(np.diag(excluded), 0.0)
    assert np.allclose(excluded.sum(axis=1), 1.0)


def test_feature_permutation_equivariance() -> None:
    coordinates = np.array([[0.0, 1.0, 4.0], [3.0, 2.0, 0.0], [8.0, 0.0, 1.0]])
    overlap = np.array([[1.0, 0.7, 0.1], [0.7, 1.0, 0.3], [0.1, 0.3, 1.0]])
    permutation = np.array([2, 0, 1])
    original, _ = build_similarity(3, feature_coords=coordinates, overlap=overlap)
    permuted, _ = build_similarity(
        3,
        feature_coords=coordinates[permutation],
        overlap=overlap[np.ix_(permutation, permutation)],
    )
    assert np.allclose(permuted, original[np.ix_(permutation, permutation)])


def test_correlation_semantics_are_explicit() -> None:
    residuals = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    absolute, _ = build_similarity(2, residuals=residuals, correlation_mode="absolute")
    shifted, _ = build_similarity(2, residuals=residuals, correlation_mode="signed_shifted")
    assert absolute[0, 1] == pytest.approx(0.5)
    assert shifted[0, 1] == pytest.approx(0.0)

from __future__ import annotations

import numpy as np

from simcombat.simulate import simulate_dataset


def test_simulation_is_deterministic_for_a_fixed_seed() -> None:
    first = simulate_dataset(seed=206, n_features=16, batch_sizes=(8, 9, 10))
    second = simulate_dataset(seed=206, n_features=16, batch_sizes=(8, 9, 10))
    assert np.array_equal(first["observed"], second["observed"])
    assert np.array_equal(first["batch_free"], second["batch_free"])
    assert np.array_equal(first["true"]["gamma"], second["true"]["gamma"])
    assert np.array_equal(first["features"]["structural_similarity"], second["features"]["structural_similarity"])


def test_simulation_truth_equations_and_units() -> None:
    simulation = simulate_dataset(
        seed=33,
        n_features=20,
        batch_sizes=(10, 11, 12),
        n_components=2,
        batch_effect_strength=0.4,
    )
    true = simulation["true"]
    batch = simulation["batch"]
    feature_scale = true["feature_scale"]
    assert simulation["observed"].shape == simulation["batch_free"].shape == (20, 33)
    assert np.all(true["delta"] > 0)
    batch_proportions = simulation["batch_sizes"] / simulation["batch_sizes"].sum()
    assert np.allclose(batch_proportions @ true["standardized_gamma"], 0.0, atol=1e-12)
    assert np.allclose(
        true["standardized_delta"],
        true["delta"] / true["pooled_reference_delta"][None, :],
    )
    assert np.allclose(
        true["native_gamma_shift"],
        (true["gamma"] - true["pooled_reference_gamma"][None, :]) * feature_scale[None, :],
    )
    reconstructed = np.empty_like(simulation["observed"])
    for batch_index in np.unique(batch):
        indices = np.flatnonzero(batch == batch_index)
        reconstructed[:, indices] = true["biological_mean"][:, indices] + feature_scale[:, None] * (
            true["gamma"][batch_index, :, None]
            + np.sqrt(true["delta"][batch_index, :, None]) * true["epsilon"][:, indices]
        )
    assert np.allclose(reconstructed, simulation["observed"])
    assert np.allclose(
        simulation["batch_free"],
        true["biological_mean"] + feature_scale[:, None] * true["epsilon"],
    )


def test_zero_batch_effect_is_exact_paired_identity() -> None:
    simulation = simulate_dataset(
        seed=75,
        n_features=16,
        batch_sizes=(8, 9, 10),
        zero_batch_effect=True,
    )
    assert np.allclose(simulation["true"]["gamma"], 0.0)
    assert np.allclose(simulation["true"]["delta"], 1.0)
    assert np.array_equal(simulation["observed"], simulation["batch_free"])


def test_simulation_has_covariates_correlated_features_and_overlap() -> None:
    simulation = simulate_dataset(
        seed=91,
        n_features=32,
        batch_sizes=(24, 24, 24),
        feature_correlation=0.55,
    )
    residual_features = simulation["batch_free"] / simulation["true"]["feature_scale"][:, None]
    correlation = np.corrcoef(residual_features)
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    assert np.mean(np.abs(upper)) > 0.08
    assert simulation["covariates"].shape == (72, 3)
    assert simulation["features"]["coordinates_mm"].shape == (32, 3)
    assert np.any(simulation["features"]["overlap"] == 1.0)
    assert np.any(simulation["features"]["overlap"] == 0.0)
    assert np.allclose(simulation["true"]["memberships"].sum(axis=1), 1.0)


def test_similarity_informativeness_scenarios_are_distinct_and_seeded() -> None:
    common = {
        "seed": 119,
        "n_features": 24,
        "batch_sizes": (8, 9, 10),
        "n_components": 2,
    }
    scenarios = {
        name: simulate_dataset(**common, similarity_informativeness=name)["features"]["structural_similarity"]
        for name in ("informative", "partial", "uninformative", "misleading")
    }
    assert np.linalg.norm(scenarios["informative"] - scenarios["partial"]) > 0.1
    assert np.linalg.norm(scenarios["informative"] - scenarios["uninformative"]) > 0.1
    assert np.linalg.norm(scenarios["informative"] - scenarios["misleading"]) > 0.1
    replay = simulate_dataset(**common, similarity_informativeness="partial")
    assert np.array_equal(scenarios["partial"], replay["features"]["structural_similarity"])

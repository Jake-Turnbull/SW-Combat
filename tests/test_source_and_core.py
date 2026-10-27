from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from simcombat import fit_combat
from simcombat.naive_prior import construct_naive_priors


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_source_is_byte_identical_and_syntax_valid() -> None:
    current = (ROOT / "HarmonisationFunctions.py").read_bytes()
    archived = (ROOT / "legacy" / "HarmonisationFunctions.py").read_bytes()
    assert current != archived
    compile(archived, "legacy/HarmonisationFunctions.py", "exec")
    compile(current, "HarmonisationFunctions.py", "exec")
    assert hashlib.sha256(archived).hexdigest() == (
        "08a21522ffb9116cc66bfa9bf8ac98c6341b00d555c0e91391fffd042d138fcb"
    )


def _balanced_no_effect_data() -> tuple[np.ndarray, np.ndarray]:
    pattern = np.array(
        [
            [-2.0, -1.0, 0.5, 1.0, 2.0, 0.0],
            [0.2, 1.1, -0.3, 2.4, -1.5, -0.8],
            [4.0, 2.0, 3.0, 1.0, 5.0, 0.0],
            [-1.4, 0.7, 2.1, -0.9, 0.0, 1.8],
        ]
    )
    return np.tile(pattern, (1, 2)), np.repeat(["a", "b"], pattern.shape[1])


@pytest.mark.parametrize("mode", ["global", "naive"])
def test_no_batch_effect_smoke_is_finite_and_nearly_identity(mode: str) -> None:
    data, batch = _balanced_no_effect_data()
    result = fit_combat(data, batch, prior_mode=mode)
    assert result["harmonized"].shape == data.shape
    for key in ("gamma_hat", "delta_hat", "gamma_star", "delta_star", "harmonized"):
        assert np.isfinite(result[key]).all()
    assert np.allclose(result["harmonized"], data, atol=1e-5, rtol=1e-5)


def test_weighted_moments_match_hand_calculation() -> None:
    gamma_hat = np.array([[0.0, 2.0, 4.0]])
    delta_hat = np.array([[1.0, 2.0, 4.0]])
    weights = np.array(
        [[1.0, 0.25, 0.0], [0.25, 0.5, 0.25], [0.0, 0.25, 1.0]]
    )
    global_priors = (
        np.array([2.0]),
        np.array([4.0]),
        np.array([4.0]),
        np.array([6.0]),
    )
    result = construct_naive_priors(
        standardized=np.zeros((3, 6)),
        gamma_hat=gamma_hat,
        delta_hat=delta_hat,
        batches=[np.arange(6)],
        global_priors=global_priors,
        methods=["custom"],
        options={"similarity_floor": 1e-50, "min_effective": 1},
        similarity=weights,
    )
    assert np.allclose(result["weights"][0, 0], [0.8, 0.2, 0.0])
    assert np.allclose(result["weights"][0, 1], [0.25, 0.5, 0.25])
    assert np.allclose(result["weights"][0, 2], [0.0, 0.2, 0.8])
    assert result["gamma_mean"][0, 1] == pytest.approx(2.0)
    expected_gamma_var = (0.25 * (0.0 - 2.0) ** 2 + 0.5 * (2.0 - 2.0) ** 2 + 0.25 * (4.0 - 2.0) ** 2) / (1 - (0.25**2 + 0.5**2 + 0.25**2))
    assert result["gamma_variance"][0, 1] == pytest.approx(expected_gamma_var)
    assert result["effective_sample_size"][0, 1] == pytest.approx(1 / 0.375)


def test_uniform_weights_reproduce_global_moments() -> None:
    gamma_hat = np.array([[-1.0, 0.5, 2.0, 4.0]])
    delta_hat = np.array([[0.4, 0.8, 1.5, 3.0]])
    global_mean = np.mean(gamma_hat, axis=1)
    global_variance = np.var(gamma_hat, axis=1, ddof=1)
    delta_mean = np.mean(delta_hat, axis=1)
    delta_variance = np.var(delta_hat, axis=1, ddof=1)
    global_shape = (2 * delta_variance + delta_mean**2) / delta_variance
    global_scale = (delta_mean * delta_variance + delta_mean**3) / delta_variance
    uniform_similarity = np.ones((4, 4))

    result = construct_naive_priors(
        standardized=np.zeros((4, 8)),
        gamma_hat=gamma_hat,
        delta_hat=delta_hat,
        batches=[np.arange(8)],
        global_priors=(global_mean, global_variance, global_shape, global_scale),
        methods=["custom"],
        options={"similarity_floor": 0.0, "min_effective": 1},
        similarity=uniform_similarity,
        legacy_compat=False,
    )
    assert np.allclose(result["gamma_mean"], global_mean[:, None])
    assert np.allclose(result["gamma_variance"], global_variance[:, None])
    assert np.allclose(result["delta_shape"], global_shape[:, None])
    assert np.allclose(result["delta_scale"], global_scale[:, None])


def test_low_effective_size_uses_global_prior_fallback() -> None:
    gamma_hat = np.array([[-1.0, 0.5, 2.0, 4.0]])
    delta_hat = np.array([[0.4, 0.8, 1.5, 3.0]])
    global_priors = (
        np.array([0.75]),
        np.array([2.5]),
        np.array([3.5]),
        np.array([4.25]),
    )
    similarity = np.eye(4)
    result = construct_naive_priors(
        standardized=np.zeros((4, 8)),
        gamma_hat=gamma_hat,
        delta_hat=delta_hat,
        batches=[np.arange(8)],
        global_priors=global_priors,
        methods=["custom"],
        options={"similarity_floor": 0.0, "min_effective": 3},
        similarity=similarity,
        legacy_compat=False,
    )
    assert result["used_global_fallback"].all()
    assert np.all(result["gamma_mean"] == global_priors[0][:, None])
    assert np.all(result["gamma_variance"] == global_priors[1][:, None])
    assert np.all(result["delta_shape"] == global_priors[2][:, None])
    assert np.all(result["delta_scale"] == global_priors[3][:, None])


def test_invalid_prior_mode_is_not_silently_replaced() -> None:
    data, batch = _balanced_no_effect_data()
    with pytest.raises(ValueError, match="prior_mode"):
        fit_combat(data, batch, prior_mode="unimplemented")


def test_feature_permutation_equivariance_for_custom_similarity() -> None:
    rng = np.random.default_rng(811)
    data = rng.normal(size=(7, 36))
    batch = np.repeat(["a", "b", "c"], 12)
    data[:, 12:24] += np.linspace(-0.4, 0.4, 7)[:, None]
    data[:, 24:] += np.linspace(0.3, -0.3, 7)[:, None]
    affinity = np.exp(-np.abs(np.subtract.outer(np.arange(7), np.arange(7))) / 2.0)
    permutation = np.array([4, 1, 6, 0, 5, 2, 3])
    original = fit_combat(
        data,
        batch,
        prior_mode="naive",
        similarity=affinity,
        options={"weight_options": {"min_effective": 1}},
    )
    permuted = fit_combat(
        data[permutation],
        batch,
        prior_mode="naive",
        similarity=affinity[np.ix_(permutation, permutation)],
        options={"weight_options": {"min_effective": 1}},
    )
    assert np.allclose(permuted["harmonized"], original["harmonized"][permutation], atol=1e-8)


def test_unequal_batch_sizes_and_low_variance_features_remain_finite() -> None:
    rng = np.random.default_rng(412)
    batch = np.array(["small"] * 8 + ["medium"] * 13 + ["large"] * 20)
    base = rng.normal(size=(6, batch.size))
    base[0] *= 1e-5
    base[1] *= 0.05
    base[2:, 8:21] += np.linspace(-0.3, 0.4, 4)[:, None]
    result = fit_combat(base, batch, prior_mode="global")
    assert result["diagnostics"]["batch_sizes"].tolist() == [20, 13, 8]
    assert np.isfinite(result["harmonized"]).all()
    assert np.isfinite(result["gamma_star"]).all()
    assert np.isfinite(result["delta_star"]).all()


def test_covariate_effect_is_preserved_without_batch_effect() -> None:
    rng = np.random.default_rng(511)
    n_subjects = 96
    age = np.linspace(-2.0, 2.0, n_subjects)
    batch = np.repeat(["a", "b", "c"], n_subjects // 3)
    slopes = np.array([0.4, -0.7, 1.1, 0.25, -0.2])
    noise = rng.normal(scale=0.3, size=(slopes.size, n_subjects))
    data = 3.0 + slopes[:, None] * age[None, :] + noise
    result = fit_combat(data, batch, covariates=age, prior_mode="global")
    input_slopes = np.linalg.lstsq(np.column_stack((np.ones(n_subjects), age)), data.T, rcond=None)[0][1]
    output_slopes = np.linalg.lstsq(
        np.column_stack((np.ones(n_subjects), age)), result["harmonized"].T, rcond=None
    )[0][1]
    assert np.allclose(output_slopes, input_slopes, atol=0.12)


def test_input_validation_rejects_singleton_batch_and_confounding() -> None:
    rng = np.random.default_rng(93)
    data = rng.normal(size=(5, 18))
    batch_singleton = np.array(["a"] + ["b"] * 17)
    with pytest.raises(ValueError, match="at least two subjects"):
        fit_combat(data, batch_singleton)
    batch = np.repeat(["a", "b", "c"], 6)
    confounded_covariate = (batch == "b").astype(float)
    with pytest.raises(ValueError, match="rank deficient or confounded"):
        fit_combat(data, batch, covariates=confounded_covariate)

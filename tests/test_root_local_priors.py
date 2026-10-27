from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "harmonisation_functions", ROOT / "HarmonisationFunctions.py"
)
_HF = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_HF)


def _inputs() -> tuple[np.ndarray, ...]:
    gamma_hat = np.array([[0.0, 1.0, 3.0, 5.0]])
    delta_hat = np.array([[0.4, 0.8, 1.5, 2.8]])
    standardized = np.array(
        [
            [-1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0],
            [-2.0, 0.0, 2.0, 4.0, -1.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 0.0, -1.0, 0.0, 2.0, 0.0, -2.0],
            [1.0, -1.0, 2.0, -2.0, 2.0, -2.0, 1.0, -1.0],
        ]
    )
    batches = [np.arange(4), np.arange(4, 8)]
    global_priors = (
        np.array([2.25]),
        np.array([4.916666666666667]),
        np.array([4.0]),
        np.array([3.0]),
    )
    return standardized, gamma_hat, delta_hat, batches, global_priors


def _construct(trust: float, **options: object) -> dict[str, object]:
    standardized, gamma_hat, delta_hat, batches, global_priors = _inputs()
    return _HF._construct_local_priors(
        standardized,
        gamma_hat,
        delta_hat,
        np.ones(gamma_hat.shape[1]),
        batches,
        weight_methods=["custom"],
        weight_opts={
            "similarity_matrix": np.eye(gamma_hat.shape[1]),
            "trust": trust,
            "min_effective": 1,
            **options,
        },
        global_priors=global_priors,
    )


def test_trust_zero_is_uniform_and_reproduces_global_moments() -> None:
    result = _construct(0.0)
    standardized, gamma_hat, delta_hat, _, _ = _inputs()
    expected_gamma_mean = np.mean(gamma_hat, axis=1)
    expected_gamma_variance = np.var(gamma_hat, axis=1, ddof=1)
    expected_delta_mean = np.mean(delta_hat, axis=1)
    expected_delta_variance = np.var(delta_hat, axis=1, ddof=1)
    expected_shape = (2 * expected_delta_variance + expected_delta_mean**2) / expected_delta_variance
    expected_scale = (
        expected_delta_mean * expected_delta_variance + expected_delta_mean**3
    ) / expected_delta_variance

    assert np.allclose(result["weights"], 1.0 / gamma_hat.shape[1])
    assert np.allclose(result["scale_weights"], 1.0 / gamma_hat.shape[1])
    assert np.allclose(result["gamma_bar_local"], expected_gamma_mean[:, None])
    assert np.allclose(result["t2_local"], expected_gamma_variance[:, None])
    assert np.allclose(result["a_prior_local"], expected_shape[:, None])
    assert np.allclose(result["b_prior_local"], expected_scale[:, None])
    assert np.allclose(result["weights"].sum(axis=2), 1.0)
    assert standardized.shape[0] == gamma_hat.shape[1]


def test_trust_half_is_convex_rowwise_blend_for_mean_and_scale() -> None:
    local = _construct(1.0)
    midpoint = _construct(0.5)
    uniform = np.full_like(local["weights"], 0.25)
    assert np.allclose(midpoint["weights"], 0.5 * local["weights"] + 0.5 * uniform)
    assert np.allclose(
        midpoint["scale_weights"], 0.5 * local["scale_weights"] + 0.5 * uniform
    )
    assert np.allclose(midpoint["weights"].sum(axis=2), 1.0)
    assert np.allclose(midpoint["scale_weights"].sum(axis=2), 1.0)
    assert np.allclose(midpoint["weights"][0], midpoint["scale_weights"][0]) is False


def test_trust_one_preserves_local_weights_and_exposes_ess() -> None:
    result = _construct(1.0)
    for key in ("weights", "scale_weights"):
        assert np.all(result[key] >= 0.0)
        assert np.allclose(result[key].sum(axis=2), 1.0)
    assert result["trust"] == 1.0
    assert np.all(result["effective_sample_size"] >= 1.0)
    assert np.all(result["scale_effective_sample_size"] >= 1.0)


def test_similarity_floor_is_applied_before_trust_blend() -> None:
    result = _construct(0.5, similarity_floor=0.01)
    assert np.allclose(result["weights"].sum(axis=2), 1.0)
    local_diagonal = 1.0 / (1.0 + 3.0 * 0.01)
    local_off_diagonal = 0.01 * local_diagonal
    expected = 0.5 * np.array(
        [local_diagonal, local_off_diagonal, local_off_diagonal, local_off_diagonal]
    ) + 0.125
    assert np.allclose(result["weights"][0, 0], expected)


@pytest.mark.parametrize("trust", [-0.01, 1.01, np.nan, np.inf])
def test_invalid_trust_is_rejected(trust: float) -> None:
    with pytest.raises(ValueError, match="trust"):
        _construct(trust)


def test_low_ess_fallback_and_spread_floor_activation_are_reported() -> None:
    standardized, gamma_hat, delta_hat, batches, global_priors = _inputs()
    result = _HF._construct_local_priors(
        standardized,
        gamma_hat,
        delta_hat,
        np.ones(gamma_hat.shape[1]),
        batches,
        weight_methods=["custom"],
        weight_opts={
            "similarity_matrix": np.eye(gamma_hat.shape[1]),
            "trust": 1.0,
            "min_effective": 4.1,
            "mean_spread_floor": 20.0,
            "scale_spread_floor": 10.0,
        },
        global_priors=global_priors,
    )
    assert result["used_global_fallback_mean"].all()
    assert result["used_global_fallback_scale"].all()
    assert not result["mean_spread_floor_applied"].any()
    assert not result["scale_spread_floor_applied"].any()
    assert result["mean_spread_floor"] == 20.0
    assert result["scale_spread_floor"] == 10.0


def test_spread_floors_activate_without_ess_fallback() -> None:
    result = _construct(0.0, mean_spread_floor=20.0, scale_spread_floor=10.0)
    assert result["mean_spread_floor_applied"].all()
    assert result["scale_spread_floor_applied"].all()
    assert np.all(result["t2_local"] >= 20.0)
    assert np.all(result["a_prior_local"] > 2.0)


def test_scale_prior_global_mix_zero_preserves_current_moments_and_one_uses_global_width() -> None:
    baseline = _construct(1.0)
    explicit_zero = _construct(1.0, scale_prior_global_mix=0.0)
    global_width = _construct(1.0, scale_prior_global_mix=1.0)
    _, _, delta_hat, _, global_priors = _inputs()

    assert np.allclose(explicit_zero["a_prior_local"], baseline["a_prior_local"])
    assert np.allclose(explicit_zero["b_prior_local"], baseline["b_prior_local"])
    assert np.allclose(global_width["gamma_bar_local"], baseline["gamma_bar_local"])
    assert np.allclose(global_width["scale_variance_local"], np.var(delta_hat, axis=1, ddof=1)[:, None])
    local_mean = np.sum(
        global_width["scale_weights"][0] * delta_hat[0][None, :], axis=1
    )
    global_variance = np.var(delta_hat[0], ddof=1)
    expected_shape = (2.0 * global_variance + local_mean**2) / global_variance
    expected_scale = (
        local_mean * global_variance + local_mean**3
    ) / global_variance
    assert np.allclose(global_width["a_prior_local"][0], expected_shape)
    assert np.allclose(global_width["b_prior_local"][0], expected_scale)
    assert global_width["scale_prior_global_mix"] == 1.0
    assert global_priors[2].size == 1


@pytest.mark.parametrize("mix", [-0.01, 1.01, np.nan, np.inf])
def test_invalid_scale_prior_global_mix_is_rejected(mix: float) -> None:
    with pytest.raises(ValueError, match="scale_prior_global_mix"):
        _construct(0.5, scale_prior_global_mix=mix)


def test_invalid_spread_floor_is_rejected() -> None:
    with pytest.raises(ValueError, match="mean_spread_floor"):
        _construct(0.5, mean_spread_floor=-1.0)


def test_root_combat_modular_trust_zero_matches_global_mode() -> None:
    rng = np.random.default_rng(89)
    batch = np.repeat(["a", "b", "c"], 12)
    data = rng.normal(size=(8, batch.size))
    data[:, 12:24] += np.linspace(-0.4, 0.4, 8)[:, None]
    data[:, 24:] *= np.linspace(0.8, 1.2, 8)[:, None]
    global_result = _HF.combat_modular(data, batch, prior_mode="global")
    local_result = _HF.combat_modular(
        data,
        batch,
        prior_mode="local",
        prior_weight_methods=["custom"],
        prior_weight_opts={"similarity_matrix": np.eye(8), "trust": 0.0},
    )
    assert np.allclose(
        local_result["priors"]["local_priors"]["gamma_bar_local"],
        local_result["priors"]["global_priors"]["gamma_bar"][:, None],
    )
    assert np.allclose(
        local_result["priors"]["local_priors"]["t2_local"],
        local_result["priors"]["global_priors"]["t2"][:, None],
    )
    assert np.allclose(local_result["bayesdata"], global_result["bayesdata"], atol=1e-8)


def test_root_combat_modular_delegates_local_likelihood_and_mixture() -> None:
    rng = np.random.default_rng(90)
    batch = np.repeat(["a", "b"], 12)
    data = rng.normal(size=(8, batch.size))
    data[:4, 12:] += 0.35
    similarity = np.eye(8) + 0.1
    local_likelihood = _HF.combat_modular(
        data,
        batch,
        prior_mode="local_likelihood",
        prior_weight_opts={"similarity_matrix": similarity, "trust": 0.5},
    )
    mixture = _HF.combat_modular(
        data,
        batch,
        prior_mode="mixture",
        prior_weight_opts={
            "similarity_matrix": similarity,
            "mixture_options": {"n_components": 2, "max_iterations": 8},
        },
    )
    assert np.isfinite(local_likelihood["bayesdata"]).all()
    assert local_likelihood["priors"]["local_likelihood_priors"]["weights"].shape == (2, 8, 8)
    assert np.isfinite(mixture["bayesdata"]).all()
    assert mixture["priors"]["mixture_priors"]["memberships"].shape == (8, 2)
    assert mixture["hist"]["mode"] == "mixture_quadrature_posterior"


def test_mean_and_scale_trust_are_independent_and_default_to_trust() -> None:
    base = _construct(0.5)
    same = _construct(0.5, mean_trust=0.5, scale_trust=0.5)
    split = _construct(0.5, mean_trust=1.0, scale_trust=0.0)
    assert np.allclose(base["weights"], same["weights"])
    assert np.allclose(base["scale_weights"], same["scale_weights"])
    assert np.allclose(split["weights"], _construct(1.0)["weights"])
    assert np.allclose(split["scale_weights"], 0.25)
    assert split["mean_trust"] == 1.0 and split["scale_trust"] == 0.0


@pytest.mark.parametrize("name", ["mean_trust", "scale_trust"])
def test_invalid_split_trust_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        _construct(0.5, **{name: 1.5})


def test_scale_prior_modes_default_matches_moment_and_modes_widen_or_recentre() -> None:
    baseline = _construct(1.0)
    moment = _construct(1.0, scale_prior_mode="moment")
    assert np.allclose(baseline["a_prior_local"], moment["a_prior_local"])
    inflated = _construct(1.0, scale_prior_mode="sampling_inflated")
    assert np.all(inflated["scale_variance_local"] > baseline["scale_variance_local"])
    _, _, delta_hat, _, _ = _inputs()
    cv = _construct(1.0, scale_prior_mode="global_cv")
    cv2 = np.var(delta_hat, axis=1, ddof=1) / np.mean(delta_hat, axis=1) ** 2
    local_mean = np.sum(cv["scale_weights"][0] * delta_hat[0][None, :], axis=1)
    assert np.allclose(cv["scale_variance_local"][0], cv2[0] * local_mean**2)
    with pytest.raises(ValueError, match="scale_prior_mode"):
        _construct(1.0, scale_prior_mode="bogus")


def test_scale_exclude_self_zeroes_diagonal_and_gaussian_kernel_is_symmetric() -> None:
    excl = _construct(0.5, scale_exclude_self=True)
    assert np.allclose(np.diagonal(excl["scale_weights"], axis1=1, axis2=2), 0.0)
    assert np.allclose(excl["scale_weights"].sum(axis=2), 1.0)
    gauss = _construct(1.0, variance_similarity_kernel="gaussian_log")
    assert np.allclose(np.diagonal(gauss["scale_weights"], axis1=1, axis2=2), gauss["scale_weights"].max(axis=2))
    legacy = _construct(1.0)
    assert not np.allclose(gauss["scale_weights"], legacy["scale_weights"])
    with pytest.raises(ValueError, match="variance_similarity_kernel"):
        _construct(1.0, variance_similarity_kernel="bogus")

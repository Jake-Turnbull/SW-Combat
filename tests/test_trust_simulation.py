from __future__ import annotations

import numpy as np
import pytest

from simcombat.trust_simulation import SHAPE_FAMILIES, simulate_trust_scenario


@pytest.mark.parametrize(
    ("mean_shape", "log_delta_shape"),
    [(mean, scale) for mean in SHAPE_FAMILIES for scale in SHAPE_FAMILIES],
)
def test_all_shape_pairs_are_reproducible_balanced_and_retain_realized_truth(
    mean_shape: str, log_delta_shape: str
) -> None:
    options = {
        "seed": 2026,
        "mean_shape": mean_shape,
        "log_delta_shape": log_delta_shape,
        "n_features": 12,
        "batch_sizes": (8, 10, 12),
        "sample_effect_sd": 0.2,
        "covariate_noise_sd": 0.2,
    }
    first = simulate_trust_scenario(**options)
    second = simulate_trust_scenario(**options)

    assert np.array_equal(first["observed"], second["observed"])
    assert np.array_equal(first["batch_free"], second["batch_free"])
    assert first["true"]["mean_shape"] == mean_shape
    assert first["true"]["log_delta_shape"] == log_delta_shape
    assert np.bincount(first["batch"]).tolist() == [8, 10, 12]
    assert np.isfinite(first["observed"]).all()
    assert np.isfinite(first["true"]["gamma_sample"]).all()
    assert np.all(first["true"]["delta_sample"] > 0)
    assert np.any(np.ptp(first["true"]["gamma_sample"], axis=1) > 0)
    assert np.any(np.ptp(first["true"]["covariate_effect_noise"], axis=1) > 0)

    feature_scale = first["features"]["feature_scale"]
    truth = first["true"]
    reconstructed = (
        first["batch_free"]
        + feature_scale[:, None] * truth["gamma_sample"]
        + feature_scale[:, None]
        * (np.sqrt(truth["delta_sample"]) - 1.0)
        * truth["residual"]
    )
    assert np.allclose(first["observed"], reconstructed)


def test_shape_families_have_distinct_generating_distributions() -> None:
    outcomes = {}
    for shape in SHAPE_FAMILIES:
        simulation = simulate_trust_scenario(
            seed=84,
            mean_shape=shape,
            log_delta_shape=shape,
            n_features=400,
            batch_sizes=(6, 6),
        )
        outcomes[shape] = simulation["true"]["mean_feature_effect"]
    assert np.std(outcomes["unimodal"]) == pytest.approx(1.0)
    assert not np.array_equal(outcomes["bimodal"], outcomes["zero_centered_bimodal"])
    assert len({tuple(np.round(value[:20], 5)) for value in outcomes.values()}) == 4

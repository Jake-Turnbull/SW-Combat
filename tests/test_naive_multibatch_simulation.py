from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.naive_multibatch_simulation import (
    METHODS,
    FAMILIES,
    run_benchmark,
    simulate_multibatch_dataset,
)


def test_multibatch_simulator_is_reproducible_and_reconstructs_observed() -> None:
    options = {
        "n_per_batch": 12,
        "n_batches": 4,
        "n_features": 12,
        "seed": 91,
        "scenario": "correlated_shared",
    }
    first = simulate_multibatch_dataset(**options)
    second = simulate_multibatch_dataset(**options)

    assert np.array_equal(first["observed"], second["observed"])
    assert np.array_equal(first["clean"], second["clean"])
    assert np.array_equal(first["truth"]["gamma_sample"], second["truth"]["gamma_sample"])
    assert np.bincount(first["batch"]).tolist() == [12, 12, 12, 12]
    assert first["truth"]["delta_sample"].shape == (12, 48)
    assert np.all(first["truth"]["delta_sample"] > 0)
    reconstructed = first["signal"] + first["truth"]["gamma_sample"] + np.sqrt(
        first["truth"]["delta_sample"]
    ) * first["residual"]
    assert np.allclose(first["observed"], reconstructed)
    assert np.allclose(first["clean"], first["signal"] + first["residual"])


def test_multibatch_smoke_writes_metrics_effects_and_requested_figures(tmp_path) -> None:
    metrics, summary, paired = run_benchmark(
        sample_sizes=[6, 8],
        batch_counts=[2, 3],
        seeds=[21, 22],
        feature_counts=[8],
        scenarios=["bimodal"],
        output=tmp_path,
    )
    expected_methods = {entry[0] for entry in METHODS}
    assert set(metrics["method"]) == expected_methods
    assert set(metrics["n_batches"]) == {2, 3}
    assert set(metrics["n_per_batch"]) == {6, 8}
    assert metrics.groupby(["scenario", "n_batches", "n_per_batch", "seed"])["method"].nunique().eq(len(METHODS)).all()
    assert np.isfinite(metrics[["rmse_to_clean", "mae_to_clean", "rmse_to_observed", "mae_to_observed"]]).all().all()
    assert np.isfinite(metrics.loc[metrics["method"] != "Raw", ["gamma_rmse", "delta_rmse", "log_delta_rmse"]]).all().all()
    global_rows = metrics[metrics["method"] == "Global ComBat"].set_index(
        ["scenario", "n_batches", "n_per_batch", "seed"]
    )
    trust_zero = metrics[metrics["method"] == "Correlation trust=0"].set_index(
        ["scenario", "n_batches", "n_per_batch", "seed"]
    )
    assert np.allclose(global_rows["rmse_to_clean"], trust_zero["rmse_to_clean"])
    assert np.allclose(global_rows["gamma_rmse"], trust_zero["gamma_rmse"])
    assert np.allclose(global_rows["delta_rmse"], trust_zero["delta_rmse"])
    scale_zero = metrics[metrics["method"] == "Corr mean=0.5 scale=0"]
    assert len(scale_zero) == len(global_rows)
    assert set(metrics["family"]) == set(FAMILIES)
    local_rows = metrics[metrics["family"] != "Baseline"]
    assert local_rows["scale_prior_mode"].isin({"moment", "global_cv", "sampling_inflated"}).all()
    assert metrics.loc[metrics["method"] != "Raw", "delta_prior_coverage90"].between(0, 1).all()
    assert (summary["delta_prior_sd_mean"].dropna() > 0).all()
    estimates = pd.read_csv(tmp_path / "batch_effect_estimates.csv")
    key = ["scenario", "n_batches", "n_per_batch", "seed", "batch", "feature"]
    glob_prior = estimates[estimates["method"] == "Global ComBat"].set_index(key)
    scale_zero_prior = estimates[estimates["method"] == "Corr mean=0.5 scale=0"].set_index(key)
    assert np.allclose(glob_prior["a_prior_used"], scale_zero_prior["a_prior_used"])
    assert np.allclose(glob_prior["b_prior_used"], scale_zero_prior["b_prior_used"])
    assert {
        "gamma_true", "gamma_estimated", "gamma_hat", "delta_true", "delta_estimated",
        "delta_hat", "log_delta_true", "log_delta_estimated", "a_prior_used",
        "b_prior_used",
    }.issubset(estimates.columns)
    assert len(summary) > 0 and len(paired) > 0
    figures = list((tmp_path / "figures").glob("*.png"))
    pdfs = list((tmp_path / "figures").glob("*.pdf"))
    assert any("figure1_feature_correlations_and_effects" in path.name for path in figures)
    assert any("figure2_similarity_matrices_low_n" in path.name for path in figures)
    assert any("figure2_similarity_matrices_high_n" in path.name for path in figures)
    assert any("figure3_rmse_by_sample_size" in path.name for path in figures)
    assert figures and pdfs
    assert all(path.stat().st_size > 0 for path in figures + pdfs)

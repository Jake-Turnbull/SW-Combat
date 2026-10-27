from __future__ import annotations

import numpy as np

from simcombat.benchmark import METHODS, run_benchmark
from simcombat.metrics import paired_bootstrap_interval, primary_metrics


def test_primary_metrics_have_expected_zero_error_values() -> None:
    truth = np.array([[1.0, 2.0, 3.0], [4.0, 3.0, 2.0]])
    scale = np.array([2.0, 5.0])
    metrics = primary_metrics(
        truth,
        truth,
        gamma_hat=np.zeros((2, 2)),
        delta_hat=np.ones((2, 2)),
        gamma_true=np.zeros((2, 2)),
        delta_true=np.ones((2, 2)),
        feature_scale=scale,
    )
    assert metrics["rmse_native"] == 0.0
    assert metrics["mae_standardized"] == 0.0
    assert metrics["absolute_log_variance_ratio"] == 0.0


def test_replicate_bootstrap_is_deterministic() -> None:
    differences = np.array([-0.2, 0.0, 0.1, 0.25, -0.05])
    first = paired_bootstrap_interval(differences, seed=8, n_bootstrap=500)
    second = paired_bootstrap_interval(differences, seed=8, n_bootstrap=500)
    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_smoke_benchmark_records_every_method_and_seed(tmp_path) -> None:
    checkpoint = tmp_path / "replicates.partial.csv"
    config = {
        "seeds": [5, 8],
        "bootstrap_seed": 100,
        "n_bootstrap": 200,
        "scenario": {
            "n_features": 12,
            "batch_sizes": [8, 8, 8],
            "n_components": 2,
            "batch_effect_strength": 0.25,
        },
        "method_options": {
            "mixture": {"mixture_options": {"n_components": 2, "max_iterations": 10}},
            "local_likelihood": {
                "local_likelihood_options": {"global_prior_penalty": 0.2}
            },
        },
        "checkpoint_path": str(checkpoint),
    }
    replicates, summary, extra = run_benchmark(config)
    assert len(replicates) == len(config["seeds"]) * len(METHODS)
    assert extra["metadata"]["all_runs_recorded"]
    assert set(replicates["method"]) == set(METHODS)
    assert set(replicates["seed"]) == set(config["seeds"])
    assert set(replicates["status"]).issubset({"ok", "failed"})
    assert len(summary) == len(METHODS)
    paired = extra["paired_comparisons"]
    assert set(paired["method"]).issubset(set(METHODS) - {"global"})
    assert (paired["n_paired_replicates"] == len(config["seeds"])).all()
    resumed, _, resumed_extra = run_benchmark(config)
    assert len(resumed) == len(replicates)
    assert resumed_extra["metadata"]["all_runs_recorded"]


def test_multi_scenario_pairs_are_stratified() -> None:
    config = {
        "seeds": [17],
        "n_bootstrap": 50,
        "scenarios": [
            {
                "name": "informative",
                "n_features": 8,
                "batch_sizes": [6, 6, 6],
                "similarity_informativeness": "informative",
            },
            {
                "name": "misleading",
                "n_features": 8,
                "batch_sizes": [6, 6, 6],
                "similarity_informativeness": "misleading",
            },
        ],
        "method_options": {
            "mixture": {"mixture_options": {"n_components": 2, "max_iterations": 5}}
        },
    }
    replicates, summary, extra = run_benchmark(config)
    assert len(replicates) == 2 * len(METHODS)
    assert extra["metadata"]["all_runs_recorded"]
    assert set(summary["scenario"]) == {"informative", "misleading"}
    paired = extra["paired_comparisons"]
    assert set(paired["scenario"]) == {"informative", "misleading"}
    assert (paired["n_paired_replicates"] == 1).all()

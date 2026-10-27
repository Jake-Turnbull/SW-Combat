from __future__ import annotations

import json

import numpy as np

from simcombat.trust_benchmark import run_trust_benchmark, write_benchmark


def test_trust_grid_aggregation_and_figure_exports(tmp_path) -> None:
    config = {
        "calibration_seeds": [101],
        "seeds": [202],
        "n_per_batch": [4],
        "n_batches": 2,
        "trust_values": [0.0, 0.5, 1.0],
        "similarity_methods": ["custom"],
        "spread_floor_quantile": 0.05,
        "scenario": {
            "n_features": 8,
            "batch_effect_strength": 0.25,
            "sample_effect_sd": 0.1,
            "covariate_noise_sd": 0.1,
            "feature_correlation": 0.2,
        },
    }
    result = run_trust_benchmark(config)
    assert result["replicates"]["mean_shape"].nunique() == 4
    assert result["replicates"]["log_delta_shape"].nunique() == 4
    local_rows = result["replicates"][result["replicates"]["method"] == "sw_combat"]
    assert set(local_rows["trust"]) == {0.0, 0.5, 1.0}
    assert result["paired"]["local_worsened_rmse"].isin([True, False]).all()
    assert np.isfinite(result["spread_floors"]["mean"])
    assert np.isfinite(result["spread_floors"]["scale"])
    assert not set(result["calibration"]["calibration_seeds"]).intersection(config["seeds"])

    write_benchmark(result, config, tmp_path)
    assert (tmp_path / "replicates.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "paired_comparisons.csv").exists()
    assert (tmp_path / "PLOT_GUIDE.md").exists()
    png_files = list((tmp_path / "figures").glob("*.png"))
    pdf_files = list((tmp_path / "figures").glob("*.pdf"))
    assert png_files and pdf_files
    assert all(path.stat().st_size > 0 for path in png_files + pdf_files)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["scenario_count"] == 16

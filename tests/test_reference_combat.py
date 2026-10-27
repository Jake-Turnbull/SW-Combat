from __future__ import annotations

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat

from simcombat import fit_combat
from simcombat.reference import run_neurocombat_smoke


def test_global_combat_agrees_with_neurocombat_on_simple_dataset() -> None:
    rng = np.random.default_rng(1771)
    n_features = 20
    batch = np.repeat(["site_a", "site_b", "site_c"], 40)
    biological = rng.normal(size=(n_features, batch.size))
    batch_means = {"site_a": -0.35, "site_b": 0.0, "site_c": 0.4}
    batch_scales = {"site_a": 0.8, "site_b": 1.0, "site_c": 1.25}
    data = np.vstack(
        [
            biological[feature]
            * np.array([batch_scales[level] for level in batch])
            + np.array([batch_means[level] for level in batch])
            for feature in range(n_features)
        ]
    )
    ours = fit_combat(data, batch, prior_mode="global")["harmonized"]
    covariates = pd.DataFrame({"batch": batch})
    reference = neuroCombat(
        dat=data,
        covars=covariates,
        batch_col="batch",
        categorical_cols=[],
        continuous_cols=[],
        eb=True,
        parametric=True,
    )["data"]
    assert ours.shape == reference.shape
    relative_rmse = np.sqrt(np.mean((ours - reference) ** 2)) / np.std(reference)
    assert relative_rmse < 0.2


def test_reference_smoke_compares_both_methods_to_paired_truth(tmp_path) -> None:
    replicates, summary, paired = run_neurocombat_smoke(
        [71, 72],
        output_directory=tmp_path,
        scenario_options={"n_features": 12, "batch_sizes": (8, 8, 8)},
        n_bootstrap=100,
    )
    assert len(replicates) == 4
    assert set(replicates["method"]) == {"local_global_combat", "neuroCombat"}
    assert set(replicates["seed"]) == {71, 72}
    assert np.isfinite(replicates[["rmse_native", "rmse_standardized"]]).all().all()
    assert len(summary) == 2
    assert paired.loc[0, "n_paired_seeds"] == 2
    assert (tmp_path / "replicates.csv").exists()
    assert (tmp_path / "report.md").exists()

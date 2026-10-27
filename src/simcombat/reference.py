"""Paired smoke comparison against the independent neuroCombat implementation."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import fit_combat
from .metrics import paired_bootstrap_interval
from .simulate import simulate_dataset


def run_neurocombat_smoke(
    seeds: list[int],
    *,
    output_directory: str | Path | None = None,
    scenario_options: dict[str, Any] | None = None,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 8801,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare native and standardized RMSE against the same simulated Y0.

    Both implementations receive identical observed measurements, batch labels,
    and continuous age covariate. The counterfactual ``batch_free`` matrix is
    paired by construction and reuses the observed-noise draw.
    """
    try:
        from neuroCombat import neuroCombat
    except ImportError as exc:
        raise RuntimeError(
            "The optional neuroCombat package is required for this reference smoke comparison."
        ) from exc

    scenario = {
        "n_features": 24,
        "batch_sizes": (20, 20, 20),
        "n_components": 2,
        "effect_heterogeneity": "heterogeneous",
        "similarity_informativeness": "informative",
        "feature_correlation": 0.35,
        "batch_effect_strength": 0.35,
    }
    if scenario_options:
        scenario.update(scenario_options)

    rows: list[dict[str, object]] = []
    for seed in seeds:
        simulation = simulate_dataset(seed=int(seed), **scenario)
        age = simulation["covariates"]["age"].to_numpy()
        local = fit_combat(
            simulation["observed"],
            simulation["batch"],
            covariates=pd.DataFrame({"age": age}),
            prior_mode="global",
        )
        reference_covariates = pd.DataFrame(
            {"batch": simulation["batch"], "age": age}
        )
        reference = neuroCombat(
            dat=simulation["observed"],
            covars=reference_covariates,
            batch_col="batch",
            categorical_cols=[],
            continuous_cols=["age"],
            eb=True,
            parametric=True,
        )["data"]

        for method, harmonized in (
            ("local_global_combat", local["harmonized"]),
            ("neuroCombat", np.asarray(reference, dtype=float)),
        ):
            error = harmonized - simulation["batch_free"]
            standardized_error = error / simulation["true"]["feature_scale"][:, None]
            rows.append(
                {
                    "seed": int(seed),
                    "method": method,
                    "rmse_native": float(np.sqrt(np.mean(error**2))),
                    "mae_native": float(np.mean(np.abs(error))),
                    "rmse_standardized": float(np.sqrt(np.mean(standardized_error**2))),
                    "mae_standardized": float(np.mean(np.abs(standardized_error))),
                    "mean_signed_bias_native": float(np.mean(error)),
                    "mean_absolute_bias_native": float(
                        np.mean(np.abs(np.mean(error, axis=1)))
                    ),
                }
            )

    replicates = pd.DataFrame(rows)
    summary = (
        replicates.groupby("method", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            rmse_native_mean=("rmse_native", "mean"),
            rmse_native_sd=("rmse_native", "std"),
            rmse_standardized_mean=("rmse_standardized", "mean"),
            rmse_standardized_sd=("rmse_standardized", "std"),
            mae_native_mean=("mae_native", "mean"),
            mean_signed_bias_native=("mean_signed_bias_native", "mean"),
            mean_absolute_bias_native=("mean_absolute_bias_native", "mean"),
        )
    )
    paired = replicates.pivot(
        index="seed", columns="method", values="rmse_standardized"
    )
    differences = (
        paired["local_global_combat"] - paired["neuroCombat"]
    ).to_numpy()
    estimate, lower, upper = paired_bootstrap_interval(
        differences,
        seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
    )
    comparison = pd.DataFrame(
        [
            {
                "difference": "local_global_combat - neuroCombat",
                "metric": "rmse_standardized",
                "mean_paired_difference": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "n_paired_seeds": len(differences),
            }
        ]
    )

    if output_directory is not None:
        destination = Path(output_directory)
        destination.mkdir(parents=True, exist_ok=True)
        replicates.to_csv(destination / "replicates.csv", index=False)
        summary.to_csv(destination / "summary.csv", index=False)
        comparison.to_csv(destination / "paired_comparison.csv", index=False)
        version = importlib.metadata.version("neuroCombat")
        metadata = {
            "seeds": [int(seed) for seed in seeds],
            "scenario": scenario,
            "reference_package": "neuroCombat",
            "reference_version": version,
            "covariates": ["age"],
            "truth": "paired simulated batch_free using the same noise draw",
            "bootstrap_seed": bootstrap_seed,
            "n_bootstrap": n_bootstrap,
            "interpretation": "smoke comparison only; not a superiority claim",
        }
        (destination / "metadata.json").write_text(
            pd.Series(metadata).to_json(indent=2), encoding="utf-8"
        )
        report_lines = [
            "# neuroCombat Reference Smoke Comparison",
            "",
            f"Seeds: `{metadata['seeds']}`",
            f"Scenario: `{scenario}`",
            "",
            "Both methods used the same simulated observations, batch assignments, continuous age covariate, and paired batch-free truth. RMSE is descriptive at smoke scale and is not a superiority claim.",
            "",
            summary.to_string(index=False),
            "",
            comparison.to_string(index=False),
            "",
        ]
        (destination / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return replicates, summary, comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[8101, 8102, 8103, 8104, 8105])
    parser.add_argument("--output-directory", default="results/neurocombat_smoke")
    args = parser.parse_args(argv)
    _, summary, comparison = run_neurocombat_smoke(
        args.seeds, output_directory=args.output_directory
    )
    print(summary.to_string(index=False))
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproducible four-method simulation benchmark command line interface."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import fit_combat
from .metrics import paired_bootstrap_interval, primary_metrics
from .simulate import simulate_dataset

METHODS = ("global", "naive", "local_likelihood", "mixture")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        for index in range(len(columns))
    ]
    header = "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def run_benchmark(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    seeds = list(config["seeds"])
    scenarios = config.get("scenarios")
    if scenarios is None:
        scenarios = [{"name": "primary", **dict(config.get("scenario", {}))}]
    normalized_scenarios = []
    for scenario in scenarios:
        scenario = dict(scenario)
        scenario_name = str(scenario.pop("name", f"scenario_{len(normalized_scenarios) + 1}"))
        normalized_scenarios.append((scenario_name, scenario))
    method_options = dict(config.get("method_options", {}))
    checkpoint_path_value = config.get("checkpoint_path")
    checkpoint_path = Path(checkpoint_path_value) if checkpoint_path_value else None
    replicate_rows: list[dict[str, Any]] = []
    completed_pairs: set[tuple[str, int]] = set()
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_frame = pd.read_csv(checkpoint_path)
        replicate_rows.extend(checkpoint_frame.to_dict(orient="records"))
        if not checkpoint_frame.empty:
            completed_pairs = {
                (str(scenario), int(seed))
                for scenario, seed in checkpoint_frame[["scenario", "seed"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
                if set(
                    checkpoint_frame.loc[
                        (checkpoint_frame["scenario"] == scenario)
                        & (checkpoint_frame["seed"] == seed),
                        "method",
                    ]
                )
                == set(METHODS)
            }
    for scenario_name, scenario_parameters in normalized_scenarios:
        for seed in seeds:
            if (scenario_name, int(seed)) in completed_pairs:
                print(f"Skipping completed scenario={scenario_name} seed={seed}")
                continue
            simulation = simulate_dataset(seed=int(seed), **scenario_parameters)
            seed_rows: list[dict[str, Any]] = []
            for method in METHODS:
                started = time.perf_counter()
                row: dict[str, Any] = {
                    "scenario": scenario_name,
                    "seed": int(seed),
                    "method": method,
                    "status": "ok",
                    "failure": "",
                }
                try:
                    fit_options = method_options.get(method, {})
                    result = fit_combat(
                        simulation["observed"],
                        simulation["batch"],
                        covariates=simulation["covariates"],
                        prior_mode=method,
                        similarity=simulation["features"]["structural_similarity"],
                        options=fit_options,
                    )
                    batch_proportions = simulation["batch_sizes"] / simulation["batch_sizes"].sum()
                    reference_gamma = batch_proportions @ simulation["true"]["gamma"]
                    reference_delta = batch_proportions @ simulation["true"]["delta"]
                    standardized_gamma_truth = (
                        simulation["true"]["feature_scale"][None, :]
                        * (simulation["true"]["gamma"] - reference_gamma[None, :])
                        / np.sqrt(result["diagnostics"]["pooled_variance"])[None, :]
                    )
                    standardized_delta_truth = (
                        simulation["true"]["feature_scale"][None, :] ** 2
                        * simulation["true"]["delta"]
                        / result["diagnostics"]["pooled_variance"][None, :]
                    )
                    row.update(
                        primary_metrics(
                            result["harmonized"],
                            simulation["batch_free"],
                            result["gamma_star"],
                            result["delta_star"],
                            standardized_gamma_truth,
                            standardized_delta_truth,
                            simulation["true"]["feature_scale"],
                        )
                    )
                    row["posterior_nonconverged_batches"] = sum(
                        not values.get("converged", True)
                        for values in result["diagnostics"]["posterior"].values()
                    )
                    row["optimizer_fallback_rate"] = result["prior_details"].get(
                        "optimizer_fallback_rate", np.nan
                    )
                    if method == "mixture":
                        row["memberships"] = json.dumps(
                            np.asarray(result["prior_details"]["memberships"]).tolist()
                        )
                except Exception as exc:
                    row["status"] = "failed"
                    row["failure"] = f"{type(exc).__name__}: {exc}"
                    row["posterior_nonconverged_batches"] = np.nan
                    row["optimizer_fallback_rate"] = np.nan
                row["runtime_seconds"] = time.perf_counter() - started
                seed_rows.append(row)
            replicate_rows.extend(seed_rows)
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_frame = pd.DataFrame(replicate_rows)
                temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
                checkpoint_frame.to_csv(temporary_path, index=False)
                temporary_path.replace(checkpoint_path)
            print(f"Completed scenario={scenario_name} seed={seed}")

    replicates = pd.DataFrame(replicate_rows)
    metric_columns = [
        "rmse_native",
        "mae_native",
        "rmse_standardized",
        "mae_standardized",
        "mean_signed_bias_native",
        "mean_absolute_bias_native",
        "mean_signed_bias_standardized",
        "mean_absolute_bias_standardized",
        "variance_ratio_mean",
        "absolute_log_variance_ratio",
        "variance_mae",
        "gamma_rmse",
        "gamma_mae",
        "delta_log_rmse",
        "delta_log_mae",
        "runtime_seconds",
    ]
    summary_rows: list[dict[str, Any]] = []
    for scenario_name, _ in normalized_scenarios:
        for method in METHODS:
            method_rows = replicates[
                (replicates["scenario"] == scenario_name)
                & (replicates["method"] == method)
                & (replicates["status"] == "ok")
            ]
            summary: dict[str, Any] = {
                "scenario": scenario_name,
                "method": method,
                "n_replicates": int(len(method_rows)),
                "n_failures": int(
                    (
                        (replicates["scenario"] == scenario_name)
                        & (replicates["method"] == method)
                        & (replicates["status"] == "failed")
                    ).sum()
                ),
            }
            for metric in metric_columns:
                summary[f"{metric}_mean"] = float(method_rows[metric].mean()) if len(method_rows) else np.nan
                summary[f"{metric}_sd"] = float(method_rows[metric].std(ddof=1)) if len(method_rows) > 1 else np.nan
            summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows)

    paired_rows: list[dict[str, Any]] = []
    for scenario_name, _ in normalized_scenarios:
        global_rmse = replicates[
            (replicates["scenario"] == scenario_name)
            & (replicates["method"] == "global")
            & (replicates["status"] == "ok")
        ][["seed", "rmse_standardized"]].rename(columns={"rmse_standardized": "global_rmse"})
        for method in METHODS:
            if method == "global":
                continue
            method_rmse = replicates[
                (replicates["scenario"] == scenario_name)
                & (replicates["method"] == method)
                & (replicates["status"] == "ok")
            ][["seed", "rmse_standardized"]].rename(columns={"rmse_standardized": "method_rmse"})
            paired = global_rmse.merge(method_rmse, on="seed", how="inner")
            if paired.empty:
                continue
            mean_difference, lower, upper = paired_bootstrap_interval(
                paired["method_rmse"].to_numpy() - paired["global_rmse"].to_numpy(),
                seed=int(config.get("bootstrap_seed", 0)),
                n_bootstrap=int(config.get("n_bootstrap", 2000)),
            )
            paired_rows.append(
                {
                    "scenario": scenario_name,
                    "method": method,
                    "reference": "global",
                    "metric": "rmse_standardized",
                    "mean_paired_difference": mean_difference,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_paired_replicates": int(len(paired)),
                }
            )
    paired_frame = pd.DataFrame(paired_rows)
    metadata = {
        "config": config,
        "methods": list(METHODS),
        "replicate_count_per_scenario": len(seeds),
        "scenario_count": len(normalized_scenarios),
        "method_count": len(METHODS),
        "all_runs_recorded": len(replicates) == len(seeds) * len(METHODS) * len(normalized_scenarios),
        "paired_truth_uses_same_noise": True,
        "tuning_protocol": config.get("tuning_protocol", "fixed options; no tuning performed"),
    }
    return replicates, summary_frame, {"paired_comparisons": paired_frame, "metadata": metadata}


def _write_report(
    config: dict[str, Any],
    replicates: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    destination: Path,
) -> None:
    lines = [
        "# Simulation Benchmark Report",
        "",
        "This report summarizes completed simulation replicates. It does not claim a method is superior without considering paired uncertainty intervals.",
        "",
        f"Seeds: `{config['seeds']}`",
        f"Scenario: `{json.dumps(config.get('scenario', {}), sort_keys=True)}`",
        "",
        "## Primary Summary",
        "",
        _markdown_table(summary),
        "",
        "## Paired Standardized RMSE Differences vs Global",
        "",
        _markdown_table(paired) if not paired.empty else "No paired comparisons were available.",
        "",
        "## Failures",
        "",
        _markdown_table(replicates.loc[replicates["status"] == "failed", ["seed", "method", "failure"]])
        if (replicates["status"] == "failed").any()
        else "No method failures.",
        "",
        "## Limitations",
        "",
        "The simulation uses a paired Gaussian counterfactual and a diagonal working model for feature-level prior fitting. Similarity can be uninformative or misleading by construction. Mixture labels are shared between location and scale; mixture correction is a posterior-mean plug-in, not a full posterior predictive transform. This report does not replace external validation on real multi-site MRI data.",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def _save_figure(figure: plt.Figure, destination: Path, name: str) -> None:
    figure.savefig(destination / f"{name}.svg", bbox_inches="tight")
    figure.savefig(destination / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(destination / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_benchmark(
    config: dict[str, Any],
    replicates: pd.DataFrame,
    destination: Path,
) -> None:
    """Write the planned descriptive and method-comparison figures."""
    destination.mkdir(parents=True, exist_ok=True)
    scenario = dict(
        config.get("scenarios", [{"name": "primary", **dict(config.get("scenario", {}))}])[0]
    )
    scenario.pop("name", None)
    example = simulate_dataset(seed=int(config["seeds"][0]), **scenario)
    method_colors = {
        "global": "#28666E",
        "naive": "#7C6A0A",
        "local_likelihood": "#C14953",
        "mixture": "#4F6D7A",
    }

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    coordinates = example["features"]["coordinates_mm"]
    family_codes, families = pd.factorize(example["features"]["families"])
    scatter = axes[0].scatter(
        coordinates[:, 0], coordinates[:, 1], c=family_codes, cmap="viridis", s=30
    )
    axes[0].set(title="MRI-inspired feature layout", xlabel="x (mm)", ylabel="y (mm)")
    colorbar = figure.colorbar(scatter, ax=axes[0], ticks=np.arange(len(families)))
    colorbar.ax.set_yticklabels(families)
    standardized_truth = example["batch_free"] / example["true"]["feature_scale"][:, None]
    axes[1].imshow(np.corrcoef(standardized_truth), cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set(title="Batch-free feature correlation", xlabel="Feature", ylabel="Feature")
    _save_figure(figure, destination, "A_layout_and_correlation")

    gamma = example["true"]["gamma"]
    log_delta = np.log(example["true"]["delta"])
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for axis, values, title, color_map in (
        (axes[0], gamma, "True standardized gamma", "coolwarm"),
        (axes[1], log_delta, "True log delta", "viridis"),
    ):
        image = axis.imshow(values, aspect="auto", cmap=color_map)
        axis.set(title=title, xlabel="Feature", ylabel="Batch")
        figure.colorbar(image, ax=axis)
    _save_figure(figure, destination, "B_true_batch_effects")

    memberships = example["true"]["memberships"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(example["features"]["structural_similarity"], cmap="viridis")
    axes[0].set(title="Structural similarity", xlabel="Feature", ylabel="Feature")
    axes[1].imshow(memberships, aspect="auto", cmap="magma", vmin=0, vmax=1)
    axes[1].set(title="Planted memberships", xlabel="Component", ylabel="Feature")
    mixture_row = replicates[
        (replicates["method"] == "mixture") & (replicates["seed"] == int(config["seeds"][0]))
    ]
    if not mixture_row.empty and "memberships" in mixture_row and pd.notna(mixture_row.iloc[0]["memberships"]):
        fitted_memberships = np.asarray(json.loads(mixture_row.iloc[0]["memberships"]))
        axes[2].imshow(fitted_memberships, aspect="auto", cmap="magma", vmin=0, vmax=1)
        axes[2].set(title="Fitted memberships", xlabel="Component", ylabel="Feature")
    else:
        axes[2].text(0.5, 0.5, "Fitted memberships are not exported in replicate CSV", ha="center", va="center")
        axes[2].set_axis_off()
    _save_figure(figure, destination, "C_similarity_and_memberships")

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for index, method in enumerate(METHODS):
        values = replicates.loc[
            (replicates["method"] == method) & (replicates["status"] == "ok"),
            "rmse_standardized",
        ].to_numpy()
        axis.scatter(np.full(values.size, index), values, color=method_colors[method], alpha=0.8)
        if values.size:
            mean, lower, upper = paired_bootstrap_interval(
                values,
                seed=int(config.get("bootstrap_seed", 0)) + index,
                n_bootstrap=int(config.get("n_bootstrap", 2000)),
            )
            axis.errorbar(index, mean, yerr=[[mean - lower], [upper - mean]], fmt="D", color="black", capsize=4)
    axis.set_xticks(np.arange(len(METHODS)), labels=METHODS, rotation=15)
    axis.set(title="Paired replicate standardized RMSE", ylabel="RMSE (per-feature standardized units)")
    _save_figure(figure, destination, "D_paired_rmse")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for method in METHODS:
        selected = replicates[(replicates["method"] == method) & (replicates["status"] == "ok")]
        axes[0].scatter(
            np.full(len(selected), method),
            selected["mean_signed_bias_standardized"],
            color=method_colors[method],
            label=method,
        )
        axes[1].scatter(
            np.full(len(selected), method),
            np.log(selected["variance_ratio_mean"].clip(lower=1e-12)),
            color=method_colors[method],
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="Mean signed bias", ylabel="Standardized bias")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="Mean log variance ratio", ylabel="log(Var(harmonized) / Var(truth))")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=20)
        axis.set_xlabel("Method")
    _save_figure(figure, destination, "E_bias_and_variance")

    sensitivity = config.get("sensitivity_scenarios", [])
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    if sensitivity:
        sensitivity_frame = pd.DataFrame(sensitivity)
        for method in METHODS:
            subset = sensitivity_frame[sensitivity_frame["method"] == method]
            if len(subset):
                axis.plot(subset["setting"], subset["rmse_standardized"], marker="o", label=method)
        axis.legend()
        axis.set_ylabel("Standardized RMSE")
    else:
        axis.text(0.5, 0.5, "No sensitivity runs configured; no sensitivity claim is made.", ha="center", va="center")
        axis.set_axis_off()
    axis.set_title("Sensitivity results require separate configured runs")
    _save_figure(figure, destination, "F_sensitivity")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    failures = replicates.assign(failed=(replicates["status"] == "failed")).groupby("method")["failed"].mean()
    runtimes = replicates[replicates["status"] == "ok"].groupby("method")["runtime_seconds"].mean()
    axes[0].bar(failures.index, failures.values, color=[method_colors[method] for method in failures.index])
    axes[0].set(title="Failure rate", ylabel="Fraction of configured replicates", ylim=(0, 1))
    axes[1].bar(runtimes.index, runtimes.values, color=[method_colors[method] for method in runtimes.index])
    axes[1].set(title="Mean runtime", ylabel="Seconds per replicate")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=20)
    _save_figure(figure, destination, "G_failure_and_runtime")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON benchmark configuration.")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_directory = Path(config.get("output_directory", "results"))
    output_directory.mkdir(parents=True, exist_ok=True)
    config["checkpoint_path"] = str(output_directory / "replicates.partial.csv")
    replicates, summary, extra = run_benchmark(config)
    replicates.to_csv(output_directory / "replicates.csv", index=False)
    partial_path = output_directory / "replicates.partial.csv"
    if partial_path.exists():
        partial_path.unlink()
    membership_directory = output_directory / "memberships"
    membership_directory.mkdir(parents=True, exist_ok=True)
    for row in replicates[
        (replicates["method"] == "mixture")
        & (replicates["status"] == "ok")
        & replicates["memberships"].notna()
    ].itertuples(index=False):
        matrix = np.asarray(json.loads(row.memberships), dtype=float)
        pd.DataFrame(matrix).to_csv(
            membership_directory / f"seed_{int(row.seed)}.csv", index=False
        )
    summary.to_csv(output_directory / "summary.csv", index=False)
    extra["paired_comparisons"].to_csv(output_directory / "paired_comparisons.csv", index=False)
    (output_directory / "metadata.json").write_text(
        json.dumps(extra["metadata"], indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    _write_report(
        config,
        replicates,
        summary,
        extra["paired_comparisons"],
        output_directory / "report.md",
    )
    _plot_benchmark(config, replicates, output_directory / "figures")
    print(f"Wrote {len(replicates)} method runs to {output_directory}")
    return 0 if extra["metadata"]["all_runs_recorded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

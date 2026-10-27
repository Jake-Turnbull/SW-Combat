"""Trust-grid benchmark for root SW-ComBat and package local methods."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .trust_simulation import SHAPE_FAMILIES, simulate_trust_scenario


def _load_root_harmonizer() -> Any:
    root = Path(__file__).resolve().parents[2]
    source = root / "HarmonisationFunctions.py"
    spec = importlib.util.spec_from_file_location("root_harmonisation_functions", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load root HarmonisationFunctions.py from {source}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calibrate_floors(config: dict[str, Any]) -> tuple[float, float, dict[str, Any]]:
    calibration_seeds = [int(seed) for seed in config.get("calibration_seeds", [])]
    held_out_seeds = {int(seed) for seed in config["seeds"]}
    if held_out_seeds.intersection(calibration_seeds):
        raise ValueError("Calibration seeds must be disjoint from held-out evaluation seeds.")
    if not calibration_seeds:
        raise ValueError("At least one disjoint calibration seed is required for spread floors.")
    scenario = dict(config["scenario"])
    q = float(config.get("spread_floor_quantile", 0.05))
    if not 0.0 <= q <= 1.0:
        raise ValueError("spread_floor_quantile must lie in [0, 1].")
    mean_spreads: list[float] = []
    scale_spreads: list[float] = []
    for seed in calibration_seeds:
        for mean_shape in SHAPE_FAMILIES:
            for log_shape in SHAPE_FAMILIES:
                simulation = simulate_trust_scenario(
                    seed=seed,
                    mean_shape=mean_shape,
                    log_delta_shape=log_shape,
                    **scenario,
                )
                for indices in simulation["batches"]:
                    mean_spreads.append(
                        float(np.var(np.mean(simulation["true"]["gamma_sample"][:, indices], axis=1), ddof=1))
                    )
                    scale_spreads.append(
                        float(np.var(np.mean(simulation["true"]["delta_sample"][:, indices], axis=1), ddof=1))
                    )
    return (
        float(np.quantile(mean_spreads, q)),
        float(np.quantile(scale_spreads, q)),
        {
            "calibration_seeds": calibration_seeds,
            "held_out_seeds": sorted(held_out_seeds),
            "quantile": q,
            "mean_spread_samples": len(mean_spreads),
            "scale_spread_samples": len(scale_spreads),
            "method": "quantile of across-feature variance of realized sample-averaged gamma and delta, across all 16 shape pairs and batches",
        },
    )


def _metrics(
    harmonized: np.ndarray,
    simulation: dict[str, Any],
    gamma_estimate: np.ndarray,
    delta_estimate: np.ndarray,
    pooled_variance: np.ndarray,
) -> dict[str, float]:
    truth = simulation["batch_free"]
    error = harmonized - truth
    feature_scale = simulation["features"]["feature_scale"]
    normalized_error = error / feature_scale[:, None]
    batch_indices = simulation["batches"]
    per_batch_variance_ratios = []
    per_batch_signed_bias = []
    for indices in batch_indices:
        truth_variance = np.var(truth[:, indices], axis=1, ddof=1)
        fitted_variance = np.var(harmonized[:, indices], axis=1, ddof=1)
        per_batch_variance_ratios.extend(
            (fitted_variance / np.maximum(truth_variance, np.finfo(float).tiny)).tolist()
        )
        per_batch_signed_bias.extend(
            (np.mean(error[:, indices], axis=1) / feature_scale).tolist()
        )
    pooled_scale = np.sqrt(np.maximum(np.var(truth / feature_scale[:, None], axis=1, ddof=1), 1e-12))
    signed_cohens_d = np.mean(normalized_error, axis=1) / pooled_scale

    batch_mean_gamma = np.stack(
        [np.mean(simulation["true"]["gamma_sample"][:, indices], axis=1) for indices in batch_indices]
    )
    batch_mean_delta = np.stack(
        [np.mean(simulation["true"]["delta_sample"][:, indices], axis=1) for indices in batch_indices]
    )
    proportions = simulation["batch_sizes"] / simulation["batch_sizes"].sum()
    gamma_reference = proportions @ batch_mean_gamma
    delta_reference = proportions @ batch_mean_delta
    gamma_truth_standardized = (
        (batch_mean_gamma - gamma_reference[None, :])
        * feature_scale[None, :]
        / np.sqrt(pooled_variance)[None, :]
    )
    delta_truth_standardized = (
        batch_mean_delta * feature_scale[None, :] ** 2 / pooled_variance[None, :]
    )
    eps = np.finfo(float).tiny
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "rmse_feature_scaled": float(np.sqrt(np.mean(normalized_error**2))),
        "mae_feature_scaled": float(np.mean(np.abs(normalized_error))),
        "mean_shift_bias_cohens_d": float(np.mean(signed_cohens_d)),
        "mean_absolute_cohens_d": float(np.mean(np.abs(signed_cohens_d))),
        "variance_ratio_to_truth": float(np.mean(per_batch_variance_ratios)),
        "variance_ratio_mae": float(np.mean(np.abs(np.asarray(per_batch_variance_ratios) - 1.0))),
        "gamma_parameter_rmse": float(np.sqrt(np.mean((gamma_estimate - gamma_truth_standardized) ** 2))),
        "gamma_parameter_mae": float(np.mean(np.abs(gamma_estimate - gamma_truth_standardized))),
        "log_delta_parameter_rmse": float(np.sqrt(np.mean((np.log(np.maximum(delta_estimate, eps)) - np.log(np.maximum(delta_truth_standardized, eps))) ** 2))),
        "log_delta_parameter_mae": float(np.mean(np.abs(np.log(np.maximum(delta_estimate, eps)) - np.log(np.maximum(delta_truth_standardized, eps))))),
        "mean_batch_signed_bias_feature_scaled": float(np.mean(per_batch_signed_bias)),
    }


def _run_root_fit(harmonizer: Any, simulation: dict[str, Any], mode: str, trust: float, floors: tuple[float, float], similarity_method: str) -> dict[str, Any]:
    options = {
        "trust": trust,
        "min_effective": 1,
        "mean_spread_floor": floors[0],
        "scale_spread_floor": floors[1],
    }
    weight_methods = [similarity_method]
    if similarity_method == "custom":
        options["similarity_matrix"] = simulation["features"]["structural_similarity"]
    covariates = pd.DataFrame({"age": simulation["covariates"][:, 0]})
    output = harmonizer.combat_modular(
        simulation["observed"],
        simulation["batch"],
        covariates,
        prior_mode=mode,
        prior_weight_methods=weight_methods if mode == "local" else None,
        prior_weight_opts=options if mode == "local" else None,
        covbat_mode=False,
        return_priors=True,
    )
    priors = output["priors"]
    local = priors.get("local_priors", {})
    return {
        "harmonized": np.asarray(output["bayesdata"]),
        "gamma": np.asarray(priors["gamma_star"]),
        "delta": np.asarray(priors["delta_star"]),
        "pooled_variance": np.asarray(output["standardisation"]["var_pooled"]),
        "prior": local,
    }


def _run_root_method_fit(
    harmonizer: Any, simulation: dict[str, Any], mode: str, trust: float
) -> dict[str, Any]:
    options: dict[str, Any] = {"trust": trust}
    options["similarity_matrix"] = simulation["features"]["structural_similarity"]
    if mode == "local_likelihood":
        options.update({
            "similarity_floor": 0.01,
            "use_residual_correlation": True,
            "correlation_mode": "signed_shifted",
        })
    else:
        options["mixture_options"] = {"n_components": 2, "max_iterations": 20, "seed": 0}
    covariates = pd.DataFrame({"age": simulation["covariates"][:, 0]})
    output = harmonizer.combat_modular(
        simulation["observed"],
        simulation["batch"],
        covariates,
        prior_mode=mode,
        prior_weight_methods=["custom"] if mode == "local_likelihood" else None,
        prior_weight_opts=options,
        covbat_mode=False,
        return_priors=True,
    )
    if mode == "local_likelihood":
        prior = output["priors"]["local_likelihood_priors"]
    else:
        prior = output["priors"]["mixture_priors"]
    return {
        "harmonized": np.asarray(output["bayesdata"]),
        "gamma": np.asarray(output["priors"]["gamma_star"]),
        "delta": np.asarray(output["priors"]["delta_star"]),
        "pooled_variance": np.asarray(output["standardisation"]["var_pooled"]),
        "prior": prior,
    }


def run_trust_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    mean_floor, scale_floor, calibration = _calibrate_floors(config)
    harmonizer = _load_root_harmonizer()
    trusts = [float(value) for value in config.get("trust_values", [0.0, 0.5, 1.0])]
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in trusts):
        raise ValueError("Every trust value must lie in [0, 1].")
    methods = list(config.get("similarity_methods", ["correlation_similarity", "magnitude_similarity", "custom"]))
    rows: list[dict[str, Any]] = []
    representatives: dict[tuple[str, str, int], dict[str, Any]] = {}
    for n_per_batch in config["n_per_batch"]:
        scenario = dict(config["scenario"])
        scenario["batch_sizes"] = tuple([int(n_per_batch)] * int(config.get("n_batches", 3)))
        for seed in config["seeds"]:
            for mean_shape in SHAPE_FAMILIES:
                for log_shape in SHAPE_FAMILIES:
                    simulation = simulate_trust_scenario(
                        seed=int(seed), mean_shape=mean_shape, log_delta_shape=log_shape, **scenario
                    )
                    scenario_key = {
                        "mean_shape": mean_shape,
                        "log_delta_shape": log_shape,
                        "n_per_batch": int(n_per_batch),
                        "seed": int(seed),
                        "n_features": int(scenario["n_features"]),
                    }
                    global_result = _run_root_fit(harmonizer, simulation, "global", 0.0, (mean_floor, scale_floor), "custom")
                    global_metrics = _metrics(
                        global_result["harmonized"], simulation, global_result["gamma"],
                        global_result["delta"], global_result["pooled_variance"],
                    )
                    rows.append({**scenario_key, **global_metrics, "method": "global", "trust": 0.0, "similarity_method": "none", "mean_ess": np.nan, "scale_ess": np.nan, "fallback_frequency": 0.0, "mean_spread_floor_activation": 0.0, "scale_spread_floor_activation": 0.0})
                    for method in methods:
                        for trust in trusts:
                            result = _run_root_fit(harmonizer, simulation, "local", trust, (mean_floor, scale_floor), method)
                            prior = result["prior"]
                            metrics = _metrics(result["harmonized"], simulation, result["gamma"], result["delta"], result["pooled_variance"])
                            rows.append({**scenario_key, **metrics, "method": "sw_combat", "trust": trust, "similarity_method": method, "mean_ess": float(np.mean(prior["effective_sample_size"])), "scale_ess": float(np.mean(prior["scale_effective_sample_size"])), "fallback_frequency": float(np.mean(prior["used_global_fallback"])), "mean_spread_floor_activation": float(np.mean(prior["mean_spread_floor_applied"])), "scale_spread_floor_activation": float(np.mean(prior["scale_spread_floor_applied"]))})
                            if trust == 1.0:
                                representatives.setdefault((mean_shape, log_shape, int(n_per_batch)), {"simulation": simulation, "similarity": method, "weights": prior.get("weights"), "scale_weights": prior.get("scale_weights")})
                    for trust in trusts:
                        result = _run_root_method_fit(harmonizer, simulation, "local_likelihood", trust)
                        local_weights = np.asarray(result["prior"].get("fitting_weights", np.ones((1, 1, 1))))
                        local_ess = 1.0 / np.maximum(np.sum(local_weights**2, axis=2), np.finfo(float).tiny)
                        rows.append({**scenario_key, **_metrics(result["harmonized"], simulation, result["gamma"], result["delta"], result["pooled_variance"]), "method": "local_likelihood", "trust": trust, "similarity_method": "correlation", "mean_ess": float(np.mean(local_ess)), "scale_ess": np.nan, "fallback_frequency": float(result["prior"].get("optimizer_fallback_rate", 0.0)), "mean_spread_floor_activation": 0.0, "scale_spread_floor_activation": 0.0})
                    mixture = _run_root_method_fit(harmonizer, simulation, "mixture", 1.0)
                    rows.append({**scenario_key, **_metrics(mixture["harmonized"], simulation, mixture["gamma"], mixture["delta"], mixture["pooled_variance"]), "method": "mixture", "trust": np.nan, "similarity_method": "structural_membership", "mean_ess": np.nan, "scale_ess": np.nan, "fallback_frequency": float(mixture["prior"].get("optimizer_fallback_rate", 0.0)), "mean_spread_floor_activation": 0.0, "scale_spread_floor_activation": 0.0})
    replicates = pd.DataFrame(rows)
    global_frame = replicates[replicates.method == "global"][["mean_shape", "log_delta_shape", "n_per_batch", "seed", "rmse", "mae"]]
    paired = replicates[replicates.method != "global"].merge(
        global_frame, on=["mean_shape", "log_delta_shape", "n_per_batch", "seed"], suffixes=("", "_global"), validate="many_to_one"
    )
    paired["rmse_difference_vs_global"] = paired["rmse"] - paired["rmse_global"]
    paired["mae_difference_vs_global"] = paired["mae"] - paired["mae_global"]
    paired["local_worsened_rmse"] = paired["rmse_difference_vs_global"] > 0
    summary = replicates.groupby(["mean_shape", "log_delta_shape", "n_per_batch", "method", "trust", "similarity_method"], dropna=False).agg(
        replicates=("seed", "nunique"), rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), mae_mean=("mae", "mean"),
        cohens_d_mean=("mean_shift_bias_cohens_d", "mean"), variance_ratio_mean=("variance_ratio_to_truth", "mean"),
        mean_ess=("mean_ess", "mean"), scale_ess=("scale_ess", "mean"), fallback_frequency=("fallback_frequency", "mean"),
        mean_spread_floor_activation=("mean_spread_floor_activation", "mean"), scale_spread_floor_activation=("scale_spread_floor_activation", "mean"),
    ).reset_index()
    paired_summary = paired.groupby(["mean_shape", "log_delta_shape", "n_per_batch", "method", "trust", "similarity_method"], dropna=False).agg(
        fraction_local_worsened=("local_worsened_rmse", "mean"), mean_rmse_difference=("rmse_difference_vs_global", "mean"), mean_mae_difference=("mae_difference_vs_global", "mean"),
    ).reset_index()
    return {"replicates": replicates, "summary": summary, "paired": paired, "paired_summary": paired_summary, "representatives": representatives, "calibration": calibration, "spread_floors": {"mean": mean_floor, "scale": scale_floor}, "trust_values": trusts}


def _save(figure: plt.Figure, folder: Path, name: str) -> None:
    figure.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def _plot_results(result: dict[str, Any], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    metrics = result["replicates"]
    method_colors = {"global": "#263D42", "sw_combat": "#D95D39", "local_likelihood": "#3B7A57", "mixture": "#D4A72C"}
    for group_kind in ("mean_shape", "log_delta_shape"):
        for family in SHAPE_FAMILIES:
            subset = metrics[metrics[group_kind] == family]
            if subset.empty:
                continue
            figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
            representative = next(value["simulation"] for key, value in result["representatives"].items() if key[0 if group_kind == "mean_shape" else 1] == family)
            correlation = np.corrcoef(representative["batch_free"] / representative["features"]["feature_scale"][:, None])
            image = axes[0].imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
            axes[0].set(title=f"Feature correlation ({group_kind}={family})", xlabel="Feature", ylabel="Feature")
            figure.colorbar(image, ax=axes[0])
            for batch_index, indices in enumerate(representative["batches"]):
                axes[1].plot(representative["true"]["gamma_batch"][batch_index], alpha=0.8, label=f"Batch {batch_index}")
                axes[2].plot(representative["true"]["log_delta_batch"][batch_index], alpha=0.8, label=f"Batch {batch_index}")
            axes[1].set(title="Additive gamma effects", xlabel="Feature", ylabel="Gamma")
            axes[2].set(title="Log-delta effects", xlabel="Feature", ylabel="Log delta")
            axes[1].legend(frameon=False, fontsize=7)
            axes[2].legend(frameon=False, fontsize=7)
            _save(figure, folder, f"{group_kind}_{family}_correlation_effects")

            figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
            representative_key = next(key for key in result["representatives"] if key[0 if group_kind == "mean_shape" else 1] == family)
            weight_details = result["representatives"][representative_key]
            similarity = weight_details["simulation"]["features"]["structural_similarity"]
            for axis, matrix, title in zip(axes, (similarity, weight_details["weights"][0], weight_details["scale_weights"][0]), ("Structural similarity", "Mean weights", "Scale weights")):
                im = axis.imshow(matrix, cmap="viridis", vmin=0)
                axis.set(title=title, xlabel="Source feature", ylabel="Target feature")
                figure.colorbar(im, ax=axis)
            _save(figure, folder, f"{group_kind}_{family}_similarity_weights")

            figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
            for method in ("global", "sw_combat", "local_likelihood", "mixture"):
                method_rows = subset[subset.method == method]
                if method == "sw_combat" or method == "local_likelihood":
                    for trust in result["trust_values"]:
                        if trust == 0.0:
                            continue
                        trust_rows = method_rows[method_rows.trust == trust]
                        grouped = trust_rows.groupby("n_per_batch")["rmse"].mean()
                        axes[0].plot(grouped.index, grouped.values, marker="o", label=f"{method}, t={trust:g}", color=method_colors[method], linestyle={0.0:"-", 0.5:"--", 1.0:":"}.get(trust, "-"))
                else:
                    grouped = method_rows.groupby("n_per_batch")["rmse"].mean()
                    label = "global (also SW trust=0)" if method == "global" else method
                    axes[0].plot(grouped.index, grouped.values, marker="D", label=label, color=method_colors[method], linewidth=2.5, zorder=5)
            axes[0].set(title="RMSE by per-batch N", xlabel="N per batch", ylabel="RMSE")
            axes[0].legend(frameon=False, fontsize=7)
            paired_rows = result["paired"]
            paired_rows = paired_rows[paired_rows[group_kind] == family]
            for (method, trust), group in paired_rows.groupby(["method", "trust"], dropna=False):
                if method in {"sw_combat", "local_likelihood"} and trust == 0.0:
                    continue
                if method == "mixture":
                    label = "mixture"
                else:
                    label = f"{method}, t={trust:g}"
                means = group.groupby("n_per_batch")["rmse_difference_vs_global"].mean()
                axes[1].plot(means.index, means.values, marker="o", label=label, color=method_colors.get(method, "black"))
            axes[1].axhline(0, color="black", linewidth=0.8)
            axes[1].set(title="Paired RMSE change vs global", xlabel="N per batch", ylabel="RMSE difference")
            axes[1].legend(frameon=False, fontsize=7)
            figure.tight_layout()
            _save(figure, folder, f"{group_kind}_{family}_performance_trust")

            figure, axes = plt.subplots(2, 2, figsize=(13, 8))
            metric_specs = (
                ("rmse", "RMSE to paired batch-free truth"),
                ("mae", "MAE to paired batch-free truth"),
                ("mean_shift_bias_cohens_d", "Mean-shift bias (Cohen's d)"),
                ("variance_ratio_to_truth", "Variance ratio to truth"),
            )
            for axis, (metric, title) in zip(axes.flat, metric_specs):
                for method in ("global", "sw_combat", "local_likelihood", "mixture"):
                    method_rows = subset[subset.method == method]
                    if method in {"sw_combat", "local_likelihood"}:
                        for trust in result["trust_values"]:
                            if trust == 0.0:
                                continue
                            trust_rows = method_rows[method_rows.trust == trust]
                            grouped = trust_rows.groupby("n_per_batch")[metric].mean()
                            axis.plot(
                                grouped.index,
                                grouped.values,
                                marker="o",
                                color=method_colors[method],
                                linestyle={0.0: "-", 0.5: "--", 1.0: ":"}.get(trust, "-"),
                                label=f"{method} t={trust:g}",
                            )
                    else:
                        grouped = method_rows.groupby("n_per_batch")[metric].mean()
                        label = "global (also SW trust=0)" if method == "global" else method
                        axis.plot(
                            grouped.index,
                            grouped.values,
                            marker="o",
                            color=method_colors[method],
                            label=label,
                        )
                axis.set(title=title, xlabel="N per batch", ylabel=metric)
                axis.grid(axis="y", alpha=0.2)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False)
            figure.tight_layout(rect=(0, 0.11, 1, 1))
            _save(figure, folder, f"{group_kind}_{family}_metrics_all_methods")


def _write_plot_guide(folder: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Trust Benchmark Plot Guide", "",
        "Each plot is emitted as both PNG (300 dpi) and PDF. Files are split by additive-mean or log-delta scenario family so the four shape regimes remain legible.", "",
        "## Correlation and Effects", "",
        "`*_correlation_effects` places the realized feature correlation matrix beside feature-level batch gamma and log-delta profiles for one paired representative simulation. Gamma is additive; log delta is the log variance multiplier.", "",
        "## Similarity and Weights", "",
        "`*_similarity_weights` compares the input structural similarity with the selected representative's target-row mean-prior weights and independently constructed scale-prior weights. Rows are target features; columns are pooled source features. The 0.01 floor is imposed on local similarity before the trust blend.", "",
        "## Performance and Trust", "",
        "`*_performance_trust` shows reconstruction RMSE against the paired batch-free truth across per-batch sample counts, followed by paired RMSE change relative to global ComBat. Positive paired differences mean local weighting worsened error. Mixture has no trust control in its current API and is shown as a separate baseline.", "",
        "## Calibration", "",
        f"Prior-spread floors were fixed before evaluation at mean variance {result['spread_floors']['mean']:.8g} and scale variance {result['spread_floors']['scale']:.8g}, using disjoint calibration seeds and the configured quantile. Per-run activation rates are in `replicates.csv` and aggregated activation rates in `summary.csv`.", "",
        "Metric CSVs also include RMSE/MAE, mean-shift Cohen's d, variance ratio to truth, gamma/log-delta parameter errors, mean/scale effective feature counts, and global fallback frequency. The design is synthetic; sample-level effects intentionally violate the feature/batch-constant effect assumption of standard ComBat.", "",
    ]
    (folder / "PLOT_GUIDE.md").write_text("\n".join(lines), encoding="utf-8")


def write_benchmark(result: dict[str, Any], config: dict[str, Any], destination: str | Path) -> None:
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    for key, name in (("replicates", "replicates.csv"), ("summary", "summary.csv"), ("paired", "paired_comparisons.csv"), ("paired_summary", "paired_summary.csv")):
        result[key].to_csv(folder / name, index=False)
    metadata = {"config": config, "calibration": result["calibration"], "spread_floors": result["spread_floors"], "trust_values": result["trust_values"], "scenario_count": 16, "paired_truth": "sample-level realized batch effects and covariate noise retained; residuals scaled by sqrt(delta)"}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _plot_results(result, folder / "figures")
    _write_plot_guide(folder, result)
    (folder / "report.md").write_text(
        "# Trust-Calibrated SW-ComBat Benchmark\n\n"
        f"Evaluation seeds: `{config['seeds']}`. Per-batch N: `{config['n_per_batch']}`. Crossed scenario count: 16.\n\n"
        f"Mean/scale prior-spread floors were calibrated on disjoint seeds and fixed before held-out scoring: `{result['spread_floors']}`. Activation is reported per replicate and in summaries.\n\n"
        "Positive values in `paired_comparisons.csv` indicate higher RMSE or MAE than global ComBat. No performance superiority is inferred from smoke-sized grids. See `PLOT_GUIDE.md` for figure definitions.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_trust_benchmark(config)
    destination = config.get("output_directory", "results/trust_benchmark")
    write_benchmark(result, config, destination)
    print(f"Wrote trust benchmark outputs to {destination}")
    print(f"Calibrated spread floors: {result['spread_floors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

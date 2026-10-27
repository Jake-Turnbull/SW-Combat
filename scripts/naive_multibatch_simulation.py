#!/usr/bin/env python3
"""Multi-batch sample-size benchmark for naive similarity-weighted ComBat."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import invgamma

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HarmonisationFunctions import combat_modular


class MethodSpec(NamedTuple):
    label: str
    mode: str
    weight_methods: tuple[str, ...] | None
    mean_trust: float | None
    scale_trust: float | None
    family: str
    prior_opts: tuple[tuple[str, Any], ...] = ()


BASELINE = "Baseline"
SCALE_TRUST = "Scale trust sweep"
MEAN_TRUST = "Mean trust sweep"
SCALE_PRIOR = "Scale prior variants"
CORRELATION = ("correlation_similarity",)

_baseline = (
    MethodSpec("Raw", "raw", None, None, None, BASELINE),
    MethodSpec("Global ComBat", "global", None, None, None, BASELINE),
    MethodSpec("Mean + variance", "local", ("magnitude_similarity", "variance_similarity"), 1.0, 1.0, BASELINE),
    *(
        MethodSpec(f"Correlation trust={t:g}", "local", CORRELATION, t, t, BASELINE)
        for t in (0.0, 0.1, 0.5, 0.75)
    ),
)
_scale_trust = tuple(
    MethodSpec(f"Corr mean={m:g} scale={s:g}", "local", CORRELATION, m, s, SCALE_TRUST)
    for m, s in ((0.5, 0.0), (0.5, 0.1), (0.5, 0.25), (0.5, 0.75), (0.5, 1.0), (0.0, 0.5), (0.0, 1.0))
)
_mean_trust = tuple(
    MethodSpec(f"Corr mean={m:g} scale=0", "local", CORRELATION, m, 0.0, MEAN_TRUST)
    for m in (0.1, 0.75, 1.0)
)
_prior_variants = tuple(
    MethodSpec(f"Corr scale={s:g} {name}", "local", CORRELATION, 0.5, s, SCALE_PRIOR, opts)
    for s in (0.5, 0.75)
    for name, opts in (
        ("global-cv", (("scale_prior_mode", "global_cv"),)),
        ("sampling-inflated", (("scale_prior_mode", "sampling_inflated"),)),
        ("exclude-self", (("scale_exclude_self", True),)),
        ("gaussian-kernel", (("variance_similarity_kernel", "gaussian_log"),)),
        (
            "gaussian+exclude+global-cv",
            (
                ("variance_similarity_kernel", "gaussian_log"),
                ("scale_exclude_self", True),
                ("scale_prior_mode", "global_cv"),
            ),
        ),
    )
)
METHODS = _baseline + _scale_trust + _mean_trust + _prior_variants
FAMILIES = (BASELINE, SCALE_TRUST, MEAN_TRUST, SCALE_PRIOR)
REFERENCE_LABELS = ("Raw", "Global ComBat")


def _family_specs(family: str) -> list[MethodSpec]:
    """Methods of a family plus the shared Raw/Global reference lines."""
    extras = [] if family == BASELINE else [m for m in METHODS if m.label in REFERENCE_LABELS]
    if family == MEAN_TRUST:
        extras += [m for m in METHODS if m.label == "Corr mean=0.5 scale=0"]
    return extras + [m for m in METHODS if m.family == family]


def _family_suffix(family: str) -> str:
    return "" if family == BASELINE else "_" + family.lower().replace(" ", "_")
SCENARIOS = ("simple", "opposing", "correlated_shared", "bimodal", "trimodal", "heavy_tailed")


def _feature_groups(n_features: int, n_groups: int = 5) -> np.ndarray:
    assignment = np.empty(n_features, dtype=int)
    for group_index, indices in enumerate(np.array_split(np.arange(n_features), n_groups)):
        assignment[indices] = group_index
    return assignment


def _shape_profile(rng: np.random.Generator, n_features: int, scenario: str) -> np.ndarray:
    if scenario == "simple":
        profile = rng.normal(0.0, 1.0, n_features)
    elif scenario == "opposing":
        signs = np.where(np.arange(n_features) % 2, -1.0, 1.0)
        rng.shuffle(signs)
        profile = signs * np.abs(rng.normal(0.8, 0.18, n_features))
    elif scenario == "correlated_shared":
        groups = _feature_groups(n_features)
        group_effect = rng.normal(0.0, 1.0, groups.max() + 1)
        profile = group_effect[groups] + rng.normal(0.0, 0.12, n_features)
    elif scenario in {"bimodal", "heavy_tailed"}:
        signs = np.where(np.arange(n_features) % 2, -1.0, 1.0)
        rng.shuffle(signs)
        profile = signs + rng.normal(0.0, 0.12, n_features)
        if scenario == "heavy_tailed":
            profile += rng.standard_t(3, n_features) * 0.18
    elif scenario == "trimodal":
        centers = np.array([-1.0, 0.0, 1.0])
        profile = centers[np.arange(n_features) % 3] + rng.normal(0.0, 0.12, n_features)
    else:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}.")
    profile -= profile.mean()
    standard_deviation = float(profile.std())
    return profile / standard_deviation if standard_deviation > 0 else profile


def simulate_multibatch_dataset(
    *,
    n_per_batch: int,
    n_batches: int,
    n_features: int,
    seed: int,
    scenario: str = "bimodal",
    additive_strength: float = 5.0,
    log_delta_strength: float = 0.2,
    sample_effect_sd: float = 0.05,
    feature_correlation: float = 0.35,
) -> dict[str, Any]:
    """Generate balanced multi-batch data with paired clean truth and realized effects."""
    if n_per_batch < 2 or n_batches < 2 or n_features < 5:
        raise ValueError("Require n_per_batch >= 2, n_batches >= 2, and n_features >= 5.")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}.")
    if min(additive_strength, log_delta_strength, sample_effect_sd) < 0:
        raise ValueError("Effect strengths and sample_effect_sd must be nonnegative.")
    if not 0.0 <= feature_correlation < 1.0:
        raise ValueError("feature_correlation must lie in [0, 1).")

    rng = np.random.default_rng(seed)
    n_subjects = n_per_batch * n_batches
    batch = np.repeat(np.arange(n_batches), n_per_batch)
    batch_indices = [np.flatnonzero(batch == level) for level in range(n_batches)]
    groups = _feature_groups(n_features)
    feature_scale = rng.uniform(7.0, 11.0, n_features)

    age = rng.normal(size=n_subjects)
    sex = rng.binomial(1, 0.5, n_subjects).astype(float)
    disease_probability = 1.0 / (1.0 + np.exp(-(-0.5 + 0.3 * age + 0.15 * sex)))
    disease = rng.binomial(1, disease_probability).astype(float)
    head_size = 1450.0 + 95.0 * sex - 20.0 * disease - 8.0 * age + rng.normal(0.0, 38.0, n_subjects)
    age_z = (age - age.mean()) / (age.std() + 1e-12)

    intercept = rng.normal(1000.0, 80.0, n_features)
    age_beta = rng.normal(-4.0, 0.7, n_features)
    sex_beta = rng.normal(2.0, 0.5, n_features)
    disease_beta = rng.normal(-2.5, 0.5, n_features)
    biological_mean = (
        intercept[:, None]
        + np.outer(age_beta, age_z)
        + np.outer(sex_beta, sex)
        + np.outer(disease_beta, disease)
    )

    latent_count = max(2, min(8, n_features // 5))
    latent = rng.normal(size=(latent_count, n_subjects))
    loadings = rng.normal(size=(n_features, latent_count))
    shared = loadings @ latent / np.sqrt(latent_count)
    independent = rng.normal(size=(n_features, n_subjects))
    standardized_residual = (
        np.sqrt(feature_correlation) * shared
        + np.sqrt(1.0 - feature_correlation) * independent
    )
    standardized_residual /= np.maximum(
        standardized_residual.std(axis=1, keepdims=True), np.finfo(float).eps
    )
    residual = feature_scale[:, None] * standardized_residual
    clean = biological_mean + residual

    batch_contrast = np.linspace(-1.0, 1.0, n_batches)
    gamma_profile = additive_strength * _shape_profile(rng, n_features, scenario)
    log_delta_profile = log_delta_strength * _shape_profile(
        rng, n_features, scenario if scenario != "simple" else "simple"
    )
    gamma_batch = batch_contrast[:, None] * gamma_profile[None, :] * feature_scale[None, :]
    log_delta_batch = batch_contrast[:, None] * log_delta_profile[None, :]
    gamma_sample = np.zeros((n_features, n_subjects), dtype=float)
    log_delta_sample = np.zeros_like(gamma_sample)
    for batch_index, indices in enumerate(batch_indices):
        gamma_sample[:, indices] = gamma_batch[batch_index, :, None] + rng.normal(
            0.0, sample_effect_sd * feature_scale[:, None], size=(n_features, n_per_batch)
        )
        log_delta_sample[:, indices] = log_delta_batch[batch_index, :, None] + rng.normal(
            0.0, sample_effect_sd, size=(n_features, n_per_batch)
        )
    delta_sample = np.exp(log_delta_sample)
    observed = biological_mean + gamma_sample + np.sqrt(delta_sample) * residual

    return {
        "observed": observed,
        "clean": clean,
        "signal": biological_mean,
        "residual": residual,
        "standardized_residual": standardized_residual,
        "batch": batch,
        "batches": batch_indices,
        "batch_sizes": np.full(n_batches, n_per_batch, dtype=int),
        "covariates": pd.DataFrame({"age": age, "sex": sex, "disease": disease, "head_size": head_size}),
        "feature_scale": feature_scale,
        "feature_groups": groups,
        "correlation": np.corrcoef(standardized_residual),
        "truth": {
            "gamma_profile": gamma_profile,
            "log_delta_profile": log_delta_profile,
            "gamma_batch": gamma_batch,
            "log_delta_batch": log_delta_batch,
            "gamma_sample": gamma_sample,
            "log_delta_sample": log_delta_sample,
            "delta_sample": delta_sample,
        },
        "n_per_batch": int(n_per_batch),
        "n_batches": int(n_batches),
        "seed": int(seed),
        "scenario": scenario,
    }


def _run_method(simulation: dict[str, Any], spec: MethodSpec) -> dict[str, Any]:
    if spec.mode == "raw":
        return {
            "harmonized": simulation["observed"],
            "gamma_star": None,
            "delta_star": None,
            "gamma_hat": None,
            "delta_hat": None,
            "pooled_variance": None,
            "local": None,
            "a_prior": None,
            "b_prior": None,
            "spec": spec,
        }
    weight_options: dict[str, Any] = {"min_effective": 1}
    if spec.mean_trust is not None:
        weight_options["mean_trust"] = spec.mean_trust
    if spec.scale_trust is not None:
        weight_options["scale_trust"] = spec.scale_trust
    weight_options.update(dict(spec.prior_opts))
    output = combat_modular(
        simulation["observed"],
        simulation["batch"],
        simulation["covariates"],
        prior_mode="global" if spec.mode == "global" else "local",
        prior_weight_methods=list(spec.weight_methods or ()),
        prior_weight_opts=weight_options if spec.mode == "local" else None,
        covbat_mode=False,
        return_priors=True,
    )
    local = output["priors"].get("local_priors")
    gamma_hat = np.asarray(output["priors"]["gamma_hat"])
    used = output["priors"]["priors_used"]
    # Global priors are per-batch vectors; broadcast to (batch, feature).
    a_prior = np.array(np.broadcast_to(np.asarray(used["a_prior"]).reshape(gamma_hat.shape[0], -1), gamma_hat.shape))
    b_prior = np.array(np.broadcast_to(np.asarray(used["b_prior"]).reshape(gamma_hat.shape[0], -1), gamma_hat.shape))
    return {
        "harmonized": np.asarray(output["bayesdata"]),
        "gamma_star": np.asarray(output["priors"]["gamma_star"]),
        "delta_star": np.asarray(output["priors"]["delta_star"]),
        "gamma_hat": gamma_hat,
        "delta_hat": np.asarray(output["priors"]["delta_hat"]),
        "pooled_variance": np.asarray(output["standardisation"]["var_pooled"]),
        "local": local,
        "a_prior": a_prior,
        "b_prior": b_prior,
        "spec": spec,
    }


def _prior_diagnostics(fit: dict[str, Any], delta_truth: np.ndarray) -> dict[str, float]:
    """How wide the delta prior is relative to the realised raw-estimate error."""
    a, b = fit["a_prior"], fit["b_prior"]
    valid = (a > 2.0) & (b > 0.0)
    sd = np.where(valid, b / (np.maximum(a - 1.0, 1e-12) * np.sqrt(np.maximum(a - 2.0, 1e-12))), np.nan)
    lower = np.where(valid, invgamma.ppf(0.05, np.where(valid, a, 3.0), scale=np.where(valid, b, 1.0)), np.nan)
    upper = np.where(valid, invgamma.ppf(0.95, np.where(valid, a, 3.0), scale=np.where(valid, b, 1.0)), np.nan)
    inside = np.where(valid, (delta_truth >= lower) & (delta_truth <= upper), np.nan)
    return {
        "delta_prior_sd_mean": float(np.nanmean(sd)) if valid.any() else np.nan,
        "delta_prior_coverage90": float(np.nanmean(inside)) if valid.any() else np.nan,
        "delta_raw_rmse": float(np.sqrt(np.mean((fit["delta_hat"] - delta_truth) ** 2))),
    }


def _metrics(fit: dict[str, Any], simulation: dict[str, Any]) -> dict[str, float]:
    harmonized = fit["harmonized"]
    error_clean = harmonized - simulation["clean"]
    error_observed = harmonized - simulation["observed"]
    metrics = {
        "rmse_to_clean": float(np.sqrt(np.mean(error_clean**2))),
        "mae_to_clean": float(np.mean(np.abs(error_clean))),
        "rmse_to_observed": float(np.sqrt(np.mean(error_observed**2))),
        "mae_to_observed": float(np.mean(np.abs(error_observed))),
    }
    if fit["gamma_star"] is None:
        for name in ("gamma_rmse", "gamma_mae", "delta_rmse", "delta_mae", "log_delta_rmse", "log_delta_mae",
                     "delta_prior_sd_mean", "delta_prior_coverage90", "delta_raw_rmse"):
            metrics[name] = np.nan
        return metrics

    proportions = simulation["batch_sizes"] / simulation["batch_sizes"].sum()
    gamma_batch_mean = np.stack([
        np.mean(simulation["truth"]["gamma_sample"][:, indices], axis=1)
        for indices in simulation["batches"]
    ])
    delta_batch_mean = np.stack([
        np.mean(simulation["truth"]["delta_sample"][:, indices], axis=1)
        for indices in simulation["batches"]
    ])
    gamma_truth = (gamma_batch_mean - proportions @ gamma_batch_mean)[None, :, :]
    gamma_truth = gamma_truth[0] / np.sqrt(fit["pooled_variance"])[None, :]
    delta_truth = delta_batch_mean * simulation["feature_scale"][None, :] ** 2
    delta_truth /= fit["pooled_variance"][None, :]
    gamma_error = fit["gamma_star"] - gamma_truth
    delta_error = fit["delta_star"] - delta_truth
    metrics.update({
        "gamma_rmse": float(np.sqrt(np.mean(gamma_error**2))),
        "gamma_mae": float(np.mean(np.abs(gamma_error))),
        "delta_rmse": float(np.sqrt(np.mean(delta_error**2))),
        "delta_mae": float(np.mean(np.abs(delta_error))),
        "log_delta_rmse": float(np.sqrt(np.mean((np.log(np.maximum(fit["delta_star"], 1e-12)) - np.log(np.maximum(delta_truth, 1e-12))) ** 2))),
        "log_delta_mae": float(np.mean(np.abs(np.log(np.maximum(fit["delta_star"], 1e-12)) - np.log(np.maximum(delta_truth, 1e-12))))),
        **_prior_diagnostics(fit, delta_truth),
    })
    return metrics


def _metric_row(simulation: dict[str, Any], fit: dict[str, Any]) -> dict[str, Any]:
    spec: MethodSpec = fit["spec"]
    local = fit["local"]
    options = dict(spec.prior_opts)
    return {
        "scenario": simulation["scenario"],
        "n_per_batch": simulation["n_per_batch"],
        "n_batches": simulation["n_batches"],
        "n_total": int(simulation["n_per_batch"] * simulation["n_batches"]),
        "n_features": int(simulation["observed"].shape[0]),
        "seed": simulation["seed"],
        "method": spec.label,
        "family": spec.family,
        "weight_methods": "+".join(spec.weight_methods) if spec.weight_methods else "none",
        "trust": spec.mean_trust if spec.mean_trust is not None else np.nan,
        "mean_trust": spec.mean_trust if spec.mean_trust is not None else np.nan,
        "scale_trust": spec.scale_trust if spec.scale_trust is not None else np.nan,
        "scale_prior_mode": options.get("scale_prior_mode", "moment" if local else "none"),
        "variance_kernel": options.get("variance_similarity_kernel", "legacy" if local else "none"),
        "scale_exclude_self": bool(options.get("scale_exclude_self", False)),
        **_metrics(fit, simulation),
        "mean_ess": float(np.mean(local["effective_sample_size"])) if local else np.nan,
        "scale_ess": float(np.mean(local["scale_effective_sample_size"])) if local else np.nan,
        "fallback_frequency": float(np.mean(local["used_global_fallback"])) if local else 0.0,
    }


def _fit_sweep_case(simulation: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    snapshots: dict[str, Any] = {}
    for spec in METHODS:
        fit = _run_method(simulation, spec)
        local = fit["local"]
        rows.append(_metric_row(simulation, fit))
        label = spec.label
        if local:
            snapshots[label] = {
                "mean": np.asarray(local["weights"]),
                "scale": np.asarray(local["scale_weights"]),
            }
        else:
            uniform = np.full(
                (simulation["n_batches"], simulation["observed"].shape[0], simulation["observed"].shape[0]),
                1.0 / simulation["observed"].shape[0],
            )
            snapshots[label] = {"mean": uniform, "scale": uniform}
    return rows, snapshots


def _effect_rows(simulation: dict[str, Any], fits: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    feature_scale = simulation["feature_scale"]
    proportions = simulation["batch_sizes"] / simulation["batch_sizes"].sum()
    gamma_mean = np.stack([
        np.mean(simulation["truth"]["gamma_sample"][:, indices], axis=1)
        for indices in simulation["batches"]
    ])
    delta_mean = np.stack([
        np.mean(simulation["truth"]["delta_sample"][:, indices], axis=1)
        for indices in simulation["batches"]
    ])
    for label, fit in fits.items():
        if fit["gamma_star"] is None:
            continue
        gamma_truth = (gamma_mean - proportions @ gamma_mean) / np.sqrt(fit["pooled_variance"])[None, :]
        delta_truth = delta_mean * feature_scale[None, :] ** 2 / fit["pooled_variance"][None, :]
        local = fit.get("local")
        for batch_idx in range(simulation["n_batches"]):
            for feature_idx in range(feature_scale.size):
                a_prior = float(fit["a_prior"][batch_idx, feature_idx])
                b_prior = float(fit["b_prior"][batch_idx, feature_idx])
                prior_variance = (
                    b_prior**2 / ((a_prior - 1.0) ** 2 * (a_prior - 2.0))
                    if a_prior > 2.0 else np.nan
                )
                rows.append({
                    "scenario": simulation["scenario"],
                    "n_per_batch": simulation["n_per_batch"],
                    "n_batches": simulation["n_batches"],
                    "n_features": feature_scale.size,
                    "seed": simulation["seed"],
                    "method": label,
                    "family": fit["spec"].family,
                    "mean_trust": fit["spec"].mean_trust,
                    "scale_trust": fit["spec"].scale_trust,
                    "batch": batch_idx,
                    "feature": feature_idx,
                    "gamma_true": float(gamma_truth[batch_idx, feature_idx]),
                    "gamma_estimated": float(fit["gamma_star"][batch_idx, feature_idx]),
                    "gamma_error": float(fit["gamma_star"][batch_idx, feature_idx] - gamma_truth[batch_idx, feature_idx]),
                    "gamma_hat": float(fit["gamma_hat"][batch_idx, feature_idx]),
                    "delta_true": float(delta_truth[batch_idx, feature_idx]),
                    "delta_estimated": float(fit["delta_star"][batch_idx, feature_idx]),
                    "delta_error": float(fit["delta_star"][batch_idx, feature_idx] - delta_truth[batch_idx, feature_idx]),
                    "delta_hat": float(fit["delta_hat"][batch_idx, feature_idx]),
                    "log_delta_true": float(np.log(max(delta_truth[batch_idx, feature_idx], 1e-12))),
                    "log_delta_estimated": float(np.log(max(fit["delta_star"][batch_idx, feature_idx], 1e-12))),
                    "log_delta_error": float(
                        np.log(max(fit["delta_star"][batch_idx, feature_idx], 1e-12))
                        - np.log(max(delta_truth[batch_idx, feature_idx], 1e-12))
                    ),
                    "gamma_batch_truth_original": float(simulation["truth"]["gamma_batch"][batch_idx, feature_idx]),
                    "log_delta_batch_truth": float(simulation["truth"]["log_delta_batch"][batch_idx, feature_idx]),
                    "delta_batch_truth": float(np.exp(simulation["truth"]["log_delta_batch"][batch_idx, feature_idx])),
                    "a_prior_used": a_prior,
                    "b_prior_used": b_prior,
                    "scale_prior_variance_used": (
                        float(local["scale_variance_local"][batch_idx, feature_idx])
                        if local is not None else np.nan
                    ),
                    "inverse_gamma_prior_variance": prior_variance,
                })
    return rows


def _save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    figure.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _line_with_band(axis: Any, subset: pd.DataFrame, label: str, metric: str) -> None:
    summary = subset[subset["method"] == label].groupby("n_per_batch")[metric].agg(["mean", "std", "count"])
    if summary.empty or summary["mean"].isna().all():
        return
    half = 1.96 * summary["std"].fillna(0) / np.sqrt(summary["count"].clip(lower=1))
    axis.plot(summary.index, summary["mean"], marker="o", label=label)
    axis.fill_between(summary.index, summary["mean"] - half, summary["mean"] + half, alpha=0.1)


def _make_family_figures(
    metrics: pd.DataFrame,
    representatives: dict[tuple[Any, ...], Any],
    scenario: str,
    n_batches: int,
    sizes: list[int],
    output: Path,
    prefix: str,
) -> None:
    """Figures 2-5 for each method family, so every plot holds only a handful of lines."""
    subset = metrics[(metrics["n_batches"] == n_batches) & (metrics["scenario"] == scenario)]
    for family in FAMILIES:
        specs = _family_specs(family)
        labels = [spec.label for spec in specs]
        suffix = _family_suffix(family)
        for sample_key, size_label in ((sizes[0], "low_n"), (sizes[-1], "high_n")):
            snap = representatives[(scenario, n_batches, int(sample_key))]["weights"]
            local_labels = [label for label in labels if label != "Raw"]
            figure, axes = plt.subplots(
                len(local_labels), 2, figsize=(8, max(6, 1.8 * len(local_labels))),
                constrained_layout=True, squeeze=False,
            )
            for row_idx, label in enumerate(local_labels):
                for col_idx, effect in enumerate(("mean", "scale")):
                    image = axes[row_idx, col_idx].imshow(snap[label][effect][0], cmap="viridis", vmin=0)
                    title = (
                        f"{label}: uniform global pooling, batch 0"
                        if label == "Global ComBat"
                        else f"{label}: batch 0 {effect} weights"
                    )
                    axes[row_idx, col_idx].set_title(title, fontsize=7)
                    axes[row_idx, col_idx].set(xlabel="Source feature", ylabel="Target feature")
                    figure.colorbar(image, ax=axes[row_idx, col_idx], fraction=0.046)
            figure.suptitle(f"{family}: {scenario}, {n_batches} batches, N={sample_key} per batch")
            _save_figure(figure, output, f"{prefix}_figure2_similarity_matrices_{size_label}{suffix}")

        panels = (
            ("figure3_rmse_by_sample_size", (("rmse_to_clean", "RMSE to clean batch-free truth"), ("mae_to_clean", "MAE to clean batch-free truth"))),
            ("figure4_batch_effect_parameter_errors", (("gamma_rmse", "Gamma batch-effect RMSE"), ("log_delta_rmse", "Log-delta batch-effect RMSE"))),
            ("figure5_delta_prior_width", (("delta_prior_coverage90", "Delta prior 90% interval covers truth"), ("delta_prior_sd_mean", "Mean delta prior sd"))),
        )
        for stem, metric_panels in panels:
            figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
            for axis, (metric, title) in zip(axes, metric_panels):
                for label in labels:
                    _line_with_band(axis, subset, label, metric)
                axis.set(title=title, xlabel="N per batch", ylabel=metric)
                axis.grid(axis="y", alpha=0.2)
                axis.legend(frameon=False, fontsize=7)
            if stem.startswith("figure5"):
                axes[0].axhline(0.9, color="k", linestyle=":", linewidth=0.8)
            figure.suptitle(f"{family}: {scenario}, {n_batches} batches")
            _save_figure(figure, output, f"{prefix}_{stem}{suffix}")


def _make_figures(metrics: pd.DataFrame, representatives: dict[tuple[Any, ...], Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    scenarios = list(dict.fromkeys(metrics["scenario"]))
    batch_counts = sorted(int(value) for value in metrics["n_batches"].unique())
    for scenario in scenarios:
        for n_batches in batch_counts:
            representative_sizes = sorted(
                size for candidate, count, size in representatives
                if candidate == scenario and count == n_batches
            )
            if not representative_sizes:
                continue
            low_n = representative_sizes[0]
            sim = representatives[(scenario, n_batches, low_n)]["simulation"]
            figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
            corr_image = axes[0].imshow(sim["correlation"], vmin=-1, vmax=1, cmap="coolwarm")
            axes[0].set(title=f"Feature correlation: {scenario}", xlabel="Feature", ylabel="Feature")
            figure.colorbar(corr_image, ax=axes[0])
            for batch_idx in range(n_batches):
                axes[1].plot(sim["truth"]["gamma_batch"][batch_idx] / sim["feature_scale"], label=f"Batch {batch_idx}", marker=".")
                axes[2].plot(sim["truth"]["log_delta_batch"][batch_idx], label=f"Batch {batch_idx}", marker=".")
            axes[1].set(title="True additive effects", xlabel="Feature", ylabel="Gamma / feature scale")
            axes[2].set(title="True log-delta effects", xlabel="Feature", ylabel="Log delta")
            axes[1].legend(frameon=False, fontsize=7, ncol=2)
            axes[2].legend(frameon=False, fontsize=7, ncol=2)
            prefix = f"{scenario}_n_batches_{n_batches}"
            _save_figure(figure, output, f"{prefix}_figure1_feature_correlations_and_effects")

            _make_family_figures(metrics, representatives, scenario, n_batches, representative_sizes, output, prefix)
            gamma = sim["truth"]["gamma_batch"]
            log_delta = sim["truth"]["log_delta_batch"]

            figure, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)

            image = axes[0].imshow(gamma, aspect="auto", cmap="coolwarm")
            axes[0].set(
                title="Simulated location effects",
                xlabel="Feature",
                ylabel="Batch",
                yticks=np.arange(n_batches),
            )
            figure.colorbar(image, ax=axes[0], label="Additive effect")

            image = axes[1].imshow(log_delta, aspect="auto", cmap="coolwarm")
            axes[1].set(
                title="Simulated log-scale effects",
                xlabel="Feature",
                ylabel="Batch",
                yticks=np.arange(n_batches),
            )
            figure.colorbar(image, ax=axes[1], label="Log variance multiplier")

            _save_figure(figure, output, f"{prefix}_figure_simulated_parameters")


def run_benchmark(
    *,
    sample_sizes: list[int],
    batch_counts: list[int],
    seeds: list[int],
    feature_counts: list[int],
    scenarios: list[str],
    output: Path,
    additive_strength: float = 5.0,
    log_delta_strength: float = 0.2,
    sample_effect_sd: float = 0.05,
    feature_correlation: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    effect_records: list[dict[str, Any]] = []
    representatives: dict[tuple[Any, ...], Any] = {}
    for scenario in scenarios:
        for n_batches in batch_counts:
            for n_per_batch in sample_sizes:
                for n_features in feature_counts:
                    for seed in seeds:
                        simulation = simulate_multibatch_dataset(
                            n_per_batch=n_per_batch,
                            n_batches=n_batches,
                            n_features=n_features,
                            seed=seed,
                            scenario=scenario,
                            additive_strength=additive_strength,
                            log_delta_strength=log_delta_strength,
                            sample_effect_sd=sample_effect_sd,
                            feature_correlation=feature_correlation,
                        )
                        fits: dict[str, dict[str, Any]] = {}
                        snapshots = {}
                        for spec in METHODS:
                            label = spec.label
                            fit = _run_method(simulation, spec)
                            fits[label] = fit
                            local = fit["local"]
                            metric_rows.append(_metric_row(simulation, fit))
                            if local:
                                snapshots[label] = {
                                    "mean": np.asarray(local["weights"]),
                                    "scale": np.asarray(local["scale_weights"]),
                                }
                            else:
                                uniform = np.full((n_batches, n_features, n_features), 1.0 / n_features)
                                snapshots[label] = {"mean": uniform, "scale": uniform}
                        effect_records.extend(_effect_rows(simulation, fits))
                        representative_key = (scenario, n_batches, n_per_batch)
                        if n_features == feature_counts[0] and seed == seeds[0]:
                            representatives[representative_key] = {"simulation": simulation, "weights": snapshots}

    metrics = pd.DataFrame(metric_rows)
    metric_keys = ["scenario", "n_per_batch", "n_batches", "n_features", "seed"]
    global_metrics = metrics[metrics["method"] == "Global ComBat"][metric_keys + ["rmse_to_clean", "mae_to_clean"]]
    paired = metrics[metrics["method"].isin([spec.label for spec in METHODS if spec.label not in {"Raw", "Global ComBat"}])].merge(
        global_metrics, on=metric_keys, suffixes=("", "_global"), validate="many_to_one"
    )
    paired["rmse_difference_vs_global"] = paired["rmse_to_clean"] - paired["rmse_to_clean_global"]
    paired["mae_difference_vs_global"] = paired["mae_to_clean"] - paired["mae_to_clean_global"]
    paired["local_worsened_rmse"] = paired["rmse_difference_vs_global"] > 0
    group_columns = [
        "scenario", "n_per_batch", "n_batches", "n_features", "method", "family", "trust",
        "mean_trust", "scale_trust", "scale_prior_mode", "variance_kernel", "scale_exclude_self",
    ]
    summary = metrics.groupby(group_columns, dropna=False).agg(
        n_seeds=("seed", "nunique"),
        rmse_to_clean_mean=("rmse_to_clean", "mean"),
        rmse_to_clean_sd=("rmse_to_clean", "std"),
        mae_to_clean_mean=("mae_to_clean", "mean"),
        rmse_to_observed_mean=("rmse_to_observed", "mean"),
        gamma_rmse_mean=("gamma_rmse", "mean"),
        gamma_mae_mean=("gamma_mae", "mean"),
        delta_rmse_mean=("delta_rmse", "mean"),
        log_delta_rmse_mean=("log_delta_rmse", "mean"),
        delta_prior_sd_mean=("delta_prior_sd_mean", "mean"),
        delta_prior_coverage90=("delta_prior_coverage90", "mean"),
        delta_raw_rmse_mean=("delta_raw_rmse", "mean"),
        mean_ess=("mean_ess", "mean"),
        scale_ess=("scale_ess", "mean"),
        fallback_frequency=("fallback_frequency", "mean"),
    ).reset_index()
    paired_summary = paired.groupby(group_columns, dropna=False).agg(
        fraction_local_worsened=("local_worsened_rmse", "mean"),
        rmse_difference_mean=("rmse_difference_vs_global", "mean"),
        mae_difference_mean=("mae_difference_vs_global", "mean"),
    ).reset_index()
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "replicates.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    paired.to_csv(output / "paired_comparisons.csv", index=False)
    paired_summary.to_csv(output / "paired_summary.csv", index=False)
    pd.DataFrame(effect_records).to_csv(output / "batch_effect_estimates.csv", index=False)
    _make_figures(metrics, representatives, output / "figures")
    return metrics, summary, paired


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/naive_multibatch")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=list(range(10, 401, 10)), help="Subjects per batch.")
    parser.add_argument("--batch-counts", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--features", type=int, nargs="+", default=[50])
    parser.add_argument("--scenarios", choices=SCENARIOS, nargs="+", default=["bimodal", "correlated_shared"])
    parser.add_argument("--additive-strength", type=float, default=2.5)
    parser.add_argument("--log-delta-strength", type=float, default=0.4)
    parser.add_argument("--sample-effect-sd", type=float, default=0.1)
    parser.add_argument("--feature-correlation", type=float, default=0.6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if any(value < 2 for value in args.sample_sizes + args.batch_counts):
        raise SystemExit("sample sizes and batch counts must be at least two")
    if any(value < 5 for value in args.features):
        raise SystemExit("feature counts must be at least five")
    config = {
        "sample_sizes_per_batch": args.sample_sizes,
        "batch_counts": args.batch_counts,
        "seeds": args.seeds,
        "feature_counts": args.features,
        "scenarios": args.scenarios,
        "methods": [
            {
                "label": spec.label, "family": spec.family, "mode": spec.mode,
                "weight_methods": spec.weight_methods, "mean_trust": spec.mean_trust,
                "scale_trust": spec.scale_trust, "prior_opts": dict(spec.prior_opts),
            }
            for spec in METHODS
        ],
        "effect_parameters": {
            "additive_strength": args.additive_strength,
            "log_delta_strength": args.log_delta_strength,
            "sample_effect_sd": args.sample_effect_sd,
            "feature_correlation": args.feature_correlation,
        },
        "truth_metric": "paired clean batch-free signal for reconstruction; realized batch-mean gamma/delta in pooled standardized units for parameter errors",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    run_benchmark(
        sample_sizes=args.sample_sizes,
        batch_counts=args.batch_counts,
        seeds=args.seeds,
        feature_counts=args.features,
        scenarios=args.scenarios,
        output=args.output,
        additive_strength=args.additive_strength,
        log_delta_strength=args.log_delta_strength,
        sample_effect_sd=args.sample_effect_sd,
        feature_correlation=args.feature_correlation,
    )
    print(f"Outputs written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*unstable below.*")
        raise SystemExit(main())

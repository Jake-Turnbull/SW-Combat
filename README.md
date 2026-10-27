# simcombat

Research implementation of global parametric ComBat and similarity-weighted empirical Bayes ComBat. Input arrays use features by subjects orientation. The original script is preserved in `legacy/` and is not imported by the package.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m simcombat.benchmark --config configs/smoke.json
python -m simcombat.benchmark --config configs/pilot.json
python -m simcombat.reference --seeds 8101 8102 8103 8104 8105
```

`fit_combat` currently implements `prior_mode="global"`, `prior_mode="naive"`, `prior_mode="local_likelihood"`, and `prior_mode="mixture"`. The mixture uses shared structural feature memberships for additive and multiplicative priors and a quadrature-integrated component posterior; its correction is a posterior-mean plug-in transformation, not a full posterior predictive transformation. Mixture fitting currently uses deterministic generalized-EM updates of raw-effect summaries with a diagonal sampling-uncertainty working likelihood; graph regularization and Monte Carlo calibration are not implemented yet.

## Trust-Calibrated Root SW-ComBat

The upstream `HarmonisationFunctions.combat_modular(prior_mode="local")` path accepts `prior_weight_opts["trust"]`, defaulting to `0.5`. For each target feature, trust blends its normalized local row with uniform pooling:

$$
w_t = t\,w_{\mathrm{local}} + (1-t)/p.
$$

The blend is applied independently to mean and scale weights after local similarity scores receive the configured `similarity_floor` (default `0.01`) and are normalized. Consequently trust `0` is exactly uniform regardless of the similarity matrix; trust `1` preserves the normalized local weights. Correlation similarity is calculated within each batch and uses signed-shifted correlation, `(r + 1) / 2`. The existing root ratio/log variance score remains scale-only.

Optional `mean_spread_floor` and `scale_spread_floor` bounds in `prior_weight_opts` constrain the local weighted gamma and delta dispersions. They are unset by default. Outputs report boolean per-prior activation arrays, so the bounds are auditable. In the trust benchmark, these bounds are calibrated from separate development seeds over all 16 shape pairs and fixed before held-out scoring; calibration and evaluation seeds must not overlap.

The new simulation crosses four feature-level additive mean shapes and four log-delta shapes independently: unimodal, bimodal, trimodal, and zero-centered bimodal. It adds sample-varying batch effects and sample-varying covariate-effect noise, records realized effects as truth, and scales residuals by `sqrt(delta)`. Global and root SW-ComBat are compared with the existing package local-likelihood and mixture estimators; those estimators retain their existing implementation/API and have different similarity mechanisms. The benchmark reports reconstruction RMSE/MAE, Cohen's d mean-shift bias, variance ratio, parameter errors, effective feature counts, fallback frequency, spread-floor activation, and the fraction of paired seeds for which local RMSE is worse than global.

Quick run (small held-out grid; calibration seed is disjoint):

```bash
python -m simcombat.trust_benchmark --config configs/trust_smoke.json
```

Outputs are written to `results/trust_smoke/`: replicate and aggregate CSVs, metadata, paired comparisons, scenario-family PNG/PDF plots, a report, and `PLOT_GUIDE.md`. The plots separate mean-shape and log-delta-shape families and include correlation/effect profiles, structural and mean/scale weight matrices, trust/per-batch-N performance, and RMSE/MAE/Cohen's d/variance-ratio comparisons across SW-ComBat, local likelihood, mixture, and global ComBat.

## Multi-Batch Naive-Prior Sweep

Run the separate Bimodal-style benchmark for the root naïve weighted-moment local priors across multiple batches and per-batch sample sizes:

```bash
python scripts/naive_multibatch_simulation.py \
	--output results/naive_multibatch_quick \
	--sample-sizes 10 20 40 \
	--batch-counts 2 3 \
	--seeds 1 2 \
	--features 50 \
	--scenarios bimodal
```

The default sweep uses per-batch N values from 10 through 400, batch counts 2/3/5, three seeds, 50 features, and bimodal/correlated-shared scenarios. It compares raw data, global ComBat, a combined `magnitude_similarity` + `variance_similarity` configuration, and `correlation_similarity` at trust 0, 0.1, 0.5, and 0.75. Experimental `scale-mix` variants of each local method blend local weighted delta dispersion with batch-global dispersion before deriving inverse-gamma `a,b`; `scale-mix=0` is the existing method, while larger values test a wider, less locally determined scale prior. Variance similarity contributes to scale-prior weights only; mean and scale weights remain independent.

Replicate and summary metrics include RMSE/MAE to paired clean batch-free truth, distance to observed unharmonized data, and gamma/delta parameter errors. `batch_effect_estimates.csv` contains per-case, per-batch, per-feature raw (`gamma_hat`, `delta_hat`) and EB-adjusted predictions, standardized truth and errors, simulated batch-level truth, and inverse-gamma prior width diagnostics. Estimated and standardized truth columns share the method's fitted pooled-variance scale; `log_delta_*` columns compare log standardized variances. `correlation_matrix_comparisons.csv` contains the full raw within-batch correlation matrices from ComBat-standardized data and the matching realized underlying residual correlations; `correlation_agreement.csv` summarizes off-diagonal agreement and error by case, batch, and method. The truth correlation is a finite-sample within-batch realization, not a population covariance.

Aggregate figures retain the feature/effect, low/high-N weight, performance, and parameter-error views. The `figures/cases/` directory adds three PNG/PDF files for every scenario × batch-count × N × feature-count × seed: a per-batch truth/observed/difference correlation comparison, gamma/log-delta truth and EB-estimate heatmaps for each method, and per-batch mean/scale weight matrices for each method. This is intentionally a large output for full sweeps; all case-level predictions and matrices are also available in CSV.

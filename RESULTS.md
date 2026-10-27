## neuroCombat Smoke Comparison

A separate five-seed smoke comparison used `neuroCombat 0.2.12` and local global ComBat on the same simulated observations, batch assignments, continuous age covariate, and paired batch-free truth. Mean native-unit RMSE was `24.3653` for local ComBat and `24.5248` for neuroCombat. Mean per-feature-standardized RMSE was `0.378881` and `0.379490`, respectively. The paired standardized-RMSE difference (local minus neuroCombat) was `-0.000609`, with a five-seed replicate-bootstrap interval of `[-0.001081, -0.000037]`.

These are small smoke-sample differences from one scenario and are not a general superiority claim. Full per-seed values, scenario settings, and package version are in `results/neurocombat_smoke/`.
# Simulation Results

## Scope

The final evaluation used 50 fixed seeds in each of six scenarios: homogeneous effects with informative similarity and small batches; heterogeneous effects with informative, partial, uninformative, or misleading similarity at moderate batch size; and heterogeneous informative effects with small batches. Each replicate was generated once and passed to all four methods. Counterfactual truth used the same biological signal and noise draw as observed data. Parameter errors were aligned to each fit's pooled-reference standardization.

The authoritative raw output is `results/study_final/replicates.csv`; scenario summaries and replicate-bootstrap paired comparisons are `results/study_final/summary.csv` and `results/study_final/paired_comparisons.csv`. The study recorded 1,200 method-runs, 50 seeds per scenario, and zero method failures. The 12-seed pilot is development-only and is excluded from these evaluation summaries.

## Primary Outcome

Across the six scenarios, arithmetic means of scenario-level standardized RMSE were approximately:

| Method | Mean scenario RMSE | Range across scenarios |
| --- | ---: | ---: |
| Global parametric ComBat | 0.336 | 0.129-0.390 |
| Naive weighted moments | 0.339 | 0.133-0.392 |
| Locally weighted likelihood | 0.331 | 0.105-0.383 |
| Soft-membership mixture | 0.325 | 0.079-0.377 |

These are descriptive summaries across the specified synthetic scenarios, not a population-weighted estimand. In the heterogeneous moderate-batch scenarios, the paired standardized-RMSE difference from global was about `+0.00074` for naive moments, `-0.0040` for local likelihood, and `-0.0060` for the mixture in the informative scenario. The differences varied little across partial/uninformative/misleading labels because the current similarity pipeline does not use similarity to update local likelihood or mixture memberships in a way that makes those methods responsive to all those structural changes. Scenario-specific paired intervals are in the CSV; they do not establish superiority outside this generator.

In the homogeneous informative small-batch scenario, global RMSE was about `0.129`, naive `0.133`, local likelihood `0.105`, and mixture `0.079`. This is a substantial result within this one generator setting, but the mixture's absolute mean bias and variance discrepancy also differ from global; method choice should not be reduced to RMSE alone.

## Parameter Outcomes

Across scenario summaries, posterior gamma RMSE means were approximately 0.090 (global), 0.093 (naive), 0.073 (local likelihood), and 0.052 (mixture). Mean log-delta RMSE values were approximately 0.243, 0.245, 0.237, and 0.235, respectively. These metrics compare posterior-adjusted estimates against simulator truth aligned to the fitted pooled variance/reference convention. They are not evidence of calibrated posterior uncertainty.

## Limitations

- The generator is synthetic and semi-realistic; it is not a substitute for validation on external multi-site MRI data.
- Feature errors are correlated, while local likelihood fitting uses a diagonal uncertainty working model.
- The mixture uses shared feature memberships for location and scale and generalized-EM summary updates; it does not use graph regularization and is sensitive to initialization and component identifiability.
- Mixture posterior correction uses component-integrated posterior means as a plug-in ComBat transformation, not a full posterior predictive transformation.
- Current global/naive/local/mixture methods share preprocessing and posterior conventions, but the mixture parameter-fitting objective is a working model; see implementation notes.
- Hyperparameters were fixed before final evaluation seeds; the pilot is not included in the final results. Additional factorial axes (heavy-tailed errors, stronger confounding, K=3-4, feature-scale sensitivity) remain future work.
- No method is claimed superior in general. Results support only scenario-conditional comparisons under the exact documented generator/configuration.
- The global implementation was compared with the independent `neuroCombat` parametric implementation on a seeded balanced Gaussian example; relative RMSE was below 0.2 reference standard deviations. This is a simple regression check, not comprehensive equivalence validation.

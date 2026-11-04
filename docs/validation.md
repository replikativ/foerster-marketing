# Validation

[README](../README.md) · [Model](model.md) · [API](api.md) · [Study](../study/README.md) · [Warehouse](warehouse.md) · [Performance](performance.md)

Foerster Marketing is validated in two ways: a regression suite that checks the library against independent references, and a study that fits the same model as PyMC-Marketing and requires both to agree.
Everything here is synthetic.
Nothing on this page establishes how the library behaves on real retailer data.

## Regression suite

`clojure -M:jvm:test` runs 58 tests with 886 assertions in sequential fresh JVMs, on Foerster 0.1.137, Raster 0.2.1720 and foerster-raster 0.1.14.
The tests compare the model with independent finite convolutions and densities, check runtime and AOT gradients against each other and against finite differences, refit with changed data, check that results are invariant to the unit of currency, compare allocations with exhaustive grids, and round-trip posteriors through EDN and JSON.

## The Kettle & Pine study

The [study](../study/README.md) fits three versions of Kettle & Pine's history with PyMC-Marketing 1.3.0 (PyMC 6.3.2) and Foerster Marketing, with identical data, priors and scaling.
All results below come from one run of `study/run_study.py` at the published revision, with no changes to the source during the run.
The receipts are in [`results/kettle-and-pine/`](../results/kettle-and-pine/): [`process-staggered-co-scheduled-calibrated.json`](../results/kettle-and-pine/process-staggered-co-scheduled-calibrated.json) records every stage's command, exit status and resource use, and [`inputs.json`](../results/kettle-and-pine/inputs.json) the hashes of every source file.
The posteriors themselves (about 30 MB per case) are attached to the GitHub release rather than stored in git.
The receipts name revision `8f2426c`, the same tree before the results and these documents were added; every source hash in `inputs.json` matches this repository's files, except `build.clj` and the `:build` alias in `deps.edn`, which later gained only the GitHub release task used by CI.

### Same density

Before sampling, both engines evaluate the log density, its gradient, original-unit means and pointwise likelihoods at three fixed points, also with changed observations.
The largest differences across all three cases are 4.3e-13 for the log density, 7.2e-14 for the scaled gradient and 1.1e-11 for original-unit means ([`numerical-comparison.json`](../results/kettle-and-pine/calibrated/numerical-comparison.json) per case).

### Converged fits

Each engine runs four chains with 2,000 warm-up and 4,000 retained draws per chain, from the same prior-based starting point.
The gates are R-hat below 1.01, bulk and tail ESS of at least 100, and no divergences.

| Case | Foerster max R-hat | Foerster min ESS (bulk / tail) | PyMC max R-hat | PyMC min ESS (bulk / tail) | Divergences | Depth-limit hits (Foerster / PyMC) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Staggered | 1.0009 | 8,013 / 8,047 | 1.0013 | 5,895 / 7,079 | 0 / 0 | 0 / 0 |
| Overlapping | 1.0005 | 7,532 / 8,092 | 1.0009 | 7,539 / 8,101 | 0 / 0 | 4 / 1 |
| Overlapping + lift test | 1.0010 | 7,052 / 8,488 | 1.0008 | 6,837 / 7,611 | 0 / 0 | 0 / 0 |

On the 26 held-out weeks, the actual revenue lies inside the 90% predictive interval in 23 weeks for the staggered history and in 22 for both overlapping versions.

### Same posterior

Every parameter mean agrees between the engines within 1.6, 2.0 and 2.6 combined Monte Carlo standard errors in the three cases; the gate is five.
Each engine's recommended plan is evaluated on both posteriors, and the expected gains agree within 1.3 combined standard errors ([`posterior-comparison.json`](../results/kettle-and-pine/calibrated/full/posterior-comparison.json) per case).

### Same decision

| Case | Foerster plan (EUR/week) | PyMC plan (EUR/week) | Expected gain of Foerster's plan, 90% interval (EUR) |
| --- | --- | --- | --- |
| Staggered | 1,386 / 2,016 / 1,598 | 1,403 / 2,005 / 1,592 | 1,307 (−870 to 3,231) |
| Overlapping | 1,382 / 2,406 / 1,212 | 1,375 / 2,402 / 1,223 | 1,678 (−1,186 to 4,265) |
| Overlapping + lift test | 1,372 / 2,013 / 1,615 | 1,355 / 2,000 / 1,645 | 1,559 (−904 to 3,831) |

Plans are weekly spend on search, social and video; the current plan is 2,000 / 1,800 / 1,200.
Each allocation is rechecked by an independent NumPy/SciPy optimiser and a feasible grid, and Foerster's decision reproduces exactly from the saved posterior in a fresh process, from both EDN and JSON.

### Against the hidden truth

Because the data are generated, each plan can be scored against the plan that is optimal under the true parameters ([`truth-regret.json`](../results/kettle-and-pine/truth-regret.json)).
The truth is used only after both engines have chosen.

| Case | Best possible gain | Foerster plan: true gain (regret) | PyMC plan: true gain (regret) |
| --- | ---: | ---: | ---: |
| Staggered | 2,861 | 2,567 (294) | 2,532 (329) |
| Overlapping | 3,114 | 1,682 (1,433) | 1,739 (1,375) |
| Overlapping + lift test | 3,114 | 2,787 (327) | 2,836 (278) |

When social and video always run together, both engines' plans lose almost half of the possible gain; one independent lift experiment recovers most of it.

### Warehouse

The [warehouse extension](warehouse.md) passes its own checks: the caller's warehouse state is unchanged, the capacity posterior agrees with an independent quadrature within 1.8 Monte Carlo standard errors for all three seeds, and an independent NumPy reimplementation reproduces every plan's profit exactly ([`independent-check.json`](../results/kettle-and-pine/warehouse/independent-check.json)).

## What this does not show

- **Real data.** The generator belongs to the fitted model family, so the model is correctly specified by construction. Misspecification, unobserved confounding and real campaign data are untested.
- **Repeated worlds.** Each case is one synthetic history. The study does not show that intervals are calibrated over many draws of the world.
- **Causal identification.** A good fit and a good holdout forecast can coexist with poorly identified channel effects, as the overlapping case shows.
- **Long-running services.** Compiled kernels are cached (at most 32 structures); long-lived worker memory and hard cancellation are not qualified.

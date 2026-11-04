# Contributor guide

Foerster Marketing is an experimental Bayesian marketing mix modelling library for the JVM, licensed under Apache-2.0.
Read [README.md](README.md) and [docs/validation.md](docs/validation.md) before changing any claim about what the library does or how well it does it.

## Layout

- `src/org/replikativ/marketing/`: the library. The public namespaces are `model`, `inference`, `decision`, `io` and `hierarchy`. The others (`source`, `layout`, `family`, `gradient`, `config`, `response`, `seasonality`, `calibration`) are implementation detail and marked `^:no-doc`.
- `test/`: the regression suite, run in sequential fresh JVMs by `test_runner.clj`.
- `examples/kettle_and_pine/`: the worked example and its data, including the warehouse extension.
- `study/`: the comparison with PyMC-Marketing (Python and Clojure) and its driver.
- `docs/`: API, decision workflow, warehouse, validation and performance.

## Boundaries

Compiler optimisation belongs to [Raster](https://github.com/replikativ/raster), the shared PPL–compiler adapter to foerster-raster, and sampling to [Foerster](https://github.com/replikativ/foerster).
Fix problems where they belong rather than working around them here.
Generated source and data layouts are implementation details, not statistical assumptions.

Within the library:

- `model` validates original-unit data and fixed scaling, and builds the density, mean response and pointwise likelihood.
- `inference` runs NUTS and returns portable results. Preserve chain identity and coordinate order, and never report an unavailable sampler statistic as zero.
- `decision` compares plans with history and carryover, and allocates a constant weekly budget. Use exact derivatives; there is no finite-difference fallback.
- `calibration` handles independent experiments. Never count training evidence again as an experiment.
- `io` persists posteriors as versioned EDN or tagged JSON.

## Checks

Run `clojure -M:jvm:test` and `clojure -M:format` before submitting a change.
Tests compare against independent references: finite convolutions and densities, runtime and AOT gradients, changed data, unit invariance and allocation grids.
Compiling is not a test.

CircleCI runs formatting, the regression suite (`bin/run-unittests`) and the study's density check on every push. Every push to `main` that passes deploys `0.1.<commit count>` to Clojars and creates the matching GitHub release.

For changes that can affect numbers, also run the study's density check locally (see [study/README.md](study/README.md)).
Run full fits in bounded, disposable processes, one heavy JVM at a time, and do not edit source while a study is running: the driver records source hashes and stops if they change.

## Claims

- Do not change priors, transforms, scaling or adstock semantics to improve a benchmark.
- Report convergence with the full gates: R-hat below 1.01, bulk and tail ESS of at least 100, and no divergences.
- Distinguish conditional means from predictive noise, and model uncertainty from decision regret.
- A narrower interval or a better fit does not by itself mean a better decision.
- New result claims need a receipt from a fresh run, linked from `docs/validation.md`.
- The Kettle & Pine data are synthetic and drawn from the fitted model family. Results on them do not transfer to real data.

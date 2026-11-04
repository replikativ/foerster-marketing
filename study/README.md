# Kettle & Pine: the comparison study

This directory fits the same media mix model with [PyMC-Marketing](https://www.pymc-marketing.io/en/stable/) and with Foerster Marketing, checks that both agree, and scores their recommendations against the hidden truth.
It also runs the warehouse extension, which turns the marketing posterior into a profit decision under a fulfilment backlog.
The results are summarised in [docs/validation.md](../docs/validation.md); the business story is told in the blog post [Don't buy demand you can't ship](https://simm.is/blog/dont-buy-demand-you-cant-ship).

## The data

Kettle & Pine is a fictional online homeware shop with 156 weeks of revenue and spending on search, social and video.
Demand, promotions and an annual seasonal basis enter as observed controls.
The model learns from the first 130 weeks; the remaining 26 are a chronological holdout.
The decision is how to spend EUR 5,000 a week over the next four weeks, within per-channel bounds.

Three fixtures in [`examples/kettle_and_pine/data/`](../examples/kettle_and_pine/data/) change what the data can reveal:

| Case | Training data |
| --- | --- |
| `staggered` | Social and video flights take turns. |
| `co-scheduled` | Social and video run together (spend correlation about 0.95). |
| `calibrated` | The co-scheduled history plus one independent video lift experiment: five weeks before, four treatment weeks and five follow-up weeks. |

The generator (`common.py`) belongs to the fitted model family, so this is a well-specified validation study.
It is not real retailer data, and it does not test misspecification.

## The model

Both implementations use the same model and priors ([docs/model.md](../docs/model.md) writes it out in full):

- finite, normalised six-week geometric adstock;
- Michaelis–Menten saturation;
- additive intercept and controls, Gaussian observation noise;
- fixed scaling of EUR 1,000 for spend and EUR 10,000 for revenue.

PyMC-Marketing calls the saturation amplitude `alpha` and the half-saturation point `lam`; Foerster Marketing calls them `beta` and `kappa`.
Adstock `alpha` is the weekly retention in both.

The lift experiment has its own treatment and baseline spend paths, so its likelihood includes the carryover over time.
PyMC-Marketing's `add_lift_test_measurements` constrains a steady-state saturation contrast instead, which its own lift-test notebook notes.
`pymc_model.py` therefore adds the temporal likelihood with the model's own `mmm.adstock` and `mmm.saturation` components.
For the true parameters, the temporal lift is EUR 4,122 and the steady-state approximation EUR 3,550.

## What is checked

1. **Same density.** Before sampling, `pymc_audit.py` and `foerster_study.clj check` evaluate log density, gradient, original-unit means and pointwise likelihoods at three fixed points, including with changed observations. `compare_checks.py` requires agreement with each other and with an independent NumPy reference in `common.py`.
2. **Converged fits.** Four chains, 2,000 warm-up and 4,000 retained draws each. R-hat below 1.01, bulk and tail ESS of at least 100, and no divergences. Both engines start from the same prior-based point; the hidden truth never initialises a fit.
3. **Same posterior.** `compare_posteriors.py` requires every parameter mean to agree within five combined Monte Carlo standard errors and evaluates both engines' plans on both posteriors.
4. **Same decision.** Each engine optimises on a thinned, deterministic subset of draws (every ⌈draws per chain / 256⌉-th draw of each chain) and evaluates the plan on all draws. `analysis.py` and `foerster_evaluation.py` recheck each allocation with an independent optimiser and grid.
5. **Hidden truth.** `truth_regret.py` scores each recommended plan against the plan that is optimal under the true parameters. The truth is used only after both engines have chosen.
6. **Warehouse.** `warehouse_study.clj` infers the reduced fulfilment capacity with SMC over the Spindel warehouse model and simulates profit for each weekly budget. `warehouse_check.py` recomputes the capacity posterior by quadrature and every profit independently.

Budgets are per week in both libraries.
The PyMC allocator repeats one weekly allocation over the four decision weeks, so EUR 5,000 a week is a EUR 20,000 campaign.

## Run it

You need Java 25 and the Clojure CLI for the Foerster side, and Python 3.12 with [`requirements.txt`](requirements.txt) for the PyMC side and the checks.
From the repository root, the driver runs every stage in a fresh process and stops at the first failed check:

```sh
python study/run_study.py --output results/my-study
```

A full run takes about an hour and a half on a shared workstation, mostly the six fits.
It never overwrites an existing output directory and records the source hashes, commands, exit codes and resource use of every stage in `process-*.json`.
`--cases staggered` runs one case; the warehouse stages run when `calibrated` is included.

The stages can also be run by hand:

```sh
F=examples/kettle_and_pine/data/calibrated.json
O=results/my-study/calibrated
python study/pymc_audit.py $F $O/pymc-expanded-check.json
clojure -M:jvm:study -m foerster-study check $F $O
python study/compare_checks.py $O

python study/pymc_model.py $F $O/full/pymc --profile full
python study/pymc_decision.py $F $O/full/pymc
clojure -M:jvm:study -m foerster-study fit $F $O/full/foerster full
clojure -M:jvm:study -m foerster-study decision $F $O/full/foerster
clojure -M:jvm:study -m foerster-study replay $F $O/full/foerster
python study/foerster_evaluation.py $F $O/full/foerster
python study/compare_posteriors.py $O/full

clojure -J-Xmx3g -M:jvm:study -m warehouse-study \
  examples/kettle_and_pine/data/warehouse.json $F $O/full/foerster results/my-study/warehouse
python study/warehouse_check.py check \
  examples/kettle_and_pine/data/warehouse.json results/my-study/warehouse $F
```

`--profile pilot` and the `pilot` profile run 100 draws to check the mechanics; they are not converged fits.

## Files

| File | Role |
| --- | --- |
| `common.py` | Fixture generator, hidden truth, NumPy reference model, shared gates |
| `pymc_model.py`, `pymc_decision.py` | PyMC-Marketing fit and budget allocation |
| `pymc_audit.py` | PyMC density and gradient check at fixed points |
| `foerster_study.clj` | Foerster Marketing check, fit, decision and replay |
| `compare_checks.py`, `compare_posteriors.py` | Cross-engine agreement |
| `analysis.py`, `foerster_evaluation.py` | Independent allocation and holdout checks |
| `truth_regret.py` | Scoring against the hidden truth |
| `warehouse_study.clj`, `warehouse_check.py` | Warehouse decision and its independent check |
| `figures.py` | Study figures |
| `run_study.py` | Driver |

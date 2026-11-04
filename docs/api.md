# Public API and model contracts

[README](../README.md) · [Model](model.md) · [Kettle & Pine example](../examples/kettle_and_pine/core.clj) · [Decision workflow](decision-workflow.md) · [Validation](validation.md)

## Build and fit

The runtime namespaces are `org.replikativ.marketing.model`, `.inference`,
`.decision`, `.io`, and `.hierarchy`. Other namespaces are implementation detail
and marked as such. The numerical compiler (Raster) and probability engine
(Foerster) are pinned dependencies; callers need no compiler internals.

1. `model/specification` validates channels, supplied controls, response family,
   lag length, fixed scaling, priors, seasonality and observation likelihood.
2. `model/prepare` validates original-unit observations and constructs runtime
   numerical inputs. `:x` is period-by-channel, `:controls` period-by-control,
   and `:y` one outcome per period. All rows must align in time and order.
3. `model/build` creates the numerical model. Defaults are lagged density,
   automatic split/packed data layout and automatic scalar/array gradient layout.
4. `inference/fit` runs NUTS and returns detached posterior chains, named
   coordinates, diagnostics, sampler statistics and timings.
5. `inference/diagnostics-summary` keys R-hat, bulk/tail ESS, MCSE and quantiles
   by parameter name (for example `[:beta "search"]`) and reports divergences,
   depth-limit visits and `:passes?` against `inference/default-gates`
   (R-hat < 1.01, ESS >= 100, no divergences). Pass your own gates as a second
   argument.

The `:jvm` alias enables `jdk.incubator.vector`. The repository currently targets
Java 25. Fit options include `:draws`, `:tune`, `:chains`, `:seed`,
`:target-accept`, `:max-depth`, `:timeout-ms` (default two hours; `nil` for
none), `:init` and
`:gradient-backend` (`:aot` by default, or explicit runtime AD), `:metric`
(`:variance` by default, or `:fisher`), and boolean `:jitter` (default `false`).
Unknown options are rejected. Marketing explicitly disables Foerster's default
per-chain jitter to preserve the supplied unconstrained starting point. With
`:jitter true`, that point is the center of Foerster's per-chain jitter.
The sampler metadata records the supplied point, metric and jitter setting.
Results retain Foerster's Markov-chain inference contract, including that no
evidence estimate is available. Chain lengths and sampler-statistic alignment
are checked before detaching the result and survive EDN/JSON persistence.
`:draws` excludes warm-up; use at least two chains. `:init` is unconstrained,
whereas returned chain positions are constrained natural parameters. Prefer
`model/initial-position` to constructing a positional initialization manually.
Four requested chains do not imply four sampling threads; see [Limits](#limits).

A first AOT fit prepares an executable; subsequent compatible fits can reuse it.
The fit budget includes gradient preparation, but does not include the prior
`model/prepare` and `model/build` calls. Cancellation/worker termination has not
been qualified as a hard resource deadline.

## Units and response semantics

Supply fixed positive scaling: one `:scaling :channel` value per channel and a
`:scaling :target` outcome scale. These are supplied conventions; the package
does not learn preprocessing on the holdout. Predictors and likelihoods use
normalized coordinates internally. Predictions, decision budgets, experiment
estimates and scenario totals use original units. Prior parameters use internal
scaled coordinates, including control coefficients; user controls are not
implicitly standardized.

Geometric adstock uses lags `0` through `l-max-1`, normalized by the sum of those
geometric weights. Spend before the first supplied row is zero. Supply actual
prehistory when that assumption would be wrong.

| Family | Response to normalized effective spend `z` | Shape meaning |
|---|---|---|
| `:logistic` | `beta * tanh(lambda*z/2)` | `lambda` controls curvature |
| `:michaelis-menten` | `beta*z/(kappa+z)` | `kappa` is effective spend at half ceiling |
| `:none` | `beta*z` | No shape parameter |

`beta` is positive. Intercept and controls are additive. Priors must match the
support of each latent. Gaussian noise is the default; a Student-t likelihood
uses `{:distribution :student-t :nu 4.0}` with fixed finite `nu > 2`.
Fourier seasonality appends sine/cosine controls; explicit
`:time-index` values are required when configured and must preserve phase across
training and prediction. Use the parameter names returned with a model/posterior
rather than fixed numeric offsets, which differ by response family and controls.

## Prediction and paired scenarios

`model/mean-response` takes a built model and one constrained parameter vector
and returns original-unit conditional means. It does not draw observation noise.
`model/pointwise-log-likelihood` returns original-unit observation log densities,
excluding priors and experiment evidence. For a new history or prediction path,
prepare it with the same canonical specification, then combine its data/inputs
with the existing built model. Keep structural layouts compatible; rebuild when
structure changes.

```clojure
(decision/compare-plans built restored
  {:x [[1100.0 900.0] [1100.0 900.0]] :controls future-controls}
  {:x [[900.0 1100.0] [900.0 1100.0]] :controls future-controls})
```

Both plans need the same nonempty horizon and identical controls. The API keeps
up to `l-max-1` rows of training history, appends zero-spend carryover periods,
and compares the plans using paired posterior draws. Optional time indices must
immediately follow training history. The returned `:active`, `:carryover` and
`:total` summaries describe conditional-mean differences, not predictive noise
or uncertainty about the future controls. In a seasonal model future phase is
continued from the training time index. An optional fifth argument
`{:max-draws n}` evaluates on a thinned posterior.

`decision/plan-responses` returns the per-draw weekly conditional-mean response
paths of one plan, over its horizon plus `l-max-1` carryover weeks, in original
units. Use it when a plan's demand feeds a larger simulation, as in the
[warehouse extension](warehouse.md).

## Allocation

`decision/allocate` maximizes posterior expected response for a constant weekly
channel allocation. Provide positive `:budget`, positive integer `:horizon`,
future controls, and optional feasible `:lower`/`:upper` bounds per channel.
The budget is weekly, not a total over the horizon. The response objective
includes the active horizon and `l-max-1` zero-spend carryover periods. There is
no arbitrary time-varying budget optimizer, profit/cost utility or risk objective.

Inspect `:converged?`, `:projected-gradient-residual`, `:iterations`, feasibility
and finite results. Optimization uses budget fractions and response divided by
`:scaling :target`; the residual is the maximum fraction-coordinate change after
a projected unit step of that normalized objective gradient.
`:objective-response-scale` records the target normalization. Returned
`:expected-response` and `:gradient` retain original outcome and spend units.
The residual therefore depends on the declared target normalization; compare it
only under the same convention. An iteration-limit exit is not success.
Derivatives come from Raster AD; finite differences in the tests check them
independently. Allocation is iterative, so large posteriors are usually thinned:
`:max-draws n` (or `decision/thin`) keeps at most `n` draws, evenly spaced within
each chain. This API currently supports flat models and explicitly rejects geo
models.

## Persistence and reconstruction

```clojure
(posterior-io/write! "fit.edn" posterior)
(posterior-io/write! "fit.json" posterior :json)
(posterior-io/read! "fit.edn")
(posterior-io/read! "fit.json" :json)
```

Files preserve chains, canonical specification, names, diagnostics, statistics
and timings. Tagged JSON preserves keyword identities and nonfinite diagnostics;
it is not a conventional plain JSON schema. Loading validates the format,
coordinate layout, finiteness and parameter supports.

A posterior does not contain the original training observations or executable
kernels. Store training data/history, time indices and provenance separately,
then rebuild in the consumer process. The [study](../study/README.md) stores the
fixture, the posterior and source fingerprints and checks them on a fresh replay.

## External experiment evidence

Static `:lift` observations require `:type :effective-spend-contrast` and describe
a channel's effective-spend contrast with original-unit `:x`, `:delta-x`, `:delta-y` and positive measurement `:sigma`.
Time-path `:experiments` specify baseline/treatment media schedules, shared
prehistory, a weighted observation window, signed `:estimate`, uncertainty, and
`:independent-of-training? true`. The first `l-max-1` rows provide common history
and have zero measurement weights. Experiment estimates measure the weighted
sum of the modeled contrast, not automatically an average or a per-week effect.

A top-level `:experiment-covariance` can supply a fixed symmetric positive
definite joint measurement covariance. Otherwise each experiment supplies its
own positive `:sigma`. The uncertainty is measurement error on the estimator,
not the training residual scale. Transportability and independence are supplied
assumptions. Do not count training outcomes again as independent calibration.
The calibrated Kettle & Pine case ([data](../examples/kettle_and_pine/data/calibrated.json))
shows a time-path experiment with history, treatment and follow-up weeks.

## Geo models

`hierarchy/specification`, `hierarchy/prepare` and `hierarchy/build` expose a
focused noncentered geo model. Channel amplitudes are partially pooled; adstock,
shape and controls are shared; intercept/noise vary by group. Histories are
isolated by group. This is not arbitrary multidimensional modelling, and the
decision API does not accept geo models yet.

## Limits

- **Threads.** Foerster samples with a default two-thread executor, regardless of
  the number of chains; `inference/fit` does not expose an executor option.
- **Layouts.** `model/build` chooses split or packed data layouts and scalar or
  array gradients automatically. These change representation, not the
  statistical target. Scalar AD supports at most 18 arguments.
- **Compilation.** The first fit of a model structure compiles its kernels, which
  takes minutes. Compiled definitions are cached by structure, not data, with at
  most 32 entries; recycle long-lived workers at capacity.
- **Cancellation.** `:timeout-ms` stops sampling, but is not a hard resource
  deadline. Run large fits in disposable processes.

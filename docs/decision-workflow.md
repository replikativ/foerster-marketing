# From campaign history to a spending decision

[README](../README.md) · [Model](model.md) · [API](api.md) · [Study](../study/README.md) · [Warehouse](warehouse.md) · [Validation](validation.md)

A marketing mix model supports a specific planning decision. In this package,
that decision is a constant weekly allocation across channels, under a fixed
budget and channel bounds. It returns expected response and uncertainty about
paired plans. The decision owner still needs to choose an acceptable policy and
measure what happens after adoption.

## Define the decision before fitting

Start with the outcome, horizon, current plan and feasible alternatives. Specify
whether the outcome is revenue, orders or another measured quantity. The current
allocator maximises posterior expected response; it does not turn revenue into
profit, price risk, or enforce stock and fulfilment constraints by itself; the
[warehouse extension](warehouse.md) shows how to add them with a simulation.

In the Kettle & Pine example, a retailer has EUR 20,000 for four weeks.
A constant EUR 5,000 weekly plan must respect bounds for search, social and video.
The analysis includes five later zero-spend weeks because advertising continues
to affect demand after the campaign. A fixed weekly allocation is a narrower
problem than optimising a different schedule every week.

## Decide what the observations can support

Supply aligned outcome, spending and control rows. Controls should represent the
business mechanisms that make the comparison defensible, not merely improve fit.
Consider why spend changed: managers may increase it when demand is already high,
channels may always move together, and promotions may coincide with campaigns.
These patterns can prevent observational data from separating channel effects.

The package validates dimensions, values and support-compatible priors. It does
not discover confounders or establish a causal graph. A good holdout prediction
can coexist with poor channel identification. Independently measured lift can
help only when its definition, uncertainty and population support the proposed
use. Training evidence and experiment evidence must not be counted twice.

## Fit the response and inspect it

The model combines a baseline, supplied controls, optional seasonality, finite
geometric carryover and a selected response family. Michaelis–Menten response
approaches a channel-specific ceiling as effective spending grows. Explicit
scaling fixes the units used by priors and latent coordinates; original-unit
observations and predictions remain visible at the API boundary.

Use `model/specification`, `model/prepare`, `model/build` and `inference/fit` as
shown in the [API guide](api.md). A posterior is a collection of chains, not proof
that sampling succeeded. Inspect finite diagnostics, chain mixing, effective
sample sizes, divergences and depth-limit visits before using decisions. Our full
example gates require R-hat below 1.01, bulk and tail ESS at least 100 and zero
divergences; satisfying those gates does not establish model validity.

Reserve later observations for evaluation, with controls and spending appropriate
to the question. The study's holdout conditions on observed held-out inputs. Its
forward decision uses declared future controls, rather than secretly knowing
future demand. Separate model checking from selecting the spending plan.

## Compare the same plans under uncertain response

`decision/compare-plans` evaluates a baseline and candidate together for each
posterior draw. It retains relevant training history and includes subsequent
carryover. Pairing isolates the modelled difference between the plans.

`decision/allocate` chooses the feasible constant weekly allocation with the
largest posterior expected response. Inspect convergence, residuals and bounds;
a stopped optimizer is not necessarily a successful optimizer. Model-driven
selection can still be optimistic, even with excellent sampler diagnostics.

The interval for a paired conditional-mean gain describes uncertainty about the
response curve. Future observation noise adds another source of variation. A
positive mean gain is not a guarantee of improvement, and a narrower interval
alone does not prove a better decision. A real rollout needs a decision rule and
prospective evidence appropriate to the business.

## Retain an executable record

Save the posterior using the versioned EDN or tagged JSON format. Keep the input
revision, specification, units, priors, experiment definitions, sampler settings,
diagnostics and selected plans together. Loading a posterior in a fresh process
and reproducing its decision helps catch dependence on hidden runtime state.
It does not establish that an old result is valid after source or data changes.

The [study](../study/README.md) records provenance and refuses incompatible
replay. The [validation guide](validation.md) separates numerical checks, sampler
checks, independent optimiser checks and what the synthetic study cannot show.

## Choose the implementation boundary

For a Python analytics workflow, [PyMC-Marketing](https://www.pymc-marketing.io/en/stable/)
provides a broader established foundation. This package covers a focused subset
in Clojure and compares matched models with PyMC; it does not claim framework
parity. The [paired study](../study/README.md) documents the shared fixtures,
priors, scaling and numerical comparisons.

Its place in our simulation stack is useful when response estimates need to
meet operational state. The [warehouse extension](warehouse.md) evaluates the
demand of each campaign budget against an existing order backlog and an inferred,
uncertain fulfilment capacity. It needs a model of how orders arrive and ship,
the current warehouse state and an economic objective, none of which this MMM
library supplies by itself.

The underlying [Foerster](https://github.com/replikativ/foerster) runtime runs
particle inference in forks of managed application state, so the warehouse
simulation executes the application's own code without changing the live state.
A [running-queue demonstration](https://github.com/replikativ/foerster/blob/v0.1.137/notebooks/foerster/queue.clj)
shows the same mechanism. Foerster also supports SCMC, which updates a posterior
as observations arrive instead of refitting; this package currently fits the
marketing model in batch with NUTS.

## Reproduce the example that answers your question

- Run [`examples/kettle_and_pine/core.clj`](../examples/kettle_and_pine/core.clj)
  for the shortest path from data to a recommended plan.
- Use the [study](../study/README.md) for the three-case comparison with
  PyMC-Marketing and the hidden-truth scores. The fixtures are synthetic.
- Use the [API guide](api.md) to construct a new workflow. Run substantial fits
  in bounded processes and budget for the compilation and memory costs in
  [performance](performance.md). Short pilot runs are not converged fits.

# Foerster Marketing

Bayesian marketing mix modelling on the JVM.
Fit how sales respond to advertising, with carryover and diminishing returns, calibrate the model with lift experiments, and allocate a weekly budget across channels with the uncertainty in view.

Foerster Marketing is a focused Clojure library built on [Foerster](https://github.com/replikativ/foerster), our probabilistic programming system, and [Raster](https://github.com/replikativ/raster), which compiles the model's density and gradients.
Because it runs inside ordinary JVM applications, the marketing posterior can feed simulations of the running business, such as a warehouse with an order backlog.

**Status: experimental.** The library is validated against [PyMC-Marketing](https://www.pymc-marketing.io/en/stable/) on synthetic data (below). It has not been validated on real retailer data yet.

## A worked example: Kettle & Pine

Kettle & Pine is a fictional online homeware shop with 130 weeks of spending on search, social and video, weekly revenue, demand and promotion controls, and one independent lift experiment on video.
It wants to know how to split EUR 5,000 a week over the next four weeks.
The full program is [`examples/kettle_and_pine/core.clj`](examples/kettle_and_pine/core.clj); the story is told in the blog post [Don't buy demand you can't ship](https://simm.is/blog/dont-buy-demand-you-cant-ship).

```clojure
(require '[org.replikativ.marketing.model :as model]
         '[org.replikativ.marketing.inference :as inference]
         '[org.replikativ.marketing.decision :as decision]
         '[kettle-and-pine.core :as kp])

;; Specification, training data and the decision to make, from one JSON file.
;; The specification declares three channels, four controls, six weeks of
;; geometric adstock, Michaelis–Menten saturation, fixed scaling and the priors.
(def fixture (kp/read-fixture "calibrated"))

(def built (model/build (model/prepare (:spec fixture) (:training fixture))))

;; NUTS: four chains of 1,000 draws. The first fit compiles the model's kernels.
(def posterior (inference/fit built {:draws 1000 :tune 1000 :chains 4 :seed 731
                                     :target-accept 0.95 :max-depth 10}))

(:passes? (inference/diagnostics-summary posterior))
;; => true  (R-hat < 1.01, ESS >= 100, no divergences)

(let [{:keys [weekly-budget horizon controls lower upper baseline]} (:decision fixture)
      plan (decision/allocate built posterior
                              {:budget weekly-budget :horizon horizon :controls controls
                               :lower lower :upper upper :max-draws 1024})
      weekly->plan (fn [w] {:x (vec (repeat horizon w)) :controls controls})]
  {:recommended (:allocation plan)
   :gain (:total (decision/compare-plans built posterior
                                         (weekly->plan baseline)
                                         (weekly->plan (:allocation plan))))})
;; => {:recommended [1341 2044 1614]
;;     :gain {:mean 1590 :p05 -988 :p50 1656 :p95 3915}}   ; EUR, campaign + carryover
```

The current plan spends EUR 2,000, 1,800 and 1,200 a week on search, social and video.
The model moves about EUR 650 a week out of search into social and video, and expects about EUR 1,600 more revenue over the campaign and the five weeks of carryover after it.
Most of that gain arrives after the campaign, because video keeps working for weeks, so `compare-plans` reports the campaign and carryover weeks separately as `:active` and `:carryover`.
The 90% interval still includes a loss, which is the part a single attribution figure would hide.

From the repository root, the same steps run as a program:

```sh
clojure -M:jvm:examples -m kettle-and-pine.core fit calibrated out/calibrated   # about 8 minutes
clojure -M:jvm:examples -m kettle-and-pine.core plan calibrated out/calibrated
```

### Extension: the warehouse

Now suppose the warehouse is behind: a staffing shortage has cut capacity and nearly 700 orders are waiting.
`kettle-and-pine.warehouse` infers the reduced capacity from recent shipment reports with SMC, running the application's own fulfilment code in forks of the live warehouse state, and simulates the profit of each weekly budget.
With the backlog, the best budget falls from EUR 3,750–5,000 to EUR 2,500 a week.

```sh
clojure -J-Xmx3g -M:jvm:examples -m kettle-and-pine.core warehouse out/calibrated out/warehouse
```

[docs/warehouse.md](docs/warehouse.md) describes the model, the checks and the result.

## Validated against PyMC-Marketing

The [study](study/README.md) fits Kettle & Pine with PyMC-Marketing 1.3.0 and Foerster Marketing, with identical data, priors and scaling, in three versions of its history.
Both engines must agree on the density and gradients at fixed points, pass the convergence gates, agree on every posterior mean within five combined Monte Carlo standard errors, and recommend the same plan.
Because the data are synthetic, each recommendation is also scored against the plan that is optimal under the hidden truth.

| Case | Foerster fit (max R-hat, min ESS, divergences) | Recommended plan (EUR/week) | Expected gain, 90% interval (EUR) | Gap to PyMC's plan | Regret vs. hidden truth (EUR) |
| --- | --- | --- | --- | --- | --- |
| Staggered flights | 1.001, 8,013, 0 | 1,386 / 2,016 / 1,598 | 1,307 (−870 to 3,231) | ≤ 17 per channel | 294 of 2,861 |
| Overlapping flights | 1.000, 7,532, 0 | 1,382 / 2,406 / 1,212 | 1,678 (−1,186 to 4,265) | ≤ 12 per channel | 1,433 of 3,114 |
| Overlapping + lift test | 1.001, 7,052, 0 | 1,372 / 2,013 / 1,615 | 1,559 (−904 to 3,831) | ≤ 31 per channel | 327 of 3,114 |

When social and video always run together, the data cannot separate them, and both engines' plans realise only about half of the possible gain.
One independent lift experiment recovers most of it.
[docs/validation.md](docs/validation.md) lists every check and links the receipts.

PyMC-Marketing currently fits this model about three times faster than Foerster Marketing (4 instead of 11–15 minutes per fit); [docs/performance.md](docs/performance.md) has the measurements.

## Install

Foerster Marketing needs Java 25 and the Clojure CLI.
Add it from [Clojars](https://clojars.org/org.replikativ/foerster-marketing):

```clojure
{:deps {org.replikativ/foerster-marketing {:mvn/version "0.1.1"}}}
```

The JVM needs `--add-modules=jdk.incubator.vector` and `--enable-native-access=ALL-UNNAMED`; the `:jvm` alias in [`deps.edn`](deps.edn) sets them.

## What the library covers

- Finite, normalised geometric adstock; Michaelis–Menten, logistic or linear response; priors you choose.
- An additive baseline with supplied controls, Fourier seasonality, Gaussian or Student-t noise.
- Lift-test calibration with static effective-spend contrasts or full treatment and control time paths, with optional joint measurement covariance.
- A focused geo hierarchy with partially pooled channel amplitudes.
- Paired plan comparisons over the campaign and its carryover, and budget allocation for a constant weekly split under channel bounds.
- Versioned EDN and JSON posteriors with named coordinates.

It does not cover the breadth of PyMC-Marketing: there is no customer lifetime value, customer choice, time-varying media effects or automatic causal identification.
Future controls, causal assumptions and whether an experiment transfers to the business come from the analyst.
If your team works in Python, PyMC-Marketing is the better starting point, and we contribute to it.

## Documentation

- [Model](docs/model.md): the media mix model as a graph and equations, with the corresponding PyMC-Marketing names.
- [API](docs/api.md): building, fitting, diagnostics, plan comparison, allocation and persistence.
- [Decision workflow](docs/decision-workflow.md): from a business question to a defensible spending decision.
- [Study](study/README.md), [validation](docs/validation.md) and [performance](docs/performance.md): the comparison with PyMC-Marketing.
- [Warehouse](docs/warehouse.md): the fulfilment extension.

## Develop

```sh
clojure -M:jvm:test     # regression suite
clojure -M:format       # formatting check (clojure -M:ffix fixes)
clojure -T:build jar
```

Contributions are welcome; [AGENTS.md](AGENTS.md) describes the repository's conventions.
Licensed under [Apache-2.0](LICENSE).

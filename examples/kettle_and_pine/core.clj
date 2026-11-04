(ns kettle-and-pine.core
  "Kettle & Pine, a worked example of Foerster Marketing.

  Kettle & Pine, a synthetic online retailer, spends on search, social and
  video. It has 130 weeks of spend and revenue, weekly demand and promotion
  controls, and one independent lift experiment on video. The question: how
  should it split EUR 5,000 per week over the next four weeks?

  The program follows the library's workflow:

    specification -> prepare -> build -> inference/fit -> diagnostics
                  -> decision/allocate -> decision/compare-plans

  Run it from the repository root:

    clojure -M:jvm:examples -m kettle-and-pine.core fit calibrated out/calibrated
    clojure -M:jvm:examples -m kettle-and-pine.core plan calibrated out/calibrated
    clojure -M:jvm:examples -m kettle-and-pine.core warehouse out/calibrated out/warehouse"
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [kettle-and-pine.warehouse :as warehouse]
            [org.replikativ.marketing.decision :as decision]
            [org.replikativ.marketing.inference :as inference]
            [org.replikativ.marketing.io :as posterior-io]
            [org.replikativ.marketing.model :as model]))

;; ---------------------------------------------------------------------------
;; Data

(defn read-fixture
  "Read a Kettle & Pine case: a name (staggered, co-scheduled or calibrated)
  or the path of a JSON file in the same format. The file holds the model
  `:spec`, the `:training` data for `model/prepare` and the `:decision` to
  make (weekly budget, horizon, future controls, channel bounds and the
  current plan as `:baseline`). `model/specification` validates the spec and
  turns JSON strings such as \"michaelis-menten\" into keywords."
  [case-or-path]
  (let [file (if (.exists (io/file case-or-path))
               (io/file case-or-path)
               (io/file "examples/kettle_and_pine/data" (str case-or-path ".json")))
        fixture (json/read-str (slurp file) :key-fn keyword)]
    (update fixture :spec model/specification)))

(defn build
  "Prepare the training data for the specification and compile the model.
  Compiled kernels are cached, so building the same specification again is cheap."
  [fixture]
  (model/build (model/prepare (:spec fixture) (:training fixture))))

;; ---------------------------------------------------------------------------
;; Fitting

(def sampler
  "NUTS settings: four chains, 1,000 retained draws each after 1,000 tuning
  steps. The model is compiled on the first call, which takes a few minutes."
  {:draws 1000 :tune 1000 :chains 4 :seed 731 :target-accept 0.95 :max-depth 10})

(defn print-diagnostics
  "Print R-hat and effective sample sizes per parameter and the overall gate."
  [summary]
  (println "parameter                      R-hat   ESS bulk  ESS tail")
  (doseq [[name {:keys [rhat ess-bulk ess-tail]}] (:parameters summary)]
    (println (format "%-28s %7.4f %10.0f %9.0f" (pr-str name) rhat ess-bulk ess-tail)))
  (println "divergences:" (:divergences summary)
           "| passes R-hat < 1.01, ESS >= 100, no divergences:" (:passes? summary)))

(defn fit!
  "Fit a case and save its posterior as EDN in `out-dir`."
  [case-name out-dir]
  (let [built (build (read-fixture case-name))
        posterior (inference/fit built sampler)
        file (io/file out-dir "posterior.edn")]
    (print-diagnostics (inference/diagnostics-summary posterior))
    (io/make-parents file)
    (posterior-io/write! file posterior)
    (println "saved" (str file))))

(defn read-posterior
  "Read the posterior saved by `fit!`; `io/read!` validates it."
  [posterior-dir]
  (posterior-io/read! (io/file posterior-dir "posterior.edn")))

;; ---------------------------------------------------------------------------
;; Deciding

(defn recommend
  "Allocate the weekly budget and compare the result with the current plan.

  `decision/allocate` maximizes the posterior expected revenue of a constant
  weekly split within the channel bounds. Allocation is iterative, so it uses
  1,024 evenly thinned draws. `decision/compare-plans` then evaluates the
  current and recommended plans on every posterior draw. Both plans share
  the training history, and the comparison includes the weeks of adstock
  carryover after the campaign."
  [built posterior {:keys [weekly-budget horizon controls lower upper baseline]}]
  (let [allocation (decision/allocate built posterior
                                      {:budget weekly-budget :horizon horizon :controls controls
                                       :lower lower :upper upper :max-draws 1024})
        plan (fn [weekly] {:x (vec (repeat horizon weekly)) :controls controls})]
    (when-not (:converged? allocation)
      (throw (ex-info "Allocation did not converge" allocation)))
    {:allocation allocation
     :comparison (decision/compare-plans built posterior (plan baseline) (plan (:allocation allocation)))}))

(defn plan!
  "Recommend a weekly split for a case from its saved posterior and print it."
  [case-name posterior-dir]
  (let [fixture (read-fixture case-name)
        posterior (read-posterior posterior-dir)
        summary (inference/diagnostics-summary posterior)
        _ (print-diagnostics summary)
        _ (when-not (:passes? summary)
            (println "WARNING: the posterior fails the convergence gates; treat the plan with caution."))
        {:keys [allocation comparison]} (recommend (build fixture) posterior (:decision fixture))]
    (println "channels:            " (get-in fixture [:spec :channels]))
    (println "current weekly plan: " (get-in fixture [:decision :baseline]))
    (println "recommended plan:    " (mapv #(Math/round (double %)) (:allocation allocation)))
    (println "expected revenue over the campaign and carryover:"
             (Math/round (double (:expected-response allocation))))
    (println "revenue gained by switching (campaign + carryover weeks):")
    (pprint (select-keys comparison [:active :carryover :total]))
    {:allocation allocation :comparison (dissoc comparison :draws)}))

;; ---------------------------------------------------------------------------
;; Fulfilment: the same posterior in a larger decision (docs/warehouse.md)

(defn warehouse!
  "Choose a campaign budget against the warehouse backlog, using the calibrated
  case's posterior in `posterior-dir`; writes report.json and runs.json."
  [posterior-dir out-dir]
  (let [fixture (read-fixture "calibrated")]
    (warehouse/run! (warehouse/read-scenario "examples/kettle_and_pine/data/warehouse.json")
                    {:built (build fixture) :posterior (read-posterior posterior-dir)
                     :campaign (:decision fixture) :posterior-directory posterior-dir}
                    out-dir)))

(defn -main
  [& [command a b]]
  (try
    (case command
      "fit" (fit! a b)
      "plan" (plan! a b)
      "warehouse" (prn (warehouse! a b))
      (println "usage: fit <case> <out-dir> | plan <case> <posterior-dir> | warehouse <posterior-dir> <out-dir>"))
    (finally (shutdown-agents))))

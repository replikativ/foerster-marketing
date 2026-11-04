(ns kettle-and-pine.warehouse
  "How much should Kettle & Pine spend on the next campaign, given the
  warehouse as it stands? (docs/warehouse.md describes the scenario.)

  `ship-week!` is the retailer's fulfilment logic: ordinary Clojure over an
  atom of warehouse state. Inference and decision runs call the same function
  on Spindel runtime atoms, so each particle works on its own fork of the
  warehouse.

  1. Allocate each candidate weekly budget with the saved marketing posterior
     and record every draw's weekly revenue path (`decision/plan-responses`).
  2. Infer the reduced warehouse capacity from recent shipments with SMC, each
     particle replaying `ship-week!` in a fork of the warehouse as it stood
     when the disruption began.
  3. Simulate every budget forward from a fork of the live warehouse, pairing
     a capacity particle with a demand draw, and score profit."
  (:refer-clojure :exclude [run!])
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :refer [normal discrete]]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.marketing.decision :as decision]
            [org.replikativ.spindel.atom :as sa]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

;; ---------------------------------------------------------------------------
;; The application: fulfilment and profit

(defn step
  "One week: ship up to capacity, then a share of the orders still waiting cancel."
  [backlog arrivals capacity cancellation-rate]
  (let [available (+ (double backlog) (double arrivals))
        shipped (min available (max 0.0 (double capacity)))
        remaining (- available shipped)
        cancelled (* (double cancellation-rate) remaining)]
    {:shipped shipped :cancelled cancelled :backlog (- remaining cancelled)}))

(defn ship-week!
  "Advance the warehouse by one week and return the orders shipped."
  [warehouse {:keys [week arrivals]} capacity cancellation-rate]
  (let [{:keys [shipped] :as result} (step (:backlog @warehouse) arrivals capacity cancellation-rate)]
    (swap! warehouse (fn [state]
                       (-> state
                           (assoc :backlog (:backlog result))
                           (update :log conj (assoc result :week week :arrivals arrivals
                                                    :capacity capacity)))))
    shipped))

(defn campaign-profit
  "Contribution after delay costs and advertising, for the weeks in `log`."
  [log {:keys [average-order-value margin delay-cost-per-order-week]} weekly-budget campaign-weeks]
  (- (* margin average-order-value (reduce + (map :shipped log)))
     (* delay-cost-per-order-week (reduce + (map :backlog log)))
     (* campaign-weeks weekly-budget)))

;; ---------------------------------------------------------------------------
;; Helpers

(defn read-scenario
  "Read a warehouse scenario (examples/kettle_and_pine/data/warehouse.json)."
  [path]
  (json/read-str (slurp path) :key-fn keyword))

(defn- write-json! [path value]
  (io/make-parents path)
  (spit path (json/write-str value)))

(defn- summary [xs]
  (let [v (vec (sort xs)) n (count v)
        q (fn [p] (let [i (* p (dec n)) a (int i) b (min (inc a) (dec n))]
                    (+ (nth v a) (* (- i a) (- (nth v b) (nth v a))))))]
    {:mean (/ (reduce + v) n) :p05 (q 0.05) :p50 (q 0.5) :p95 (q 0.95)}))

;; ---------------------------------------------------------------------------
;; 1. Demand under each budget

(def decision-draws
  "Posterior draws used for allocation and demand paths: 256 per chain, the
  same deterministic selection as the published study."
  1024)

(defn- scaled-bounds [d budget]
  (let [s (/ budget (:weekly-budget d))]
    [(mapv #(* s %) (:lower d)) (mapv #(* s %) (:upper d))]))

(defn- allocation [built posterior d budget]
  (if (zero? budget)
    (vec (repeat (count (:baseline d)) 0.0))
    (let [[lower upper] (scaled-bounds d budget)
          a (decision/allocate built posterior {:budget budget :horizon (:horizon d) :controls (:controls d)
                                                :lower lower :upper upper :tolerance 1e-6 :max-iterations 500
                                                :max-draws decision-draws})]
      (when-not (:converged? a) (throw (ex-info "Allocation did not converge" {:budget budget})))
      (:allocation a))))

(defn- revenue-paths [built posterior d weekly]
  (:draws (decision/plan-responses built posterior
                                   {:x (vec (repeat (:horizon d) weekly)) :controls (:controls d)}
                                   {:max-draws decision-draws})))

;; ---------------------------------------------------------------------------
;; 2. Capacity from recent shipments

(defn capacity-model [warehouse weeks {:keys [prior-mean prior-sd weekly-sd report-sd]} cancel]
  (spin
   (let [reduced (sample (normal prior-mean prior-sd) :id :reduced-capacity)]
     (doseq [{:keys [week reported-shipped] :as w} weeks]
       (let [capacity (sample (normal reduced weekly-sd) :id [:capacity week])
             shipped (ship-week! warehouse w capacity cancel)]
         (observe (normal shipped report-sd) reported-shipped :id [:shipped week])))
     {:reduced-capacity reduced :warehouse @warehouse})))

(defn- weighted [measure]
  (let [values (mapv m/get-value (m/get-contexts measure))
        lw (m/get-log-weights measure)
        top (apply max lw)
        w (mapv #(Math/exp (- % top)) lw)
        total (reduce + w)
        w (mapv #(/ % total) w)
        xs (mapv :reduced-capacity values)
        mean (reduce + (map * w xs))
        sd (Math/sqrt (reduce + (map (fn [wi x] (* wi (Math/pow (- x mean) 2))) w xs)))
        ess (/ 1.0 (reduce + (map #(* % %) w)))
        unique (count (distinct xs))]
    {:values values :weights w :capacities xs :mean mean :sd sd :ess ess :unique unique
     :mcse (/ sd (Math/sqrt (min ess unique)))}))

(defn- infer-capacity [scenario seed particles]
  (let [world (sp/create-execution-context)
        start {:backlog (get-in scenario [:disruption-start :backlog]) :log []}
        cancel (get-in scenario [:economics :weekly-cancellation-rate])]
    (try
      (sp/with-context world
        (let [warehouse (sa/atom start)
              posterior @(infer/smc-infer (capacity-model warehouse (:disruption-weeks scenario)
                                                          (:capacity scenario) cancel)
                                          particles {:world-policy :fork :seed seed})
              result (weighted posterior)]
          (when-not (= start @warehouse) (throw (ex-info "Inference changed the caller's warehouse" {})))
          (when-not (every? #(= 3 (count (get-in % [:warehouse :log]))) (:values result))
            (throw (ex-info "Invalid particle history" {})))
          (assoc result :seed seed :caller-unchanged true)))
      (finally (ctx/stop-context! world)))))

;; ---------------------------------------------------------------------------
;; 3. Budgets against the live warehouse

(defn- evaluate-budgets!
  "Ordinary application code: reset the (forked) warehouse and run each plan."
  [warehouse start capacities paths draw budgets economics campaign-weeks]
  (let [cancel (:weekly-cancellation-rate economics)
        aov (:average-order-value economics)]
    (mapv (fn [k budget]
            (reset! warehouse start)
            (doseq [[t revenue] (map-indexed vector (nth (nth paths k) draw))]
              (ship-week! warehouse {:week t :arrivals (/ revenue aov)} (nth capacities t) cancel))
            (campaign-profit (:log @warehouse) economics budget campaign-weeks))
          (range) budgets)))

(defn decision-model [warehouse start {normal-capacity :normal
                                       :keys [capacity-values capacity-weights reduced-weeks weekly-sd
                                              paths budgets economics campaign-weeks]}]
  (spin
   (let [i (sample (discrete capacity-weights) :id :capacity-particle)
         reduced (nth capacity-values i)
         draw (sample (discrete (vec (repeat (count (first paths)) 1.0))) :id :demand-draw)
         capacities (loop [t 0 acc []]
                      (if (< t (count (first (first paths))))
                        (recur (inc t) (conj acc (if (< t reduced-weeks)
                                                   (sample (normal reduced weekly-sd) :id [:capacity t])
                                                   normal-capacity)))
                        acc))
         profits (evaluate-budgets! warehouse start capacities paths draw budgets economics campaign-weeks)]
     {:draw draw :reduced-capacity reduced :capacities capacities :profits profits})))

(defn- evaluate-state [state options runs seed]
  (let [world (sp/create-execution-context)
        start {:backlog (:start-backlog state) :log []}]
    (try
      (sp/with-context world
        (let [warehouse (sa/atom start)
              measure @(infer/importance-sampling
                        (decision-model warehouse start (assoc options :reduced-weeks (:reduced-weeks state)))
                        runs {:world-policy :fork :seed seed})
              values (mapv m/get-value (m/get-contexts measure))]
          (when-not (= start @warehouse) (throw (ex-info "Decision runs changed the live warehouse" {})))
          (when-not (apply = (m/get-log-weights measure)) (throw (ex-info "Unexpected unequal weights" {})))
          values))
      (finally (ctx/stop-context! world)))))

(defn- state-report [state values budgets reference-budget]
  (let [k-ref (.indexOf ^java.util.List budgets reference-budget)
        plans (mapv (fn [k budget]
                      (let [ps (mapv #(nth (:profits %) k) values)
                            diffs (mapv #(- (nth (:profits %) k) (nth (:profits %) k-ref)) values)]
                        {:budget budget :profit (summary ps) :versus-reference (summary diffs)
                         :probability-better (/ (count (filter pos? diffs)) (double (count diffs)))}))
                    (range) budgets)
        best (apply max-key #(get-in % [:profit :mean]) plans)]
    (assoc state :plans plans :best-budget (:budget best) :reference-budget reference-budget)))

(defn run!
  "Evaluate the scenario's candidate budgets against the warehouse and write
  `report.json` and `runs.json` (revenue paths and every simulated run) to
  `output-dir`. `built` and `posterior` are the fitted marketing model;
  `campaign` is the fixture's :decision block (reference budget, horizon,
  controls and channel bounds); `posterior-directory` is recorded in the report."
  [scenario {:keys [built posterior campaign posterior-directory]} output-dir]
  (when (.exists (io/file output-dir "report.json")) (throw (ex-info "Refusing to overwrite report" {})))
  (let [d campaign
        budgets (mapv double (:budgets scenario))
        allocations (mapv #(allocation built posterior d %) budgets)
        paths (mapv #(revenue-paths built posterior d %) allocations)
        capacity (mapv #(infer-capacity scenario % 4000) [17 31 73])
        cap (first capacity)
        c (:capacity scenario)
        options {:capacity-values (:capacities cap) :capacity-weights (:weights cap)
                 :normal (:normal c) :weekly-sd (:weekly-sd c) :paths paths :budgets budgets
                 :economics (:economics scenario) :campaign-weeks (:horizon scenario)}
        states [{:name "backlog" :start-backlog (get-in scenario [:current :backlog])
                 :reduced-weeks (:remaining-reduced-weeks c)
                 :reduced-mean-truth (:hidden-true-reduced-capacity scenario)}
                {:name "clear" :start-backlog 0.0 :reduced-weeks 0
                 :reduced-mean-truth (:normal c)}]
        evaluated (mapv #(evaluate-state % options 4000 101) states)
        report {:scenario-case (:fixture-case scenario)
                :posterior-directory (str posterior-directory)
                :allocations (mapv (fn [b a] {:budget b :weekly-allocation a}) budgets allocations)
                :demand-draws (count (first paths))
                :capacity {:seeds (mapv #(select-keys % [:seed :mean :sd :ess :unique :mcse :caller-unchanged]) capacity)}
                :states (mapv (fn [s vs]
                                (-> (state-report s vs budgets (double (:weekly-budget d)))
                                    (update :plans (fn [ps] (mapv #(assoc %2 :weekly-allocation %1) allocations ps)))))
                              states evaluated)}]
    (write-json! (str output-dir "/runs.json")
                 {:demand-paths paths
                  :states (mapv (fn [s vs] (assoc s :budgets budgets
                                                  :runs (mapv #(select-keys % [:draw :capacities :profits]) vs)))
                                states evaluated)})
    (write-json! (str output-dir "/report.json") report)
    (mapv #(select-keys % [:name :best-budget]) (:states report))))

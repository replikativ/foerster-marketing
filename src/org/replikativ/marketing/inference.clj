(ns org.replikativ.marketing.inference
  "NUTS inference returning portable chains, and convergence diagnostics.

  `fit` samples a built model with Foerster's NUTS and returns a plain-data
  posterior (see `io` for persistence); `diagnostics-summary` checks it."
  (:require [org.replikativ.marketing.config :as config]
            [org.replikativ.marketing.model :as model]
            [org.replikativ.marketing.layout :as layout]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as infer :refer [sample]]
            [org.replikativ.foerster.diagnostics :as diagnostics]
            [org.replikativ.foerster.measure :as measure]
            [org.replikativ.spindel.core :as sp :refer [spin]])
  (:import [java.lang.management ManagementFactory]
           [com.sun.management OperatingSystemMXBean]))

(defn- cpu-nanos []
  (let [bean (ManagementFactory/getOperatingSystemMXBean)]
    (when (instance? OperatingSystemMXBean bean)
      (.getProcessCpuTime ^OperatingSystemMXBean bean))))

(defn- cpu-seconds [started ended]
  (when (and started ended (<= 0 started ended))
    (/ (- ended started) 1e9)))

(defn parameter-names
  "Parameter names of a built model, in posterior draw order, e.g. [:alpha \"search\"]."
  [model]
  (layout/names (:parameter-layout model)))

(defn- detached-samples [posterior chains draws dimension]
  (let [particles (measure/get-particles posterior)
        expected (vec (repeat chains draws))
        contract (:inference-contract posterior)]
    (when-not (and (= :markov-chain (:semantics contract))
                   (= :unavailable (get-in contract [:normalizer :status])))
      (throw (ex-info "MMM inference requires a Markov-chain result without an evidence estimate"
                      {:inference-contract contract})))
    (when-not (and (= expected (:chain-lengths posterior))
                   (= (* chains draws) (count particles)))
      (throw (ex-info "Retained draws do not match the requested chains"
                      {:expected-chain-lengths expected :actual-chain-lengths (:chain-lengths posterior)
                       :particles (count particles)})))
    (let [flat (mapv (comp measure/get-value first) particles)
          stats (mapv (comp :sampler-stats first) particles)]
      (when-not (every? #(and (sequential? %) (= dimension (count %))
                              (every? (fn [x] (and (number? x) (Double/isFinite (double x)))) %)) flat)
        (throw (ex-info "Retained parameter values have invalid dimensions or nonfinite values" {})))
      (when-not (every? #(and (map? %) (boolean? (:diverging %)) (= 1 (:blocks %))) stats)
        (throw (ex-info "MMM inference requires sampler statistics for its one complete density block" {})))
      {:flat flat :stats stats :contract contract})))

(defn fit
  "Fit a complete density block and detach its result from execution worlds.
  :draws excludes :tune. :timeout-ms bounds the whole fit, including gradient
  compilation (default 2 hours; nil disables the bound); on timeout the fit's
  world is cancelled and an exception is thrown.
  :gradient-backend defaults to :aot; :runtime selects raster runtime AD.
  :metric is :variance (default) or :fisher. :jitter defaults to false to keep
  the supplied initial position; true enables Foerster's per-chain jitter.
  Sampler diagnostics remain aligned with their retained chain and draw."
  ([model] (fit model {}))
  ([model {:keys [draws tune chains seed target-accept max-depth timeout-ms init gradient-backend metric jitter]
           :as options :or {draws 1000 tune 1000 chains 4 seed 42 target-accept 0.9
                            max-depth 8 timeout-ms 7200000 gradient-backend :aot metric :variance jitter false}}]
   (when-let [unknown (seq (remove #{:draws :tune :chains :seed :target-accept :max-depth
                                     :timeout-ms :init :gradient-backend :metric :jitter} (keys options)))]
     (throw (ex-info "Unknown sampler options" {:options (vec unknown)})))
   (when-not (and (integer? draws) (pos? draws) (integer? tune) (pos? tune)
                  (integer? chains) (>= chains 2) (integer? seed) (< 0.0 target-accept 1.0)
                  (#{:variance :fisher} metric) (boolean? jitter)
                  (integer? max-depth) (pos? max-depth)
                  (or (nil? timeout-ms) (and (number? timeout-ms) (pos? timeout-ms))) (#{:aot :runtime} gradient-backend))
     (throw (ex-info "Invalid sampler configuration" {})))
   (let [root (sp/create-execution-context)
         init (or init (model/initial-position model))
         cpu-started (cpu-nanos)
         started (System/nanoTime)
         task (future
                (sp/with-context root
                  (let [b (model/inference-block model gradient-backend)
                        prepared (System/nanoTime)
                        cpu-prepared (cpu-nanos)
                        posterior @(infer/infer
                                    (spin (let [theta (sample (block/block-dist b (:inputs model))
                                                              :id :parameters :init init)]
                                            (block/constrain b theta)))
                                    {:method :nuts :iterations (+ draws tune) :burn tune
                                     :chains chains :seed seed :target-accept target-accept :max-depth max-depth
                                     :metric metric :jitter jitter})
                        sampled (System/nanoTime)
                        cpu-sampled (cpu-nanos)
                        _ (when-not (= seed (:seed posterior))
                            (throw (ex-info "This foerster build does not honor explicit Markov seeds" {})))
                        {:keys [flat stats contract]} (detached-samples posterior chains draws (:dimension model))
                        diagnostics (mapv (fn [j] (diagnostics/summary posterior #(nth % j)))
                                          (range (:dimension model)))]
                    {:format :foerster-marketing/posterior :version 1
                     :inference-contract contract
                     :spec (:spec model) :parameter-names (parameter-names model)
                     :chains (mapv vec (partition draws flat))
                     :diagnostics diagnostics
                     :sampler-statistics {:availability :available
                                          :divergences (count (filter :diverging stats))
                                          :depth-limit-visits (count (filter :reached-max-depth? stats))
                                          :by-chain (mapv vec (partition draws stats))}
                     :timing {:sampling-cpu-seconds (cpu-seconds cpu-prepared cpu-sampled)
                              :gradient-preparation-cpu-seconds (cpu-seconds cpu-started cpu-prepared)
                              :gradient-preparation-seconds (/ (- prepared started) 1e9)
                              :sampling-seconds (/ (- sampled prepared) 1e9)
                              :diagnostics-seconds (/ (- (System/nanoTime) sampled) 1e9)}
                     :sampler {:gradient-parameter-layout (when (= gradient-backend :aot) (:gradient-parameter-layout model :array))
                               :density-kernel (:density-kernel model :scan) :gradient-backend gradient-backend :method :nuts :seed seed :draws draws :tune tune :chains chains
                               :target-accept target-accept :max-depth max-depth :metric metric :jitter jitter
                               :initial-position (vec init)}
                     :elapsed-seconds (/ (- (System/nanoTime) started) 1e9)})))]
     (try
       (let [result (if timeout-ms (deref task timeout-ms ::timeout) @task)]
         (when (= ::timeout result)
           (future-cancel task)
           (throw (ex-info "MMM fit timed out" {:timeout-ms timeout-ms})))
         result)
       (finally
         (future-cancel task)
         (sp/stop-context! root))))))

(def default-gates
  "Convergence gates used by `diagnostics-summary`: R-hat strictly below
  1.01, bulk and tail ESS of at least 100 for every parameter, no divergences."
  {:max-rhat 1.01 :min-ess 100.0 :max-divergences 0})

(defn diagnostics-summary
  "Summarize convergence of a posterior from `fit` (or `io/read!`).

  Returns
  - :parameters, a map from parameter name (e.g. [:beta \"search\"]) to
    {:mean :sd :rhat :ess-bulk :ess-tail :mcse :quantiles}, in draw order;
  - :max-rhat, :min-ess-bulk and :min-ess-tail over all parameters (nil when
    any of them is not finite) and :finite?;
  - :divergences and :depth-limit-visits from the sampler statistics;
  - :passes? when every parameter and the divergence count meet `gates`
    (default `default-gates`), and the :gates that were applied."
  ([posterior] (diagnostics-summary posterior default-gates))
  ([posterior gates]
   (let [{:keys [max-rhat min-ess max-divergences]} (merge default-gates gates)
         ds (:diagnostics posterior)
         names (:parameter-names posterior)
         _ (when-not (and (seq ds) (= (count ds) (count names)))
             (throw (ex-info "Posterior has no per-parameter diagnostics aligned with its parameter names"
                             {:diagnostics (count ds) :parameter-names (count names)})))
         finite (every? config/finite? (mapcat #(map % [:rhat :ess-bulk :ess-tail]) ds))
         divergences (get-in posterior [:sampler-statistics :divergences])]
     {:parameters (apply array-map (mapcat (fn [name d] [name (dissoc d :draws :chains)]) names ds))
      :finite? finite
      :max-rhat (when finite (apply max (map :rhat ds)))
      :min-ess-bulk (when finite (apply min (map :ess-bulk ds)))
      :min-ess-tail (when finite (apply min (map :ess-tail ds)))
      :divergences divergences
      :depth-limit-visits (get-in posterior [:sampler-statistics :depth-limit-visits])
      :passes? (boolean (and finite (integer? divergences) (<= divergences max-divergences)
                             (every? #(< (:rhat %) max-rhat) ds)
                             (every? #(>= (:ess-bulk %) min-ess) ds)
                             (every? #(>= (:ess-tail %) min-ess) ds)))
      :gates {:max-rhat max-rhat :min-ess min-ess :max-divergences max-divergences}})))

(ns foerster-study
  "Foerster side of the Kettle & Pine comparison with PyMC-Marketing.

  Usage (from the repository root; study/run_study.py drives all cases):

    clojure -M:jvm:study -m foerster-study check|fit|decision|replay <fixture.json> <dir> [pilot|full]

  - check:    log density, gradient, mean response and pointwise likelihood
              at the fixture's unconstrained points -> foerster-check.json
  - fit:      NUTS fit -> posterior.edn, posterior.json, diagnostics.json,
              foerster-report.json
  - decision: allocation and paired plan contrast from the saved posterior,
              merged into foerster-report.json
  - replay:   rebuild in a fresh JVM, re-run the decision from the persisted
              posterior and require identical results -> native-replay.json"
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [kettle-and-pine.core :as kettle-and-pine]
            [org.replikativ.marketing.model :as model]
            [org.replikativ.marketing.inference :as inference]
            [org.replikativ.marketing.decision :as decision]
            [org.replikativ.marketing.io :as posterior-io]
            [org.replikativ.foerster.block :as block])
  (:import [java.security MessageDigest]))

(def pymc-matched-start
  "Shared unconstrained starting point, identical to the one study/pymc_model.py
  asserts for PyMC: each parameter at its prior mean, in unconstrained
  coordinates. alpha: logit 0.5 = 0 (Beta(2,2)); beta and kappa: the log of
  the LogNormal means, mu + sigma^2/2 = 0.18 and -0.32; intercept 1.0;
  sigma: log 0.2 (the HalfNormal scale); controls 0. The order is the model's
  parameter layout: alpha[3], beta[3], kappa[3], intercept, sigma, control[4]."
  [0.0 0.0 0.0 0.18 0.18 0.18 -0.32 -0.32 -0.32 1.0 (Math/log 0.2) 0.0 0.0 0.0 0.0])

(def profiles
  {:pilot {:draws 100 :tune 100 :chains 2 :seed 731 :target-accept 0.9 :max-depth 8 :timeout-ms 900000 :init pymc-matched-start}
   :full {:draws 4000 :tune 2000 :chains 4 :seed 731 :target-accept 0.95 :max-depth 10 :timeout-ms 7200000 :init pymc-matched-start}})

(def decision-draws
  "Allocation uses 256 evenly spaced draws per chain (1,000 of the 16,000 for
  the full profile); study/pymc_decision.py applies the same rule. All draws
  remain available for diagnostics and the final plan contrast."
  1024)

(defn- require! [condition message data]
  (when-not condition (throw (ex-info message data))))

(defn- sha256 [text]
  (format "%064x" (BigInteger. 1 (.digest (MessageDigest/getInstance "SHA-256") (.getBytes ^String text "UTF-8")))))

(defn- source-hashes []
  (into (sorted-map)
        (for [root ["src" "examples"] file (file-seq (io/file root))
              :when (and (.isFile file) (str/ends-with? (.getName file) ".clj"))]
          [(.getPath file) (sha256 (slurp file))])))

(defn- write-json! [path value]
  (io/make-parents path)
  (spit path (json/write-str value)))

(defn- gate-report
  "The study's diagnostics record (also read by the Python side)."
  [posterior]
  (let [s (inference/diagnostics-summary posterior)]
    {:finite? (:finite? s) :max-rhat (:max-rhat s) :min-ess-bulk (:min-ess-bulk s) :min-ess-tail (:min-ess-tail s)
     :divergences (:divergences s) :depth-limit-visits (:depth-limit-visits s)
     :passes-tutorial-gates? (:passes? s)}))

(defn- provenance [fixture-path]
  {:fixture-sha256 (sha256 (slurp fixture-path))
   :deps-sha256 (sha256 (slurp "deps.edn"))
   :source-sha256 (sha256 (pr-str (source-hashes)))
   :runner-sha256 (sha256 (slurp "study/foerster_study.clj"))
   :java-version (System/getProperty "java.version")
   :available-processors (.availableProcessors (Runtime/getRuntime))
   :jvm-options (vec (filter #(re-find #"^(?:-X|-XX:|--add-modules|--enable-native-access)" %)
                             (.getInputArguments (java.lang.management.ManagementFactory/getRuntimeMXBean))))
   :foerster-resource (str (io/resource "org/replikativ/foerster/core.cljc"))
   :raster-resource (str (io/resource "raster/ad/reverse.clj"))
   :bridge-resource (str (io/resource "org/replikativ/foerster_raster/block.clj"))})

(defn check! [fixture-path directory]
  (let [f (kettle-and-pine/read-fixture fixture-path)
        built (kettle-and-pine/build f)
        b (model/inference-block built :aot)
        evaluate (block/capability b :value+grad)
        points (mapv (fn [q]
                       (let [[value gradient] (evaluate (double-array q) (:inputs built))
                             theta (block/constrain b q)]
                         {:q q :theta theta :logp value :gradient (vec gradient)
                          :mean (model/mean-response built theta)
                          :pointwise (model/pointwise-log-likelihood built theta)}))
                     (:unconstrained-points f))
        changed (model/prepare (:spec f) (update (:training f) :y #(mapv (fn [y] (+ y 100.0)) %)))
        [changed-value changed-gradient] (evaluate (double-array (:q (first points))) (:inputs changed))
        report {:engine :foerster-marketing :case (:case f) :points points
                :changed-data {:logp changed-value :gradient (vec changed-gradient)}
                :parameter-names (:parameter-names built)
                :data-layout (:data-layout built)
                :gradient-parameter-layout (:gradient-parameter-layout built)
                :provenance (provenance fixture-path)}]
    (write-json! (str directory "/foerster-check.json") report)
    (require! (not= changed-value (:logp (first points))) "Changed observations were not used" {})
    {:case (:case f) :points (count points) :checked true}))

(defn- allocation-options [fixture]
  (let [d (:decision fixture)]
    {:budget (:weekly-budget d) :horizon (:horizon d)
     :controls (:controls d) :lower (:lower d) :upper (:upper d)
     :tolerance 1e-6 :max-iterations 500 :max-draws decision-draws}))

(defn fit! [fixture-path directory profile]
  (let [f (kettle-and-pine/read-fixture fixture-path)
        _ (require! (profiles profile) "Unknown study profile" {})
        _ (require! (not (.exists (io/file directory "posterior.edn"))) "Refusing to overwrite posterior" {})
        started (System/nanoTime)
        built (kettle-and-pine/build f)
        build-seconds (/ (- (System/nanoTime) started) 1e9)
        posterior (inference/fit built (profiles profile))
        diagnostics (gate-report posterior)
        report {:engine :foerster-marketing :case (:case f) :profile profile
                :statistically-qualified (and (= profile :full) (:passes-tutorial-gates? diagnostics))
                :diagnostics diagnostics :timing {:build-seconds build-seconds :fit (:timing posterior)}
                :sampler (assoc (:sampler posterior)
                                :initial-unconstrained-position pymc-matched-start
                                :initial-natural-position (block/constrain (:block built) pymc-matched-start))
                :provenance (provenance fixture-path)}]
    (io/make-parents (str directory "/posterior.edn"))
    (posterior-io/write! (str directory "/posterior.edn") posterior)
    (posterior-io/write! (str directory "/posterior.json") posterior :json)
    (write-json! (str directory "/diagnostics.json") diagnostics)
    (write-json! (str directory "/foerster-report.json") report)
    (when (= profile :full)
      (require! (:passes-tutorial-gates? diagnostics) "Full fit failed statistical gates" diagnostics))
    ;; Decision compilation/evaluation runs in its own process, as for PyMC.
    (select-keys report [:case :profile :diagnostics :timing])))

(defn- saved-state [fixture-path directory]
  (let [f (kettle-and-pine/read-fixture fixture-path)
        report (json/read-str (slurp (io/file directory "foerster-report.json")) :key-fn keyword)
        expected (provenance fixture-path)
        keys [:fixture-sha256 :deps-sha256 :runner-sha256 :source-sha256]
        _ (require! (= (select-keys expected keys) (select-keys (:provenance report) keys))
                    "Saved campaign result has stale source, dependencies or fixture" {})
        posterior (posterior-io/read! (io/file directory "posterior.edn"))
        json-posterior (posterior-io/read! (io/file directory "posterior.json") :json)
        fields [:spec :parameter-names :chains :sampler-statistics :sampler :inference-contract]
        _ (require! (= (select-keys posterior fields) (select-keys json-posterior fields))
                    "Native posterior formats disagree" {})
        diagnostics (gate-report posterior)]
    (when (= "full" (:profile report))
      (require! (and (:statistically-qualified report) (:passes-tutorial-gates? diagnostics))
                "Saved full posterior fails statistical gates" diagnostics))
    {:fixture f :report report :posterior posterior :diagnostics diagnostics :provenance expected}))

(defn- evaluate-decision [built posterior fixture]
  (let [started (System/nanoTime)
        allocation (decision/allocate built posterior (allocation-options fixture))
        _ (require! (:converged? allocation) "Study allocation failed to converge" allocation)
        d (:decision fixture)
        baseline {:x (vec (repeat (:horizon d) (:baseline d))) :controls (:controls d)}
        candidate {:x (vec (repeat (:horizon d) (:allocation allocation))) :controls (:controls d)}
        contrast (decision/compare-plans built posterior baseline candidate)]
    {:allocation allocation :contrast contrast
     :optimization-draws (reduce + (map count (:chains (decision/thin posterior decision-draws))))
     :decision-seconds (/ (- (System/nanoTime) started) 1e9)}))

(defn decision! [fixture-path directory]
  (let [{:keys [fixture report posterior]} (saved-state fixture-path directory)
        started (System/nanoTime)
        built (kettle-and-pine/build fixture)
        build-seconds (/ (- (System/nanoTime) started) 1e9)
        result (evaluate-decision built posterior fixture)
        report (-> (merge report (dissoc result :decision-seconds))
                   (update :timing assoc :decision-build-seconds build-seconds
                           :decision-seconds (:decision-seconds result)))]
    (write-json! (str directory "/foerster-report.json") report)
    (select-keys report [:case :profile :allocation :timing])))

(defn replay!
  "Rebuild and replay the native persisted decision in a fresh JVM."
  [fixture-path directory]
  (let [{:keys [fixture report posterior diagnostics provenance]} (saved-state fixture-path directory)
        built (kettle-and-pine/build fixture)
        result (evaluate-decision built posterior fixture)
        actual-allocation (get-in result [:allocation :allocation])
        saved-allocation (get-in report [:allocation :allocation])
        _ (require! (= (count actual-allocation) (count saved-allocation))
                    "Replayed allocation dimensions differ" {})
        scaled-error (fn [a b] (/ (Math/abs (- (double a) (double b))) (+ 1.0 (Math/abs (double b)))))
        allocation-error (apply max (map scaled-error actual-allocation saved-allocation))
        draw-values (fn [contrast] (mapcat #(map % [:active :tail :total]) (:draws contrast)))
        actual (vec (draw-values (:contrast result)))
        saved (vec (draw-values (:contrast report)))
        _ (require! (= (count actual) (count saved)) "Replayed posterior draw count differs" {})
        contrast-error (apply max (map scaled-error actual saved))
        output {:case (:case fixture) :passes true :profile (:profile report)
                :allocation-max-scaled-error allocation-error :contrast-max-scaled-error contrast-error
                :posterior-formats-agree true :diagnostics diagnostics
                :posterior-edn-sha256 (sha256 (slurp (io/file directory "posterior.edn")))
                :posterior-json-sha256 (sha256 (slurp (io/file directory "posterior.json")))
                :provenance provenance}]
    (require! (and (<= allocation-error 1e-10) (<= contrast-error 1e-10))
              "Native decision changed after persistence and fresh-process rebuild" output)
    (write-json! (str directory "/native-replay.json") output)
    (select-keys output [:case :passes :allocation-max-scaled-error :contrast-max-scaled-error])))

(defn -main [mode fixture-path directory & [profile]]
  (try
    (case mode "check" (check! fixture-path directory)
          "fit" (fit! fixture-path directory (keyword (or profile "pilot")))
          "decision" (decision! fixture-path directory)
          "replay" (replay! fixture-path directory)
          (throw (ex-info "Use check, fit, decision or replay" {:mode mode})))
    (finally (shutdown-agents))))

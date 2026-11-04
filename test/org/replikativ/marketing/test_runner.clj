(ns org.replikativ.marketing.test-runner
  "Run compiler-heavy test groups in sequential, bounded JVMs."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :as t])
  (:import [java.lang.management ManagementFactory]
           [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]
           [java.util.concurrent TimeUnit]))

(def test-namespaces
  '[org.replikativ.marketing.michaelis-menten-test
    org.replikativ.marketing.response-test
    org.replikativ.marketing.model-test
    org.replikativ.marketing.io-test
    org.replikativ.marketing.advanced-test
    org.replikativ.marketing.hierarchy-test
    org.replikativ.marketing.flexibility-test
    org.replikativ.marketing.decision-test
    org.replikativ.marketing.inference-test
    org.replikativ.marketing.compiled-gradient-test])

(defn- resources []
  {:kernel-cache (when-let [model-ns (find-ns 'org.replikativ.marketing.model)]
                   (when-let [info (ns-resolve model-ns 'kernel-cache-info)] (info)))
   :loaded-classes (.getLoadedClassCount (ManagementFactory/getClassLoadingMXBean))
   :metaspace (into {} (for [pool (ManagementFactory/getMemoryPoolMXBeans)
                             :when (= "Metaspace" (.getName pool))]
                         [:used-bytes (.getUsed (.getUsage pool))]))})

(defn- run-group [namespaces]
  (doseq [n namespaces] (require n))
  (let [before (resources)
        result (apply t/run-tests namespaces)]
    (assoc result :namespaces namespaces :resources-before before :resources-after (resources))))

(defn- worker! [receipt namespaces]
  (try
    (let [result (run-group namespaces)]
      (spit receipt (pr-str result))
      (System/exit (if (zero? (+ (:fail result) (:error result))) 0 1)))
    (finally (shutdown-agents))))

(defn- stop-worker! [^Process process]
  (when (.isAlive process)
    (.destroy process)
    (when-not (.waitFor process 5 TimeUnit/SECONDS)
      (.destroyForcibly process)
      (.waitFor process 5 TimeUnit/SECONDS))))

(defn- isolated-group! [directory index namespaces]
  (let [receipt (io/file directory (str "worker-" index ".edn"))
        java (str (io/file (System/getProperty "java.home") "bin" "java"))
        command (into [java]
                      (concat (.getInputArguments (ManagementFactory/getRuntimeMXBean))
                              ["-cp" (System/getProperty "java.class.path")
                               "clojure.main" "-m" "org.replikativ.marketing.test-runner"
                               "--worker" (str receipt)]
                              (map str namespaces)))
        process (.start (.inheritIO (ProcessBuilder. ^java.util.List command)))
        runtime (Runtime/getRuntime)
        hook (Thread. #(stop-worker! process))]
    (.addShutdownHook runtime hook)
    (try
      (println "Test worker" index "namespaces" namespaces)
      (flush)
      (when-not (.waitFor process 20 TimeUnit/MINUTES)
        (throw (ex-info "Numerical test worker exceeded 20 minutes" {:namespaces namespaces})))
      (when-not (.isFile receipt)
        (throw (ex-info "Test worker exited without a result" {:namespaces namespaces :exit (.exitValue process)})))
      (let [result (edn/read-string (slurp receipt))]
        (when-not (= (.exitValue process) (if (zero? (+ (:fail result) (:error result))) 0 1))
          (throw (ex-info "Test worker result disagrees with exit status" {:result result :exit (.exitValue process)})))
        (prn (select-keys result [:namespaces :resources-before :resources-after]))
        result)
      (finally
        (try (.removeShutdownHook runtime hook) (catch IllegalStateException _))
        (stop-worker! process)))))

(defn -main [& args]
  (if (= "--worker" (first args))
    (worker! (second args) (mapv symbol (drop 2 args)))
    (try
      (let [directory (or (System/getenv "MARKETING_TEST_RESULTS")
                          (str (Files/createTempDirectory "marketing-tests-" (make-array FileAttribute 0))))
            _ (.mkdirs (io/file directory))
            results (mapv (fn [index group] (isolated-group! directory index group))
                          (range) (partition-all 2 test-namespaces))
            totals (apply merge-with + (map #(select-keys % [:test :pass :fail :error]) results))]
        (prn (assoc totals :assertions (+ (:pass totals) (:fail totals) (:error totals)) :worker-results directory))
        (System/exit (if (zero? (+ (:fail totals) (:error totals))) 0 1)))
      (finally (shutdown-agents)))))

(ns org.replikativ.marketing.inference-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.marketing.inference :as inference]
            [org.replikativ.marketing.io :as posterior-io]
            [org.replikativ.marketing.model-test :as reference]
            [org.replikativ.foerster.measure :as measure]))

(deftest portable-seeded-inference
  (let [model @reference/fitted-model
        ;; The public fit budget includes cold AOT preparation as well as sampling.
        opts {:draws 8 :tune 10 :chains 2 :seed 731 :max-depth 2 :timeout-ms 300000}
        ;; nil disables the timeout and must not change the draws.
        a (inference/fit model opts) b (inference/fit model (assoc opts :timeout-ms nil))]
    (is (= (:chains a) (:chains b)))
    (is (= :aot (get-in a [:sampler :gradient-backend])))
    (is (= :markov-chain (get-in a [:inference-contract :semantics])))
    (is (= :unavailable (get-in a [:inference-contract :normalizer :status])))
    (is (false? (get-in a [:sampler :jitter])))
    (is (= :variance (get-in a [:sampler :metric])))
    (is (= [8 8] (mapv count (:chains a))))
    (is (= (:dimension model) (count (:parameter-names a))))
    (is (every? #(= (:dimension model) (count %)) (mapcat identity (:chains a))))
    (is (= :available (get-in a [:sampler-statistics :availability])))
    (is (= [8 8] (mapv count (get-in a [:sampler-statistics :by-chain]))))
    (is (every? boolean? (map :diverging (mapcat identity (get-in a [:sampler-statistics :by-chain])))))
    (is (nil? (:block a)))
    (is (nil? (:particles a)))
    (doseq [format [:edn :json]]
      (let [file (java.io.File/createTempFile "mmm-fitted-contract-" ".data")
            fields [:inference-contract :sampler :chains :sampler-statistics]]
        (try
          (posterior-io/write! file a format)
          (is (= (select-keys a fields)
                 (select-keys (posterior-io/read! file format) fields)))
          (finally (.delete file)))))))

(deftest sampler-options-are-not-silently-ignored
  (doseq [options [{:metric :unknown} {:jitter :yes} {:jitetr true} {:timeout-ms 0} {:timeout-ms "1"}]]
    (is (thrown? clojure.lang.ExceptionInfo (inference/fit @reference/fitted-model options)))))

(deftest detached-results-respect-the-chain-contract
  (let [particle (assoc (measure/sample-particle [0.2] {}) :sampler-stats {:diverging false :blocks 1})
        good (assoc (measure/markov-measure (vec (repeat 4 [particle 0.0]))) :chain-lengths [2 2])]
    (is (= 4 (count (:flat (#'inference/detached-samples good 2 2 1)))))
    (doseq [bad [(assoc good :chain-lengths [1 3])
                 (assoc-in good [:inference-contract :semantics] :weighted-population)
                 (update good :particles pop)
                 (assoc-in good [:particles 0 0 :sampler-stats :blocks] 0)
                 (assoc-in good [:particles 0 0 :result] [Double/NaN])]]
      (is (thrown? clojure.lang.ExceptionInfo (#'inference/detached-samples bad 2 2 1))))))

(deftest diagnostics-summary-names-parameters-and-applies-gates
  (let [good {:rhat 1.001 :ess-bulk 900.0 :ess-tail 800.0 :mean 0.5 :sd 0.1 :draws 2000 :chains 4}
        posterior {:parameter-names [[:alpha "search"] [:alpha "video"] [:sigma]]
                   :diagnostics [good (assoc good :rhat 1.004 :ess-bulk 400.0) (assoc good :ess-tail 300.0)]
                   :sampler-statistics {:divergences 0 :depth-limit-visits 3}}
        summary (inference/diagnostics-summary posterior)]
    (is (= [[:alpha "search"] [:alpha "video"] [:sigma]] (keys (:parameters summary))))
    (is (= (dissoc good :draws :chains) (get-in summary [:parameters [:alpha "search"]])))
    (is (= {:finite? true :max-rhat 1.004 :min-ess-bulk 400.0 :min-ess-tail 300.0
            :divergences 0 :depth-limit-visits 3 :passes? true :gates inference/default-gates}
           (dissoc summary :parameters)))
    (is (false? (:passes? (inference/diagnostics-summary (assoc-in posterior [:sampler-statistics :divergences] 1)))))
    (is (false? (:passes? (inference/diagnostics-summary posterior {:min-ess 500.0}))))
    (is (true? (:passes? (inference/diagnostics-summary (assoc-in posterior [:diagnostics 1 :rhat] 1.02) {:max-rhat 1.05}))))
    (let [nonfinite (inference/diagnostics-summary (assoc-in posterior [:diagnostics 2 :rhat] Double/NaN))]
      (is (= [false nil false] (map nonfinite [:finite? :max-rhat :passes?]))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (inference/diagnostics-summary (update posterior :parameter-names pop))))))

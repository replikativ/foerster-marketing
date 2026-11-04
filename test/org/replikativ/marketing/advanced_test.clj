(ns org.replikativ.marketing.advanced-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.marketing.model :as m]
            [org.replikativ.marketing.model-test :as ref]
            [org.replikativ.marketing.decision :as decision]
            [org.replikativ.marketing.calibration :as calibration]))

(def packed (delay (m/build (m/prepare ref/spec ref/data) {:data-layout :packed})))
(defn invoke [model theta prepared]
  (vec (apply @(:aot-gradient model) ((:args model) (double-array theta) (:inputs prepared)))))

(deftest packed-data-and-reuse
  (let [model @packed f @(:aot-gradient model)
        changed (-> ref/data (update :x #(into % %)) (update :y #(into % %))
                    (update :controls #(into % %)))
        second-model (m/build (m/prepare ref/spec changed) {:data-layout :packed})]
    (is (identical? f @(:aot-gradient second-model)))
    (is (identical? (:density-var model) (:density-var second-model)))
    (is (= :array (:gradient-parameter-layout model)))
    (doseq [data [ref/data changed]]
      (let [result (invoke model ref/theta (m/prepare ref/spec data))]
        (is (ref/close? (first result) (ref/reference-lp ref/spec data ref/theta) 1e-9))
        (doseq [j (range (count ref/theta))]
          (let [fd (/ (- (ref/reference-lp ref/spec data (update ref/theta j + 1e-6))
                         (ref/reference-lp ref/spec data (update ref/theta j - 1e-6))) 2e-6)]
            (is (ref/close? (nth result (inc j)) fd 1e-5))))))
    (is (every? true? (map #(ref/close? %1 %2 1e-12)
                           (m/mean-response @ref/fitted-model ref/theta) (m/mean-response model ref/theta))))
    (let [objective (decision/allocation-objective model {:spec ref/spec :chains [[ref/theta]]} 2 [[0.2] [0.3]])
          result (objective [0.4 0.2])]
      (doseq [j [0 1]]
        (let [fd (/ (- (:value (objective (update [0.4 0.2] j + 1e-6)))
                       (:value (objective (update [0.4 0.2] j - 1e-6)))) 2e-6)]
          (is (ref/close? (nth (:gradient result) j) fd 1e-5)))))))

(def baseline (:x ref/data))
(def treatment (mapv (fn [i row] (if (#{3 4} i) (update row 0 + 0.4) row)) (range) baseline))
(def experiments
  [{:id "pulse-active" :baseline baseline :treatment treatment :weights [0 0 0 1 1 1 1 0]
    :estimate 0.4 :sigma 0.2 :independent-of-training? true}
   {:id "pulse-cooldown" :baseline baseline :treatment treatment :weights [0 0 0 0 0 1 1 1]
    :estimate 0.1 :sigma 0.3 :independent-of-training? true}])
(def covariance [[0.04 0.01] [0.01 0.09]])
(def experiment-data (assoc ref/data :experiments experiments :experiment-covariance covariance))
(def experiment-model (delay (m/build (m/prepare ref/spec experiment-data))))

(defn experiment-effects [theta rows]
  (mapv (fn [{:keys [baseline treatment weights]}]
          (let [mean (fn [x] (ref/reference-mean ref/spec {:x x :controls (vec (repeat (count x) [0.0]))} theta))]
            (reduce + (map #(* %1 (- %2 %3)) weights (mean treatment) (mean baseline))))) rows))
(defn experiment-lp [theta rows cov]
  (let [[d0 d1] (map - (map :estimate rows) (experiment-effects theta rows))
        a (get-in cov [0 0]) b (get-in cov [0 1]) c (get-in cov [1 1]) det (- (* a c) (* b b))
        q (/ (+ (* c d0 d0) (* a d1 d1) (* -2.0 b d0 d1)) det)]
    (+ (* -1.0 (Math/log (* 2.0 Math/PI))) (* -0.5 (Math/log det)) (* -0.5 q)
       (* 2.0 (Math/log (get-in ref/spec [:scaling :target]))))))

(deftest window-matched-joint-experiment-gradient
  (let [model @experiment-model
        reference (fn [theta data] (+ (ref/reference-lp ref/spec ref/data theta)
                                      (experiment-lp theta (:experiments data) (:experiment-covariance data))))
        changed (-> experiment-data (update-in [:experiments 0 :estimate] + 0.1)
                    (assoc :experiment-covariance [[0.09 -0.01] [-0.01 0.16]])
                    (update-in [:experiments 0 :treatment 4 0] + 0.2)
                    (assoc-in [:experiments 1 :weights 6] 0.5))]
    (doseq [data [experiment-data changed]
            :let [actual (invoke model ref/theta (m/prepare ref/spec data))]]
      (is (ref/close? (first actual) (reference ref/theta data) 1e-9))
      (doseq [j (range (count ref/theta))]
        (let [fd (/ (- (reference (update ref/theta j + 1e-6) data)
                       (reference (update ref/theta j - 1e-6) data)) 2e-6)]
          (is (ref/close? (nth actual (inc j)) fd 1e-5)))))))

(deftest invalid-experiment-semantics
  (doseq [rows [(assoc-in experiments [0 :weights 0] 1)
                (assoc-in experiments [0 :independent-of-training?] false)
                (assoc-in experiments [0 :treatment 0 0] 9.0)
                (assoc-in experiments [0 :estimate] Double/NaN)]]
    (is (thrown? clojure.lang.ExceptionInfo (calibration/prepare ref/spec rows covariance))))
  (is (thrown? clojure.lang.ExceptionInfo
               (calibration/prepare ref/spec experiments [[1.0 2.0] [2.0 1.0]]))))

(def wide-spec (m/specification {:channels ["media"] :controls (mapv str (range 15)) :l-max 2}))
(def wide-data {:x [[0.1] [0.3] [0.2] [0.4]] :y [0.2 0.3 0.25 0.4]
                :controls (vec (repeat 4 (vec (repeat 15 0.1))))})
(def wide-model (delay (m/build (m/prepare wide-spec wide-data))))
(deftest packed-model-exceeds-old-public-arity
  (let [model @wide-model theta (into [0.4 0.5 2.0 0.2 0.1] (repeat 15 0.05))
        result (let [[v grad] (apply @(:runtime-gradient model) ((:args model) (double-array theta) (:inputs model)))]
                 (into [v] grad))]
    (is (= :packed (:data-layout model)))
    (is (= 21 (count result)))
    (is (every? #(Double/isFinite (double %)) result))
    (is (ref/close? (first result) (ref/reference-lp wide-spec wide-data theta) 1e-9))
    (doseq [j (range (count theta))]
      (let [fd (/ (- (ref/reference-lp wide-spec wide-data (update theta j + 1e-6))
                     (ref/reference-lp wide-spec wide-data (update theta j - 1e-6))) 2e-6)]
        (is (ref/close? (nth result (inc j)) fd 1e-5))))))

(deftest reject-incompatible-experiment-layout
  (let [changed (-> experiment-data (update-in [:experiments 0 :baseline] conj (last baseline))
                    (update-in [:experiments 0 :treatment] conj (last treatment))
                    (update-in [:experiments 0 :weights] conj 0))
        prepared (m/prepare ref/spec changed)]
    (is (thrown? clojure.lang.ExceptionInfo
                 ((:args @experiment-model) (double-array ref/theta) (:inputs prepared))))))

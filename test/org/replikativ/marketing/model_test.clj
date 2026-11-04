(ns org.replikativ.marketing.model-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.marketing.model :as m]
            [org.replikativ.foerster.block :as block]))

(def data
  {:x [[0.2 0.5] [1.1 0.2] [0.4 0.3] [0.0 0.7] [0.7 0.1] [1.4 0.9] [0.1 0.0] [0.5 0.8]]
   :y [0.4 0.8 0.7 0.6 0.8 1.1 0.9 0.7]
   :controls [[0.0] [0.1] [0.2] [0.3] [0.4] [0.5] [0.6] [0.7]]
   :lift [{:type :effective-spend-contrast :channel :tv :x 0.4 :delta-x 0.3 :delta-y 0.1 :sigma 0.1}
          {:type :effective-spend-contrast :channel :search :x 0.5 :delta-x -0.2 :delta-y -0.03 :sigma 0.12}]})
(def spec (m/specification {:channels [:tv :search] :controls [:trend] :l-max 4
                            :scaling {:channel [2.0 3.0] :target 2.0}}))
(def fitted-model (delay (m/build (m/prepare spec data))))
(def theta [0.6 0.3 1.2 0.7 2.0 1.4 0.3 0.2 0.1])
(defn close? [a b tol] (<= (Math/abs (- a b)) (* tol (max 1.0 (Math/abs a) (Math/abs b)))))
(defn normal-lp [v mu sd]
  (let [r (/ (- v mu) sd)] (- -0.9189385332046727 (Math/log sd) (* 0.5 r r))))

(defn reference-mean
  "Independent direct finite convolution."
  [spec data theta]
  (let [c (count (:channels spec)) l (:l-max spec) sx (get-in spec [:scaling :channel])
        sy (get-in spec [:scaling :target])]
    (mapv (fn [t]
            (* sy (+ (nth theta (* 3 c))
                     (reduce + (map * (drop (+ (* 3 c) 2) theta) (nth (:controls data) t [])))
                     (reduce + (for [j (range c)
                                     :let [a (nth theta j) norm (reduce + (map #(Math/pow a %) (range l)))
                                           z (/ (reduce + (for [k (range l) :when (>= (- t k) 0)]
                                                            (* (Math/pow a k) (/ (get-in data [:x (- t k) j]) (nth sx j))))) norm)]]
                                 (* (nth theta (+ c j)) (Math/tanh (* 0.5 (nth theta (+ (* 2 c) j)) z))))))))
          (range (count (:x data))))))

(defn reference-lp [spec data theta]
  (let [c (count (:channels spec)) sx (get-in spec [:scaling :channel]) sy (get-in spec [:scaling :target])
        prior (+ (normal-lp (nth theta (* 3 c)) 0.0 1.0)
                 (normal-lp (nth theta (inc (* 3 c))) 0.0 0.5) (Math/log 2.0)
                 (reduce + (map #(normal-lp % 0.0 1.0) (drop (+ (* 3 c) 2) theta)))
                 (reduce + (for [j (range c)
                                 :let [a (nth theta j) b (nth theta (+ c j)) lm (nth theta (+ (* 2 c) j))]]
                             (+ (Math/log 6.0) (Math/log a) (Math/log (- 1.0 a))
                                (normal-lp b 0.0 1.0) (Math/log 2.0)
                                (- (* 2.0 (Math/log lm)) lm (Math/log 2.0))))))
        likelihood (reduce + (map #(normal-lp (/ %1 sy) (/ %2 sy) (nth theta (inc (* 3 c))))
                                  (:y data) (reference-mean spec data theta)))
        calibration (reduce + (for [{:keys [channel x delta-x delta-y sigma]} (:lift data)
                                    :let [j (.indexOf (:channels spec) channel)
                                          f (fn [xx] (* (nth theta (+ c j))
                                                        (Math/tanh (* 0.5 (nth theta (+ (* 2 c) j)) (/ xx (nth sx j))))))]]
                                (normal-lp (/ delta-y sy) (- (f (+ x delta-x)) (f x)) (/ sigma sy))))]
    (+ prior likelihood calibration)))

(defn transformed-reference [model q]
  (let [th (block/constrain (:block model) q) c (count (get-in model [:spec :channels]))
        jac (+ (reduce + (for [j (range c) :let [a (nth th j)]] (+ (Math/log a) (Math/log (- 1.0 a)))))
               (reduce + (subvec q c (* 3 c))) (nth q (inc (* 3 c))))]
    (+ (reference-lp (:spec model) (:data model) th) jac)))

(deftest full-vector-density-and-transformed-gradient
  (let [model @fitted-model f (block/capability (:block model) :value+grad)]
    (doseq [q [(m/initial-position model) [-1.0 0.7 -0.3 0.2 0.5 0.8 0.4 -1.2 -0.1]]]
      (let [[v grad] (f (double-array q) (:inputs model))]
        (is (close? v (transformed-reference model q) 1e-10))
        (doseq [j (range (count q))]
          (let [fd (/ (- (transformed-reference model (update q j + 1e-6))
                         (transformed-reference model (update q j - 1e-6))) 2e-6)]
            (is (close? (aget grad j) fd 1e-5) (str "coordinate " j))))))))

(deftest convolution-prediction-and-priors
  (let [model @fitted-model]
    (doseq [a [0.0 1e-8 0.6 0.99999999 1.0]]
      (let [th (assoc theta 0 a)]
        (is (every? true? (map #(close? %1 %2 1e-10)
                               (m/mean-response model th) (reference-mean spec data th))))))
    (is (close? (apply @(:density-var model) ((:args model) (double-array theta) (:inputs model)))
                (reference-lp spec data theta) 1e-10))
    (is (every? true? (map #(close? %1 %2 1e-10)
                           (m/pointwise-log-likelihood model theta)
                           (map #(normal-lp %1 %2 0.4) (:y data) (reference-mean spec data theta)))))))

(deftest same-block-with-new-runtime-data
  (let [model @fitted-model new-data (update data :y #(mapv inc %))
        prepared (m/prepare spec new-data) lp (block/capability (:block model) :log-density)
        q (double-array (m/initial-position model))]
    (is (not= (lp q (:inputs model)) (lp q (:inputs prepared))))))

(deftest reject-ambiguous-data
  (doseq [bad [(assoc data :y [0.1]) (assoc-in data [:x 0 0] -1.0)
               (assoc-in data [:y 0] Double/NaN) (assoc data :dates [1 1])
               (assoc-in data [:lift 0 :sigma] 0.0)
               (assoc-in data [:lift 0 :type] :total-experiment-lift)]]
    (is (thrown? clojure.lang.ExceptionInfo (m/prepare spec bad)))))

(deftest scan-and-lagged-density-equivalence
  (let [prepared (m/prepare spec data)
        scan (m/build prepared {:density-kernel :scan})
        lagged (m/build prepared {:density-kernel :lagged})]
    (doseq [a [1e-8 0.3 0.6 0.99999999]
            :let [th (assoc theta 0 a)
                  f (fn [model]
                      (apply @(:density-var model) ((:args model) (double-array th) (:inputs model))))]]
      (is (close? (f scan) (f lagged) 1e-10))
      (is (close? (f lagged) (reference-lp spec data th) 1e-10)))
    (is (= (:spec scan) (:spec lagged)))
    (is (not (realized? (get-in (m/prepare spec data) [:inputs :x0_lags]))))))

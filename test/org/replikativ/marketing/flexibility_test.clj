(ns org.replikativ.marketing.flexibility-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.marketing.model :as m]
            [org.replikativ.marketing.model-test :as old]
            [org.replikativ.marketing.decision :as decision]
            [org.replikativ.marketing.seasonality :as seasonality]
            [org.replikativ.foerster.block :as block])
  (:import [org.apache.commons.math3.special Gamma]))

(def settings
  {:channels [:tv :search] :controls [:trend] :l-max 4
   :scaling {:channel [2.0 3.0] :target 2.0}
   :seasonality {:period 12.0 :order 1}
   :likelihood {:distribution :student-t :nu 4.0}
   :priors {:alpha [{:alpha 2.0 :beta 3.0} {:alpha 3.0 :beta 2.0}]
            :beta [{:distribution :gamma :shape 3.0 :rate 4.0} {:sigma 0.8}]
            :lambda [{:distribution :log-normal :mu 0.5 :sigma 0.6}
                     {:distribution :gamma :shape 2.5 :rate 1.2}]
            :intercept {:mu 0.2 :sigma 0.5}
            :sigma {:distribution :log-normal :mu -1.6 :sigma 0.4}
            :control [{:sigma 0.5} {:sigma 0.3} {:sigma 0.3}]}})
(def spec (m/specification settings))
(def data (assoc (update-in old/data [:y 2] + 2.0) :time-index (vec (range 40 48))))
(def theta (into old/theta [0.15 -0.08]))
(def model (delay (m/build (m/prepare spec data))))

(defn regressors [spec data]
  (if-let [{:keys [period order]} (:seasonality spec)]
    (mapv (fn [row t]
            (into row (mapcat (fn [k] (let [phase (/ (* 2.0 Math/PI k t) period)]
                                        [(Math/sin phase) (Math/cos phase)])) (range 1 (inc order)))))
          (:controls data) (:time-index data))
    (:controls data)))

(defn reference-mean [spec data theta]
  (old/reference-mean spec (assoc data :controls (regressors spec data)) theta))

(defn prior-lp [value p]
  (case (:distribution p)
    :normal (old/normal-lp value (:mu p) (:sigma p))
    :half-normal (+ (old/normal-lp value 0.0 (:sigma p)) (Math/log 2.0))
    :log-normal (- (old/normal-lp (Math/log value) (:mu p) (:sigma p)) (Math/log value))
    :beta (+ (- (Gamma/logGamma (+ (:alpha p) (:beta p))) (Gamma/logGamma (:alpha p)) (Gamma/logGamma (:beta p)))
             (* (dec (:alpha p)) (Math/log value)) (* (dec (:beta p)) (Math/log (- 1.0 value))))
    :gamma (+ (* (:shape p) (Math/log (:rate p))) (- (Gamma/logGamma (:shape p)))
              (* (dec (:shape p)) (Math/log value)) (- (* (:rate p) value)))))

(defn student-lp [value mu scale nu]
  (let [r (/ (- value mu) scale)]
    (- (Gamma/logGamma (/ (+ nu 1.0) 2.0)) (Gamma/logGamma (/ nu 2.0))
       (* 0.5 (Math/log (* nu Math/PI))) (Math/log scale)
       (* (/ (+ nu 1.0) 2.0) (Math/log1p (/ (* r r) nu))))))

(defn reference-lp [spec data theta]
  (let [c (count (:channels spec)) sy (get-in spec [:scaling :target])
        sigma (nth theta (inc (* 3 c)))
        groups [[:alpha (subvec theta 0 c)] [:beta (subvec theta c (* 2 c))]
                [:lambda (subvec theta (* 2 c) (* 3 c))]
                [:intercept [(nth theta (* 3 c))]] [:sigma [sigma]]
                [:control (subvec theta (+ (* 3 c) 2))]]
        prior (reduce + (for [[group values] groups [j value] (map-indexed vector values)
                              :let [p (get-in spec [:priors group]) p (if (vector? p) (nth p j) p)]]
                          (prior-lp value p)))
        lp (fn [y mu] (if-let [nu (get-in spec [:likelihood :nu])]
                        (student-lp (/ y sy) (/ mu sy) sigma nu)
                        (old/normal-lp (/ y sy) (/ mu sy) sigma)))
        likelihood (reduce + (map lp (:y data) (reference-mean spec data theta)))
        calibration (reduce + (for [{:keys [channel x delta-x delta-y sigma]} (:lift data)
                                    :let [j (.indexOf (:channels spec) channel)
                                          sx (get-in spec [:scaling :channel j])
                                          f (fn [v] (* (nth theta (+ c j))
                                                       (Math/tanh (/ (* (nth theta (+ (* 2 c) j)) v) (* 2.0 sx)))))]]
                                (old/normal-lp (/ delta-y sy) (- (f (+ x delta-x)) (f x)) (/ sigma sy))))]
    (+ prior likelihood calibration)))

(deftest validate-configuration-and-default-compatibility
  (is (= old/spec (m/specification old/spec)))
  (is (= spec (m/specification spec)))
  (is (= [:trend "seasonality/sin-1" "seasonality/cos-1"] (m/control-names spec)))
  (doseq [bad [(assoc settings :likelihood {:distribution :student-t :nu 2.0})
               (assoc settings :likelihood {:distribution :student-t :nu Double/NaN})
               (assoc-in settings [:priors :sigma] {:distribution :normal})
               (assoc-in settings [:priors :alpha] [{:alpha 2.0}])
               (assoc-in settings [:priors :beta] {:sigma -1.0})
               (assoc-in settings [:priors :control] {:sigma 1.0 :typo 2.0})
               (assoc settings :seasonality {:period 2.0 :order 1})
               (assoc settings :seasonality {:period 12.0 :order 1.5})]]
    (is (thrown? clojure.lang.ExceptionInfo (m/specification bad))))
  (doseq [bad [(dissoc data :time-index) (assoc data :time-index [0 1])
               (assoc data :time-index [0 1 2 4 5 6 7 8])]]
    (is (thrown? clojure.lang.ExceptionInfo (m/prepare spec bad)))))

(deftest seasonal-prediction-and-likelihood-units
  (let [model @model]
    (is (every? true? (map #(old/close? %1 %2 1e-10)
                           (m/mean-response model theta) (reference-mean spec data theta))))
    (is (every? true? (map #(old/close? %1 %2 1e-10)
                           (m/pointwise-log-likelihood model theta)
                           (map #(student-lp %1 %2 0.4 4.0) (:y data) (reference-mean spec data theta)))))
    (is (old/close? (apply @(:density-var model) ((:args model) (double-array theta) (:inputs model)))
                    (reference-lp spec data theta) 1e-10))
    (is (every? true? (map #(old/close? %1 %2 1e-10)
                           (first (seasonality/basis {:period 12.0 :order 1} [48])) [0.0 1.0])))))

(deftest configured-aot-gradient-and-changed-data
  (let [model @model f @(:aot-gradient model)
        cases [data (-> data (update :time-index #(mapv (partial + 3) %))
                        (update :y #(mapv (partial + 0.03) %)))]]
    (is (= :array (:gradient-parameter-layout model)))
    (doseq [d cases th [theta (-> theta (assoc 0 0.3) (assoc 7 0.35))]
            :let [inputs (:inputs (m/prepare spec d))
                  actual (vec (apply f ((:args model) (double-array th) inputs)))]]
      (is (old/close? (first actual) (reference-lp spec d th) 1e-9))
      (doseq [j (range (count th))]
        (let [fd (/ (- (reference-lp spec d (update th j + 1e-6))
                       (reference-lp spec d (update th j - 1e-6))) 2e-6)]
          (is (old/close? (nth actual (inc j)) fd 1e-5) (str "configured natural coordinate " j)))))))

(deftest seasonal-scenario-continuity
  (let [model @model posterior {:spec spec :chains [[theta]]}
        baseline {:x [[0.2 0.1] [0.2 0.1]] :controls [[0.8] [0.9]]}
        candidate (assoc baseline :x [[0.6 0.1] [0.6 0.1]])
        response (decision/compare-plans model posterior baseline candidate)
        all-data (fn [plan] {:x (into (:x data) (concat (:x plan) (repeat 3 [0.0 0.0])))
                             :controls (into (:controls data) (concat (:controls plan) (repeat 3 [0.0])))
                             :time-index (vec (range 40 53))})
        a (drop 8 (reference-mean spec (all-data baseline) theta))
        b (drop 8 (reference-mean spec (all-data candidate) theta))]
    (is (old/close? (get-in response [:total :mean]) (reduce + (map - b a)) 1e-9))
    (is (thrown? clojure.lang.ExceptionInfo
                 (decision/compare-plans model posterior (assoc baseline :time-index [0 1]) candidate)))
    (let [objective (decision/allocation-objective model posterior 2 [[0.8] [0.9]])
          result (objective [0.4 0.2])]
      (doseq [j [0 1]]
        (let [fd (/ (- (:value (objective (update [0.4 0.2] j + 1e-6)))
                       (:value (objective (update [0.4 0.2] j - 1e-6)))) 2e-6)]
          (is (old/close? (nth (:gradient result) j) fd 1e-5)))))))

(deftest student-t-extreme-finite-observation
  (let [model @model d (assoc-in data [:y 0] 1e200)
        inputs (:inputs (m/prepare spec d))
        result (vec (apply @(:aot-gradient model) ((:args model) (double-array theta) inputs)))
        updated (merge model (m/prepare spec d))]
    (is (every? #(Double/isFinite (double %)) result))
    (is (every? #(Double/isFinite (double %)) (m/pointwise-log-likelihood updated theta)))))

(def boundary-model
  (delay (m/build (m/prepare (m/specification (-> settings (assoc :controls [] :priors {})
                                                  (dissoc :likelihood)))
                             (assoc data :controls (vec (repeat 8 [])))))))

(deftest qualified-scalar-signature-boundary
  (let [model @boundary-model
        th (into (subvec theta 0 8) [0.15 -0.08])
        actual (vec (apply @(:aot-gradient model) ((:args model) (double-array th) (:inputs model))))
        spec (:spec model) data (:data model)]
    (is (= :scalar (:gradient-parameter-layout model)))
    (is (old/close? (first actual) (reference-lp spec data th) 1e-9))
    (doseq [j (range (count th))]
      (let [fd (/ (- (reference-lp spec data (update th j + 1e-6))
                     (reference-lp spec data (update th j - 1e-6))) 2e-6)]
        (is (old/close? (nth actual (inc j)) fd 1e-5))))))

(deftest sequential-inputs-preserve-layout
  (let [as-lists (-> data (update :x #(apply list (map (partial apply list) %)))
                     (update :controls #(apply list (map (partial apply list) %)))
                     (update :y #(apply list %)))
        a (m/prepare spec data) b (m/prepare spec as-lists)]
    (is (= (:data a) (:data b)))
    (doseq [key [:u0 :u1 :u2 :x0 :x1 :x0_lags :x1_lags]]
      (is (= (vec (force (get-in a [:inputs key])))
             (vec (force (get-in b [:inputs key])))) (str key)))
    (is (= (mapv first (:controls data)) (vec (get-in b [:inputs :u0]))))))

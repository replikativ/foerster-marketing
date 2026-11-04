(ns org.replikativ.marketing.hierarchy-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.marketing.hierarchy :as geo]
            [org.replikativ.marketing.model :as m]
            [org.replikativ.marketing.model-test :as ref]
            [org.replikativ.marketing.layout :as layout]
            [org.replikativ.marketing.advanced-test :as advanced]
            [org.replikativ.marketing.flexibility-test :as flex]))
(def spec (geo/specification (assoc (dissoc ref/spec :priors) :groups ["north" "south"])))
(def data {:panels {"north" (-> ref/data (update :x #(subvec % 0 5)) (update :y #(subvec % 0 5)) (update :controls #(subvec % 0 5)))
                    "south" (-> ref/data (update :x #(vec (reverse %))))}})
(def theta [0.35 0.65 2.0 1.3 -0.6 -0.9 0.25 0.35 0.7 -0.5 -0.7 0.5 0.5 0.7 0.1 0.15 0.15])
(def model (delay (geo/build (geo/prepare spec data))))
(defn flat-theta [theta g]
  (let [[a0 a1 l0 l1 mu0 mu1 tau0 tau1 z00 z01 z10 z11 b0 b1 s0 s1 gamma] theta
        z (if (zero? g) [z00 z01] [z10 z11])]
    [a0 a1 (Math/exp (+ mu0 (* tau0 (first z)))) (Math/exp (+ mu1 (* tau1 (second z))))
     l0 l1 (if (zero? g) b0 b1) (if (zero? g) s0 s1) gamma]))
(defn means [theta data]
  (vec (mapcat (fn [g name] (ref/reference-mean ref/spec (get-in data [:panels name]) (flat-theta theta g))) (range) (:groups spec))))
(defn reference-lp [theta data]
  (let [p (:priors spec) h (:hierarchy spec)
        prior (reduce + (concat (map #(flex/prior-lp % (:alpha p)) (take 2 theta))
                                (map #(flex/prior-lp % (:lambda p)) (subvec theta 2 4))
                                (map #(ref/normal-lp % (:location-mu h) (:location-sigma h)) (subvec theta 4 6))
                                (map #(+ (ref/normal-lp % 0.0 (:scale-sigma h)) (Math/log 2.0)) (subvec theta 6 8))
                                (map #(ref/normal-lp % 0.0 1.0) (subvec theta 8 12))
                                (map #(flex/prior-lp % (:intercept p)) (subvec theta 12 14))
                                (map #(flex/prior-lp % (:sigma p)) (subvec theta 14 16))
                                [(flex/prior-lp (nth theta 16) (:control p))]))
        sy (get-in spec [:scaling :target])
        observed (reduce + (for [[g name] (map-indexed vector (:groups spec))
                                 :let [panel (get-in data [:panels name]) th (flat-theta theta g)]
                                 [y mu] (map vector (:y panel) (ref/reference-mean ref/spec panel th))]
                             (ref/normal-lp (/ y sy) (/ mu sy) (nth th 7))))
        lift (reduce + (for [[g name] (map-indexed vector (:groups spec))
                             {:keys [channel x delta-x delta-y sigma]} (get-in data [:panels name :lift])
                             :let [th (flat-theta theta g) j (.indexOf (:channels spec) channel)
                                   sx (get-in spec [:scaling :channel j])
                                   f (fn [x] (* (nth th (+ 2 j)) (Math/tanh (/ (* (nth th (+ 4 j)) x) (* 2.0 sx)))))]]
                         (ref/normal-lp (/ delta-y sy) (- (f (+ x delta-x)) (f x)) (/ sigma sy))))]
    (+ prior observed lift)))

(deftest geo-layout-and-runtime-gradient
  (let [model @model f (fn [& args] (let [[v grad] (apply @(:runtime-gradient model) args)] (double-array (into [v] grad))))
        changed (update-in data [:panels "south"] (fn [panel] (-> panel (update :x #(subvec % 0 3))
                                                                  (update :y #(subvec % 0 3))
                                                                  (update :controls #(subvec % 0 3)))))]
    (is (= spec (geo/specification spec)))
    (is (= 17 (:dimension model) (count (layout/names (geo/parameter-layout spec)))))
    (is (= :array (:gradient-parameter-layout model)))
    (doseq [data [data changed] :let [prepared (geo/prepare spec data)
                                      actual (vec (apply f ((:args model) (double-array theta) (:inputs prepared))))]]
      (is (ref/close? (first actual) (reference-lp theta data) 1e-9))
      (doseq [j (range (count theta))]
        (let [fd (/ (- (reference-lp (update theta j + 1e-6) data) (reference-lp (update theta j - 1e-6) data)) 2e-6)]
          (is (ref/close? (nth actual (inc j)) fd 1e-5) (str "geo coordinate " j))))
      (is (every? true? (map #(ref/close? %1 %2 1e-10) (m/mean-response (merge model prepared) theta) (means theta data)))))
    (is (every? #(Double/isFinite (double %)) (m/pointwise-log-likelihood model theta)))
    (is (= (mapv #(subvec (flat-theta theta %) 2 4) [0 1]) (geo/amplitudes spec theta)))))

(deftest reject-ambiguous-geo-configurations
  (is (thrown? clojure.lang.ExceptionInfo (geo/specification (assoc spec :groups ["north" "north"]))))
  (is (thrown? clojure.lang.ExceptionInfo (geo/prepare spec {:panels {"north" ref/data}})))
  (is (thrown? clojure.lang.ExceptionInfo (geo/specification (assoc-in (dissoc spec :model) [:priors :beta] {:sigma 1.0})))))

(deftest reject-panel-local-covariance
  (is (thrown? clojure.lang.ExceptionInfo
               (geo/prepare spec (assoc-in data [:panels "north" :experiment-covariance] [[1.0]])))))

(deftest geo-experiments-share-cross-group-covariance
  (let [rows [(first advanced/experiments) (second advanced/experiments)]
        d (-> data (assoc-in [:panels "north" :experiments] [(first rows)])
              (assoc-in [:panels "south" :experiments] [(second rows)])
              (assoc :experiment-covariance advanced/covariance))
        model (geo/build (geo/prepare spec d))
        reference (fn [theta]
                    (let [effects (mapv (fn [g row] (first (advanced/experiment-effects (flat-theta theta g) [row]))) [0 1] rows)
                          residuals (mapv - (mapv :estimate rows) effects)
                          [d0 d1] residuals
                          [[a b] [_ c]] advanced/covariance
                          det (- (* a c) (* b b))
                          q (/ (+ (* c d0 d0) (* a d1 d1) (* -2.0 b d0 d1)) det)]
                      (+ (reference-lp theta data) (- (Math/log (* 2.0 Math/PI)))
                         (* -0.5 (Math/log det)) (* -0.5 q)
                         (* 2.0 (Math/log (get-in spec [:scaling :target]))))))
        [v grad] (apply @(:runtime-gradient model) ((:args model) (double-array theta) (:inputs model)))]
    (is (ref/close? v (reference theta) 1e-9))
    (doseq [j (range (count theta))]
      (let [fd (/ (- (reference (update theta j + 1e-6)) (reference (update theta j - 1e-6))) 2e-6)]
        (is (ref/close? (nth grad j) fd 1e-5))))))

(deftest many-geo-lifts-use-a-compact-runtime-reduction
  (let [many (update data :panels
                     (fn [panels] (into {} (map (fn [[name panel]]
                                                  [name (update panel :lift #(vec (mapcat (fn [row] (repeat 3 row)) %)))]) panels))))
        model (geo/build (geo/prepare spec many))
        changed (update-in many [:panels "south" :lift] #(mapv (fn [row] (-> row (update :x + 0.2) (update :delta-y + 0.01) (update :sigma * 1.1))) %))]
    (doseq [data [many changed]
            :let [prepared (geo/prepare spec data)
                  [v grad] (apply @(:runtime-gradient model) ((:args model) (double-array theta) (:inputs prepared)))]]
      (is (ref/close? v (reference-lp theta data) 1e-9))
      (doseq [j (range (count theta))]
        (let [fd (/ (- (reference-lp (update theta j + 1e-6) data) (reference-lp (update theta j - 1e-6) data)) 2e-6)]
          (is (ref/close? (nth grad j) fd 1e-5)))))))

(deftest geo-lift-selection-with-different-channel-and-group-counts
  (let [spec (geo/specification {:channels ["media"] :controls [] :l-max 2
                                 :groups ["a" "b" "c"] :scaling {:channel [3.0] :target 2.0}})
        base (m/specification {:channels ["media"] :controls [] :l-max 2
                               :scaling {:channel [3.0] :target 2.0}})
        panel {:x [[0.2] [0.6] [0.4]] :y [0.4 0.5 0.6]
               :lift [{:type :effective-spend-contrast :channel "media" :x 0.2 :delta-x 0.3 :delta-y 0.07 :sigma 0.05}]}
        data {:panels (zipmap (:groups spec) (repeat panel))}
        plain (update data :panels #(into {} (map (fn [[g p]] [g (assoc p :lift [])]) %)))
        theta [0.4 1.7 -0.5 0.3 -1.0 0.0 1.0 0.2 0.3 0.4 0.1 0.12 0.15]
        with-lift (geo/build (geo/prepare spec data)) without (geo/build (geo/prepare spec plain))
        evaluate (fn [model] (apply @(:runtime-gradient model) ((:args model) (double-array theta) (:inputs model))))
        [v grad] (evaluate with-lift) [v0 grad0] (evaluate without)
        effect (fn [theta]
                 (reduce + (for [g (range 3)
                                 :let [beta (Math/exp (+ (nth theta 2) (* (nth theta 3) (nth theta (+ 4 g)))))
                                       curve (fn [x] (* beta (Math/tanh (/ (* (nth theta 1) x) 6.0))))]]
                             (ref/normal-lp 0.035 (- (curve 0.5) (curve 0.2)) 0.025))))]
    (is (ref/close? (- v v0) (effect theta) 1e-9))
    (doseq [j (range (count theta))]
      (let [fd (/ (- (effect (update theta j + 1e-6)) (effect (update theta j - 1e-6))) 2e-6)]
        (is (ref/close? (- (nth grad j) (nth grad0 j)) fd 1e-5))))
    (let [expected (mapcat (fn [g]
                             (ref/reference-mean base panel
                                                 [(first theta) (Math/exp (+ (nth theta 2) (* (nth theta 3) (nth theta (+ 4 g)))))
                                                  (nth theta 1) (nth theta (+ 7 g)) (nth theta (+ 10 g))])) (range 3))]
      (is (every? true? (map #(ref/close? %1 %2 1e-10) expected (m/mean-response with-lift theta)))))))

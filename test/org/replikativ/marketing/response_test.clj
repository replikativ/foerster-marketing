(ns org.replikativ.marketing.response-test
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.marketing.model :as m]
            [org.replikativ.marketing.hierarchy :as geo]
            [org.replikativ.marketing.layout :as layout]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.marketing.inference :as inference]
            [org.replikativ.marketing.io :as io]
            [org.replikativ.marketing.decision :as decision]
            [org.replikativ.marketing.model-test :as ref]
            [org.replikativ.marketing.flexibility-test :as flex]))

(def spec (m/specification {:channels [:tv :search] :controls [:trend] :l-max 3
                            :saturation :none :scaling {:channel [2.0 3.0] :target 4.0}}))
;; This independent reference deliberately encodes the no-saturation coordinate
;; order directly instead of consuming production layout/response constructors.
(def theta [0.3 0.6 0.7 0.4 0.2 0.3 -0.1])
(def data {:x [[0.0 0.1] [0.5 0.3] [1.0 0.6] [0.4 0.2] [0.2 0.7] [0.6 0.4]]
           :controls [[0.1] [0.3] [-0.2] [0.0] [0.2] [0.4]] :y [0.8 1.0 1.3 0.9 1.2 1.1]
           :lift [{:type :effective-spend-contrast :channel :tv :x 0.4 :delta-x 0.2 :delta-y 0.3 :sigma 0.2}
                  {:type :effective-spend-contrast :channel :search :x 0.8 :delta-x -0.3 :delta-y -0.1 :sigma 0.3}]})
(defn means [spec data theta]
  (let [c (count (:channels spec)) l (:l-max spec) sx (get-in spec [:scaling :channel]) sy (get-in spec [:scaling :target])]
    (mapv (fn [t]
            (* sy (+ (nth theta (* 2 c))
                     (reduce + (map * (drop (+ (* 2 c) 2) theta) (nth (:controls data) t [])))
                     (reduce + (for [j (range c)
                                     :let [a (nth theta j) weights (mapv #(Math/pow a %) (range l))]]
                                 (* (nth theta (+ c j))
                                    (/ (reduce + (for [k (range l) :when (>= t k)]
                                                   (* (nth weights k) (/ (get-in data [:x (- t k) j]) (nth sx j)))))
                                       (reduce + weights)))))))) (range (count (:x data))))))
(def experiments
  [{:id :active :baseline (:x data) :treatment (update-in (:x data) [3 0] + 0.6)
    :weights [0 0 0 1 1 0] :estimate 0.4 :sigma 0.2 :independent-of-training? true}
   {:id :tail :baseline (:x data) :treatment (update-in (:x data) [3 0] + 0.6)
    :weights [0 0 0 0 0.5 1] :estimate 0.1 :sigma 0.3 :independent-of-training? true}])
(def covariance [[0.04 0.01] [0.01 0.09]])
(def temporal-data (assoc data :experiments experiments :experiment-covariance covariance))
(defn experiment-delta [spec theta row]
  (let [path (fn [x] (means spec {:x x :controls (vec (repeat (count x) []))}
                            (assoc (vec (take (+ (* 2 (count (:channels spec))) 2) theta))
                                   (* 2 (count (:channels spec))) 0.0)))]
    (reduce + (map #(* %1 (- %2 %3)) (:weights row) (path (:treatment row)) (path (:baseline row))))))
(defn joint-lp [spec residual cov]
  (let [[u v] residual a (get-in cov [0 0]) b (get-in cov [0 1]) d (get-in cov [1 1])
        det (- (* a d) (* b b))]
    (+ (- (Math/log (* 2.0 Math/PI))) (* -0.5 (Math/log det))
       (* -0.5 (/ (+ (* d u u) (* a v v) (* -2.0 b u v)) det))
       (* 2.0 (Math/log (get-in spec [:scaling :target]))))))
(defn evidence-lp [spec data theta]
  (let [c (count (:channels spec)) sy (get-in spec [:scaling :target])
        observation (reduce + (map #(ref/normal-lp (/ %1 sy) (/ %2 sy) (nth theta (inc (* 2 c))))
                                   (:y data) (means spec data theta)))
        lift (reduce + (for [row (:lift data) :let [j (.indexOf (:channels spec) (:channel row))]]
                         (ref/normal-lp (/ (:delta-y row) sy)
                                        (* (nth theta (+ c j)) (/ (:delta-x row) (get-in spec [:scaling :channel j])))
                                        (/ (:sigma row) sy))))]
    (+ observation lift
       (if (seq (:experiments data))
         (joint-lp spec (mapv #(- (:estimate %) (experiment-delta spec theta %)) (:experiments data))
                   (:experiment-covariance data)) 0.0))))
(defn flat-lp [spec data theta]
  (let [c (count (:channels spec)) groups [[:alpha (subvec theta 0 c)] [:beta (subvec theta c (* 2 c))]
                                           [:intercept [(nth theta (* 2 c))]] [:sigma [(nth theta (inc (* 2 c)))]]
                                           [:control (subvec theta (+ (* 2 c) 2))]]]
    (+ (evidence-lp spec data theta)
       (reduce + (for [[group values] groups [j x] (map-indexed vector values)
                       :let [prior (get-in spec [:priors group])]]
                   (flex/prior-lp x (if (vector? prior) (nth prior j) prior)))))))
(def flat (delay (m/build (m/prepare spec data))))
(def packed (delay (m/build (m/prepare spec data) {:data-layout :packed})))
(def scan (delay (m/build (m/prepare spec data) {:density-kernel :scan})))
(def temporal (delay (m/build (m/prepare spec temporal-data))))
(defn check-runtime [model theta prepared reference]
  (let [args ((:args model) (double-array theta) (:inputs prepared))
        [value grad] (apply @(:runtime-gradient model) args)]
    (is (ref/close? value (reference theta) 1e-9))
    (doseq [j (range (count theta))]
      (let [fd (/ (- (reference (update theta j + 1e-6)) (reference (update theta j - 1e-6))) 2e-6)]
        (is (ref/close? (nth grad j) fd 1e-5))))))

(deftest family-layout-and-persistence
  (is (= spec (m/specification (assoc spec :saturation "none"))))
  (doseq [settings [{:saturation :hill} {:saturation :none :priors {:lambda {}}}]]
    (is (thrown? clojure.lang.ExceptionInfo (m/specification (merge {:channels [:tv]} settings)))))
  (let [model @flat]
    (is (= 7 (:dimension model) (count (m/initial-position model)) (count (inference/parameter-names model))))
    (is (= #{:alpha :beta :intercept :sigma :control} (set (keys (:priors spec)))))
    (is (not-any? #(= :lambda (first %)) (:parameter-names model)))
    (is (= {:alpha 0 :beta 2 :intercept 4 :sigma 5 :control 6} (layout/offsets (:parameter-layout model))))
    (is (ref/close? (first (m/pointwise-log-likelihood model theta))
                    (ref/normal-lp (first (:y data)) (first (means spec data theta)) (* 4.0 0.3)) 1e-12))
    (doseq [format [:edn :json]]
      (let [file (java.io.File/createTempFile "mmm-none-" ".data")
            posterior {:format :foerster-marketing/posterior :version 1 :spec spec
                       :parameter-names (:parameter-names model) :chains [[theta]]}]
        (try (io/write! file posterior format)
             (is (= posterior (io/read! file format)))
             (is (thrown? clojure.lang.ExceptionInfo (io/validate (assoc-in posterior [:chains 0 0] (conj theta 1.0)))))
             (finally (.delete file)))))))

(deftest independent-flat-density-and-gradients
  (doseq [model [@flat @packed @scan]]
    (is (every? true? (map #(ref/close? %1 %2 1e-12) (m/mean-response model theta) (means spec data theta))))
    (check-runtime model theta (m/prepare spec data) #(flat-lp spec data %)))
  (let [model @flat result (vec (apply @(:aot-gradient model) ((:args model) (double-array theta) (:inputs model))))]
    (is (ref/close? (first result) (flat-lp spec data theta) 1e-9))
    (let [[_ runtime] (apply @(:runtime-gradient model) ((:args model) (double-array theta) (:inputs model)))]
      (is (every? true? (map #(ref/close? %1 %2 1e-10) (rest result) runtime)))))
  (doseq [d [temporal-data (-> temporal-data (update-in [:experiments 0 :treatment 3 1] + 0.2)
                               (assoc :experiment-covariance [[0.09 -0.01] [-0.01 0.16]])
                               (update :x conj [0.2 0.1]) (update :controls conj [0.3]) (update :y conj 1.1))]]
    (check-runtime @temporal theta (m/prepare spec d) #(flat-lp spec d %))))

(def geo-spec (geo/specification {:channels [:tv] :controls [:trend] :groups [:west :east]
                                  :l-max 3 :saturation :none :scaling {:channel [2.0] :target 4.0}}))
(def geo-theta [0.4 -0.3 0.25 -0.2 0.3 0.1 0.2 0.3 0.4 -0.1])
(defn panel [j]
  (let [rows (mapv #(vector (nth % j)) (:x data))
        row {:id (keyword (str "pulse-" j)) :baseline rows :treatment (update-in rows [3 0] + 0.5)
             :weights [0 0 0 1 1 1] :estimate 0.25 :sigma 0.2 :independent-of-training? true}]
    {:x rows :y (:y data) :controls (:controls data) :experiments [row]
     :lift [(assoc (first (:lift data)) :delta-x (- 0.2 (* j 0.3)))]}))
(def geo-data {:panels {:west (panel 0) :east (panel 1)} :experiment-covariance [[0.04 -0.01] [-0.01 0.09]]})
(def geo-model (delay (geo/build (geo/prepare geo-spec geo-data))))
(defn geo-flat-theta [theta j]
  [(nth theta 0) (Math/exp (+ (nth theta 1) (* (nth theta 2) (nth theta (+ 3 j)))))
   (nth theta (+ 5 j)) (nth theta (+ 7 j)) (nth theta 9)])
(def panel-spec (m/specification (dissoc geo-spec :groups :hierarchy :model)))
(defn geo-lp [theta data]
  (+ (flex/prior-lp (first theta) (get-in panel-spec [:priors :alpha]))
     (ref/normal-lp (nth theta 1) (Math/log 0.5) 0.7)
     (ref/normal-lp (nth theta 2) 0.0 0.5) (Math/log 2.0)
     (reduce + (map #(ref/normal-lp % 0.0 1.0) (subvec theta 3 5)))
     (reduce + (map #(ref/normal-lp % 0.0 1.0) (subvec theta 5 7)))
     (reduce + (map #(+ (ref/normal-lp % 0.0 0.5) (Math/log 2.0)) (subvec theta 7 9)))
     (ref/normal-lp (nth theta 9) 0.0 1.0)
     (reduce + (for [[j group] (map-indexed vector [:west :east])]
                 (evidence-lp panel-spec (dissoc (get-in data [:panels group]) :experiments) (geo-flat-theta theta j))))
     (joint-lp geo-spec (mapv (fn [j group]
                                (let [row (first (get-in data [:panels group :experiments]))]
                                  (- (:estimate row) (experiment-delta panel-spec (geo-flat-theta theta j) row))))
                              (range) [:west :east]) (:experiment-covariance data))))

(deftest independent-geo-density-and-gradients
  (let [model @geo-model]
    (is (= 10 (:dimension model) (count (m/initial-position model)) (count (:parameter-names model))))
    (is (not (contains? (:priors geo-spec) :lambda)))
    (is (not (contains? (:priors geo-spec) :beta)))
    (is (not-any? #(= :lambda (first %)) (:parameter-names model)))
    (is (every? true? (map #(ref/close? %1 %2 1e-12) (m/mean-response model geo-theta)
                           (mapcat (fn [j group] (means panel-spec (get-in geo-data [:panels group]) (geo-flat-theta geo-theta j)))
                                   (range) [:west :east]))))
    (doseq [d [geo-data (-> geo-data (update-in [:panels :east :lift 0 :delta-y] + 0.1)
                            (update-in [:panels :west :experiments 0 :treatment 4 0] + 0.3)
                            (assoc :experiment-covariance [[0.09 0.01] [0.01 0.16]]))]]
      (check-runtime model geo-theta (geo/prepare geo-spec d) #(geo-lp % d)))
    (let [posterior {:format :foerster-marketing/posterior :version 1 :spec geo-spec
                     :parameter-names (:parameter-names model) :chains [[geo-theta]]}]
      (is (= posterior (io/validate posterior))))
    (is (every? true? (map-indexed (fn [i lp]
                                     (ref/close? lp (ref/normal-lp (nth (:y data) (mod i 6))
                                                                   (nth (m/mean-response model geo-theta) i)
                                                                   (* 4.0 (nth geo-theta (+ 7 (quot i 6))))) 1e-10))
                                   (m/pointwise-log-likelihood model geo-theta))))))

(deftest no-saturation-decisions-with-full-carryover
  (let [model @flat posterior {:spec spec :chains [[theta]]}
        controls [[0.2] [0.1]] objective (decision/allocation-objective model posterior 2 controls)
        value (objective [0.5 0.9])]
    ;; Entire finite lag tail is included: each new spend unit contributes once.
    (is (every? true? (map #(ref/close? %1 %2 1e-10) (:gradient value) [2.8 (/ 3.2 3.0)])))
    (doseq [j [0 1]]
      (let [fd (/ (- (:value (objective (update [0.5 0.9] j + 1e-6)))
                     (:value (objective (update [0.5 0.9] j - 1e-6)))) 2e-6)]
        (is (ref/close? fd (nth (:gradient value) j) 1e-5))))
    (let [allocation (decision/allocate model posterior {:budget 1.4 :horizon 2 :controls controls
                                                         :lower [0.1 0.2] :upper [1.2 1.3]})]
      (is (:converged? allocation))
      (is (every? true? (map #(ref/close? %1 %2 1e-6) (:allocation allocation) [1.2 0.2]))))
    (let [plans (decision/compare-plans model posterior {:x [[0.7 0.7] [0.7 0.7]] :controls controls}
                                        {:x [[1.2 0.2] [1.2 0.2]] :controls controls})]
      (is (= 2 (:carryover-periods plans)))
      (is (ref/close? (get-in plans [:total :mean]) (* 8.0 (+ (* 0.7 (/ 0.5 2.0)) (* 0.4 (/ -0.5 3.0)))) 1e-10)))))

(defn transformed-lp [q]
  (let [a (mapv #(/ 1.0 (+ 1.0 (Math/exp (- %)))) (subvec q 0 2))
        theta (vec (concat a (map #(Math/exp %) (subvec q 2 4))
                           [(nth q 4) (Math/exp (nth q 5)) (nth q 6)]))
        jac (+ (reduce + (map #(+ (Math/log %) (Math/log (- 1.0 %))) a))
               (nth q 2) (nth q 3) (nth q 5))]
    (+ (flat-lp spec data theta) jac)))

(deftest no-saturation-transformed-gradient
  (let [model @flat]
    (doseq [backend [:runtime :aot]
            q [(m/initial-position model) [-1.2 0.7 -0.6 -0.2 0.3 -1.0 -0.1]]
            :let [f (block/capability (m/inference-block model backend) :value+grad)
                  [value grad] (f (double-array q) (:inputs model))]]
      (is (ref/close? value (transformed-lp q) 1e-9))
      (doseq [j (range (count q))]
        (let [fd (/ (- (transformed-lp (update q j + 1e-6))
                       (transformed-lp (update q j - 1e-6))) 2e-6)]
          (is (ref/close? (nth grad j) fd 1e-5)))))))

(deftest no-saturation-geo-transformed-gradient
  (let [model @geo-model
        reference (fn [q]
                    (let [a (/ 1.0 (+ 1.0 (Math/exp (- (nth q 0)))))
                          theta (-> q (assoc 0 a) (update 2 #(Math/exp %))
                                    (update 7 #(Math/exp %)) (update 8 #(Math/exp %)))
                          jac (+ (Math/log a) (Math/log (- 1.0 a))
                                 (nth q 2) (nth q 7) (nth q 8))]
                      (+ (geo-lp theta geo-data) jac)))
        q (m/initial-position model)
        [value grad] ((block/capability (:block model) :value+grad) (double-array q) (:inputs model))]
    (is (ref/close? value (reference q) 1e-9))
    (doseq [j (range (count q))]
      (let [fd (/ (- (reference (update q j + 1e-6)) (reference (update q j - 1e-6))) 2e-6)]
        (is (ref/close? (nth grad j) fd 1e-5))))))

(deftest packed-scan-layout-with-changed-row-count
  (let [model (m/build (m/prepare spec data) {:data-layout :packed :density-kernel :scan})]
    (doseq [d [data (-> data (update :x conj [1.7 0.05])
                        (update :controls conj [-0.8]) (update :y conj 1.6)
                        (assoc-in [:controls 1 0] 1.3))]]
      (let [prepared (m/prepare spec d)]
        (is (every? true? (map #(ref/close? %1 %2 1e-12)
                               (m/mean-response (assoc model :inputs (:inputs prepared)) theta)
                               (means spec d theta))))
        (check-runtime model theta prepared #(flat-lp spec d %))))))

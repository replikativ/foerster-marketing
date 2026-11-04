(ns org.replikativ.marketing.michaelis-menten-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.marketing.model :as m]
            [org.replikativ.marketing.hierarchy :as geo]
            [org.replikativ.marketing.layout :as layout]
            [org.replikativ.marketing.io :as io]
            [org.replikativ.marketing.decision :as decision]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.marketing.model-test :as ref]
            [org.replikativ.marketing.flexibility-test :as flex]
            [org.replikativ.marketing.response-test :as linear]))

(def spec (m/specification (-> linear/spec (assoc :saturation :michaelis-menten) (dissoc :priors))))
(def theta [0.3 0.6 0.7 0.4 0.8 1.1 0.2 0.3 -0.1])
(def data linear/data)
(def temporal-data linear/temporal-data)
;; Independent mathematical reference. No production response or layout helpers.
(defn means [spec data theta]
  (let [c (count (:channels spec)) l (:l-max spec) sx (get-in spec [:scaling :channel]) sy (get-in spec [:scaling :target])]
    (mapv (fn [t]
            (* sy (+ (nth theta (* 3 c))
                     (reduce + (map * (drop (+ (* 3 c) 2) theta) (nth (:controls data) t [])))
                     (reduce + (for [j (range c)
                                     :let [weights (mapv #(Math/pow (nth theta j) %) (range l))
                                           z (/ (reduce + (for [k (range l) :when (>= t k)]
                                                            (* (nth weights k) (/ (get-in data [:x (- t k) j]) (nth sx j)))))
                                                (reduce + weights))]]
                                 (* (nth theta (+ c j)) (/ z (+ (nth theta (+ (* 2 c) j)) z))))))))
          (range (count (:x data))))))
(defn delta [spec theta row]
  (let [p (fn [x] (means spec {:x x :controls (vec (repeat (count x) []))}
                         (assoc (vec (take (+ (* 3 (count (:channels spec))) 2) theta)) (* 3 (count (:channels spec))) 0.0)))]
    (reduce + (map #(* %1 (- %2 %3)) (:weights row) (p (:treatment row)) (p (:baseline row))))))
(defn evidence-lp [spec data theta]
  (let [c (count (:channels spec)) sy (get-in spec [:scaling :target])
        obs (reduce + (map #(ref/normal-lp (/ %1 sy) (/ %2 sy) (nth theta (inc (* 3 c)))) (:y data) (means spec data theta)))
        lift (reduce + (for [row (:lift data)
                             :let [j (.indexOf (:channels spec) (:channel row))
                                   f (fn [x] (let [z (/ x (get-in spec [:scaling :channel j]))]
                                               (* (nth theta (+ c j)) (/ z (+ (nth theta (+ (* 2 c) j)) z)))))]]
                         (ref/normal-lp (/ (:delta-y row) sy) (- (f (+ (:x row) (:delta-x row))) (f (:x row))) (/ (:sigma row) sy))))]
    (+ obs lift (if (seq (:experiments data))
                  (linear/joint-lp spec (mapv #(- (:estimate %) (delta spec theta %)) (:experiments data)) (:experiment-covariance data))
                  0.0))))
(defn flat-lp [spec data theta]
  (let [c (count (:channels spec)) groups [[:alpha (subvec theta 0 c)] [:beta (subvec theta c (* 2 c))]
                                           [:kappa (subvec theta (* 2 c) (* 3 c))] [:intercept [(nth theta (* 3 c))]]
                                           [:sigma [(nth theta (inc (* 3 c)))]] [:control (subvec theta (+ (* 3 c) 2))]]]
    (+ (evidence-lp spec data theta)
       (reduce + (for [[group values] groups [j x] (map-indexed vector values)
                       :let [prior (get-in spec [:priors group])]]
                   (flex/prior-lp x (if (vector? prior) (nth prior j) prior)))))))
(def flat (delay (m/build (m/prepare spec data))))
(def scan (delay (m/build (m/prepare spec data) {:density-kernel :scan})))
(def temporal (delay (m/build (m/prepare spec temporal-data))))
(defn check-aot [model theta prepared]
  (let [args ((:args model) (double-array theta) (:inputs prepared))
        actual (vec (apply @(:aot-gradient model) args)) [_ runtime] (apply @(:runtime-gradient model) args)]
    (is (every? true? (map #(ref/close? %1 %2 1e-10) (rest actual) runtime)))
    (is (ref/close? (first actual) (apply @(:density-var model) ((:args model) (double-array theta) (:inputs prepared))) 1e-10))))

(deftest kappa-layout-priors-and-wire-contract
  (is (= :michaelis-menten (:saturation spec)))
  (is (= spec (m/specification spec)))
  (is (= spec (m/specification (assoc spec :saturation "michaelis-menten"))))
  (is (= {:distribution :log-normal :mu 0.0 :sigma 0.7} (get-in spec [:priors :kappa])))
  (is (not (contains? (:priors spec) :lambda)))
  (is (= [:kappa :tv] (nth (:parameter-names @flat) 4)))
  (is (= 0.0 (nth (m/initial-position @flat) 4)))
  (is (= 9 (:dimension @flat)))
  (is (thrown? clojure.lang.ExceptionInfo (m/specification (assoc-in spec [:priors :lambda] {}))))
  (let [s (m/specification (assoc-in spec [:priors :kappa] [{:sigma 0.4} {:distribution :gamma :shape 2.0 :rate 1.0}]))]
    (is (= [:log-normal :gamma] (mapv :distribution (get-in s [:priors :kappa])))))
  (doseq [format [:edn :json]]
    (let [file (java.io.File/createTempFile "mmm-kappa-" ".data")
          posterior {:format :foerster-marketing/posterior :version 1 :spec spec :parameter-names (:parameter-names @flat) :chains [[theta]]}]
      (try (io/write! file posterior format)
           (is (= posterior (io/read! file format)))
           (is (thrown? clojure.lang.ExceptionInfo (io/validate (assoc-in posterior [:chains 0 0 4] 0.0))))
           (finally (.delete file))))))

(deftest independent-flat-and-packed-gradients
  (doseq [model [@flat @scan]]
    (is (every? true? (map #(ref/close? %1 %2 1e-12) (m/mean-response model theta) (means spec data theta))))
    (linear/check-runtime model theta (m/prepare spec data) #(flat-lp spec data %)))
  (check-aot @flat theta (m/prepare spec data))
  (doseq [d [temporal-data (-> temporal-data (update-in [:experiments 0 :treatment 3 0] + 0.3)
                               (assoc :experiment-covariance [[0.09 -0.01] [-0.01 0.16]])
                               (update :x conj [0.2 0.1]) (update :controls conj [0.3]) (update :y conj 1.1))]]
    (linear/check-runtime @temporal theta (m/prepare spec d) #(flat-lp spec d %))
    (check-aot @temporal theta (m/prepare spec d))))

(deftest packed-scalar-density-and-changed-data
  (let [spec (m/specification {:channels ["search" "social" "video"]
                               :controls ["demand" "promotion" "season-a" "season-b"] :l-max 2
                               :saturation :michaelis-menten})
        data {:x [[0.2 0.3 0.4] [0.5 0.2 0.1] [0.1 0.4 0.2]]
              :controls [[0.2 0.1 -0.1 0.3] [0.1 0.0 0.1 0.2] [0.3 0.2 -0.2 0.1]] :y [0.4 0.5 0.3]}
        theta [0.3 0.4 0.5 0.7 0.8 0.9 0.6 0.8 1.0 0.2 0.3 0.1 -0.1 0.05 0.02]
        built (m/build (m/prepare spec data) {:data-layout :packed :gradient-parameter-layout :scalar})]
    (is (= :scalar (:gradient-parameter-layout built)))
    (is (= 18 (quot (count (get-in built [:density-program :scalar-args])) 3)))
    (doseq [d [data (-> data (update :x conj [0.4 0.2 0.1])
                        (update :controls conj [0.1 0.2 0.0 -0.1]) (update :y conj 0.6))]
            :let [p (m/prepare spec d)
                  actual (vec (apply @(:aot-gradient built) ((:args built) (double-array theta) (:inputs p))))]]
      (is (= 16 (count actual)))
      (is (ref/close? (first actual) (flat-lp spec d theta) 1e-9))
      (doseq [j (range (count theta))]
        (let [fd (/ (- (flat-lp spec d (update theta j + 1e-6))
                       (flat-lp spec d (update theta j - 1e-6))) 2e-6)]
          (is (ref/close? (nth actual (inc j)) fd 1e-5)))))
    ;; Additional evidence exceeds scalar arity; retain the array path rather
    ;; than forcing a wider interface or dropping calibration factors.
    (let [lifted (assoc data :lift [{:type :effective-spend-contrast :channel "search"
                                     :x 0.2 :delta-x 0.1 :delta-y 0.05 :sigma 0.1}])
          automatic (m/build (m/prepare spec lifted) {:data-layout :packed})]
      (is (= :array (:gradient-parameter-layout automatic)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (m/build (m/prepare spec lifted) {:data-layout :packed :gradient-parameter-layout :scalar}))))))

(deftest kappa-means-half-ceiling-in-normalized-spend
  (let [d (assoc data :x (vec (repeat 6 [(* 2.0 0.8) (* 3.0 1.1)])) :controls (vec (repeat 6 [0.0])))
        model (m/build (m/prepare spec d))]
    (is (ref/close? (last (m/mean-response model theta)) (* 4.0 (+ 0.2 (* 0.5 (+ 0.7 0.4)))) 1e-12))))

(def geo-spec (geo/specification (-> linear/geo-spec (assoc :saturation :michaelis-menten) (dissoc :priors))))
(def geo-theta [0.4 0.8 -0.3 0.25 -0.2 0.3 0.1 0.2 0.3 0.4 -0.1])
(def geo-data linear/geo-data)
(def panel-spec (m/specification (dissoc geo-spec :groups :hierarchy :model)))
(def geo-model (delay (geo/build (geo/prepare geo-spec geo-data))))
(defn panel-theta [theta j]
  [(nth theta 0) (Math/exp (+ (nth theta 2) (* (nth theta 3) (nth theta (+ 4 j)))))
   (nth theta 1) (nth theta (+ 6 j)) (nth theta (+ 8 j)) (nth theta 10)])
(defn geo-lp [theta data]
  (+ (flex/prior-lp (first theta) (get-in panel-spec [:priors :alpha]))
     (flex/prior-lp (nth theta 1) (get-in panel-spec [:priors :kappa]))
     (ref/normal-lp (nth theta 2) (Math/log 0.5) 0.7)
     (ref/normal-lp (nth theta 3) 0.0 0.5) (Math/log 2.0)
     (reduce + (map #(ref/normal-lp % 0.0 1.0) (subvec theta 4 6)))
     (reduce + (map #(ref/normal-lp % 0.0 1.0) (subvec theta 6 8)))
     (reduce + (map #(+ (ref/normal-lp % 0.0 0.5) (Math/log 2.0)) (subvec theta 8 10)))
     (ref/normal-lp (nth theta 10) 0.0 1.0)
     (reduce + (for [[j group] (map-indexed vector [:west :east])]
                 (evidence-lp panel-spec (dissoc (get-in data [:panels group]) :experiments) (panel-theta theta j))))
     (linear/joint-lp geo-spec (mapv (fn [j group]
                                       (let [row (first (get-in data [:panels group :experiments]))]
                                         (- (:estimate row) (delta panel-spec (panel-theta theta j) row))))
                                     (range) [:west :east]) (:experiment-covariance data))))

(deftest shared-kappa-with-geo-amplitudes
  (let [model @geo-model]
    (is (= 11 (:dimension model) (count (m/initial-position model)) (count (:parameter-names model))))
    (is (= [:kappa :tv] (nth (:parameter-names model) 1)))
    (is (every? true? (map #(ref/close? %1 %2 1e-12) (m/mean-response model geo-theta)
                           (mapcat (fn [j group] (means panel-spec (get-in geo-data [:panels group]) (panel-theta geo-theta j)))
                                   (range) [:west :east]))))
    (doseq [d [geo-data (-> geo-data (update-in [:panels :east :lift 0 :delta-y] + 0.1)
                            (update-in [:panels :west :experiments 0 :treatment 4 0] + 0.3)
                            (assoc :experiment-covariance [[0.09 0.01] [0.01 0.16]]))]]
      (linear/check-runtime model geo-theta (geo/prepare geo-spec d) #(geo-lp % d)))
    (let [posterior {:format :foerster-marketing/posterior :version 1 :spec geo-spec :chains [[geo-theta]]}]
      (is (= posterior (io/validate posterior))))))

(deftest transformed-flat-and-geo-gradients
  (doseq [[model reference positive alpha]
          [[@flat #(flat-lp spec data %) [2 3 4 5 7] [0 1]]
           [@geo-model #(geo-lp % geo-data) [1 3 8 9] [0]]]
          :let [q (m/initial-position model)
                density (fn [q]
                          (let [th (reduce #(update %1 %2 (fn [x] (/ 1.0 (+ 1.0 (Math/exp (- x)))))) q alpha)
                                th (reduce #(update %1 %2 (fn [x] (Math/exp x))) th positive)
                                jac (+ (reduce + (map #(nth q %) positive))
                                       (reduce + (map #(+ (Math/log (nth th %)) (Math/log (- 1.0 (nth th %)))) alpha)))]
                            (+ (reference th) jac)))
                [value grad] ((block/capability (:block model) :value+grad) (double-array q) (:inputs model))]]
    (is (ref/close? value (density q) 1e-9))
    (doseq [j (range (count q))]
      (let [fd (/ (- (density (update q j + 1e-6)) (density (update q j - 1e-6))) 2e-6)]
        (is (ref/close? (nth grad j) fd 1e-5))))))

(deftest concave-allocation-agrees-with-independent-grid
  (let [model @flat posterior {:spec spec :chains [[theta]]} h 3 controls [[0.2] [0.1] [0.3]]
        history (vec (take-last 2 (:x data)))
        oracle (fn [budgets]
                 (reduce + (drop 2 (means spec {:x (vec (concat history (repeat h budgets) (repeat 2 [0.0 0.0])))
                                                :controls (vec (concat (take-last 2 (:controls data)) controls (repeat 2 [0.0])))} theta))))
        f (decision/allocation-objective model posterior h controls) value (f [0.6 0.8])
        plan (decision/allocate model posterior {:budget 1.4 :horizon h :controls controls :tolerance 1e-6})
        grid (apply max (map #(oracle [(/ % 1000.0) (- 1.4 (/ % 1000.0))]) (range 1401)))]
    (is (ref/close? (:value value) (oracle [0.6 0.8]) 1e-10))
    (doseq [j [0 1]]
      (let [fd (/ (- (oracle (update [0.6 0.8] j + 1e-6)) (oracle (update [0.6 0.8] j - 1e-6))) 2e-6)]
        (is (ref/close? (nth (:gradient value) j) fd 1e-5))))
    (is (:converged? plan))
    (is (>= (:expected-response plan) (- grid 1e-6)))
    (let [a {:x (vec (repeat h [0.6 0.8])) :controls controls}
          b {:x (vec (repeat h (:allocation plan))) :controls controls}
          compared (decision/compare-plans model posterior a b)]
      (is (ref/close? (get-in compared [:total :mean]) (- (oracle (:allocation plan)) (oracle [0.6 0.8])) 1e-10)))))

(ns org.replikativ.marketing.decision-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.marketing.model :as model]
            [org.replikativ.marketing.decision :as decision]
            [org.replikativ.marketing.model-test :as reference]
            [raster.ad.reverse :as rev]))

(defn posterior [m]
  {:spec (:spec m) :parameter-names (:parameter-names m) :chains [[reference/theta (assoc reference/theta 2 0.8)]]})

(deftest response-gradient-and-global-allocation
  (let [m @reference/fitted-model p (posterior m)
        controls [[0.8] [0.9] [1.0] [1.1]]
        f (decision/allocation-objective m p 4 controls)
        budget [0.6 0.4] result (f budget)]
    (doseq [j [0 1]]
      (let [fd (/ (- (:value (f (update budget j + 1e-6)))
                     (:value (f (update budget j - 1e-6)))) 2e-6)]
        (is (reference/close? fd (nth (:gradient result) j) 1e-6))))
    (let [answer (decision/allocate m p {:budget 1.0 :horizon 4 :controls controls
                                         :lower [0.1 0.1] :upper [0.8 0.9] :tolerance 1e-6})
          grid (mapv #(vector % (- 1.0 %)) (map #(/ % 100.0) (range 10 81)))]
      (is (:converged? answer))
      (is (reference/close? 1.0 (reduce + (:allocation answer)) 1e-12))
      (is (every? true? (map <= [0.1 0.1] (:allocation answer) [0.8 0.9])))
      (is (>= (:expected-response answer) (- (apply max (map (comp :value f) grid)) 1e-8))))))

(deftest paired-scenarios-match-independent-history-and-tail
  (let [m @reference/fitted-model p (posterior m)
        controls [[0.8] [0.9]]
        baseline {:x [[0.2 0.3] [0.2 0.3]] :controls controls}
        candidate {:x [[0.6 0.1] [0.6 0.1]] :controls controls}
        actual (decision/compare-plans m p baseline candidate)
        make-data (fn [plan]
                    {:x (vec (concat (get-in m [:data :x]) (:x plan) (repeat 3 [0.0 0.0])))
                     :controls (vec (concat (get-in m [:data :controls]) controls (repeat 3 [0.0])))})
        expected (mapv (fn [theta]
                         (reduce + (map -
                                        (drop 8 (reference/reference-mean (:spec m) (make-data candidate) theta))
                                        (drop 8 (reference/reference-mean (:spec m) (make-data baseline) theta)))))
                       (mapcat identity (:chains p)))]
    (is (= 3 (:carryover-periods actual)))
    (is (every? true? (map #(reference/close? %1 %2 1e-10) expected (map :total (:draws actual)))))
    (is (zero? (get-in (decision/compare-plans m p baseline baseline) [:total :mean])))))

(deftest refuse-infeasible-or-mismatched-requests
  (let [m @reference/fitted-model p (posterior m)]
    (is (thrown? clojure.lang.ExceptionInfo
                 (decision/allocate m p {:budget 1.0 :horizon 2 :lower [0.6 0.6]})))
    (is (thrown? clojure.lang.ExceptionInfo
                 (decision/allocate m p {:budget 1.0 :horizon 2 :upper [0.2 0.2]})))
    (is (thrown? clojure.lang.ExceptionInfo
                 (decision/allocation-objective m (assoc p :spec {}) 2 [[0.0] [0.0]])))))

(deftest allocation-is-independent-of-spend-units
  (let [m @reference/fitted-model p (posterior m)
        controls [[0.8] [0.9]]
        spec (model/specification (assoc (select-keys (:spec m) [:channels :controls :l-max])
                                         :scaling {:channel [2000.0 3000.0] :target 2.0}))
        data (-> reference/data
                 (update :x #(mapv (fn [row] (mapv (fn [x] (* 1000.0 x)) row)) %))
                 (update :lift #(mapv (fn [row] (-> row (update :x * 1000.0) (update :delta-x * 1000.0))) %)))
        scaled (model/build (model/prepare spec data))
        a (decision/allocate m p {:budget 1.0 :horizon 2 :controls controls})
        b (decision/allocate scaled (assoc p :spec spec) {:budget 1000.0 :horizon 2 :controls controls})]
    (is (and (:converged? a) (:converged? b)))
    (is (every? true? (map #(reference/close? %1 (/ %2 1000.0) 1e-6) (:allocation a) (:allocation b))))
    (is (reference/close? (:expected-response a) (:expected-response b) 1e-8))))

(deftest invalid-tolerance-is-rejected-before-objective-preparation
  (let [m @reference/fitted-model p (posterior m)]
    (with-redefs [decision/allocation-objective
                  (fn [& _] (throw (AssertionError. "Invalid tolerance reached objective preparation")))]
      (doseq [tolerance [Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY Double/NaN 0.0 -1.0 nil "0.1"]]
        (is (thrown? clojure.lang.ExceptionInfo
                     (decision/allocate m p {:budget 1.0 :horizon 2 :tolerance tolerance})))))))

(deftest invalid-posterior-is-rejected-before-differentiation
  (let [m @reference/fitted-model p (posterior m)
        bad-positions (mapv (fn [[index value]] (assoc reference/theta index value))
                            [[0 0.0] [0 1.0] [2 -0.1] [4 0.0] [7 0.0]
                             [6 Double/NaN] [8 Double/POSITIVE_INFINITY]])]
    (with-redefs [rev/value+grad
                  (fn [& _] (throw (AssertionError. "Invalid posterior reached differentiation")))]
      (doseq [theta bad-positions]
        (is (thrown? clojure.lang.ExceptionInfo
                     (decision/allocation-objective m (assoc p :chains [[theta]]) 2 [[0.0] [0.0]]))))
      (doseq [names [false [] (vec (reverse (:parameter-names m)))]]
        (is (thrown? clojure.lang.ExceptionInfo
                     (decision/allocation-objective m (assoc p :parameter-names names)
                                                    2 [[0.0] [0.0]])))))))

(deftest allocation-is-independent-of-response-and-currency-units
  ;; Strictly concave analytic oracle. At x+y=2 solve
  ;; 4/(1+4*x) = 2/(1+y): x=5/6, y=7/6.
  (let [answer (fn [response-unit spend-unit]
                 (let [m {:spec {:channels [:a :b] :scaling {:target response-unit}}}
                       objective (fn [budgets]
                                   (let [[x y] (mapv #(/ % spend-unit) budgets)]
                                     {:value (* response-unit (+ (Math/log1p (* 4.0 x)) (* 2.0 (Math/log1p y))))
                                      :gradient (mapv #(* (/ response-unit spend-unit) %)
                                                      [(/ 4.0 (+ 1.0 (* 4.0 x))) (/ 2.0 (+ 1.0 y))])}))]
                   (with-redefs [decision/allocation-objective (fn [& _] objective)]
                     (decision/allocate m {} {:budget (* 2.0 spend-unit) :horizon 1
                                              :lower [(* 0.1 spend-unit) (* 0.1 spend-unit)]
                                              :upper [(* 1.9 spend-unit) (* 1.9 spend-unit)]
                                              :tolerance 1e-8 :max-iterations 500}))))
        expected [(/ 5.0 6.0) (/ 7.0 6.0)]
        reference (answer 1.0 1.0)]
    (doseq [response-unit [1e-9 1.0 1e6] spend-unit [1e-3 1.0 1000.0]
            :let [result (answer response-unit spend-unit)]]
      (is (:converged? result))
      (is (< (:projected-gradient-residual result) 1e-8))
      (is (every? true? (map #(reference/close? %1 (/ %2 spend-unit) 1e-7) expected (:allocation result))))
      (is (reference/close? (:expected-response reference) (/ (:expected-response result) response-unit) 1e-10))
      (is (every? true? (map #(reference/close? %1 (* %2 (/ spend-unit response-unit)) 1e-7)
                             (:gradient reference) (:gradient result)))))))

(deftest thinning-keeps-evenly-spaced-draws-per-chain
  (let [chain (fn [offset n] (mapv #(vector (+ offset %)) (range n)))
        p {:chains [(chain 0 10) (chain 100 10)] :spec :unchanged}]
    (is (= p (decision/thin p nil)))
    (is (= p (decision/thin p 40)))
    ;; Two chains share 8 draws: 4 per chain, every ceil(10/4)=3rd draw.
    (is (= {:chains [[[0] [3] [6] [9]] [[100] [103] [106] [109]]] :spec :unchanged}
           (decision/thin p 8)))
    ;; Same rule as the study's 256 draws per chain of 4,000 (step 16).
    (is (= [250 250 250 250]
           (mapv count (:chains (decision/thin {:chains (vec (repeat 4 (chain 0 4000)))} 1024)))))
    (doseq [bad [0 1 -4 2.5 "8"]]
      (is (thrown? clojure.lang.ExceptionInfo (decision/thin p bad))))))

(deftest plan-responses-match-independent-paths-and-pair-with-contrasts
  (let [m @reference/fitted-model p (posterior m)
        controls [[0.8] [0.9]]
        baseline {:x [[0.2 0.3] [0.2 0.3]] :controls controls}
        candidate {:x [[0.6 0.1] [0.6 0.1]] :controls controls}
        responses (decision/plan-responses m p candidate)
        expected (mapv (fn [theta]
                         (vec (drop 8 (reference/reference-mean
                                       (:spec m)
                                       {:x (vec (concat (get-in m [:data :x]) (:x candidate) (repeat 3 [0.0 0.0])))
                                        :controls (vec (concat (get-in m [:data :controls]) controls (repeat 3 [0.0])))}
                                       theta))))
                       (mapcat identity (:chains p)))]
    (is (= {:distribution :posterior-conditional-mean :horizon 2 :carryover-periods 3}
           (dissoc responses :draws)))
    (is (= 2 (count (:draws responses))))
    (is (every? #(= 5 (count %)) (:draws responses)))
    (is (every? true? (map #(reference/close? %1 %2 1e-10)
                           (mapcat identity expected) (mapcat identity (:draws responses)))))
    (let [base (decision/plan-responses m p baseline)
          contrast (decision/compare-plans m p baseline candidate)]
      (is (= (mapv #(reduce + (map - %1 %2)) (:draws responses) (:draws base))
             (mapv :total (:draws contrast)))))
    (is (= 1 (count (:draws (decision/plan-responses m p candidate {:max-draws 1})))))
    (is (thrown? clojure.lang.ExceptionInfo (decision/plan-responses m p {:x [] :controls []})))))

(deftest max-draws-option-equals-explicit-thinning
  (let [m @reference/fitted-model
        p (assoc (posterior m) :chains [[reference/theta (assoc reference/theta 2 0.8) (assoc reference/theta 2 1.5)]])
        thinned (decision/thin p 2)
        controls [[0.8] [0.9]]
        options {:budget 1.0 :horizon 2 :controls controls :lower [0.1 0.1] :upper [0.8 0.9]}
        baseline {:x [[0.2 0.3] [0.2 0.3]] :controls controls}
        candidate {:x [[0.6 0.1] [0.6 0.1]] :controls controls}]
    (is (= 2 (count (first (:chains thinned)))))
    (is (= (decision/allocate m thinned options)
           (decision/allocate m p (assoc options :max-draws 2))))
    (is (= (decision/compare-plans m thinned baseline candidate)
           (decision/compare-plans m p baseline candidate {:max-draws 2})))))

(ns org.replikativ.marketing.compiled-gradient-test
  (:require [clojure.data.json :as json]
            [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [org.replikativ.marketing.model :as model]
            [org.replikativ.marketing.gradient :as gradient]
            [org.replikativ.marketing.model-test :as reference]
            [raster.ad.reverse :as rev]
            [org.replikativ.foerster.block :as block]))

(defn- load-fixture
  "Flat model fixture (spec, data with lift rows, two parameter points)."
  []
  (walk/postwalk (fn [x] (if (and (map? x) (= "effective-spend-contrast" (:type x)))
                           (assoc x :type :effective-spend-contrast) x))
                 (json/read-str (slurp "test/fixtures/flat-mmm.json") :key-fn keyword)))

(deftest invoked-aot-gradient-on-marketing-fixture
  ;; Compile once, then vary parameters and runtime data. Check against an
  ;; independent convolution/density, not just the same AD in another backend.
  (let [fixture (load-fixture)
        m (model/build (model/prepare (model/specification (:spec fixture)) (:data fixture)))
        f @(:aot-gradient m)
        array-f (gradient/compile-gradient m :parameter-layout :array)
        aot-block (model/inference-block m :aot)
        runtime (rev/value+grad (:density-var m) :wrt [0])
        changed (-> (:data m)
                    (update :x #(mapv (fn [row] (update row 0 + 0.07)) %))
                    (update :y #(mapv (fn [y] (+ y 0.02)) %))
                    (update :controls #(mapv (fn [row] (mapv (fn [u] (* u 0.9)) row)) %))
                    (update :lift #(mapv (fn [row] (update row :delta-y + 0.01)) %)))
        inputs-for #(get (model/prepare (:spec m) %) :inputs)
        invoke (fn [theta data]
                 (vec (apply f ((:args m) (double-array theta) (inputs-for data)))))]
    (is (= :scalar (:gradient-parameter-layout m)))
    (doseq [data [(:data m) changed] theta (:points fixture)]
      (let [args ((:args m) (double-array theta) (inputs-for data))
            [v grad] (apply runtime args)
            actual (invoke theta data)
            c (count (get-in m [:spec :channels]))
            q (mapv (fn [j x]
                      (cond (< j c) (Math/log (/ x (- 1.0 x)))
                            (or (< j (* 3 c)) (= j (inc (* 3 c)))) (Math/log x)
                            :else x)) (range) theta)
            transformed ((block/capability aot-block :value+grad) (double-array q) (inputs-for data))
            ref-model (assoc m :data data)]
        (is (= (inc (:dimension m)) (count actual)))
        (is (every? true? (map #(reference/close? %1 %2 1e-9) actual (vec (apply array-f args)))))
        (is (reference/close? (first transformed) (reference/transformed-reference ref-model q) 1e-9))
        (doseq [j (range (:dimension m))]
          (let [fd (/ (- (reference/transformed-reference ref-model (update q j + 1e-6))
                         (reference/transformed-reference ref-model (update q j - 1e-6))) 2e-6)]
            (is (reference/close? (aget ^doubles (second transformed) j) fd 1e-5)
                (str "transformed coordinate " j))))
        (is (reference/close? (first actual) (reference/reference-lp (:spec m) data theta) 1e-9))
        (is (every? true? (map #(reference/close? %1 %2 1e-9)
                               actual (cons v (vec grad)))))
        (doseq [j (range (:dimension m))]
          (let [fd (/ (- (reference/reference-lp (:spec m) data (update theta j + 1e-6))
                         (reference/reference-lp (:spec m) data (update theta j - 1e-6))) 2e-6)]
            (is (reference/close? (nth actual (inc j)) fd 1e-5)
                (str "natural coordinate " j))))))
    (is (not= (invoke (first (:points fixture)) (:data m))
              (invoke (first (:points fixture)) changed)))))

(deftest large-model-keeps-array-ad
  (let [fixture (load-fixture)
        spec (-> (:spec fixture)
                 (assoc :channels ["tv" "search" "display" "radio"])
                 (assoc-in [:scaling :channel] [1.0 1.0 1.0 1.0]))
        data (update (:data fixture) :x #(mapv (fn [row] (into row row)) %))
        m (model/build (model/prepare (model/specification spec) data))]
    (testing "An otherwise valid larger model does not exceed JVM scalar arity"
      (is (= :array (:gradient-parameter-layout m)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Scalar AD signature"
                            (gradient/parameter-layout m :scalar)))
      (is (Double/isFinite (double (apply @(:density-var m)
                                          ((:args m) (double-array [0.4 0.5 0.3 0.6 0.6 0.4 0.3 0.2 2.0 1.5 1.8 2.1 0.5 0.1 0.15]) (:inputs m)))))))))

(deftest pruning-density-inputs-preserves-automatic-gradient-choice
  (let [spec (model/specification {:channels [:tv :search] :controls [:a :b :c] :l-max 2})
        data {:x [[0.2 0.3] [0.4 0.1]] :controls [[0.0 0.1 0.2] [0.1 0.0 -0.1]] :y [0.4 0.5]}
        built (model/build (model/prepare spec data))]
    ;; The complete model interface historically exceeds scalar arity, while
    ;; the density alone now fits after removing unused weights/empty lifts.
    (is (= :array (:gradient-parameter-layout built)))
    (is (= :scalar (gradient/parameter-layout built :scalar)))))

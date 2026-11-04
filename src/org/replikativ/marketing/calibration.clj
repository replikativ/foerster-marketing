(ns ^:no-doc org.replikativ.marketing.calibration
  "Window-matched treatment/control trajectories with fixed joint measurement covariance.

  Implementation detail of the public `model`, `hierarchy`, `inference`,
  `decision` and `io` namespaces; not public API and may change without notice."
  (:require [org.replikativ.marketing.config :as config]
            [org.replikativ.marketing.response :as response]
            [org.replikativ.marketing.source :refer [balanced-sum mul sub div log read-at normal]])
  (:import [org.apache.commons.math3.linear Array2DRowRealMatrix CholeskyDecomposition]))

(defn prepare [spec experiments covariance]
  (let [rows (vec experiments) e (count rows) c (count (:channels spec)) l (:l-max spec)
        sx (get-in spec [:scaling :channel]) sy (get-in spec [:scaling :target])]
    (config/require! (= e (count (set (map :id rows)))) "Experiment ids must be unique" {})
    (doseq [{:keys [id baseline treatment weights estimate sigma independent-of-training?]} rows
            :let [n (count baseline)]]
      (config/require! (and id (true? independent-of-training?)
                            (>= n l) (= n (count treatment) (count weights))
                            (every? #(= c (count %)) (concat baseline treatment))
                            (every? #(and (config/finite? %) (not (neg? %))) (mapcat identity (concat baseline treatment)))
                            (every? #(and (config/finite? %) (not (neg? %))) weights)
                            (some pos? weights)
                            (every? zero? (take (dec l) weights))
                            (= (mapv vec (take (dec l) baseline)) (mapv vec (take (dec l) treatment)))
                            (config/finite? estimate)
                            (or covariance (and (config/finite? sigma) (pos? sigma))))
                       "Invalid time-path experiment; provide shared history, an explicit window and independent external evidence"
                       {:id id}))
    (config/require! (or (nil? covariance) (pos? e)) "Covariance requires experiments" {})
    (let [cov (if covariance (mapv vec covariance)
                  (mapv (fn [i] (mapv #(if (= i %) (let [s (:sigma (nth rows i))] (* s s)) 0.0) (range e))) (range e)))
          _ (config/require! (and (= e (count cov)) (every? #(= e (count %)) cov)
                                  (every? config/finite? (mapcat identity cov)))
                             "Experiment covariance must be finite and square" {})
          lower (when (pos? e)
                  (try
                    (.getData (.getL (CholeskyDecomposition.
                                      (Array2DRowRealMatrix. (into-array (map double-array cov))) 1e-12 0.0)))
                    (catch Exception cause
                      (throw (ex-info "Experiment covariance must be symmetric positive definite" {} cause)))))
          packed (reduce
                  (fn [{:keys [values layout]} {:keys [baseline treatment weights estimate group] :as row}]
                    (let [n (count baseline) start (count values)
                          lagged (fn [x] (for [j (range c) t (range n) k (range l)]
                                           (if (>= t k) (/ (nth (nth x (- t k)) j) (nth sx j)) 0.0)))
                          descriptor {:periods n :estimate-offset start :weights-offset (inc start)
                                      :baseline-offset (+ start 1 n) :treatment-offset (+ start 1 n (* c n l))
                                      :group group}]
                      {:values (into values (concat [(/ estimate sy)] weights (lagged (mapv vec baseline)) (lagged (mapv vec treatment))))
                       :layout (conj layout descriptor)})) {:values [] :layout []} rows)
          covariance-offset (count (:values packed))]
      {:layout (mapv #(assoc % :cholesky-offset covariance-offset) (:layout packed))
       :data (double-array (concat (:values packed) (for [i (range e) j (range e)] (/ (aget ^doubles (aget lower i) j) sy))))
       :uncertainty (if covariance :joint-normal :independent-normal)})))

(defn expression
  "Emit one joint normalized likelihood. Amplitudes can depend on an experiment's geo.
  Coefficients encode normalized lag weights in the selected response coordinates."
  [spec layout amplitude shapes coefficients]
  (let [e (count layout) c (count coefficients) l (count (first coefficients))
        deltas (mapv #(symbol (str "experiment_delta_" %)) (range e))
        residuals (mapv #(symbol (str "experiment_residual_" %)) (range e))
        data 'experiments
        lower (fn [i j] (read-at data (+ (:cholesky-offset (first layout)) (* i e) j)))
        delta-bindings
        (mapcat
         (fn [j {:keys [periods weights-offset baseline-offset treatment-offset] :as descriptor}]
           (let [t (symbol (str "experiment_time_" j)) acc (symbol (str "experiment_sum_" j))
                 curve (fn [offset ch]
                         (response/weighted spec (amplitude descriptor ch) (nth shapes ch)
                                            (apply balanced-sum (for [k (range l)]
                                                                  (mul (get-in coefficients [ch k])
                                                                       (read-at data (list '+ (+ offset (* ch periods l) k) (list '* t l))))))))
                 effect (apply balanced-sum (for [ch (range c)] (sub (curve treatment-offset ch) (curve baseline-offset ch))))]
             [(nth deltas j) (list 'raster.par/reduce acc 0.0 t periods
                                   (balanced-sum acc (mul (read-at data (list '+ weights-offset t)) effect)))]))
         (range e) layout)
        residual-bindings
        (mapcat (fn [i descriptor]
                  [(nth residuals i)
                   (div (sub (sub (read-at data (:estimate-offset descriptor)) (nth deltas i))
                             (apply balanced-sum (for [j (range i)] (mul (lower i j) (nth residuals j))))) (lower i i))])
                (range e) layout)]
    (if (zero? e) 0.0
        (list 'let (vec (concat delta-bindings residual-bindings))
              (apply balanced-sum (* e -0.9189385332046727)
                     (concat (for [i (range e)] (sub 0.0 (log (lower i i))))
                             (map #(mul -0.5 (mul % %)) residuals)))))))

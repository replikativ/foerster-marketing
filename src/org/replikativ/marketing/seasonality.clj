(ns ^:no-doc org.replikativ.marketing.seasonality
  "Fourier regressors indexed by explicit absolute model periods.

  Implementation detail of the public `model`, `hierarchy`, `inference`,
  `decision` and `io` namespaces; not public API and may change without notice."
  (:require [org.replikativ.marketing.config :as config]))

(defn normalize [setting]
  (when setting
    (config/require! (and (map? setting) (= #{:period :order} (set (keys setting)))
                          (config/finite? (:period setting)) (integer? (:order setting))
                          (pos? (:order setting)) (> (:period setting) (* 2 (:order setting))))
                     "Seasonality requires period > 2*order and positive integer order" {:seasonality setting})
    {:period (double (:period setting)) :order (:order setting)}))

(defn column-names [setting]
  (vec (mapcat (fn [k] [(str "seasonality/sin-" k)
                        (str "seasonality/cos-" k)])
               (range 1 (inc (:order setting 0))))))

(defn validate-index! [index n required?]
  (config/require! (or (not required?) (some? index)) "Seasonality requires explicit absolute :time-index" {})
  (when index
    (config/require! (and (= n (count index)) (every? config/finite? index)
                          (every? (fn [[a b]] (< (Math/abs (- (- (double b) (double a)) 1.0)) 1e-9))
                                  (partition 2 1 index)))
                     "Time indices must match rows and advance by one model period" {})))

(defn basis [setting index]
  (mapv (fn [t]
          (vec (mapcat (fn [k]
                         (let [phase (* 2.0 Math/PI k (/ (Math/IEEEremainder (double t) (:period setting))
                                                         (:period setting)))]
                           [(Math/sin phase) (Math/cos phase)]))
                       (range 1 (inc (:order setting)))))) index))

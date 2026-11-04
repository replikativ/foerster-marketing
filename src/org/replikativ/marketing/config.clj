(ns ^:no-doc org.replikativ.marketing.config
  "Validated distribution configuration in fixed, scaled model coordinates.

  Implementation detail of the public `model`, `hierarchy`, `inference`,
  `decision` and `io` namespaces; not public API and may change without notice."
  (:require [org.replikativ.marketing.family :as response-family])
  (:import [org.apache.commons.math3.special Gamma]))

(def default-priors
  {:alpha {:distribution :beta :alpha 2.0 :beta 2.0}
   :beta {:distribution :half-normal :sigma 1.0}
   :lambda {:distribution :gamma :shape 3.0 :rate 1.0}
   :intercept {:distribution :normal :mu 0.0 :sigma 1.0}
   :sigma {:distribution :half-normal :sigma 0.5}
   :control {:distribution :normal :mu 0.0 :sigma 1.0}})

(defn finite? [x] (and (number? x) (Double/isFinite (double x))))
(defn require! [condition message data]
  (when-not condition (throw (ex-info message data))))
(defn- family [x] (if (string? x) (keyword x) x))
(def defaults
  {:normal {:mu 0.0 :sigma 1.0} :half-normal {:sigma 1.0}
   :gamma {:shape 3.0 :rate 1.0} :beta {:alpha 2.0 :beta 2.0}
   :log-normal {:mu 0.0 :sigma 1.0}})
(def allowed
  {:alpha #{:beta} :intercept #{:normal} :control #{:normal}
   :beta #{:half-normal :gamma :log-normal}
   :lambda #{:half-normal :gamma :log-normal}
   :sigma #{:half-normal :gamma :log-normal}})

(defn- log-gamma [x] (Gamma/logGamma (double x)))
(defn log-normalizer [prior]
  (case (:distribution prior)
    :beta (- (log-gamma (+ (:alpha prior) (:beta prior)))
             (log-gamma (:alpha prior)) (log-gamma (:beta prior)))
    :gamma (- (* (:shape prior) (Math/log (:rate prior))) (log-gamma (:shape prior)))
    0.0))

(defn- normalize-prior [group base override]
  (require! (map? override) "Prior must be a map" {:group group})
  (let [dist (family (:distribution override (:distribution base)))
        parameters (get defaults dist)
        _ (require! (contains? (get allowed group (get allowed :beta)) dist) "Prior family does not match latent support"
                    {:group group :distribution dist :allowed (get allowed group (get allowed :beta))})
        _ (require! (every? (conj (set (keys parameters)) :distribution) (keys override))
                    "Unknown prior parameter" {:group group :prior override})
        prior (merge (if (= dist (:distribution base)) base (assoc parameters :distribution dist))
                     override {:distribution dist})]
    (doseq [key (keys parameters)]
      (require! (and (finite? (get prior key)) (or (= key :mu) (pos? (get prior key))))
                "Prior parameters must be finite, with positive scales/shapes/rates" {:group group :prior prior}))
    (require! (finite? (log-normalizer prior)) "Prior normalization is not finite" {:group group :prior prior})
    (reduce #(update %1 %2 double) prior (keys parameters))))

(defn priors
  ([overrides channels controls] (priors overrides channels controls :logistic))
  ([overrides channels controls saturation]
   (let [{:keys [shape prior]} (response-family/description saturation)
         active-priors (cond-> (dissoc default-priors :lambda) shape (assoc shape prior))]
     (require! (or (nil? overrides) (map? overrides)) "Priors must be a map" {})
     (require! (every? (set (keys active-priors)) (keys overrides)) "Unknown prior group" {})
     (into {} (for [group (keys active-priors)
                    :let [setting (get overrides group {})
                          size (case group :control controls (:intercept :sigma) 1 channels)]]
                [group (if (vector? setting)
                         (do (require! (and (not (#{:intercept :sigma} group)) (= size (count setting)))
                                       "Prior vector must match coordinate count" {:group group :expected size})
                             (mapv #(normalize-prior group (get active-priors group) %) setting))
                         (normalize-prior group (get active-priors group) setting))])))))

(defn group-priors [spec group size]
  (let [setting (get-in spec [:priors group])]
    (if (vector? setting) setting (vec (repeat size setting)))))

(defn student-normalizer [nu]
  (- (log-gamma (/ (+ nu 1.0) 2.0)) (log-gamma (/ nu 2.0))
     (* 0.5 (+ (Math/log nu) (Math/log Math/PI)))))

(defn likelihood [setting]
  (let [setting (or setting {:distribution :normal})
        _ (require! (map? setting) "Likelihood must be a map" {})
        dist (family (:distribution setting))]
    (case dist
      :normal (do (require! (= #{:distribution} (set (keys setting))) "Unknown normal likelihood option" {})
                  {:distribution :normal})
      :student-t (do (require! (and (= #{:distribution :nu} (set (keys setting)))
                                    (finite? (:nu setting)) (> (:nu setting) 2.0)
                                    (finite? (student-normalizer (:nu setting))))
                               "Student-t requires fixed finite nu > 2" {:likelihood setting})
                     {:distribution :student-t :nu (double (:nu setting))})
      (throw (ex-info "Unsupported likelihood" {:likelihood setting})))))


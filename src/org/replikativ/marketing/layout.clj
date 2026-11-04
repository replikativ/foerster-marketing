(ns ^:no-doc org.replikativ.marketing.layout
  "Explicit latent offsets, supports and named coordinates shared by inference and reports.

  Implementation detail of the public `model`, `hierarchy`, `inference`,
  `decision` and `io` namespaces; not public API and may change without notice."
  (:require [org.replikativ.marketing.family :as family]))

(defn describe [latents]
  (let [entries (second
                 (reduce (fn [[offset entries] {:keys [shape] :as latent}]
                           (let [size (reduce * 1 shape)]
                             [(+ offset size) (conj entries (assoc latent :offset offset :size size))]))
                         [0 []] latents))]
    {:dimension (reduce + (map :size entries)) :latents entries
     :block {:block/id :mmm :block/target :complete-conditional :block/coordinates :constrained
             :block/latents (mapv #(select-keys % [:name :shape :support]) entries)}}))
(defn offsets [layout] (into {} (map (juxt :name :offset) (:latents layout))))
(defn names [layout]
  (vec (mapcat (fn [{:keys [name coordinates]}]
                 (if (seq coordinates)
                   (map #(into [name] %) (reduce (fn [prefixes labels]
                                                   (vec (for [prefix prefixes label labels] (conj prefix label)))) [[]] coordinates))
                   [[name]])) (:latents layout))))

(defn flat
  ([channels controls] (flat channels controls :logistic))
  ([channels controls saturation]
   (let [c (count channels) p (count controls)]
     (describe (concat [{:name :alpha :shape [c] :support [:interval 0.0 1.0] :coordinates [channels]}
                        {:name :beta :shape [c] :support :positive :coordinates [channels]}]
                       (family/shape-latents saturation channels)
                       [{:name :intercept :shape [] :support :real}
                        {:name :sigma :shape [] :support :positive}
                        {:name :control :shape [p] :support :real :coordinates [controls]}])))))
(defn initial-position
  "Expand a named map of unconstrained initial values in descriptor order."
  [layout values]
  (vec (mapcat (fn [{:keys [name size]}]
                 (when-not (contains? values name)
                   (throw (ex-info "Missing named initial value" {:name name})))
                 (repeat size (get values name))) (:latents layout))))

(defn valid-position? [layout values]
  (and (= (:dimension layout) (count values))
       (every? (fn [{:keys [offset size support]}]
                 (every? (fn [x] (and (number? x) (Double/isFinite (double x))
                                      (cond (= support :positive) (pos? x)
                                            (= support :real) true
                                            :else (let [[_ lo hi] support] (< lo x hi)))))
                         (take size (drop offset values)))) (:latents layout))))

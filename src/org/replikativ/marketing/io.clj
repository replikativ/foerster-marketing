(ns org.replikativ.marketing.io
  "Versioned EDN and lossless tagged JSON posterior persistence.
  Tagged JSON preserves keyword identifiers and nonfinite diagnostics explicitly."
  (:require [clojure.edn :as edn]
            [clojure.data.json :as json]
            [clojure.string :as str]
            [org.replikativ.marketing.layout :as layout]
            [org.replikativ.marketing.model :as model]
            [org.replikativ.marketing.hierarchy :as geo]
            [org.replikativ.marketing.config :as config]))

(defn- encode [x]
  (cond
    (keyword? x) {"$keyword" (subs (str x) 1)}
    (and (number? x) (not (Double/isFinite (double x)))) {"$nonfinite" (str x)}
    (map? x) (into {} (for [[k v] x]
                        [(cond (keyword? k) (str k) (string? k) (str "=" k)
                               :else (throw (ex-info "Posterior map keys must be keywords or strings" {:key k}))) (encode v)]))
    (sequential? x) (mapv encode x)
    :else x))
(defn- decode [x]
  (cond
    (and (map? x) (= #{"$keyword"} (set (keys x)))) (keyword (get x "$keyword"))
    (and (map? x) (= #{"$nonfinite"} (set (keys x))))
    (case (get x "$nonfinite") "NaN" Double/NaN "Infinity" Double/POSITIVE_INFINITY "-Infinity" Double/NEGATIVE_INFINITY
          (throw (ex-info "Unknown nonfinite diagnostic encoding" {})))
    (map? x) (into {} (for [[k v] x]
                        [(cond (str/starts-with? k ":") (keyword (subs k 1))
                               (str/starts-with? k "=") (subs k 1)
                               :else (throw (ex-info "Unknown tagged JSON key encoding" {:key k}))) (decode v)]))
    (vector? x) (mapv decode x)
    :else x))
(defn validate [posterior]
  (config/require! (and (= :foerster-marketing/posterior (:format posterior)) (= 1 (:version posterior)))
                   "Unknown posterior format/version" {})
  (let [spec ((if (= :hierarchical-geo (get-in posterior [:spec :model])) geo/specification model/specification) (:spec posterior))
        latent-layout (if (= :hierarchical-geo (:model spec)) (geo/parameter-layout spec)
                          (layout/flat (:channels spec) (model/control-names spec) (:saturation spec :logistic)))
        chains (:chains posterior)]
    (config/require! (and (seq chains) (every? seq chains)
                          (apply = (map count chains))
                          (every? #(layout/valid-position? latent-layout %) (mapcat identity chains)))
                     "Posterior chains must be nonempty, finite and match the parameter layout" {})
    (config/require! (or (nil? (:parameter-names posterior)) (= (layout/names latent-layout) (:parameter-names posterior)))
                     "Posterior parameter names disagree with layout" {})
    (assoc posterior :spec spec)))
(defn write!
  ([path posterior] (write! path posterior :edn))
  ([path posterior format]
   (let [posterior (validate posterior)]
     (spit path (case format
                  :edn (pr-str posterior)
                  :json (json/write-str {"wire-format" "foerster-marketing/tagged-json-1" "posterior" (encode posterior)})
                  (throw (ex-info "Unknown persistence format" {:format format}))))
     path)))
(defn read!
  ([path] (read! path :edn))
  ([path format]
   (validate
    (case format
      :edn (edn/read-string (slurp path))
      :json (let [wire (json/read-str (slurp path))]
              (config/require! (= "foerster-marketing/tagged-json-1" (get wire "wire-format")) "Unknown JSON wire format" {})
              (decode (get wire "posterior")))
      (throw (ex-info "Unknown persistence format" {:format format}))))))

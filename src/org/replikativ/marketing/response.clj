(ns ^:no-doc org.replikativ.marketing.response
  "Media-response families expressed as single-body numerical source.
  Consumers supply amplitude expressions; hierarchy and evidence stay outside.

  Implementation detail of the public `model`, `hierarchy`, `inference`,
  `decision` and `io` namespaces; not public API and may change without notice."
  (:require [org.replikativ.marketing.family :as family]
            [org.replikativ.marketing.source :as source]))

(def normalize family/normalize)
(defn shaped? [spec] (some? (family/shape-name (:saturation spec))))
(defn logistic? [spec] (= :logistic (normalize (:saturation spec))))

(defn- michaelis-menten [amplitude kappa exposure]
  (let [z (gensym "exposure_")]
    (list 'let [z exposure]
          (source/mul amplitude (source/div z (source/add kappa z))))))

(defn weighted
  "Apply the response to a lag sum; logistic sums already include 0.5*lambda.
  Other families receive normalized effective spend and the shared shape value."
  [spec amplitude shape exposure]
  (case (normalize (:saturation spec))
    :logistic (source/mul amplitude (list 'raster.math/tanh exposure))
    :none (source/mul amplitude exposure)
    :michaelis-menten (michaelis-menten amplitude shape exposure)))

(defn effective
  "Response to normalized effective spend (without applying adstock again)."
  [spec amplitude shape exposure]
  (case (normalize (:saturation spec))
    :logistic (source/saturation amplitude shape exposure)
    :none (source/mul amplitude exposure)
    :michaelis-menten (michaelis-menten amplitude shape exposure)))

(defn lag-scale [spec shape normalizer]
  (source/div (if (logistic? spec) (source/mul 0.5 shape) 1.0) normalizer))

(defn lag-coefficient [spec shape power normalizer]
  ;; Preserve the qualified logistic association in the geo kernel.
  (source/div (if (logistic? spec) (source/mul 0.5 (source/mul shape power)) power) normalizer))

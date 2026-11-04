(ns ^:no-doc org.replikativ.marketing.family
  "Statistical metadata for built-in response families, independent of numerical source.

  Implementation detail of the public `model`, `hierarchy`, `inference`,
  `decision` and `io` namespaces; not public API and may change without notice.")

(def definitions
  {:logistic {:shape :lambda :prior {:distribution :gamma :shape 3.0 :rate 1.0} :initial 2.0}
   :none {}
   :michaelis-menten {:shape :kappa :prior {:distribution :log-normal :mu 0.0 :sigma 0.7} :initial 1.0}})

(defn normalize [family]
  (let [family (if (nil? family) :logistic family)
        family (if (string? family) (keyword family) family)]
    (when-not (contains? definitions family)
      (throw (ex-info "Unsupported saturation family" {:saturation family})))
    family))
(defn description [family] (get definitions (normalize family)))
(defn shape-name [family] (:shape (description family)))
(defn shape-latents [family channels]
  (when-let [name (shape-name family)]
    [{:name name :shape [(count channels)] :support :positive :coordinates [channels]}]))
(defn initial-values [family]
  (let [{:keys [shape initial]} (description family)]
    (if shape {shape (Math/log initial)} {})))

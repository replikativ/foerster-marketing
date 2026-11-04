(ns ^:no-doc org.replikativ.marketing.gradient
  "Marketing source descriptors for the shared Foerster/Raster gradient adapter.

  Implementation detail of the public `model`, `hierarchy`, `inference`,
  `decision` and `io` namespaces; not public API and may change without notice."
  (:require [org.replikativ.foerster-raster.gradient :as gradient]))

(defn- source [model]
  {:arguments (:density-arguments model)
   :dimension (:dimension model)
   :scalar-source (when-let [args (get-in model [:density-program :scalar-args])]
                    {:args args :body (get-in model [:density-program :scalar-body])})})

(defn parameter-layout
  ([model] (parameter-layout model (:gradient-parameter-layout model :auto)))
  ([model requested] (gradient/parameter-layout (source model) requested)))

(defn compile-gradient [model & {:as options}]
  (apply gradient/compile-gradient (:density-var model) (source model)
         (mapcat identity (merge {:parameter-layout (:gradient-parameter-layout model :auto)} options))))

(defn aot-block [model compiled]
  (gradient/aot-block (:description (:block model)) (:density-var model)
                      {:args (:args model)} compiled))

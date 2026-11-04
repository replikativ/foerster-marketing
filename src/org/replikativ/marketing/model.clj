(ns org.replikativ.marketing.model
  "An additive MMM with finite geometric adstock and selectable media response.

  A model owns its fixed channel layout, scaling and compiled numerical block.
  Observations and experimental contrasts are runtime inputs. Default priors: alpha ~ Beta(2,2), beta ~ HalfNormal(1), lambda ~ Gamma(3,rate=1),
  intercept ~ Normal(0,1), sigma ~ HalfNormal(0.5), controls ~ Normal(0,1)."
  (:require [raster.core]
            [org.replikativ.marketing.layout :as layout]
            [org.replikativ.marketing.family :as family]
            [org.replikativ.marketing.response :as response]
            [org.replikativ.marketing.source :refer [add balanced-sum mul sub div log read-at normal prior-density student-penalty]]
            [org.replikativ.marketing.calibration :as calibration]
            [org.replikativ.marketing.config :as config :refer [finite? require!]]
            [org.replikativ.marketing.seasonality :as seasonality]
            [raster.numeric]
            [raster.math]
            [raster.arrays]
            [raster.par]
            [org.replikativ.foerster-raster.block :as rb]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.marketing.gradient :as gradient]))

(def prior-spec
  "Default priors by parameter group, in scaled model coordinates."
  config/default-priors)

(defn control-names
  "Control column names of a specification: `:controls` followed by any
  generated seasonality columns."
  [spec]
  (into (vec (:controls spec)) (seasonality/column-names (:seasonality spec))))

(defn specification
  "Validate a fixed layout and positive business-unit scaling constants.
  `:channels` is required. `:saturation` is :logistic (default), :none or :michaelis-menten;
  :none uses beta * normalized-adstock and has no lambda parameter/prior."
  [{:keys [channels controls l-max scaling priors likelihood seasonality saturation] :or {controls [] l-max 6} :as opts}]
  (require! (empty? (remove #{:channels :controls :l-max :scaling :priors :likelihood :seasonality :saturation} (keys opts)))
            "Unknown model specification key" {:spec opts})
  (require! (and (seq channels) (= (count channels) (count (set channels))))
            "Channels must be nonempty and unique" {:channels channels})
  (require! (= (count controls) (count (set controls))) "Controls must be unique" {})
  (require! (and (integer? l-max) (pos? l-max)) "l-max must be a positive integer" {})
  (let [saturation (response/normalize saturation)
        seasonality (seasonality/normalize seasonality)
        names (into (vec controls) (seasonality/column-names seasonality))
        _ (require! (= (count names) (count (set names))) "Control names collide with seasonal columns" {})
        priors (config/priors priors (count channels) (count names) saturation)
        likelihood (config/likelihood likelihood)
        sx (vec (or (:channel scaling) (repeat (count channels) 1.0)))
        sy (or (:target scaling) 1.0)]
    (require! (and (= (count sx) (count channels)) (every? #(and (finite? %) (pos? %)) sx)
                   (finite? sy) (pos? sy)) "Scaling constants must be finite and positive" {})
    (cond-> {:channels (vec channels) :controls (vec controls) :l-max l-max
             :scaling {:channel (mapv double sx) :target (double sy)} :priors priors}
      (not= saturation :logistic) (assoc :saturation saturation)
      seasonality (assoc :seasonality seasonality)
      (not= likelihood {:distribution :normal}) (assoc :likelihood likelihood))))

(defn prepare
  "Prepare row-major business-unit :x, :y, optional :controls and :lift rows.
  Lift rows describe effective-spend contrasts, not arbitrary experiment totals.
  Missing or nonfinite data are rejected; missing spend is never inferred as zero."
  [spec {:keys [x y controls lift dates time-index] :or {lift []} :as data}]
  (let [x (mapv vec x) y (vec y)
        n (count x) c (count (:channels spec)) p (count (:controls spec))
        controls (mapv vec (or controls (vec (repeat n []))))
        data (assoc data :x x :y y :controls controls)
        sx (get-in spec [:scaling :channel]) sy (get-in spec [:scaling :target])
        _ (seasonality/validate-index! time-index n (some? (:seasonality spec)))
        regressors (delay (if (:seasonality spec)
                            (mapv into controls (seasonality/basis (:seasonality spec) time-index)) controls))]
    (require! (pos? n) "At least one time period is required" {})
    (require! (and (= n (count y) (count controls))
                   (every? #(= c (count %)) x) (every? #(= p (count %)) controls))
              "Data dimensions disagree with the model layout" {:periods n :channels c :controls p})
    (require! (and (every? #(and (finite? %) (not (neg? %))) (mapcat identity x))
                   (every? finite? y) (every? finite? (mapcat identity controls)))
              "Inputs must be finite and media must be nonnegative" {})
    (when dates
      (require! (and (= n (count dates)) (= n (count (set dates)))
                     (= (vec dates) (vec (sort dates))))
                "Dates must be unique, increasing and match the number of rows" {}))
    (doseq [row lift]
      (require! (and (= :effective-spend-contrast (:type row))
                     (some #{(:channel row)} (:channels spec))
                     (every? finite? (map row [:x :delta-x :delta-y :sigma]))
                     (not (neg? (:x row))) (not (neg? (+ (:x row) (:delta-x row))))
                     (pos? (:sigma row)))
                "Invalid effective-spend lift contrast" {:row row}))
    (let [experiment (calibration/prepare spec (:experiments data) (:experiment-covariance data))
          prepared {:experiment-layout (:layout experiment) :spec spec :data (assoc data :controls controls :lift (vec lift)) :n n
                    :inputs (merge {:ys (double-array (map #(/ % sy) y)) :cnt (long n)
                                    :weights (double-array (repeat n 1.0))
                                    :lifts (double-array
                                            (mapcat (fn [row]
                                                      (let [j (.indexOf (:channels spec) (:channel row)) scale (nth sx j)]
                                                        [(/ (:x row) scale) (/ (:delta-x row) scale)
                                                         (/ (:delta-y row) sy) (/ (:sigma row) sy)])) lift))}
                                   (into {} (for [j (range c)]
                                              [(keyword (str "x" j))
                                               (double-array (map #(/ (nth % j) (nth sx j)) x))]))
                                   (into {} (for [j (range (count (control-names spec)))]
                                              [(keyword (str "u" j)) (double-array (map #(nth % j) @regressors))]))
                    ;; Fixed observed-spend lags are data, not parameters. Keeping
                    ;; them outside the density removes conditional scan tapes.
                                   (into {} (for [j (range c)]
                                              [(keyword (str "x" j "_lags"))
                                               (delay (double-array (for [t (range n) k (range (:l-max spec))]
                                                                      (if (>= t k)
                                                                        (/ (get-in x [(- t k) j]) (nth sx j))
                                                                        0.0))))])))}
          inputs (:inputs prepared)
          raw (double-array (mapcat #(seq (get inputs %))
                                    (concat (map #(keyword (str "x" %)) (range c))
                                            (map #(keyword (str "u" %)) (range (count (control-names spec)))))))
          packed (delay (double-array (mapcat #(seq (force (get inputs %)))
                                              (concat (map #(keyword (str "x" % "_lags")) (range c))
                                                      (map #(keyword (str "u" %)) (range (count (control-names spec))))))))]
      (assoc prepared :inputs (assoc inputs :response-data raw :density-data packed :experiments (:data experiment)
                                     :structure [:flat spec (mapv :channel lift) (:layout experiment)]
                                     :response-structure [:flat spec])))))

(defn- program-source [spec lift-channels output density-kernel experiment-layout packed?
                       {:keys [scalar?] :as options}]
  (let [balanced? (get options :balanced? packed?)
        add (if balanced? balanced-sum add)
        c (count (:channels spec)) p (count (control-names spec)) l (:l-max spec)
        lagged? (and (= output :density) (= density-kernel :lagged))
        theta 'theta cnt 'cnt idx 'i acc 'acc
        a (mapv #(symbol (str "a" %)) (range c))
        b (mapv #(symbol (str "b" %)) (range c))
        shape-name (family/shape-name (:saturation spec))
        lam (mapv #(symbol (str (if (= shape-name :kappa) "kappa" "lam") %)) (range c))
        x (mapv #(symbol (str "x" % (when lagged? "_lags"))) (range c))
        u (mapv #(symbol (str "u" %)) (range p))
        width (if lagged? l 1)
        input-at (fn [name slot index]
                   (if packed?
                     (read-at 'observations (list '+ (list '* slot cnt) index))
                     (read-at name index)))
        media-at (fn [j index] (input-at (nth x j) (* j width) index))
        control-at (fn [j index] (input-at (nth u j) (+ (* c width) j) index))
        g (mapv #(symbol (str "g" %)) (range p))
        h (mapv #(symbol (str "h" %)) (range c))
        norms (mapv #(symbol (str "norm" %)) (range c))
        powers-by-channel (mapv (fn [j] (mapv #(symbol (str "power" j "_" %)) (range (inc l)))) (range c))
        coefficients (mapv (fn [j] (mapv #(symbol (str "coefficient" j "_" %)) (range l))) (range c))
        latent-layout (layout/flat (:channels spec) (control-names spec) (:saturation spec :logistic))
        offsets (layout/offsets latent-layout)
        symbols-by-group (cond-> {:alpha a :beta b :intercept ['base] :sigma ['sigma] :control g}
                           shape-name (assoc shape-name lam))
        params (mapcat #(get symbols-by-group (:name %)) (:latents latent-layout))
        bindings (vec (mapcat (fn [s j] [s (read-at theta j)]) params (range)))
        control-value (fn [j] (if (and balanced? (not scalar?)) (read-at theta (list '+ (:control offsets) j)) (nth g j)))
        control-prior (fn [prior]
                        (if scalar?
                          ;; Match the ordered reduction, including its initial zero.
                          (reduce (fn [acc value] (add acc (prior-density value prior))) 0.0 g)
                          (list 'raster.par/reduce 'control_prior_sum 0.0 'control_index p
                                (add 'control_prior_sum (prior-density (control-value 'control_index) prior)))))
        scan-bindings
        (mapcat (fn [j]
                  (let [powers (nth powers-by-channel j)
                        power-bindings (mapcat (fn [k] [(nth powers k)
                                                        (if (zero? k) 1.0
                                                            (mul (nth a j) (nth powers (dec k))))])
                                               (range (inc l)))
                        lag-index (list 'if (list '>= idx l) (list '- idx l) 0)]
                    (concat power-bindings
                            [(nth norms j) (apply add (take l powers))]
                            (if lagged?
                              (let [scale (symbol (str "saturation_scale" j))]
                                (concat [scale (response/lag-scale spec (nth lam j) (nth norms j))]
                                        (mapcat (fn [k] [(get-in coefficients [j k])
                                                         (mul scale (nth powers k))]) (range l))))
                              [(symbol (str "out" j)) (list 'double-array cnt)
                               (nth h j) (list 'raster.par/scan (symbol (str "out" j)) acc 0.0 idx cnt 'double
                                               (list 'let ['current (media-at j idx)
                                                           'lagged (media-at j lag-index)]
                                                     (sub (add (mul (nth a j) acc) 'current)
                                                          (list 'if (list '>= idx l)
                                                                (mul (nth powers l) 'lagged) 0.0))))]))))
                (range c))
        adstocks (mapv (fn [j]
                         (if lagged?
                           (apply add (for [k (range l)]
                                        (mul (get-in coefficients [j k])
                                             (media-at j (list '+ (list '* idx l) k)))))
                           (read-at (nth h j) idx))) (range c))
        mu (apply (if (and balanced? (>= p 8)) balanced-sum add) 'base (concat (map-indexed (fn [j value] (mul value (control-at j idx))) g)
                                                                               (if lagged?
                                                                                 (map #(response/weighted spec %1 %2 %3) b lam adstocks)
                                                                                 (map #(response/effective spec %1 %2 (div %3 %4)) b lam adstocks norms))))
        legacy-prior (apply add
                            (concat [(normal 'base 0.0 1.0)
                                     (add (normal 'sigma 0.0 0.5) 0.6931471805599453)]
                                    (if balanced? [(control-prior {:distribution :normal :mu 0.0 :sigma 1.0})] (map #(normal % 0.0 1.0) g))
                                    (map (fn [alpha beta lambda]
                                           (add (Math/log 6.0) (log alpha) (log (sub 1.0 alpha))
                                                (normal beta 0.0 1.0) 0.6931471805599453
                                                -0.6931471805599453 (mul 2.0 (log lambda)) (sub 0.0 lambda))) a b lam)))
        prior (if (and (response/logistic? spec) (= (:priors spec) prior-spec)) legacy-prior
                  (apply add (concat
                              [(prior-density 'base (get-in spec [:priors :intercept]))
                               (prior-density 'sigma (get-in spec [:priors :sigma]))]
                              (if (and balanced? (map? (get-in spec [:priors :control])))
                                [(control-prior (get-in spec [:priors :control]))]
                                (map prior-density (map control-value (range p)) (config/group-priors spec :control p)))
                              (mapcat (fn [j]
                                        (concat [(prior-density (nth a j) (nth (config/group-priors spec :alpha c) j))
                                                 (prior-density (nth b j) (nth (config/group-priors spec :beta c) j))]
                                                (when (response/shaped? spec)
                                                  [(prior-density (nth lam j) (nth (config/group-priors spec shape-name c) j))]))) (range c)))))
        static-calibration (apply add
                                  (map-indexed
                                   (fn [k channel]
                                     (let [j (.indexOf (:channels spec) channel)
                                           xx (read-at 'lifts (* 4 k)) dx (read-at 'lifts (inc (* 4 k)))
                                           dy (read-at 'lifts (+ 2 (* 4 k))) sd (read-at 'lifts (+ 3 (* 4 k)))
                                           delta (sub (response/effective spec (nth b j) (nth lam j) (add xx dx))
                                                      (response/effective spec (nth b j) (nth lam j) xx))]
                                       (normal dy delta sd))) lift-channels))
        calibration (if (seq experiment-layout)
                      (add static-calibration
                           (calibration/expression spec experiment-layout (fn [_ j] (nth b j)) lam
                                                   (mapv (fn [j] (mapv #(mul (response/lag-scale spec (nth lam j) (nth norms j)) %)
                                                                       (take l (nth powers-by-channel j)))) (range c))))
                      static-calibration)
        tail (case output
               :mean (list 'raster.par/map [idx cnt] mu)
               :response-total (list 'raster.par/reduce acc 0.0 idx cnt
                                     (add acc (mul (read-at 'weights idx) mu)))
               :density (if (= :student-t (get-in spec [:likelihood :distribution]))
                          (let [nu (get-in spec [:likelihood :nu])]
                            (add prior calibration
                                 (mul (list 'double cnt) (config/student-normalizer nu))
                                 (sub 0.0 (mul (list 'double cnt) (log 'sigma)))
                                 (mul (- (/ (inc nu) 2.0))
                                      (list 'raster.par/reduce acc 0.0 idx cnt
                                            (list 'let ['residual (div (sub (read-at 'ys idx) mu) 'sigma)]
                                                  (add acc (student-penalty 'residual nu)))))))
                          (add prior calibration
                               (mul (list 'double cnt) -0.9189385332046727)
                               (sub 0.0 (mul (list 'double cnt) (log 'sigma)))
                               (mul -0.5
                                    (list 'raster.par/reduce acc 0.0 idx cnt
                                          (list 'let ['residual (div (sub (read-at 'ys idx) mu) 'sigma)]
                                                (add acc (mul 'residual 'residual))))))))
        data-names (concat (if packed? ['observations] (concat x u))
                           (if (= output :density)
                             (concat ['ys] (when (seq lift-channels) ['lifts]))
                             ['ys 'lifts 'weights]))
        args (vec (concat (if scalar? (mapcat #(vector % :- 'Double) params) [theta :- '(Array double)])
                          (mapcat #(vector % :- '(Array double)) data-names)
                          (when (and (= output :density) (seq experiment-layout)) ['experiments :- '(Array double)])
                          [cnt :- 'Long]))]
    {:args args :body (list 'let (vec (concat (when-not scalar? bindings) scan-bindings)) tail)}))

(defn- program
  ([spec lift-channels output density-kernel experiment-layout packed?]
   (program spec lift-channels output density-kernel experiment-layout packed? {}))
  ([spec lift-channels output density-kernel experiment-layout packed? options]
   (let [args [spec lift-channels output density-kernel experiment-layout packed?]
         vector-source (apply program-source (conj args options))
         dimension (:dimension (layout/flat (:channels spec) (control-names spec) (:saturation spec :logistic)))
         arity (+ dimension (dec (quot (count (:args vector-source)) 3)))
         ;; Keep the historical split alternative; avoid constructing unused
         ;; scalar programs for large packed layouts.
         scalar-source (when (or (not packed?) (<= arity org.replikativ.foerster-raster.gradient/qualified-scalar-arity))
                         (apply program-source (conj args (assoc options :scalar? true))))]
     (assoc vector-source :scalar-args (:args scalar-source) :scalar-body (:body scalar-source)))))

(def ^:private kernel-cache-limit 32)
(defonce ^:private kernel-cache (atom {}))
(defn- kernel-cache-info [] {:entries (count @kernel-cache) :limit kernel-cache-limit})
(defn ^:no-doc cached-kernel
  "Implementation detail shared with `hierarchy`.
  Reuse numerical definitions by structural key. Retention is bounded; recycle
  the worker at capacity. No observed values belong in a key or cached closure."
  [key build!]
  (let [entry (locking kernel-cache
                (or (get @kernel-cache key)
                    (do (require! (< (count @kernel-cache) kernel-cache-limit)
                                  "Numerical kernel cache is full; recycle this worker" (kernel-cache-info))
                        (let [entry (delay (build!))]
                          (swap! kernel-cache assoc key entry) entry))))]
    @entry))

(defn ^:no-doc checked-arguments
  "Implementation detail shared with `hierarchy`: guard compiled kernel
  arguments against prepared data of a different structural layout."
  [structure-key expected arguments]
  (fn [theta inputs]
    (when-not (= expected (get inputs structure-key))
      (throw (ex-info "Prepared data do not match the compiled structural layout; rebuild the model"
                      {:expected expected :actual (get inputs structure-key)})))
    (arguments theta inputs)))

(defn ^:no-doc numerical-kernel
  "Implementation detail shared with `hierarchy`.
  Compile generated single-body programs and attach a complete-density block.
  The returned structure owns no observation arrays. Used by flat and geo models."
  [{:keys [spec density-program mean-program response-program args response-args
           description dimension gradient-parameter-layout] :as kernel}]
  (let [compile! (fn [mode source]
                   (let [name (gensym (str "mmm_" (clojure.core/name mode) "_"))]
                     (binding [*ns* (the-ns 'org.replikativ.marketing.model)]
                       (eval (list 'raster.core/deftm name (:args source)
                                   :- (if (= mode :mean) '(Array double) 'Double) (:body source)))
                       (ns-resolve *ns* name))))
        density (compile! :density density-program)
        mean-fn (compile! :mean mean-program)
        response (when response-program (compile! :response-total response-program))
        b (rb/raster-block description density {:args args :theta [0] :compile? false})
        kernel (assoc kernel :density-var density :mean-var mean-fn :response-var response
                      :density-arguments (:args density-program) :block b)
        kernel (assoc kernel :gradient-parameter-layout (gradient/parameter-layout kernel))
        compiled (delay (gradient/compile-gradient kernel))]
    (assoc kernel :runtime-gradient (delay ((requiring-resolve 'raster.ad.reverse/value+grad) density :wrt [0]))
           :aot-gradient compiled :aot-block (delay (gradient/aot-block kernel @compiled)))))

(defn build
  "Compile a prepared model. :data-layout :auto retains the qualified split ABI
  for small models and packs larger inputs; :packed removes the public arity cap."
  ([prepared] (build prepared {}))
  ([prepared {:keys [density-kernel gradient-parameter-layout data-layout]
              :or {density-kernel :lagged gradient-parameter-layout :auto data-layout :auto}}]
   (require! (#{:scan :lagged} density-kernel) "Unknown density kernel" {})
   (require! (#{:auto :split :packed} data-layout) "Unknown data layout" {})
   (let [spec (:spec prepared) c (count (:channels spec)) p (count (control-names spec))
         experiment-layout (:experiment-layout prepared)
         latent-layout (layout/flat (:channels spec) (control-names spec) (:saturation spec :logistic))
         shared-input-arity (+ c p 5)
         fits? (and (empty? experiment-layout) (<= shared-input-arity 20))
         layout (if (= data-layout :auto) (if fits? :split :packed) data-layout)
         _ (require! (or (= layout :packed) fits?) "Split data signature exceeds JVM arity; use packed inputs" {})
         ;; Pruning density-only inputs must not silently switch existing models
         ;; to scalar AD. Explicit requests can use the smaller physical ABI.
         automatic-scalar-arity (+ (:dimension latent-layout) (dec shared-input-arity))
         requested (if (and (= gradient-parameter-layout :auto)
                            (or (= layout :packed)
                                (> automatic-scalar-arity org.replikativ.foerster-raster.gradient/qualified-scalar-arity)))
                     :array gradient-parameter-layout)
         lift-channels (mapv :channel (get-in prepared [:data :lift]))
         key [:abi-11 spec lift-channels density-kernel layout requested experiment-layout]
         kernel
         (cached-kernel key
                        (fn []
                          (let [source (fn [mode] (program spec lift-channels mode density-kernel experiment-layout (= layout :packed)))
                                density-program (source :density)
                                input-keys (concat (map #(keyword (str "x" %)) (range c))
                                                   (map #(keyword (str "u" %)) (range p)) [:ys :lifts :weights :cnt])
                                density-keys (mapv (comp keyword first) (rest (partition 3 (:args density-program))))
                                args (fn [theta inputs]
                                       (into [theta]
                                             (map (fn [key]
                                                    (force (get inputs (if (= key :observations)
                                                                         (if (= density-kernel :lagged) :density-data :response-data)
                                                                         key)))) density-keys)))
                                response-args (if (= layout :packed)
                                                (fn [theta inputs] [theta (:response-data inputs) (:ys inputs) (:lifts inputs) (:weights inputs) (:cnt inputs)])
                                                (fn [theta inputs] (into [theta] (map inputs input-keys))))
                                description (:block latent-layout)]
                            (numerical-kernel {:spec spec :block-description description :description description
                                               :parameter-layout latent-layout :parameter-names (layout/names latent-layout)
                                               :density-program density-program :mean-program (source :mean)
                                               :response-program (source :response-total)
                                               :args (checked-arguments :structure [:flat spec lift-channels experiment-layout] args)
                                               :response-args (checked-arguments :response-structure [:flat spec] response-args)
                                               :dimension (:dimension latent-layout) :lift-channels lift-channels :density-kernel density-kernel
                                               :data-layout layout :gradient-parameter-layout requested}))))]
     (merge prepared kernel))))

(defn inference-block
  "The requested gradient backend. AOT compilation is cached per built model."
  [model backend]
  (case backend
    :aot @(:aot-block model)
    :runtime (:block model)
    (throw (ex-info "Unknown gradient backend" {:gradient-backend backend}))))

(defn mean-response
  "Conditional mean in original target units at natural parameter values."
  [model theta]
  (mapv #(* (get-in model [:spec :scaling :target]) %)
        (apply @(:mean-var model) ((:response-args model) (double-array theta) (:inputs model)))))

(defn pointwise-log-likelihood
  "Original-unit observation log densities, excluding priors and experiments."
  [model theta]
  (let [sy (get-in model [:spec :scaling :target])
        sigma-offset (:sigma (layout/offsets (:parameter-layout model)))
        nu (get-in model [:spec :likelihood :nu])
        groups (get-in model [:data :group-index])]
    (mapv (fn [i y mu]
            (let [sigma (* sy (nth theta (+ sigma-offset (if groups (nth groups i) 0))))
                  r (/ (- y mu) sigma)]
              (if nu
                (- (config/student-normalizer nu) (Math/log sigma)
                   (* (/ (inc nu) 2.0) (* 2.0 (Math/log (Math/hypot 1.0 (/ r (Math/sqrt nu)))))))
                (- -0.9189385332046727 (Math/log sigma) (* 0.5 r r)))))
          (range) (get-in model [:data :y]) (mean-response model theta))))

(defn initial-position
  "A finite unconstrained starting position; init is not a prior draw."
  [model]
  (or (:initial-position model)
      (layout/initial-position (:parameter-layout model)
                               (merge {:alpha 0.0 :beta (Math/log 0.5)
                                       :intercept 0.5 :sigma (Math/log 0.2) :control 0.0}
                                      (family/initial-values (get-in model [:spec :saturation]))))))

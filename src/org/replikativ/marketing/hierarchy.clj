(ns org.replikativ.marketing.hierarchy
  "Geo-specific intercept/noise and non-centered log media amplitudes.
  Adstock, saturation shape and control effects are shared across groups."
  (:require [org.replikativ.marketing.model :as model]
            [org.replikativ.marketing.layout :as layout]
            [org.replikativ.marketing.family :as family]
            [org.replikativ.marketing.response :as response]
            [org.replikativ.marketing.config :as config]
            [org.replikativ.marketing.calibration :as calibration]
            [org.replikativ.marketing.source :refer [balanced-sum mul sub div log read-at normal prior-density student-penalty]]))

(defn specification [{:keys [groups hierarchy] :as settings}]
  (config/require! (and (vector? groups) (<= 2 (count groups)) (= (count groups) (count (set groups))))
                   "Hierarchical models need at least two unique named groups" {})
  (config/require! (nil? (get-in settings [:priors :beta]))
                   "Geo amplitudes use hierarchy priors, not independent beta priors" {})
  (let [base (model/specification (dissoc settings :groups :hierarchy :model))
        hierarchy (merge {:location-mu (Math/log 0.5) :location-sigma 0.7 :scale-sigma 0.5} hierarchy)]
    (config/require! (= #{:location-mu :location-sigma :scale-sigma} (set (keys hierarchy)))
                     "Unknown hierarchy option" {})
    (config/require! (and (every? config/finite? (vals hierarchy))
                          (pos? (:location-sigma hierarchy)) (pos? (:scale-sigma hierarchy)))
                     "Hierarchy priors require finite location and positive scales" {})
    (-> base (assoc :model :hierarchical-geo :groups groups :hierarchy hierarchy)
        (update :priors dissoc :beta))))

(defn parameter-layout [spec]
  (let [channels (:channels spec) groups (:groups spec) controls (model/control-names spec)
        c (count channels) g (count groups) p (count controls)]
    (layout/describe (concat [{:name :alpha :shape [c] :support [:interval 0.0 1.0] :coordinates [channels]}]
                             (family/shape-latents (:saturation spec) channels)
                             [{:name :log-amplitude-location :shape [c] :support :real :coordinates [channels]}
                              {:name :log-amplitude-scale :shape [c] :support :positive :coordinates [channels]}
                              {:name :amplitude-z :shape [g c] :support :real :coordinates [groups channels]}
                              {:name :intercept :shape [g] :support :real :coordinates [groups]}
                              {:name :sigma :shape [g] :support :positive :coordinates [groups]}
                              {:name :control :shape [p] :support :real :coordinates [controls]}]))))

(defn prepare
  "Panels are keyed by group. Each panel owns its history; periods may differ.
  Lift rows and independent external time-path experiments belong to their geo."
  [spec {:keys [panels experiment-covariance]}]
  (config/require! (= (set (:groups spec)) (set (keys panels))) "Panels must match model groups" {})
  (doseq [[group panel] panels]
    (config/require! (nil? (:experiment-covariance panel))
                     "Put experiment covariance at panel collection level, ordered by model groups and experiment rows" {:group group}))
  (let [base (model/specification (dissoc spec :groups :hierarchy :model))
        prepared (mapv #(model/prepare base (dissoc (get panels %) :experiments :experiment-covariance)) (:groups spec))
        c (count (:channels spec)) p (count (model/control-names spec))
        counts (mapv :n prepared) n (reduce + counts)
        groups (vec (mapcat #(repeat %2 %1) (range) counts))
        lifts (vec (mapcat (fn [j panel] (map #(assoc % :group j) (get-in panel [:data :lift]))) (range) prepared))
        experiments (vec (mapcat (fn [j group] (map #(assoc % :group j) (:experiments (get panels group)))) (range) (:groups spec)))
        experiment (calibration/prepare spec experiments experiment-covariance)
        packed (double-array (mapcat (fn [key] (mapcat #(seq (force (get-in % [:inputs key]))) prepared))
                                     (concat (map #(keyword (str "x" % "_lags")) (range c))
                                             (map #(keyword (str "u" %)) (range p)))))
        ys (double-array (mapcat #(seq (get-in % [:inputs :ys])) prepared))]
    {:spec spec :n n :panel-counts counts :experiment-layout (:layout experiment)
     :lift-layout (mapv #(select-keys % [:channel :group]) lifts)
     :data {:panels panels :y (vec (mapcat #(get-in % [:data :y]) prepared)) :group-index groups :lift lifts}
     :inputs {:observations packed :ys ys :groups (double-array groups)
              :lifts (double-array
                      (mapcat (fn [{:keys [group channel x delta-x delta-y sigma]}]
                                (let [j (.indexOf (:channels spec) channel)
                                      sx (get-in spec [:scaling :channel j])
                                      sy (get-in spec [:scaling :target])]
                                  [group j (/ x sx) (/ delta-x sx) (/ delta-y sy) (/ sigma sy)])) lifts))
              :experiments (:data experiment) :cnt (long n)
              :structure [:geo spec (mapv #(select-keys % [:channel :group]) lifts) (:layout experiment)]}}))

(defn- program [spec latent-layout lift-layout experiment-layout output]
  (let [c (count (:channels spec)) g (count (:groups spec)) p (count (model/control-names spec)) l (:l-max spec)
        offsets (layout/offsets latent-layout) theta 'theta i 'i acc 'acc
        at (fn [group j] (read-at theta (+ (get offsets group) j)))
        alpha (mapv #(symbol (str "alpha_" %)) (range c))
        shape-name (family/shape-name (:saturation spec))
        lambda (mapv #(symbol (str (if (= shape-name :kappa) "kappa_" "lambda_") %)) (range c))
        powers (mapv (fn [j] (mapv #(symbol (str "power_" j "_" %)) (range l))) (range c))
        norms (mapv #(symbol (str "normalizer_" %)) (range c))
        coefficients (mapv (fn [j] (mapv #(symbol (str "weight_" j "_" %)) (range l))) (range c))
        amplitudes (mapv (fn [group] (mapv #(symbol (str "amplitude_" group "_" %)) (range c))) (range g))
        baselines (mapv #(symbol (str "baseline_" %)) (range g))
        noises (mapv #(symbol (str "noise_" %)) (range g))
        control-values (mapv #(symbol (str "control_" %)) (range p))
        derived-bindings (vec (concat
                               (mapcat (fn [j] [(nth control-values j) (at :control j)]) (range p))
                               (mapcat (fn [group] (mapcat (fn [j]
                                                             [(get-in amplitudes [group j])
                                                              (list 'raster.math/exp
                                                                    (balanced-sum (at :log-amplitude-location j)
                                                                                  (mul (at :log-amplitude-scale j) (at :amplitude-z (+ (* group c) j)))))]) (range c))) (range g))
                               (mapcat (fn [group] [(nth baselines group) (at :intercept group) (nth noises group) (at :sigma group)]) (range g))))
        select-group (fn [values group] (reduce (fn [otherwise j] (list 'if (list '= group j) (nth values j) otherwise))
                                                (last values) (reverse (range (dec (count values))))))
        bindings (vec (concat (mapcat (fn [j] (concat [(nth alpha j) (at :alpha j)]
                                                      (when (response/shaped? spec) [(nth lambda j) (at shape-name j)]))) (range c))
                              (mapcat (fn [j]
                                        (concat (mapcat (fn [k] [(get-in powers [j k]) (if (zero? k) 1.0
                                                                                           (mul (nth alpha j) (get-in powers [j (dec k)])))]) (range l))
                                                [(nth norms j) (apply balanced-sum (nth powers j))]
                                                (mapcat (fn [k] [(get-in coefficients [j k])
                                                                 (response/lag-coefficient spec (nth lambda j) (get-in powers [j k]) (nth norms j))]) (range l)))) (range c))))
        amplitude (fn [group j] (if (integer? group) (get-in amplitudes [group j])
                                    (select-group (mapv #(nth % j) amplitudes) group)))
        group (list 'long (read-at 'groups i))
        sigma (select-group noises group)
        mu (apply balanced-sum (select-group baselines group)
                  (concat (for [j (range p)] (mul (nth control-values j) (read-at 'observations (list '+ (list '* (+ (* c l) j) 'cnt) i))))
                          (for [j (range c)]
                            (response/weighted spec (amplitude group j) (nth lambda j)
                                               (apply balanced-sum (for [k (range l)]
                                                                     (mul (get-in coefficients [j k])
                                                                          (read-at 'observations (list '+ (list '* (* j l) 'cnt) (list '* i l) k)))))))))
        groups-prior (fn [name count group] (apply balanced-sum (map #(prior-density (at name %1) %2) (range count) (config/group-priors spec group count))))
        prior (apply balanced-sum (concat [(groups-prior :alpha c :alpha)]
                                          (when (response/shaped? spec) [(groups-prior shape-name c shape-name)])
                                          [(groups-prior :intercept g :intercept) (groups-prior :sigma g :sigma)
                                           (groups-prior :control p :control)]
                                          (for [j (range c)] (normal (at :log-amplitude-location j) (get-in spec [:hierarchy :location-mu]) (get-in spec [:hierarchy :location-sigma])))
                                          (for [j (range c)] (balanced-sum (normal (at :log-amplitude-scale j) 0.0 (get-in spec [:hierarchy :scale-sigma])) (Math/log 2.0)))
                                          (for [j (range (* g c))] (normal (at :amplitude-z j) 0.0 1.0))))
        lift (if (seq lift-layout)
               (let [k 'lift_index total 'lift_total
                     row (fn [column] (read-at 'lifts (list '+ (list '* k 6) column)))
                     lift-group (list 'long (row 0))
                     channel (list 'long (row 1))
                     beta (select-group (mapv #(amplitude lift-group %) (range c)) channel)
                     shape (when (response/shaped? spec) (select-group lambda channel))
                     f (fn [x] (response/effective spec 'lift_amplitude 'lift_shape x))]
                 (list 'raster.par/reduce total 0.0 k (count lift-layout)
                       (list 'let (vec (concat ['lift_amplitude beta]
                                               (when (response/shaped? spec) ['lift_shape shape])
                                               ['lift_x (row 2) 'lift_dx (row 3)]))
                             (balanced-sum total (normal (row 4) (sub (f (balanced-sum 'lift_x 'lift_dx)) (f 'lift_x)) (row 5))))))
               0.0)
        observation (if-let [nu (get-in spec [:likelihood :nu])]
                      (sub (sub (config/student-normalizer nu) (log sigma))
                           (mul (/ (inc nu) 2.0) (student-penalty (div (sub (read-at 'ys i) mu) sigma) nu)))
                      (normal (read-at 'ys i) mu sigma))
        tail (if (= output :mean) (list 'raster.par/map [i 'cnt] mu)
                 (balanced-sum prior lift (calibration/expression spec experiment-layout (fn [row j] (amplitude (:group row) j)) lambda coefficients)
                               (list 'raster.par/reduce acc 0.0 i 'cnt (balanced-sum acc observation))))]
    {:args ['theta :- '(Array double) 'observations :- '(Array double) 'ys :- '(Array double)
            'groups :- '(Array double) 'lifts :- '(Array double) 'experiments :- '(Array double) 'cnt :- 'Long]
     :body (list 'let (into bindings derived-bindings) tail)}))

(defn initial-position [spec]
  (layout/initial-position (parameter-layout spec)
                           (merge {:alpha 0.0 :log-amplitude-location (get-in spec [:hierarchy :location-mu])
                                   :log-amplitude-scale (Math/log 0.3) :amplitude-z 0.0
                                   :intercept 0.5 :sigma (Math/log 0.2) :control 0.0}
                                  (family/initial-values (:saturation spec)))))
(defn build [prepared]
  (let [spec (:spec prepared) latent-layout (parameter-layout spec)
        lift-layout (:lift-layout prepared) experiment-layout (:experiment-layout prepared)
        kernel (model/cached-kernel
                [:geo-8 spec lift-layout experiment-layout]
                (fn []
                  (let [args (fn [theta inputs] [theta (:observations inputs) (:ys inputs) (:groups inputs)
                                                 (:lifts inputs) (:experiments inputs) (:cnt inputs)])]
                    (model/numerical-kernel
                     {:spec spec :description (:block latent-layout) :parameter-layout latent-layout
                      :parameter-names (layout/names latent-layout) :dimension (:dimension latent-layout)
                      :density-program (program spec latent-layout lift-layout experiment-layout :density)
                      :mean-program (program spec latent-layout lift-layout experiment-layout :mean)
                      :args (model/checked-arguments :structure [:geo spec lift-layout experiment-layout] args)
                      :response-args (model/checked-arguments :structure [:geo spec lift-layout experiment-layout] args) :data-layout :packed-geo :density-kernel :grouped-lagged :gradient-parameter-layout :array
                      :initial-position (initial-position spec) :sigma-offset (:sigma (layout/offsets latent-layout))}))))]
    (merge prepared kernel)))

(defn amplitudes [spec theta]
  (let [offsets (layout/offsets (parameter-layout spec)) c (count (:channels spec))]
    (mapv (fn [g] (mapv (fn [j] (Math/exp (+ (nth theta (+ (:log-amplitude-location offsets) j))
                                             (* (nth theta (+ (:log-amplitude-scale offsets) j))
                                                (nth theta (+ (:amplitude-z offsets) (* g c) j)))))) (range c)))
          (range (count (:groups spec))))))

(ns org.replikativ.marketing.decision
  "Decisions from a fitted posterior: future response paths, paired plan
  comparisons and concave, constant-weekly-budget allocation.

  Every function takes the built model and a posterior (from `inference/fit`
  or `io/read!`) and evaluates future plans after the model's training
  history, including the adstock carryover of past spend."
  (:require [org.replikativ.marketing.model :as model]
            [org.replikativ.marketing.layout :as layout]
            [raster.ad.reverse :as rev]))

(defn- draws [model posterior]
  (when-not (= (:spec model) (:spec posterior))
    (throw (ex-info "Posterior and model specifications differ" {})))
  (when (and (some? (:parameter-names posterior))
             (not= (layout/names (:parameter-layout model)) (:parameter-names posterior)))
    (throw (ex-info "Posterior parameter names disagree with model layout" {})))
  (let [xs (vec (mapcat identity (:chains posterior)))]
    (when-not (and (seq xs) (every? #(layout/valid-position? (:parameter-layout model) %) xs))
      (throw (ex-info "Posterior draws must be finite and match the model layout and supports" {})))
    xs))

(defn- summary [xs]
  (let [ordered (vec (sort xs))
        q (fn [p] (let [i (* p (dec (count ordered))) a (int i) b (min (inc a) (dec (count ordered)))]
                    (+ (nth ordered a) (* (- i a) (- (nth ordered b) (nth ordered a))))))]
    {:mean (/ (reduce + xs) (count xs)) :p05 (q 0.05) :p50 (q 0.5) :p95 (q 0.95)}))

(defn- plan-model [model plan tail]
  (when (= :hierarchical-geo (get-in model [:spec :model]))
    (throw (ex-info "Geo scenario/allocation requires an explicit group plan; this API currently supports flat models" {})))
  (let [spec (:spec model) c (count (:channels spec)) p (count (:controls spec))
        x (:x plan) h (count x) history (vec (take-last (dec (:l-max spec)) (get-in model [:data :x])))
        previous-controls (vec (take-last (count history) (get-in model [:data :controls])))
        controls (or (:controls plan) (vec (repeat h [])))
        future-x (vec (concat history x (repeat tail (vec (repeat c 0.0)))))
        future-controls (vec (concat previous-controls controls (repeat tail (vec (repeat p 0.0)))))
        training-index (get-in model [:data :time-index])
        future-index (when training-index
                       (mapv #(+ (double (peek (vec training-index))) %) (range 1 (inc (+ h tail)))))
        _ (when (:time-index plan)
            (when-not (and future-index (= h (count (:time-index plan)))
                           (every? true? (map #(< (Math/abs (- (double %1) (double %2))) 1e-9)
                                              (:time-index plan) (take h future-index))))
              (throw (ex-info "Plan time indices must immediately follow training history" {}))))
        indices (when training-index (into (vec (take-last (count history) training-index)) future-index))
        prepared (model/prepare spec {:x future-x :controls future-controls :time-index indices
                                      :y (vec (repeat (count future-x) 0.0))
                                      :lift (get-in model [:data :lift])})
        weights (double-array (concat (repeat (count history) 0.0) (repeat (+ h tail) 1.0)))]
    {:model (assoc (merge model prepared) :inputs (assoc (:inputs prepared) :weights weights))
     :history (count history) :horizon h :tail tail}))

(defn thin
  "Keep at most `max-draws` posterior draws, evenly spaced within each chain.
  Each of the k chains keeps every s-th draw, starting with its first, where
  s = ceil(chain length / floor(max-draws / k)). Only `:chains` changes; the
  selection is deterministic, so repeated runs see the same draws. Returns
  the posterior unchanged when `max-draws` is nil."
  [posterior max-draws]
  (if (nil? max-draws)
    posterior
    (let [chains (:chains posterior)]
      (when-not (and (integer? max-draws) (pos? max-draws) (seq chains) (<= (count chains) max-draws))
        (throw (ex-info "max-draws must be a positive integer of at least one draw per chain"
                        {:max-draws max-draws :chains (count chains)})))
      (let [per-chain (quot max-draws (count chains))]
        (assoc posterior :chains
               (mapv (fn [chain]
                       (vec (take-nth (max 1 (long (Math/ceil (/ (count chain) (double per-chain))))) chain)))
                     chains))))))

(defn- response-paths
  "Per-draw conditional mean response over a plan's horizon and carryover tail."
  [prepared positions]
  (mapv (fn [theta] (vec (drop (:history prepared) (model/mean-response (:model prepared) theta))))
        positions))

(defn plan-responses
  "Posterior weekly conditional-mean response paths for one future spend plan.
  `plan` is {:x [[spend per channel] ...] :controls [[control values] ...]}
  (plus optional :time-index), one row per future period, continuing from the
  model's training history. Each path covers the plan's periods followed by
  l-max-1 zero-spend carryover periods, in original target units.
  `:draws` holds one path per posterior draw, in chain order.
  Options: `:max-draws` thins the posterior first (see `thin`)."
  ([model posterior plan] (plan-responses model posterior plan {}))
  ([model posterior plan {:keys [max-draws]}]
   (when-not (seq (:x plan))
     (throw (ex-info "A plan needs at least one future period" {})))
   (let [tail (dec (get-in model [:spec :l-max]))
         prepared (plan-model model plan tail)]
     {:distribution :posterior-conditional-mean :horizon (:horizon prepared) :carryover-periods tail
      :draws (response-paths prepared (draws model (thin posterior max-draws)))})))

(defn compare-plans
  "Compare future spend plans using paired posterior draws and shared history.
  Includes l-max-1 zero-spend tail periods; controls in those periods cancel.
  Results describe uncertainty in conditional mean response, not observation noise.
  Options: `:max-draws` thins the posterior first (see `thin`)."
  ([model posterior baseline candidate] (compare-plans model posterior baseline candidate {}))
  ([model posterior baseline candidate {:keys [max-draws]}]
   (when-not (and (= (count (:x baseline)) (count (:x candidate)))
                  (= (:controls baseline) (:controls candidate)) (seq (:x baseline)))
     (throw (ex-info "Plans need the same nonempty horizon and the same controls" {})))
   (let [tail (dec (get-in model [:spec :l-max]))
         a (plan-model model baseline tail) b (plan-model model candidate tail)
         h (:horizon a)
         positions (draws model (thin posterior max-draws))
         deltas (mapv (fn [ma mb]
                        (let [d (mapv - mb ma)]
                          {:active (reduce + (take h d)) :tail (reduce + (drop h d))
                           :total (reduce + d)}))
                      (response-paths a positions) (response-paths b positions))]
     {:distribution :posterior-conditional-mean :horizon h :carryover-periods tail
      :active (summary (mapv :active deltas)) :carryover (summary (mapv :tail deltas))
      :total (summary (mapv :total deltas)) :draws deltas})))

(defn allocation-objective
  "Expected response and its raster AD gradient for constant weekly budgets.
  Historical spend is fixed; gradients accumulate only over decision periods."
  [model posterior horizon controls]
  (when-not (and (integer? horizon) (pos? horizon))
    (throw (ex-info "Horizon must be a positive integer" {})))
  (let [spec (:spec model) c (count (:channels spec))
        sx (get-in spec [:scaling :channel]) sy (get-in spec [:scaling :target])
        positions (draws model posterior)
        packed? (= :packed (:data-layout model))
        vg (rev/value+grad (:response-var model) :wrt (if packed? [1] (vec (range 1 (inc c)))))
        context (plan-model model {:x (vec (repeat horizon (vec (repeat c 0.0)))) :controls controls}
                            (dec (:l-max spec)))
        input-model (:model context) skip (:history context)]
    (fn [budgets]
      (when-not (and (= c (count budgets))
                     (every? #(and (number? %) (Double/isFinite (double %)) (not (neg? %))) budgets))
        (throw (ex-info "Budgets must be finite and nonnegative" {})))
      (let [inputs (reduce (fn [inputs j]
                             (let [key (keyword (str "x" j)) a (aclone (get inputs key))]
                               (doseq [t (range skip (+ skip horizon))]
                                 (aset-double a t (/ (nth budgets j) (nth sx j))))
                               (assoc inputs key a)))
                           (:inputs input-model) (range c))
            inputs (if packed?
                     (assoc inputs :response-data
                            (double-array (mapcat #(seq (get inputs %))
                                                  (concat (map #(keyword (str "x" %)) (range c))
                                                          (map #(keyword (str "u" %)) (range (count (model/control-names spec))))))))
                     inputs)
            values (mapv (fn [theta]
                           (let [result (apply vg ((:response-args model) (double-array theta) inputs))]
                             [(double (first result))
                              (mapv (fn [j]
                                      (* (/ sy (nth sx j))
                                         (reduce + (take horizon (drop (+ skip (if packed? (* j (:cnt inputs)) 0))
                                                                       (seq (nth result (if packed? 2 (+ j 2)))))))))
                                    (range c))])) positions)
            count-draws (count positions)]
        {:value (* sy (/ (reduce + (map first values)) count-draws))
         :gradient (mapv #(/ % count-draws) (apply mapv + (map second values)))}))))

(defn- project [v budget lower upper]
  (let [clip (fn [tau] (mapv #(max %2 (min %3 (- %1 tau))) v lower upper))]
    (loop [lo (apply min (map - v upper)) hi (apply max (map - v lower)) n 0]
      (let [mid (* 0.5 (+ lo hi)) x (clip mid)]
        (if (= n 80) x
            (if (> (reduce + x) budget) (recur mid hi (inc n)) (recur lo mid (inc n))))))))

(defn allocate
  "Maximize posterior expected response with a weekly budget and channel bounds.
  This objective is concave for this model. Returns a projected-gradient residual
  in budget fractions and target-scaled response units, plus :converged?; reaching
  the iteration limit is never reported as success. Internal value/gradient
  normalization makes stopping and line search independent of currency units.

  Options: `:budget` (per period, required), `:horizon` (periods, required),
  `:controls` (one row per horizon period), `:lower`/`:upper` per-channel
  weekly bounds (default 0 and budget), `:tolerance` (1e-6), `:max-iterations`
  (500) and `:max-draws`, which thins the posterior first (see `thin`); the
  objective's cost grows linearly with the number of draws."
  [model posterior {:keys [budget horizon controls lower upper tolerance max-iterations max-draws]
                    :or {tolerance 1e-6 max-iterations 500}}]
  (let [c (count (get-in model [:spec :channels]))
        lower (or lower (vec (repeat c 0.0))) upper (or upper (vec (repeat c budget)))]
    (when-not (and (number? budget) (Double/isFinite (double budget)) (pos? budget)
                   (= c (count lower) (count upper))
                   (every? true? (map #(and (number? %1) (number? %2)
                                            (Double/isFinite (double %1)) (Double/isFinite (double %2))
                                            (<= 0.0 %1 %2)) lower upper))
                   (<= (reduce + lower) budget (reduce + upper))
                   (number? tolerance) (Double/isFinite (double tolerance)) (pos? tolerance)
                   (integer? max-iterations) (pos? max-iterations))
      (throw (ex-info "Budget bounds are invalid or infeasible" {})))
    (let [raw-objective (allocation-objective model (thin posterior max-draws) horizon controls)
          response-scale (get-in model [:spec :scaling :target])
          objective (fn [fractions]
                      (let [{:keys [value gradient]} (raw-objective (mapv #(* budget %) fractions))]
                        {:value (/ value response-scale)
                         :gradient (mapv #(* (/ budget response-scale) %) gradient)}))
          lower (mapv #(/ % budget) lower) upper (mapv #(/ % budget) upper)
          start (project (vec (repeat c (/ 1.0 c))) 1.0 lower upper)]
      (loop [x start current (objective start) iteration 0 step 1.0]
        (let [stationary (project (mapv + x (:gradient current)) 1.0 lower upper)
              residual (apply max (map #(Math/abs (- %1 %2)) x stationary))]
          (if (or (< residual tolerance) (>= iteration max-iterations))
            {:allocation (mapv #(* budget %) x) :expected-response (* response-scale (:value current))
             :gradient (mapv #(* (/ response-scale budget) %) (:gradient current))
             :objective-response-scale response-scale
             :projected-gradient-residual residual :converged? (< residual tolerance)
             :iterations iteration :horizon horizon :weekly-budget budget}
            (let [[next-x next-value next-step]
                  (loop [s step]
                    (let [candidate (project (mapv #(+ %1 (* s %2)) x (:gradient current)) 1.0 lower upper)
                          delta (mapv - candidate x) result (objective candidate)
                          gain (reduce + (map * (:gradient current) delta))]
                      (if (or (>= (:value result) (+ (:value current) (* 1e-4 gain))) (< s 1e-12))
                        [candidate result s]
                        (recur (* 0.5 s)))))]
              (recur next-x next-value (inc iteration) (min (* 1.5 next-step) 100.0)))))))))

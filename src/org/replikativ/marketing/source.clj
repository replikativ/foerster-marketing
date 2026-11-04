(ns ^:no-doc org.replikativ.marketing.source
  "Construct single-body Raster source; these functions never evaluate model data.

  Implementation detail of the public `model`, `hierarchy`, `inference`,
  `decision` and `io` namespaces; not public API and may change without notice."
  (:require [org.replikativ.marketing.config :as config]))

;; These helpers construct numerical source, rather than doing numerical work.
;; Layout and lag window specialize the program; data values never enter it.
(defn add [& xs] (reduce (fn [a b] (list 'raster.numeric/+ a b)) 0.0 xs))
(defn mul [a b] (list 'raster.numeric/* a b))
(defn sub [a b] (list 'raster.numeric/- a b))
(defn div [a b] (list 'raster.numeric// a b))
(defn log [x] (list 'raster.math/log x))
(defn read-at [x i] (list 'raster.arrays/aget x i))
(defn normal [value mu sd]
  (let [r (gensym "residual_")]
    (list 'let [r (div (sub value mu) sd)]
          (sub (sub -0.9189385332046727 (log sd)) (mul 0.5 (mul r r))))))
(defn prior-density [value prior]
  (case (:distribution prior)
    :normal (normal value (:mu prior) (:sigma prior))
    :half-normal (add (normal value 0.0 (:sigma prior)) (Math/log 2.0))
    :log-normal (sub (normal (log value) (:mu prior) (:sigma prior)) (log value))
    :beta (add (config/log-normalizer prior)
               (mul (dec (:alpha prior)) (log value))
               (mul (dec (:beta prior)) (log (sub 1.0 value))))
    :gamma (add (config/log-normalizer prior)
                (mul (dec (:shape prior)) (log value))
                (sub 0.0 (mul (:rate prior) value)))))

(defn student-penalty [residual nu]
  ;; 2*log(hypot(1,r/sqrt(nu))) = log1p(r*r/nu), without square overflow.
  (mul 2.0 (log (list 'raster.math/hypot 1.0 (div residual (Math/sqrt nu))))))

(defn saturation [beta lam x]
  (mul beta (list 'raster.math/tanh (mul 0.5 (mul lam x)))))

(defn balanced-sum
  "Bound source depth for new large sums. Existing flat arithmetic keeps `add`."
  [& xs]
  (letfn [(sum [xs]
            (case (count xs)
              0 0.0
              1 (first xs)
              (let [mid (quot (count xs) 2)]
                (list 'raster.numeric/+ (sum (subvec xs 0 mid)) (sum (subvec xs mid))))))]
    (sum (vec xs))))

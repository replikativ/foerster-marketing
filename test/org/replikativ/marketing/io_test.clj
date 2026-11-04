(ns org.replikativ.marketing.io-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.marketing.io :as io]
            [org.replikativ.marketing.model-test :as ref]
            [org.replikativ.marketing.hierarchy-test :as geo]))
(deftest typed-posterior-roundtrip
  (doseq [[spec theta] [[ref/spec ref/theta] [geo/spec geo/theta]] format [:edn :json]]
    (let [file (java.io.File/createTempFile "mmm-posterior-" ".data")
          posterior {:format :foerster-marketing/posterior :version 1 :spec spec :chains [[theta]]
                     :metadata {"$keyword" "preserved" :namespaced/key :namespaced/value}}]
      (try
        (io/write! file posterior format)
        (is (= posterior (io/read! file format)))
        (finally (.delete file))))))
(deftest reject-incompatible-results
  (doseq [posterior [{:format :unknown :version 1} {:format :foerster-marketing/posterior :version 2}
                     {:format :foerster-marketing/posterior :version 1 :spec ref/spec :chains [[[1.0]]]}]]
    (is (thrown? clojure.lang.ExceptionInfo (io/validate posterior)))))

(deftest reject-reordered-or-unsupported-coordinates
  (let [base {:format :foerster-marketing/posterior :version 1 :spec ref/spec :chains [[ref/theta]]}]
    (is (thrown? clojure.lang.ExceptionInfo (io/validate (assoc base :parameter-names (vec (repeat (count ref/theta) [:wrong]))))))
    (doseq [[j value] [[0 1.2] [0 -0.1] [2 -1.0] [7 -0.1]]]
      (is (thrown? clojure.lang.ExceptionInfo (io/validate (assoc-in base [:chains 0 0 j] value)))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (io/validate {:format :foerster-marketing/posterior :version 1 :spec geo/spec
                               :chains [[(assoc geo/theta 6 -0.1)]]})))))
(deftest nonfinite-diagnostics-roundtrip
  (let [file (java.io.File/createTempFile "mmm-diagnostics-" ".json")
        posterior {:format :foerster-marketing/posterior :version 1 :spec ref/spec :chains [[ref/theta]]
                   :diagnostics [{:rhat Double/NaN :limit Double/POSITIVE_INFINITY}]}]
    (try (io/write! file posterior :json)
         (let [loaded (io/read! file :json)]
           (is (Double/isNaN (get-in loaded [:diagnostics 0 :rhat])))
           (is (= Double/POSITIVE_INFINITY (get-in loaded [:diagnostics 0 :limit]))))
         (finally (.delete file)))))

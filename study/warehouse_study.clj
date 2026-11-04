(ns warehouse-study
  "Fulfilment extension of the Kettle & Pine study (docs/warehouse.md).

  Usage (from the repository root):

    clojure -J-Xmx3g -M:jvm:study -m warehouse-study \\
      <scenario.json> <fixture.json> <foerster-posterior-dir> <out-dir>

  Reads the saved Foerster posterior of the calibrated case and the warehouse
  scenario (study/warehouse_check.py generates it), then runs the decision in
  `kettle-and-pine.warehouse`: allocation and demand paths per budget,
  SMC capacity inference (seeds 17, 31, 73; 4,000 particles) and 4,000
  paired simulations per warehouse state (seed 101). Writes report.json and
  runs.json to <out-dir>; study/warehouse_check.py checks them independently."
  (:refer-clojure :exclude [run!])
  (:require [clojure.java.io :as io]
            [kettle-and-pine.core :as kettle-and-pine]
            [kettle-and-pine.warehouse :as warehouse]
            [org.replikativ.marketing.io :as posterior-io]))

(defn run!
  "Run the warehouse decision for a scenario, fixture and posterior directory."
  [scenario-path fixture-path posterior-dir output-dir]
  (let [fixture (kettle-and-pine/read-fixture fixture-path)]
    (warehouse/run! (warehouse/read-scenario scenario-path)
                    {:built (kettle-and-pine/build fixture)
                     :posterior (posterior-io/read! (io/file posterior-dir "posterior.edn"))
                     :campaign (:decision fixture)
                     :posterior-directory posterior-dir}
                    output-dir)))

(defn -main [scenario fixture posterior-dir output-dir]
  (try (prn (run! scenario fixture posterior-dir output-dir))
       (finally (shutdown-agents))))

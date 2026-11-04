(ns build
  "Build and deploy the standalone Apache-2.0 marketing library."
  (:require [borkdude.gh-release-artifact :as gh]
            [clojure.java.io :as io]
            [clojure.tools.build.api :as b]
            [deps-deploy.deps-deploy :as deploy])
  (:import [clojure.lang ExceptionInfo]))

(def lib 'org.replikativ/foerster-marketing)
(def class-dir "target/classes")

(defn- version [options]
  (or (:version options) (str "0.1." (b/git-count-revs nil))))

(defn clean [_]
  (b/delete {:path "target"}))

(defn jar [options]
  (let [version (version options)
        basis (b/create-basis {:project "deps.edn"})
        artifact (str "target/foerster-marketing-" version ".jar")]
    (clean nil)
    (b/write-pom {:class-dir class-dir :lib lib :version version :basis basis :src-dirs ["src"]
                  :scm {:url "https://github.com/replikativ/foerster-marketing"
                        :connection "scm:git:git://github.com/replikativ/foerster-marketing.git"
                        :developerConnection "scm:git:ssh://git@github.com/replikativ/foerster-marketing.git"
                        :tag (str "v" version)}
                  :pom-data [[:description "Bayesian marketing mix modelling on the JVM, powered by Foerster and Raster."]
                             [:url "https://github.com/replikativ/foerster-marketing"]
                             [:licenses [:license [:name "Apache License 2.0"]
                                         [:url "https://www.apache.org/licenses/LICENSE-2.0"]]]
                             [:developers [:developer [:id "whilo"] [:name "Christian Weilbach"]
                                           [:email "ch_weil@topiq.es"]]]]})
    (b/copy-dir {:src-dirs ["src"] :target-dir class-dir})
    (let [license (io/file class-dir "META-INF" "LICENSE")]
      (io/make-parents license)
      (io/copy (io/file "LICENSE") license))
    (b/jar {:class-dir class-dir :jar-file artifact})
    {:lib lib :version version :basis basis :jar-file artifact :class-dir class-dir}))

(defn install [options]
  (let [artifact (jar options)]
    (b/install artifact)
    (select-keys artifact [:lib :version :jar-file])))

(defn deploy
  "Deploy to Clojars using CLOJARS_USERNAME and CLOJARS_PASSWORD."
  [options]
  (let [{:keys [jar-file] :as artifact} (jar options)]
    (deploy/deploy {:installer :remote :artifact jar-file
                    :pom-file (b/pom-path {:lib lib :class-dir class-dir})})
    (select-keys artifact [:lib :version :jar-file])))

;; GitHub release: upload the jar as an asset of the release at tag vX.Y.Z,
;; retrying with a Fibonacci backoff because GitHub's release API is
;; occasionally flaky. Same pattern as Foerster.

(defn- fib [a b]
  (lazy-seq (cons a (fib b (+ a b)))))

(defn- retry-with-fib-backoff [retries exec-fn failure?]
  (loop [idle-times (take retries (fib 1 2))]
    (let [result (exec-fn)]
      (if (and (failure? result) (seq idle-times))
        (do (println "Retrying in" (first idle-times) "s:" result)
            (Thread/sleep (* 1000 (first idle-times)))
            (recur (rest idle-times)))
        result))))

(defn release
  "Create the GitHub release at tag vX.Y.Z and upload the jar. Requires GITHUB_TOKEN."
  [options]
  (let [{:keys [version jar-file]} (jar options)
        commit (b/git-process {:git-args "rev-parse HEAD"})
        result (retry-with-fib-backoff
                10
                #(try (gh/overwrite-asset {:org "replikativ" :repo (name lib)
                                           :tag (str "v" version) :commit commit
                                           :file jar-file :content-type "application/java-archive"
                                           :draft false})
                      (catch ExceptionInfo e (assoc (ex-data e) :failure? true)))
                :failure?)]
    (when (:failure? result)
      (println "GitHub release failed:" result)
      (System/exit 1))
    (println (:url result))))

name := "MovieRecommender"

version := "1.0"

scalaVersion := "2.13.16"  // ← make sure this matches your Spark version (run `spark-shell --version` to confirm)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "4.0.1" % "provided",
  "org.apache.spark" %% "spark-sql" % "4.0.1" % "provided",
  "org.apache.spark" %% "spark-mllib" % "4.0.1" % "provided",
  "org.scala-lang" % "scala-library" % scalaVersion.value
)

assembly / mainClass := Some("Recommend") // ← replace with your main class name
assembly / assemblyJarName := "MovieRecommender-fat.jar"

// optional: reduce JAR size by excluding Spark's internal files
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

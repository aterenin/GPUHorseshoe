name := "AsyncGibbsMPI"

version := "1.0"

scalaVersion := "2.11.8"

assemblyJarName in assembly := "AsyncGibbsMPI.jar"
mainClass in assembly := Some("MCMC")
test in assembly := {}

val breezeVersion = "0.12"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % breezeVersion,
  "org.scalanlp" %% "breeze-natives" % breezeVersion
)
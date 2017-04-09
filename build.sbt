name := "Horseshoe"

version := "1.0"

scalaVersion := "2.11.8"

assemblyJarName in assembly := "Horseshoe.jar"
mainClass in assembly := Some("GPUTest")
test in assembly := {}
assemblyExcludedJars in assembly := {
  val cp = (fullClasspath in assembly).value
  cp.filter{_.data.getName.contains("jcu")}
}

val breezeVersion = "0.12"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % breezeVersion,
  "org.scalanlp" %% "breeze-natives" % breezeVersion
)
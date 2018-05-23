name := "ScalaApp"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.2.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.1"
libraryDependencies += "org.scalanlp" % "jblas" % "1.2.1"
resolvers += "Akka Repository" at "http://repo.akka.io/releases/"
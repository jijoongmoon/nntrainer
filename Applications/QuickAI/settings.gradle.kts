pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
// No java { toolchain } / kotlin { jvmToolchain } DSL is used anywhere
// in this build — every module pins its Java version through
// `android { compileOptions { sourceCompatibility / targetCompatibility } }`
// instead, and the Gradle daemon JVM is provisioned via
// gradle/gradle-daemon-jvm.properties (foojay URLs are hard-coded
// there, no plugin needed). The foojay-resolver-convention plugin
// was scaffolded originally but is dead code here; re-add it if a
// module later adopts the toolchain DSL.
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "QuickAI"
include(":QuickDotAI")
include(":LauncherApp")
include(":clientapp")
include(":SampleTestAPP")

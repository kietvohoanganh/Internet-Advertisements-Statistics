run_pipeline <- function() {
  if (!file.exists("config.R") || !file.exists("R/utils.R")) {
    stop("Run from the project root (the folder containing config.R).",
         call. = FALSE)
  }
  source("R/utils.R", local = TRUE)
  ensure_directories()
  # Invalidate the previous success marker before rerunning.
  dir.create("outputs/history", recursive = TRUE, showWarnings = FALSE)
  if (file.exists("outputs/classifier/metrics.csv") && !file.exists("outputs/history/classifier_metrics.csv")) {
    file.copy("outputs/classifier/metrics.csv", "outputs/history/classifier_metrics.csv")
    writeLines("development-exposed historical test; archived before optimization; not an external benchmark",
               "outputs/history/protocol.txt")
  }
  invalidate_validation()
  unlink("outputs/models/selected.csv")
  status <- data.frame(step = character(), status = character(),
                       seconds = numeric(), detail = character())
  record <- function(step, state, started, detail = "") {
    status <<- rbind(status, data.frame(step = step, status = state,
      seconds = as.numeric(difftime(Sys.time(), started, units = "secs")),
      detail = detail))
    utils::write.csv(status, project_file("outputs", "run_status.csv"),
                     row.names = FALSE)
  }
  started <- Sys.time()
  tryCatch(check_dependencies(), error = function(e) {
    record("dependencies", "failed", started, conditionMessage(e))
    stop(e)
  })
  warnings <- data.frame(step = character(), message = character())
  steps <- paste0("scripts/", c("01_prepare", "02_explore", "03_inference",
                               "04_train"), ".R")
  if (isTRUE(config$run_correction)) steps <- c(steps, "scripts/05_finalize.R")
  if (isTRUE(config$run_benchmark)) steps <- c(steps, "scripts/05_benchmark.R")
  steps <- c(steps, "tests/validate_outputs.R")
  on.exit(utils::write.csv(warnings, project_file("outputs", "run_warnings.csv"), row.names = FALSE))
  for (step in steps) {
    message("\nRunning ", step)
    started <- Sys.time()
    tryCatch({
      withCallingHandlers(
        sys.source(step, envir = new.env(parent = globalenv())),
        warning = function(w) {
          warnings <<- rbind(warnings, data.frame(step = step, message = conditionMessage(w)))
        })
      record(step, "passed", started)
    }, error = function(e) {
      record(step, "failed", started, conditionMessage(e))
      stop(e)
    })
  }
  message("\nFinished. See outputs/benchmark/comparison.csv and outputs/run_status.csv.")
  invisible(status)
}
run_pipeline()

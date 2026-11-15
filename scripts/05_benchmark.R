# Compare candidates on shared nested CV folds.
source("R/utils.R", local = TRUE)
source("R/data_io.R", local = TRUE)
source("R/model_metrics.R", local = TRUE)
source("R/ad_classifier.R", local = TRUE)
source("R/benchmark.R", local = TRUE)
check_dependencies()
unlink("outputs/models/selected.csv")
unlink("outputs/benchmark", recursive = TRUE)
dir.create("outputs/benchmark", recursive = TRUE)
settings <- list(seed = config$seed, outer_folds = config$benchmark_folds,
  inner_folds = config$inner_folds, rf_trees = as.list(config$rf_trees),
  jobs = config$benchmark_jobs, min_precision = config$min_precision,
  objective = config$selection_objective, correction_min_gain = config$correction_min_gain,
  reps = config$bootstrap_reps)
jsonlite::write_json(settings, "outputs/benchmark/config.json", auto_unbox = TRUE, pretty = TRUE)
run_linear_benchmark()
python <- Sys.getenv("PYTHON", unset = if (file.exists(".venv/bin/python")) ".venv/bin/python" else "python3")
status <- system2(python, c("classifier/benchmark.py", "run", "outputs/benchmark/config.json"))
if (status != 0L) stop("Nested benchmark failed; no new default classifier was installed.")
source("R/benchmark_figures.R", local = TRUE)
benchmark_figures()

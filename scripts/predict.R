# Score the model selected by training CV.
predict_file <- function(input, output) {
  source("R/utils.R", local = TRUE)
  source("R/data_io.R", local = TRUE)
  source("R/model_metrics.R", local = TRUE)
  source("R/ad_classifier.R", local = TRUE)
  source("R/benchmark.R", local = TRUE)
  manifest <- utils::read.csv(require_input("outputs/models/selected.csv", "Run make run to build the selected classifier."), stringsAsFactors = FALSE)
  if (nrow(manifest) != 1L || manifest$schema_version != 1L) stop("Invalid selected-model manifest.")
  same_hash <- function(path, expected) file.exists(path) && identical(unname(tools::md5sum(path)), expected)
  if (!same_hash("config.R", manifest$config_md5) || !same_hash(manifest$artifact, manifest$artifact_md5)) {
    stop("Stale/incompatible selected model or configuration. Run make run.")
  }
  if ("parent_artifact" %in% names(manifest) && !is.na(manifest$parent_artifact) && nzchar(manifest$parent_artifact) &&
      !same_hash(manifest$parent_artifact, manifest$parent_md5)) stop("Stale correction parent; run make run.")
  source_path <- file.path("data/raw", manifest$source_file)
  if (file.exists(source_path) && !same_hash(source_path, manifest$source_md5)) stop("Training source changed; run make run.")
  raw <- readRDS(require_input(input))
  if (!same_hash("data/processed/feature_dictionary.csv", manifest$dictionary_md5)) stop("Feature dictionary changed; run make run.")
  dictionary <- utils::read.csv(require_input("data/processed/feature_dictionary.csv"))
  validate_predictors(raw, dictionary$feature)
  stage <- tempfile("ads_predict_"); dir.create(stage)
  on.exit(unlink(stage, recursive = TRUE), add = TRUE)
  if (manifest$engine == "r") {
    fit <- readRDS(manifest$artifact)
    probability <- predict_linear_candidate(fit, raw)
    result <- data.frame(ad_score = probability,
      predicted_class = ifelse(probability >= fit$threshold, "ad", "nonad"), score_type = "probability")
  } else if (manifest$engine == "python") {
    utils::write.csv(raw[dictionary$feature], file.path(stage, "features.csv"), row.names = FALSE, na = "")
    if (manifest$model == "pattern_correction") {
      parent <- readRDS(manifest$parent_artifact)
      probability <- predict_linear_candidate(parent, raw)
      utils::write.csv(data.frame(ad_prob = probability,
        predicted_class = ifelse(probability >= parent$threshold, "ad", "nonad")),
        file.path(stage, "baseline.csv"), row.names = FALSE)
    }
    python <- Sys.getenv("PYTHON", unset = if (file.exists(".venv/bin/python")) ".venv/bin/python" else "python3")
    status <- system2(python, vapply(c("classifier/benchmark.py", "predict", stage,
      manifest$artifact, file.path(stage, "predictions.csv")), shQuote, character(1)))
    if (status != 0L) stop("Prediction failed; output was not replaced.")
    result <- utils::read.csv(file.path(stage, "predictions.csv"))
  } else stop("Unsupported selected-model engine.")
  stopifnot(nrow(result) == nrow(raw), all(is.finite(result$ad_score)),
            all(result$predicted_class %in% c("ad", "nonad")))
  result$model <- manifest$model; result$objective <- manifest$objective
  # Replace the output only after scoring succeeds.
  temporary <- tempfile("predictions_", tmpdir = dirname(output))
  on.exit(unlink(temporary), add = TRUE)
  utils::write.csv(result, temporary, row.names = FALSE)
  if (!file.rename(temporary, output)) stop("Cannot install prediction output.")
  invisible(result)
}
if (sys.nframe() == 0L) {
  arguments <- commandArgs(trailingOnly = TRUE)
  if (length(arguments) != 2L) stop("Usage: Rscript scripts/predict.R INPUT.rds OUTPUT.csv")
  predict_file(arguments[1L], arguments[2L])
}

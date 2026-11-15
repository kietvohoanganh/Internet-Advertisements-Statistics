# Reconstruct the legacy classifier log-odds.
source("R/data_io.R")
source("R/model_metrics.R")
source("R/ad_classifier.R")

export_evidence <- function(directory) {
  if (!requireNamespace("glmnet", quietly = TRUE)) stop("Install glmnet first.")
  fit <- readRDS("outputs/models/04_ad_classifier.rds")
  split <- readRDS("data/processed/split.rds")
  test <- readRDS("data/processed/analysis_raw.rds")[split$test, , drop = FALSE]
  predictions <- read.csv("outputs/tables/04_test_predictions.csv")
  dictionary <- read.csv("data/processed/feature_dictionary.csv")
  name_map <- setNames(dictionary$original_name, dictionary$feature)
  x <- classifier_matrix(test, fit$preprocessor, fit$geometry, fit$tokens)
  coefs <- as.matrix(coef(fit$model, s = fit$lambda))
  stopifnot(identical(rownames(coefs)[-1], colnames(x)), identical(split$test, predictions$row_id))
  active <- which(coefs[-1, 1] != 0)
  contrib <- sweep(as.matrix(x[, active, drop = FALSE]), 2, coefs[active + 1L, 1], "*")
  intercept <- unname(coefs[1,1])
  p <- plogis(intercept + rowSums(contrib))
  stopifnot(max(abs(p - predictions$tuned_prob)) < 1e-10,
            all(ifelse(p >= fit$threshold, "ad", "nonad") == predictions$predicted_class))
  present <- unname(name_map[colnames(contrib)])
  use <- !is.na(present) & nzchar(present)
  colnames(contrib)[use] <- present[use]
  ranked <- function(v, sign) {
    keep <- which(sign*v > 0)
    keep <- head(keep[order(sign * v[keep], decreasing = TRUE)], 3L)
    paste(sprintf("%s (%+.3f)", names(v)[keep], v[keep]), collapse = " | ")
  }
  dir.create(directory, showWarnings = FALSE, recursive = TRUE)
  write.csv(data.frame(row_id = split$test, intercept = intercept, cutoff = fit$threshold,
    log_odds = intercept + rowSums(contrib), toward_ad = apply(contrib, 1L, ranked, sign = 1),
    toward_nonad = apply(contrib, 1L, ranked, sign = -1)),
    file.path(directory, "contributions.csv"), row.names = FALSE)
}
export_batch_scores <- function(input, directory) {
  raw <- readRDS(input)
  fit <- readRDS("outputs/models/04_ad_classifier.rds")
  prediction <- predict_ads(fit, raw)
  write.csv(raw[fit$preprocessor$columns], file.path(directory, "features.csv"), row.names = FALSE)
  write.csv(prediction, file.path(directory, "baseline.csv"), row.names = FALSE)
}
if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 1L) export_evidence(args[1L]) else if (length(args) == 2L) {
    export_batch_scores(args[1L], args[2L])
  } else stop("Usage: Rscript classifier/evidence.R [input.rds] DIRECTORY")
}

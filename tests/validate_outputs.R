# Validate saved artifacts without refitting models.
validate_outputs <- function() {
  source("R/utils.R", local = TRUE)
  source("R/model_metrics.R", local = TRUE)
  source("R/data_io.R", local = TRUE)
  source("R/ad_classifier.R", local = TRUE)
  unlink("outputs/validation.txt")
  raw <- readRDS("data/processed/analysis_raw.rds")
  split <- readRDS("data/processed/split.rds")
  groups <- readRDS("data/processed/group_ids.rds")
  train <- readRDS("data/processed/train_prepared.rds")
  pred <- utils::read.csv("outputs/tables/04_test_predictions.csv")
  metrics <- utils::read.csv("outputs/tables/04_test_metrics.csv")
  odds <- utils::read.csv("outputs/tables/04_logistic_odds_ratios.csv")
  stable <- odds$stable_wald_interval
  stopifnot(all(is.finite(odds$ci_high[stable])), all(odds$ci_low[stable] > 0))
  info <- readRDS("outputs/run_info.rds")
  stopifnot(!length(intersect(split$train, split$test)),
    identical(sort(c(split$train, split$test)), seq_len(nrow(raw))),
    !length(intersect(groups[split$train], groups[split$test])),
    !anyDuplicated(train), nrow(train) == length(split$train_fit),
    identical(as.integer(pred$row_id), as.integer(split$test)),
    identical(as.integer(pred$target), raw$target[split$test]),
    identical(unname(tools::md5sum(file.path("data/raw", info$source_file))), info$source_md5))
  map <- c(logistic_baseline = "baseline_prob", logistic_lasso_1se = "lasso_prob",
           tuned_classifier = "tuned_prob")
  if (isTRUE(info$model_config$run_lasso)) {
    classifier <- readRDS("outputs/models/04_ad_classifier.rds")
    inputs <- raw[split$test, classifier$preprocessor$columns, drop = FALSE]
    replay <- predict_ads(classifier, inputs)
    oof <- utils::read.csv("outputs/tables/04_classifier_oof.csv")
    cv <- utils::read.csv("outputs/tables/04_classifier_cv.csv")
    best <- cv[1L, ]
    decision <- oof$ad_prob >= classifier$threshold
    stopifnot(identical(pred$predicted_class, replay$predicted_class),
      isTRUE(all.equal(replay$ad_prob, pred$tuned_prob, tolerance = 1e-10)),
      identical(as.integer(oof$row_id), as.integer(split$train_fit)),
      identical(oof$target, raw$target[oof$row_id]), all(is.finite(oof$ad_prob)),
      best$correct == max(cv$correct), best$tp == max(cv$tp[cv$correct == best$correct]),
      sum(decision == oof$target) == best$correct, sum(decision & oof$target == 1L) == best$tp,
      isTRUE(all.equal(classifier$lambda, best$lambda)), classifier$alpha == best$alpha,
      classifier$threshold == best$threshold, "tuned_classifier" %in% metrics$model)
  }
  for (model in metrics$model) {
    p <- if (model == "constant_train_prevalence") rep(mean(train$target), nrow(pred)) else pred[[map[[model]]]]
    threshold <- if (model == "tuned_classifier") classifier$threshold else info$model_config$threshold
    computed <- classification_metrics(pred$target, p, threshold)
    actual <- metrics[metrics$model == model, names(computed), drop = FALSE]
    stopifnot(isTRUE(all.equal(as.numeric(unlist(actual)), as.numeric(unlist(computed)), tolerance = 1e-10)),
      sum(computed[c("tp", "fp", "tn", "fn")]) == nrow(pred))
  }
  if (isTRUE(info$model_config$run_correction)) {
    corrected <- utils::read.csv("outputs/classifier/test_predictions.csv")
    corrected_metrics <- utils::read.csv("outputs/classifier/metrics.csv")
    best <- subset(corrected_metrics, partition == "test" & model == "correction")
    stopifnot(file.exists("outputs/models/similarity.joblib"),
      file.exists("outputs/models/correction.joblib"), nrow(best) == 1L,
      identical(corrected$row_id, pred$row_id), identical(corrected$target, pred$target),
      all(corrected$corrected_class %in% 0:1), best$n == nrow(pred),
      best$correct == sum(corrected$corrected_class == pred$target),
      best$tp == sum(corrected$corrected_class == 1L & pred$target == 1L),
      best$fp == sum(corrected$corrected_class == 1L & pred$target == 0L),
      best$fn == sum(corrected$corrected_class == 0L & pred$target == 1L),
      best$tn == sum(corrected$corrected_class == 0L & pred$target == 0L))
  }
  if (isTRUE(info$model_config$run_lasso)) {
    fit <- readRDS("outputs/models/04_logistic_lasso_cv.rds")
    stopifnot(length(fit$foldid) == nrow(train), identical(oof$fold, fit$foldid),
      all(vapply(base::split(fit$foldid, fit$group_ids), function(x) length(unique(x)) == 1L, logical(1))),
      all(is.finite(fit$cv$deviance)), fit$lambda.1se >= fit$lambda.min)
  }
  intervals <- utils::read.csv("outputs/tables/04_test_metric_intervals.csv")
  stopifnot(all(intervals$ci_low <= intervals$ci_high, na.rm = TRUE),
    all(intervals$valid_replicates <= intervals$bootstrap_reps))
  if (isTRUE(info$model_config$run_benchmark)) source("tests/validate_benchmark.R", local = TRUE)
  manifest <- utils::read.csv("outputs/source_checksums.csv", stringsAsFactors = FALSE)
  stopifnot(identical(unname(tools::md5sum(manifest$file)), manifest$md5))
  summary <- c(paste("Validated:", format(Sys.time(), tz = "Asia/Ho_Chi_Minh", usetz = TRUE)),
    "PASS: source/data checksums, complete disjoint split, zero overlapping predictor groups",
    "PASS: training deduplication, original test row IDs, recomputed confusion counts and metrics",
    "PASS: grouped CV folds and finite loss, bootstrap interval bounds and replicate counts",
    "PASS: computed statistical and modeling artifacts")
  writeLines(summary, "outputs/validation.txt")
  cat(paste(summary, collapse = "\n"), "\n")
  invisible(TRUE)
}
validate_outputs()

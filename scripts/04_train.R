# Fit the geometry baseline and penalized classifiers.
source("R/utils.R", local = TRUE)
source("R/model_metrics.R", local = TRUE)
source("R/data_io.R", local = TRUE)
source("R/ad_classifier.R", local = TRUE)
check_dependencies()
ensure_directories()
invalidate_validation()
train <- readRDS(require_input(project_file("data", "processed", "train_prepared.rds"),
                               "Run scripts/01_prepare.R first."))
test <- readRDS(require_input(project_file("data", "processed", "test_prepared.rds")))

# Preprocessing was fitted on training data only.
baseline_cols <- c("height", "width", "aratio", "local",
                   "height_missing", "width_missing", "aratio_missing")
baseline_cols <- baseline_cols[vapply(train[baseline_cols], function(x) {
  length(unique(x)) > 1L
}, logical(1))]
if (!length(baseline_cols)) stop("No usable baseline predictors")

baseline_formula <- stats::reformulate(baseline_cols, response = "target")
model_warnings <- character()
baseline <- withCallingHandlers(
  stats::glm(baseline_formula, data = train, family = stats::binomial(link = "logit")),
  warning = function(w) { model_warnings <<- c(model_warnings, conditionMessage(w)) })
saveRDS(baseline, project_file("outputs", "models", "04_logistic_baseline.rds"))
odds <- extract_odds_ratios(baseline)
write_table(odds, "04_logistic_odds_ratios.csv")
write_table(data.frame(converged = baseline$converged,
                       aliased_terms = sum(is.na(stats::coef(baseline))),
                       unstable_wald_terms = sum(!odds$stable_wald_interval),
                       max_wald_se = max(odds$standard_error),
                       warnings = paste(unique(model_warnings), collapse = "; ")),
            "04_model_diagnostics.csv")

predict_safe <- function(fit, data) {
  p <- as.numeric(stats::predict(fit, newdata = data, type = "response"))
  if (any(!is.finite(p))) stop("Non-finite model predictions; check separation")
  pmax(0, pmin(1, p))
}

split <- readRDS(project_file("data", "processed", "split.rds"))
groups <- readRDS(project_file("data", "processed", "group_ids.rds"))
predictions <- data.frame(row_id = split$test, target = test$target,
                          baseline_prob = predict_safe(baseline, test))
results <- classification_metrics(test$target, predictions$baseline_prob, config$threshold)
results$model <- "logistic_baseline"
message("Baseline ROC-AUC: ", sprintf("%.3f", results$roc_auc))
majority <- classification_metrics(test$target, rep(mean(train$target), nrow(test)),
                                    config$threshold)
majority$model <- "constant_train_prevalence"
results <- rbind(majority, results)

# Use the same grouped training folds for both classifiers.
if (isTRUE(config$run_lasso)) {
  raw <- readRDS(project_file("data", "processed", "analysis_raw.rds"))
  prep <- readRDS(project_file("data", "processed", "preprocessor.rds"))
  train_raw <- raw[split$train_fit, , drop = FALSE]
  cv_fit <- fit_lasso_cv(train_raw, prep,
                         groups = groups[split$train_fit],
                         k = config$cv_folds, seed = config$seed)
  dictionary <- utils::read.csv(project_file("data", "processed", "feature_dictionary.csv"))
  tuned <- fit_ad_classifier(train_raw, prep, cv_fit$foldid, dictionary)
  saveRDS(tuned$classifier, project_file("outputs", "models", "04_ad_classifier.rds"))
  write_table(tuned$scores, "04_classifier_cv.csv")
  write_table(data.frame(row_id = split$train_fit, fold = cv_fit$foldid,
    target = train_raw$target, ad_prob = tuned$oof_prob), "04_classifier_oof.csv")
  x_test <- lasso_matrix(test, cv_fit$columns)
  saveRDS(cv_fit, project_file("outputs", "models", "04_logistic_lasso_cv.rds"))
  predictions$lasso_prob <- as.numeric(stats::predict(cv_fit$model, newx = x_test,
    s = cv_fit$lambda.1se, type = "response"))
  lasso_metrics <- classification_metrics(test$target, predictions$lasso_prob,
                                          config$threshold)
  lasso_metrics$model <- "logistic_lasso_1se"
  results <- rbind(results, lasso_metrics)
  lasso_coef <- as.matrix(stats::coef(cv_fit$model, s = cv_fit$lambda.1se))
  selected <- data.frame(feature = rownames(lasso_coef),
                         coefficient = as.numeric(lasso_coef[, 1L]))
  selected <- selected[selected$coefficient != 0, , drop = FALSE]
  selected$original_name <- dictionary$original_name[match(selected$feature, dictionary$feature)]
  write_table(selected, "04_lasso_nonzero_coefficients.csv")
  write_table(data.frame(lambda_min = cv_fit$lambda.min,
                         lambda_1se = cv_fit$lambda.1se), "04_lasso_tuning.csv")
  write_table(cv_fit$cv, "04_lasso_cv.csv")
  classified <- predict_ads(tuned$classifier, raw[split$test, , drop = FALSE])
  predictions$tuned_prob <- classified$ad_prob
  predictions$predicted_class <- classified$predicted_class
  tuned_metrics <- classification_metrics(test$target, classified$ad_prob, tuned$classifier$threshold)
  tuned_metrics$model <- "tuned_classifier"
  results <- rbind(results, tuned_metrics)
  message("LASSO ROC-AUC: ", sprintf("%.3f", lasso_metrics$roc_auc))
} else {
  unlink(project_file("outputs", "models", c("04_logistic_lasso_cv.rds", "04_ad_classifier.rds")))
  unlink(project_file("outputs", "tables", c("04_lasso_nonzero_coefficients.csv",
    "04_lasso_tuning.csv", "04_lasso_cv.csv", "04_classifier_cv.csv", "04_classifier_oof.csv")))
  message("LASSO disabled in config.R; running baseline only.")
}

write_table(results[c("model", setdiff(names(results), "model"))],
            "04_test_metrics.csv")
write_table(predictions, "04_test_predictions.csv")
probabilities <- c(constant_train_prevalence = "constant_prob",
                   logistic_baseline = "baseline_prob")
evaluation <- predictions
evaluation$constant_prob <- mean(train$target)
if (isTRUE(config$run_lasso)) probabilities <- c(probabilities, logistic_lasso_1se = "lasso_prob",
                                                 tuned_classifier = "tuned_prob")
unique_metrics <- intervals <- list()
for (model in names(probabilities)) {
  prob <- evaluation[[probabilities[[model]]]]
  threshold <- results$threshold[match(model, results$model)]
  ix <- split$test_unique
  unique_metrics[[model]] <- classification_metrics(test$target[ix], prob[ix], threshold)
  unique_metrics[[model]]$model <- model
  intervals[[model]] <- bootstrap_metrics(test$target, prob, groups[split$test],
    reps = config$bootstrap_reps, seed = config$seed + 1L, threshold = threshold)
  intervals[[model]]$model <- model
}
write_table(do.call(rbind, unique_metrics), "04_test_metrics_unique.csv")
write_table(do.call(rbind, intervals), "04_test_metric_intervals.csv")
write_table(data.frame(row_id = split$train_fit, group_id = groups[split$train_fit],
  fold = if (isTRUE(config$run_lasso)) cv_fit$foldid else NA_integer_), "04_training_folds.csv")

# Historical-test curves use stepwise average precision for AP.
curves <- lapply(grep("_prob$", names(predictions), value = TRUE), function(col) {
  model_name <- sub("_prob$", "", col)
  item <- probability_curves(predictions$target, predictions[[col]])
  item$roc$model <- model_name
  item$pr$model <- model_name
  item
})
roc <- do.call(rbind, lapply(curves, `[[`, "roc"))
pr <- do.call(rbind, lapply(curves, `[[`, "pr"))
write_table(roc, "04_roc_points.csv")
write_table(pr, "04_pr_points.csv")

p_roc <- ggplot2::ggplot(roc, ggplot2::aes(x = fpr, y = tpr, color = model)) +
  ggplot2::scale_colour_manual(name = NULL,
    values = c(baseline = "#b46f54", lasso = "#718caa", tuned = "#227969"),
    labels = c(baseline = "Geometry logistic", lasso = "LASSO 1-SE", tuned = "Elastic Net")) +
  ggplot2::geom_line(linewidth = 1) +
  ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                       color = "grey70") +
  ggplot2::coord_equal() + ggplot2::labs(title = "Test ROC curves - component scores",
                                        x = "False positive rate", y = "True positive rate") +
  analysis_theme()
save_plot(p_roc, "04_roc_curve.png", 7, 6)

p_pr <- ggplot2::ggplot(pr, ggplot2::aes(x = recall, y = precision, color = model)) +
  ggplot2::scale_colour_manual(name = NULL,
    values = c(baseline = "#b46f54", lasso = "#718caa", tuned = "#227969"),
    labels = c(baseline = "Geometry logistic", lasso = "LASSO 1-SE", tuned = "Elastic Net")) +
  ggplot2::geom_step(linewidth = 1, direction = "vh") +
  ggplot2::labs(title = "Test precision-recall curves - component scores",
                x = "Recall", y = "Precision") +
  analysis_theme()
save_plot(p_pr, "04_pr_curve.png", 7, 6)
if (isTRUE(config$run_lasso)) {
  p_cv <- ggplot2::ggplot(cv_fit$cv, ggplot2::aes(x = lambda, y = deviance)) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = deviance - se, ymax = deviance + se),
                         fill = "#dbeafe") + ggplot2::geom_line(color = "#1d4ed8") +
    ggplot2::geom_vline(xintercept = cv_fit$lambda.1se, linetype = "dashed") +
    ggplot2::scale_x_log10() + analysis_theme() +
    ggplot2::labs(title = "Grouped cross-validation - LASSO", x = "Lambda (log scale)",
                  y = "Mean validation deviance +/- 1 SE")
  save_plot(p_cv, "04_lasso_cv.png", 8, 4)
} else unlink(project_file("outputs", "figures", "04_lasso_cv.png"))
info <- readRDS(project_file("outputs", "run_info.rds"))
info$model_config <- config
saveRDS(info, project_file("outputs", "run_info.rds"))
message("Models complete: outputs/tables/04_* and outputs/models/04_*")

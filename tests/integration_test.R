# Isolate the fixture from existing project outputs.
integration_test <- function() {
  source("R/utils.R", local = TRUE)
  if (!all(vapply(c("ggplot2", "glmnet"), requireNamespace, logical(1), quietly = TRUE))) {
    stop('Run scripts/setup.R with run_lasso=TRUE to install integration-test dependencies.')
  }
  original_dir <- getwd()
  fixture_dir <- tempfile("ads_pipeline_")
  dir.create(fixture_dir)
  on.exit({ setwd(original_dir); unlink(fixture_dir, recursive = TRUE) }, add = TRUE)
  stopifnot(all(file.copy(c("R", "scripts", "tests", "classifier",
                           "config.R", "requirements.txt", "Makefile"), fixture_dir, recursive = TRUE)))
  setwd(fixture_dir)
  dir.create("data/raw", recursive = TRUE)

  set.seed(42)
  n <- 200L
  x <- matrix(stats::rbinom(n * 1558L, 1L, 0.08), nrow = n)
  x[, 1L] <- sample(30:300, n, replace = TRUE)
  x[, 2L] <- sample(50:500, n, replace = TRUE)
  x[, 3L] <- x[, 2L] / x[, 1L]
  x[seq(1L, n, 9L), 1L] <- NA
  y <- ifelse(seq_len(n) %% 4L == 0L, "ad.", "nonad.")
  x[, 5L] <- as.integer(y == "ad.")
  # Include duplicate profiles with conflicting labels.
  x <- rbind(x, x[1:12, ], x[1:3, ])
  y <- c(y, y[1:12], rep("ad.", 3L))
  utils::write.table(cbind(x, y), "data/raw/ad.data", sep = ",", quote = FALSE,
                     row.names = FALSE, col.names = FALSE, na = "?")
  writeLines(c("config <- list(seed=2026L, train_fraction=.8, threshold=.5,",
               "cv_folds=5L, run_lasso=TRUE, bootstrap_reps=30L)"), "config.R")
  source("scripts/run.R", local = TRUE)

  status <- utils::read.csv("outputs/run_status.csv")
  metrics <- utils::read.csv("outputs/tables/04_test_metrics.csv")
  predictions <- utils::read.csv("outputs/tables/04_test_predictions.csv")
  split <- readRDS("data/processed/split.rds")
  lasso <- readRDS("outputs/models/04_logistic_lasso_cv.rds")
  groups <- readRDS("data/processed/group_ids.rds")
  stopifnot(nrow(status) == 5L, all(status$status == "passed"),
    nrow(metrics) == 4L, all(is.finite(predictions$lasso_prob)),
    all(is.finite(predictions$tuned_prob)), all(predictions$predicted_class %in% c("ad", "nonad")),
    identical(as.integer(predictions$row_id), as.integer(split$test)),
    all(metrics$roc_auc >= 0 & metrics$roc_auc <= 1),
    lasso$lambda.1se >= lasso$lambda.min,
    length(lasso$foldid) == length(split$train_fit),
    !length(intersect(groups[split$train], groups[split$test])),
    all(vapply(base::split(lasso$foldid, lasso$group_ids), function(x) length(unique(x)) == 1L, logical(1))),
    nrow(lasso$cv) > 1L,
    file.exists("outputs/session_info.txt"), file.exists("outputs/run_info.rds"),
    file.exists("outputs/validation.txt"))
  figures <- list.files("outputs/figures", "[.]png$", full.names=TRUE)
  stopifnot(length(figures) > 0L, all(file.info(figures)$size > 1000L))

  # A baseline-only rerun must remove stale LASSO artifacts.
  settings <- readLines("config.R")
  writeLines(sub("run_lasso=TRUE", "run_lasso=FALSE", settings, fixed = TRUE), "config.R")
  source("scripts/04_train.R", local = TRUE)
  stopifnot(!file.exists("outputs/models/04_logistic_lasso_cv.rds"),
    !file.exists("outputs/models/04_ad_classifier.rds"),
    !file.exists("outputs/tables/04_classifier_cv.csv"), !file.exists("outputs/tables/04_classifier_oof.csv"),
    !file.exists("outputs/tables/04_lasso_tuning.csv"),
    nrow(utils::read.csv("outputs/tables/04_test_metrics.csv")) == 2L)

  # Failed reruns must invalidate the previous success marker.
  unlink("data/raw/ad.data")
  failure <- tryCatch(source("scripts/run.R", local = TRUE), error = identity)
  stopifnot(inherits(failure, "error"), !file.exists("outputs/validation.txt"),
    utils::read.csv("outputs/run_status.csv")$status[1L] == "failed")
  cat("PASS: full pipeline, grouped split/CV, LASSO, bootstrap, figures, audit, rerun/failure handling\n")
}
integration_test()

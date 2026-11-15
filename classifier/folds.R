# Use nested Elastic Net predictions to train the correction rule.
source("config.R")
source("R/data_io.R")
source("R/model_metrics.R")
source("R/ad_classifier.R")

elastic_design <- function(training, validation, tokens) {
  prep <- fit_preprocessor(training)
  geometry <- fit_geometry(training)
  list(train = classifier_matrix(training, prep, geometry, tokens),
       valid = classifier_matrix(validation, prep, geometry, tokens))
}

prob_path <- function(x, y, xv, alpha, lambdas) {
  model <- fit_penalty_path(x, y, alpha, lambdas)
  as.matrix(stats::predict(model, newx = xv, s = lambdas, type = "response"))
}

train_elastic_folds <- function(directory = "outputs/training/elastic") {
  ids <- read.csv("outputs/tables/04_training_folds.csv")
  partition <- readRDS("data/processed/split.rds")
  groups <- readRDS("data/processed/group_ids.rds")
  stopifnot(identical(ids$row_id, partition$train_fit), !anyDuplicated(ids$row_id),
            identical(ids$group_id, groups[ids$row_id]),
            !length(intersect(ids$group_id, groups[partition$test])),
            identical(sort(unique(ids$fold)), seq_len(config$cv_folds)))
  raw <- readRDS("data/processed/analysis_raw.rds")[ids$row_id, , drop = FALSE]
  tokens <- token_groups(read.csv("data/processed/feature_dictionary.csv"))
  if (!length(tokens)) stop("The retained classifier requires the UCI feature dictionary.")
  alphas <- c(1, 0.35)
  lambdas <- exp(seq(log(.05), log(.0005), length.out = 17L))
  cutoffs <- (4:14) / 20
  k <- length(unique(ids$fold))
  predictions <- choices <- inner_ids <- vector("list", k)
  for (fold in seq_len(k)) {
    message("Elastic Net outer fold ", fold, "/", k)
    hold <- ids$fold == fold
    training <- raw[!hold, , drop = FALSE]
    inner <- group_partitions(training$target, ids$group_id[!hold], rep(1, k), config$seed + fold)
    stopifnot(!anyDuplicated(unique(data.frame(group = ids$group_id[!hold], fold = inner))$group))
    inner_ids[[fold]] <- data.frame(outer_fold = fold, row_id = ids$row_id[!hold],
                                    group_id = ids$group_id[!hold], inner_fold = inner)
    oof <- array(NA_real_, c(nrow(training), length(lambdas), length(alphas)))
    for (j in seq_len(k)) {
      valid <- inner == j
      design <- elastic_design(training[!valid, , drop = FALSE], training[valid, , drop = FALSE], tokens)
      for (a in seq_along(alphas)) {
        oof[valid, , a] <- prob_path(design$train, training$target[!valid], design$valid, alphas[a], lambdas)
      }
    }
    best <- rank_classifier_settings(oof, training$target, alphas, lambdas, cutoffs)[1L, ]
    design <- elastic_design(training, raw[hold, , drop = FALSE], tokens)
    path <- prob_path(design$train, training$target, design$valid, best$alpha, lambdas)
    probability <- path[, best$lambda_index]
    stopifnot(all(is.finite(probability)), all(probability >= 0 & probability <= 1))
    predictions[[fold]] <- data.frame(ids[hold, ], target = raw$target[hold],
       baseline_prob = probability, baseline = as.integer(probability >= best$threshold))
    choices[[fold]] <- data.frame(outer_fold = fold, model = "baseline", inner_n = nrow(training),
                                  best[c("alpha", "lambda", "threshold", "correct", "tp")])
  }
  rows <- do.call(rbind, predictions)
  rows <- rows[order(rows$row_id), ]
  stopifnot(identical(rows$row_id, ids$row_id))
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  write.csv(rows, file.path(directory, "predictions.csv"), row.names = FALSE)
  write.csv(do.call(rbind, choices), file.path(directory, "settings.csv"), row.names = FALSE)
  write.csv(do.call(rbind, inner_ids), file.path(directory, "inner_folds.csv"), row.names = FALSE)
}

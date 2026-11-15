# The positive class is ad (1).
source("R/penalized.R", local = TRUE)

classification_metrics <- function(y, prob, threshold = 0.5) {
  stopifnot(length(y) == length(prob), length(y) > 0L,
            all(y %in% 0:1), all(is.finite(prob)), all(prob >= 0 & prob <= 1),
            threshold > 0 && threshold < 1)
  pred <- as.integer(prob >= threshold)
  tp <- sum(pred == 1 & y == 1)
  fp <- sum(pred == 1 & y == 0)
  tn <- sum(pred == 0 & y == 0)
  fn <- sum(pred == 0 & y == 1)
  div <- function(a, b) if (b == 0) NA_real_ else a / b
  precision <- div(tp, tp + fp)
  recall <- div(tp, tp + fn)
  f1 <- div(2 * tp, 2 * tp + fp + fn)
  npos <- sum(y == 1)
  nneg <- sum(y == 0)
  auc <- if (npos && nneg) {
    (sum(rank(prob, ties.method = "average")[y == 1]) -
       npos * (npos + 1) / 2) / (npos * nneg)
  } else NA_real_
  curves <- probability_curves(y, prob)
  data.frame(accuracy = mean(pred == y), precision = precision,
             recall = recall, specificity = div(tn, tn + fp), f1 = f1,
             f2 = div(5 * tp, 5 * tp + 4 * fn + fp),
             balanced_accuracy = (recall + div(tn, tn + fp)) / 2,
             mcc = div(as.double(tp) * tn - as.double(fp) * fn, sqrt(prod(as.double(c(tp + fp, tp + fn, tn + fp, tn + fn))))),
             roc_auc = auc, average_precision = curves$average_precision,
             threshold = threshold, tp = tp, fp = fp, tn = tn, fn = fn)
}

probability_curves <- function(y, prob) {
  stopifnot(length(y) == length(prob), all(y %in% 0:1), all(is.finite(prob)))
  ord <- order(prob, decreasing = TRUE)
  sorted_p <- prob[ord]
  sorted_y <- y[ord]
  last_of_tie <- !duplicated(sorted_p, fromLast = TRUE)
  tp <- cumsum(sorted_y)[last_of_tie]
  fp <- cumsum(1L - sorted_y)[last_of_tie]
  npos <- sum(y == 1L)
  nneg <- sum(y == 0L)
  precision <- tp / (tp + fp)
  recall <- if (npos) tp / npos else rep(NA_real_, length(tp))
  fpr <- if (nneg) fp / nneg else rep(NA_real_, length(fp))
  roc <- data.frame(fpr = c(0, fpr), tpr = c(0, recall))
  pr <- data.frame(recall = c(0, recall), precision = c(1, precision))
  ap <- if (npos) sum(diff(pr$recall) * tail(pr$precision, -1L)) else NA_real_
  list(roc = roc, pr = pr, average_precision = ap)
}

extract_odds_ratios <- function(model) {
  tab <- summary(model)$coefficients
  terms <- rownames(tab)
  estimates <- tab[, "Estimate"]
  se <- tab[, "Std. Error"]
  valid <- is.finite(estimates) & is.finite(se)
  low <- exp(estimates - stats::qnorm(0.975) * se)
  high <- exp(estimates + stats::qnorm(0.975) * se)
  data.frame(term = terms, coefficient = estimates,
             standard_error = se,
             odds_ratio = exp(estimates),
             ci_low = low, ci_high = high,
             p_value = tab[, "Pr(>|z|)"],
             estimable = valid,
             stable_wald_interval = valid & se <= 10 & low > 0 & is.finite(high),
             row.names = NULL)
}

stratified_folds <- function(y, k = 5L, seed = 2026L) {
  set.seed(seed)
  folds <- integer(length(y))
  for (cls in c(0L, 1L)) {
    ix <- sample(which(y == cls))
    if (length(ix) < k) stop("Need at least k cases per class for CV")
    folds[ix] <- rep(seq_len(k), length.out = length(ix))
  }
  folds
}

# Fit preprocessing within each training fold using a shared penalty grid.
lasso_matrix <- function(data, columns) {
  Matrix::Matrix(as.matrix(data[columns]), sparse = TRUE)
}

fit_lasso_cv <- function(train_raw, prep, groups, k = 5L, seed = 2026L) {
  columns <- c(prep$columns, paste0(prep$continuous, "_missing"))
  train <- apply_preprocessor(train_raw, prep)
  x <- lasso_matrix(train, columns)
  folds <- group_partitions(train$target, groups, rep(1, k), seed)
  # Keep validation labels out of the penalty grid.
  lambda <- exp(seq(log(.1), log(.0001), length.out = 32L))
  fit <- fit_penalty_path(x, train$target, 1, lambda)
  fold_loss <- matrix(NA_real_, nrow = k, ncol = length(lambda))
  for (fold in seq_len(k)) {
    message("LASSO grouped CV fold ", fold, "/", k)
    validation <- folds == fold
    fold_prep <- fit_preprocessor(train_raw[!validation, , drop = FALSE])
    fold_train <- apply_preprocessor(train_raw[!validation, , drop = FALSE], fold_prep)
    fold_valid <- apply_preprocessor(train_raw[validation, , drop = FALSE], fold_prep)
    fold_fit <- fit_penalty_path(lasso_matrix(fold_train, columns), fold_train$target, 1, lambda)
    prob <- as.matrix(stats::predict(fold_fit, newx = lasso_matrix(fold_valid, columns),
                                     s = lambda, type = "response"))
    prob <- pmin(pmax(prob, 1e-15), 1 - 1e-15)
    y <- fold_valid$target
    fold_loss[fold, ] <- colMeans(-2 * (y * log(prob) + (1 - y) * log1p(-prob)))
  }
  cv_mean <- colMeans(fold_loss)
  cv_se <- apply(fold_loss, 2L, stats::sd) / sqrt(k)
  best <- which.min(cv_mean)
  chosen <- which(cv_mean <= cv_mean[best] + cv_se[best])[1L]
  list(model = fit, lambda.min = lambda[best], lambda.1se = lambda[chosen],
       cv = data.frame(lambda = lambda, deviance = cv_mean, se = cv_se),
       foldid = folds, group_ids = groups, columns = columns, preprocessor = prep)
}

# Resample whole groups with fixed predictions; exclude retraining uncertainty.
bootstrap_metrics <- function(y, prob, groups, reps = 500L, seed = 2027L, threshold = 0.5) {
  fields <- c("accuracy", "precision", "recall", "f1", "f2", "balanced_accuracy", "mcc", "roc_auc", "average_precision")
  indices <- split(seq_along(y), groups)
  stopifnot(length(indices) >= 2L, reps >= 20L)
  set.seed(seed)
  draws <- replicate(reps, {
    selected <- sample.int(length(indices), length(indices), replace = TRUE)
    ix <- unlist(indices[selected], use.names = FALSE)
    unlist(classification_metrics(y[ix], prob[ix], threshold)[1L, fields], use.names = FALSE)
  })
  bounds <- t(apply(draws, 1L, function(x) {
    if (all(is.na(x))) c(NA_real_, NA_real_) else
      stats::quantile(x, c(0.025, 0.975), na.rm = TRUE, names = FALSE)
  }))
  data.frame(metric = fields,
    estimate = unlist(classification_metrics(y, prob, threshold)[1L, fields], use.names = FALSE),
    ci_low = bounds[, 1L], ci_high = bounds[, 2L],
    valid_replicates = rowSums(is.finite(draws)), bootstrap_reps = reps)
}

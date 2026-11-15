# Learn geometry bins from fitting rows; token groups use the UCI dictionary.
geometry_values <- function(raw) {
  size <- ifelse(is.na(raw$height) | is.na(raw$width), NA_character_,
                 paste(raw$width, raw$height, sep = "x"))
  list(height = raw$height, width = raw$width, size = size)
}

fit_geometry <- function(raw) {
  levels <- lapply(geometry_values(raw), function(x) {
    freq <- table(x)
    names(freq)[freq >= 3L]
  })
  breaks <- lapply(raw[c("height", "width", "aratio")], function(x) {
    unique(as.numeric(stats::quantile(x, (1:9) / 10, na.rm = TRUE)))
  })
  list(levels = levels, breaks = breaks)
}

token_groups <- function(dictionary) {
  names_uci <- tolower(dictionary$original_name)
  fields <- c("url", "origurl", "ancurl", "alt", "caption")
  field <- sub("\\*.*$", "", names_uci)
  recognized <- !is.na(names_uci) & field %in% fields & grepl("*", names_uci, fixed = TRUE)
  tokens <- sub("^[^*]+\\*", "", names_uci)
  patterns <- c(ad = "(^|[^a-z])(ads?|adverts?|advertis[a-z]*)([^a-z]|$)",
                banner = "banner", sponsor = "sponsor", click = "click",
                promo = "promo|affiliate")
  groups <- lapply(fields, function(f) dictionary$feature[which(recognized & field == f)])
  names(groups) <- paste0("field_", fields)
  for (name in names(patterns)) {
    groups[[paste0("token_", name)]] <- dictionary$feature[which(recognized & grepl(patterns[[name]], tokens))]
  }
  Filter(length, groups)
}

classifier_matrix <- function(raw, prep, geometry, tokens) {
  prepared <- apply_preprocessor(raw, prep)
  columns <- c(prep$columns, paste0(prep$continuous, "_missing"))
  x <- lasso_matrix(prepared, columns)
  values <- geometry_values(raw)
  blocks <- list(x)
  # Unknown and missing values activate no indicators.
  for (name in names(values)) {
    levels <- geometry$levels[[name]]
    if (!length(levels)) next
    index <- match(values[[name]], levels)
    keep <- which(!is.na(index))
    blocks[[length(blocks) + 1L]] <- Matrix::sparseMatrix(
      i = keep, j = index[keep], x = rep(1, length(keep)),
      dims = c(nrow(raw), length(levels)),
      dimnames = list(NULL, paste0("value_", name, "_", levels)))
  }
  for (name in names(geometry$breaks)) {
    breaks <- geometry$breaks[[name]]
    measured <- which(!is.na(raw[[name]]))
    bin <- findInterval(raw[[name]][measured], breaks) + 1L
    blocks[[length(blocks) + 1L]] <- Matrix::sparseMatrix(
      i = measured, j = bin, x = rep(1, length(bin)),
      dims = c(nrow(raw), length(breaks) + 1L),
      dimnames = list(NULL, paste0("bin_", name, "_", seq_len(length(breaks) + 1L))))
  }
  if (length(tokens)) {
    counts <- lapply(tokens, function(cols) log1p(Matrix::rowSums(x[, cols, drop = FALSE])))
    blocks[[length(blocks) + 1L]] <- Matrix::Matrix(do.call(cbind, counts), sparse = TRUE)
  }
  do.call(cbind, blocks)
}

# Break accuracy ties by recall, regularization, then a central cutoff.
rank_classifier_settings <- function(oof, y, alphas, lambdas, thresholds, objective = "accuracy", min_precision = .9) {
  stopifnot(all(is.finite(oof)), objective %in% c("accuracy", "f2_precision"),
            min_precision > 0, min_precision <= 1)
  scores <- expand.grid(lambda_index = seq_along(lambdas), threshold = thresholds, alpha = alphas)
  scores$correct <- scores$tp <- 0L
  for (a in seq_along(alphas)) {
    prob <- matrix(oof[, , a], nrow = length(y))
    for (cutoff in thresholds) {
      rows <- scores$alpha == alphas[a] & scores$threshold == cutoff
      pred <- prob >= cutoff
      scores$correct[rows] <- colSums(pred == y)
      scores$tp[rows] <- colSums(pred[y == 1L, , drop = FALSE])
    }
  }
  scores$lambda <- lambdas[scores$lambda_index]
  scores$accuracy <- scores$correct / length(y)
  scores$recall <- scores$tp / sum(y == 1L)
  scores$tn <- scores$correct - scores$tp
  scores$fp <- sum(y == 0L) - scores$tn
  scores$fn <- sum(y == 1L) - scores$tp
  scores$precision <- ifelse(scores$tp + scores$fp > 0, scores$tp / (scores$tp + scores$fp), NA_real_)
  scores$f2 <- 5 * scores$tp / (5 * scores$tp + 4 * scores$fn + scores$fp)
  scores$constraint_met <- !is.na(scores$precision) & scores$precision >= min_precision
  scores$fallback <- objective == "f2_precision" & !any(scores$constraint_met)
  if (objective == "f2_precision") {
    eligible <- if (any(scores$constraint_met)) scores$constraint_met else rep(TRUE, nrow(scores))
    return(scores[order(!eligible, -scores$f2, -scores$precision, -scores$correct,
                        -scores$lambda, abs(scores$threshold - .5), -scores$alpha), ])
  }
  scores[order(-scores$correct, -scores$tp, -scores$lambda, abs(scores$threshold - 0.5),
               -scores$alpha, scores$threshold), ]
}

fit_ad_classifier <- function(raw, prep, folds, dictionary) {
  alphas <- c(1, 0.35)
  lambdas <- exp(seq(log(.05), log(.0005), length.out = 17L))
  thresholds <- (4:14) / 20
  tokens <- token_groups(dictionary)
  oof <- array(NA_real_, c(nrow(raw), length(lambdas), length(alphas)))
  for (fold in sort(unique(folds))) {
    message("Classifier grouped CV fold ", fold, "/", max(folds))
    valid <- folds == fold
    training <- raw[!valid, , drop = FALSE]
    fold_prep <- fit_preprocessor(training)
    geometry <- fit_geometry(training)
    x <- classifier_matrix(training, fold_prep, geometry, tokens)
    xv <- classifier_matrix(raw[valid, , drop = FALSE], fold_prep, geometry, tokens)
    for (a in seq_along(alphas)) {
      fit <- fit_penalty_path(x, training$target, alphas[a], lambdas)
      oof[valid, , a] <- as.matrix(stats::predict(fit, newx = xv, s = lambdas, type = "response"))
    }
  }
  scores <- rank_classifier_settings(oof, raw$target, alphas, lambdas, thresholds)
  selected <- scores[1L, ]
  geometry <- fit_geometry(raw)
  x <- classifier_matrix(raw, prep, geometry, tokens)
  fit <- fit_penalty_path(x, raw$target, selected$alpha, lambdas)
  list(classifier = list(model = fit, preprocessor = prep, geometry = geometry, tokens = tokens,
                         lambda = selected$lambda, alpha = selected$alpha, threshold = selected$threshold),
       scores = scores[, setdiff(names(scores), "lambda_index")],
       oof_prob = oof[, selected$lambda_index, match(selected$alpha, alphas)])
}

predict_ads <- function(classifier, raw) {
  if (!requireNamespace("glmnet", quietly = TRUE)) stop("Install glmnet to score the saved classifier.")
  validate_predictors(raw, classifier$preprocessor$columns)
  missing <- setdiff(classifier$preprocessor$columns, names(raw))
  if (length(missing)) stop("Missing predictor: ", missing[1L])
  x <- classifier_matrix(raw, classifier$preprocessor, classifier$geometry, classifier$tokens)
  prob <- as.numeric(stats::predict(classifier$model, newx = x, s = classifier$lambda, type = "response"))
  data.frame(ad_prob = prob, predicted_class = ifelse(prob >= classifier$threshold, "ad", "nonad"))
}

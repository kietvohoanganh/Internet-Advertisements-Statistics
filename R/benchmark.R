# Share predictor-profile folds across all candidates.
fit_outlier_policy <- function(raw, policy) {
  bounds <- lapply(raw[c("height", "width", "aratio")], function(x) {
    q <- stats::quantile(x, c(.25, .75), na.rm = TRUE)
    c(lower = unname(q[1] - 1.5 * diff(q)), upper = unname(q[2] + 1.5 * diff(q)))
  })
  list(policy = policy, bounds = bounds)
}
apply_outlier_policy <- function(raw, transformation, fitting = FALSE) {
  keep <- rep(TRUE, nrow(raw))
  for (name in names(transformation$bounds)) {
    bound <- transformation$bounds[[name]]
    if (transformation$policy == "clip") {
      raw[[name]] <- pmin(pmax(raw[[name]], bound[1]), bound[2])
    } else if (transformation$policy == "filter" && fitting) {
      keep <- keep & (is.na(raw[[name]]) | (raw[[name]] >= bound[1] & raw[[name]] <= bound[2]))
    }
  }
  if (fitting && length(unique(raw$target[keep])) != 2L) stop("Outlier policy removes an entire training class.")
  raw[keep, , drop = FALSE]
}
linear_design <- function(training, validation, dictionary, family, policy = "none") {
  transformation <- fit_outlier_policy(training, policy)
  training <- apply_outlier_policy(training, transformation, TRUE)
  validation <- apply_outlier_policy(validation, transformation)
  feature_training <- training[setdiff(names(training), c("row_id", "group_id"))]
  prep <- fit_preprocessor(feature_training)
  if (family == "logistic") {
    cols <- c("height", "width", "aratio", "local", paste0(prep$continuous, "_missing"))
    train <- apply_preprocessor(training, prep)
    cols <- cols[vapply(train[cols], function(x) length(unique(x)) > 1L, logical(1))]
    fit <- stats::glm(stats::reformulate(cols, "target"), data = train, family = stats::binomial(),
                      control = stats::glm.control(maxit = 100L))
    if (!isTRUE(fit$converged)) stop("Geometry logistic did not converge.")
    bundle <- list(kind = family, model = fit, preprocessor = prep, transformation = transformation)
    return(list(bundle = bundle, probability = as.numeric(stats::predict(fit,
      newdata = apply_preprocessor(validation, prep), type = "response"))))
  }
  geometry <- if (family == "lasso") list(levels = list(), breaks = list()) else fit_geometry(training)
  tokens <- if (family == "lasso") list() else token_groups(dictionary)
  list(train = classifier_matrix(training, prep, geometry, tokens), y = training$target,
       valid = classifier_matrix(validation, prep, geometry, tokens),
       bundle = list(kind = "penalized", preprocessor = prep, geometry = geometry, tokens = tokens,
                     transformation = transformation))
}
predict_linear_candidate <- function(bundle, raw) {
  validate_predictors(raw, bundle$preprocessor$columns)
  raw <- apply_outlier_policy(raw, bundle$transformation)
  if (bundle$kind == "logistic") {
    return(as.numeric(stats::predict(bundle$model, newdata = apply_preprocessor(raw, bundle$preprocessor), type = "response")))
  }
  x <- classifier_matrix(raw, bundle$preprocessor, bundle$geometry, bundle$tokens)
  as.numeric(stats::predict(bundle$model, newx = x, s = bundle$lambda, type = "response"))
}

benchmark_linear_case <- function(training, validation, folds, dictionary, directory, case) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  lambdas <- exp(seq(log(.05), log(.0005), length.out = 17L))
  policies <- if (isTRUE(config$outlier_sensitivity)) c("none", "clip", "filter") else "none"
  specifications <- list(logistic = c("logistic", "none"), lasso = c("lasso", "none"),
                         elastic_net = c("enet", "none"))
  for (policy in setdiff(policies, "none")) specifications[[paste0("elastic_net_", policy)]] <- c("enet", policy)
  predictions <- choices <- list()
  for (name in names(specifications)) {
    message("Benchmark ", case, ": ", name)
    started <- proc.time()[[3L]]
    family <- specifications[[name]][1L]; policy <- specifications[[name]][2L]
    alphas <- if (family == "enet") c(1, .35) else 1
    penalties <- if (family == "logistic") .01 else lambdas
    oof <- array(NA_real_, c(nrow(training), length(penalties), length(alphas)))
    for (fold in sort(unique(folds))) {
      valid <- folds == fold
      design <- linear_design(training[!valid, , drop = FALSE], training[valid, , drop = FALSE], dictionary, family, policy)
      if (family == "logistic") oof[valid, 1L, 1L] <- design$probability else {
        for (a in seq_along(alphas)) {
          model <- fit_penalty_path(design$train, design$y, alphas[a], penalties)
          oof[valid, , a] <- as.matrix(predict(model, newx = design$valid, s = penalties, type = "response"))
        }
      }
    }
    design <- linear_design(training, validation, dictionary, family, policy)
    full_fits <- if (family != "logistic") lapply(alphas, function(a) fit_penalty_path(design$train, design$y, a, penalties)) else NULL
    for (objective in c("accuracy", "f2_precision")) {
      best <- rank_classifier_settings(oof, training$target, alphas, penalties, (2:38)/40,
                                        objective, config$min_precision)[1L, ]
      bundle <- design$bundle
      if (family == "logistic") probability <- design$probability else {
        bundle$model <- full_fits[[match(best$alpha, alphas)]]
        bundle$lambda <- best$lambda; bundle$alpha <- best$alpha
        probability <- as.numeric(predict(bundle$model, newx = design$valid, s = best$lambda, type = "response"))
      }
      bundle$threshold <- best$threshold
      bundle$schema_version <- 1L; bundle$objective <- objective
      path <- file.path(directory, paste0(name, "_", objective, ".rds"))
      if (case == "full") saveRDS(bundle, path)
      inner_probability <- oof[, best$lambda_index, match(best$alpha, alphas)]
      for (part in c("inner", "validation")) {
        raw <- if (part == "inner") training else validation
        p <- if (part == "inner") inner_probability else probability
        predictions[[length(predictions) + 1L]] <- data.frame(model = name, objective = objective,
          part = part, row_id = raw$row_id, group_id = raw$group_id, target = raw$target,
          score = p, prediction = as.integer(p >= best$threshold), threshold = best$threshold)
      }
      choices[[length(choices) + 1L]] <- data.frame(model = name, objective = objective,
        alpha = best$alpha, lambda = best$lambda, threshold = best$threshold,
        precision_constraint_met = best$constraint_met, fallback = best$fallback,
        elapsed_seconds = proc.time()[[3L]] - started, artifact = if (case == "full") path else "")
    }
  }
  utils::write.csv(do.call(rbind, predictions), file.path(directory, "linear_predictions.csv"), row.names = FALSE)
  utils::write.csv(do.call(rbind, choices), file.path(directory, "linear_settings.csv"), row.names = FALSE)
}

run_linear_benchmark <- function() {
  raw <- readRDS("data/processed/analysis_raw.rds")
  split <- readRDS("data/processed/split.rds")
  groups <- readRDS("data/processed/group_ids.rds")
  dictionary <- utils::read.csv("data/processed/feature_dictionary.csv", stringsAsFactors = FALSE)
  raw$row_id <- seq_len(nrow(raw)); raw$group_id <- groups
  # Exclude row and group IDs from predictors.
  training <- raw[split$train_fit, , drop = FALSE]
  outer <- group_partitions(training$target, training$group_id, rep(1, config$benchmark_folds), config$seed)
  export_case <- function(train, valid, case, seed) {
    inner <- group_partitions(train$target, train$group_id, rep(1, config$inner_folds), seed)
    stopifnot(!length(intersect(train$group_id, valid$group_id)),
      all(vapply(base::split(inner, train$group_id), function(x) length(unique(x)) == 1L, logical(1))))
    directory <- file.path("outputs/benchmark", case)
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    partitions <- rbind(data.frame(part = "train", row_id = train$row_id, group_id = train$group_id,
      target = train$target, fold = inner), data.frame(part = "validation", row_id = valid$row_id,
      group_id = valid$group_id, target = valid$target, fold = NA_integer_))
    write.csv(partitions, file.path(directory, "partitions.csv"), row.names = FALSE)
    benchmark_linear_case(train, valid, inner, dictionary, directory, case)
  }
  for (fold in seq_len(config$benchmark_folds)) {
    export_case(training[outer != fold, , drop = FALSE], training[outer == fold, , drop = FALSE],
                 paste0("fold", fold), config$seed + fold)
  }
  export_case(training, raw[split$test, , drop = FALSE], "full", config$seed + 100L)
  write.csv(data.frame(row_id = training$row_id, group_id = training$group_id, fold = outer),
             "outputs/benchmark/outer_folds.csv", row.names = FALSE)
}

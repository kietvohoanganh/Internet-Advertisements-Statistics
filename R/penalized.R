# Start each penalty path near the null fit.
fit_penalty_path <- function(x, y, alpha, lambdas, maxit = NULL, tolerance = NULL,
                             standardize = TRUE) {
  settings <- get0("config", inherits = TRUE)
  if (is.null(maxit)) maxit <- if (is.null(settings$glmnet_maxit)) 100000L else settings$glmnet_maxit
  if (is.null(tolerance)) tolerance <- if (is.null(settings$glmnet_tolerance)) 1e-7 else settings$glmnet_tolerance
  stopifnot(nrow(x) == length(y), all(y %in% 0:1), length(unique(y)) == 2L,
            all(is.finite(x@x)), all(is.finite(lambdas)), all(lambdas > 0),
            maxit >= 1000L, tolerance > 0)
  run <- function(lambda = NULL, ...) {
    arguments <- list(x = x, y = y, family = "binomial", alpha = alpha,
                      standardize = standardize, lambda = lambda, ...)
    control <- list(maxit = as.integer(maxit), thresh = tolerance)
    if ("control" %in% names(formals(glmnet::glmnet))) arguments$control <- control
    else arguments <- c(arguments, control)
    do.call(glmnet::glmnet, arguments)
  }
  # Estimate lambda_max from this fitting fold only.
  anchor <- run(nlambda = 5L, lambda.min.ratio = .5)
  if (anchor$jerr != 0L || !length(anchor$lambda)) stop("glmnet null-path initialization failed.")
  start <- max(anchor$lambda[1L], max(lambdas))
  # Dense warm starts prevent stalls after large penalty jumps.
  # Tune only the requested penalties.
  warm <- exp(seq(log(start), log(min(lambdas)), length.out = 100L))
  path <- sort(unique(c(warm, lambdas)), decreasing = TRUE)
  fit <- run(path)
  present <- vapply(lambdas, function(s) any(abs(fit$lambda - s) <= 1e-10 * s), logical(1))
  if (fit$jerr != 0L || !all(present) || any(!is.finite(fit$beta@x))) {
    stop("Incomplete/non-finite glmnet solution; refusing interpolated or empty model predictions.")
  }
  fit$call <- quote(fit_penalty_path(x, y, alpha, lambdas))
  fit$solver <- list(maxit = maxit, tolerance = tolerance, standardize = standardize,
                     requested_lambda = lambdas, anchor = start, path_points = length(path), passes = fit$npasses)
  fit
}

source("config.R", local = TRUE)
# Keep older configurations and small fixtures usable.
defaults <- list(run_benchmark = FALSE, benchmark_folds = 3L,
  inner_folds = 3L, min_precision = .9, selection_objective = "f2_precision",
  glmnet_maxit = 100000L, glmnet_tolerance = 1e-7, rf_trees = c(128L, 256L),
  benchmark_jobs = 2L, outlier_sensitivity = FALSE, correction_min_gain = .005)
config <- utils::modifyList(defaults, config)
numeric_scalars <- c("min_precision", "benchmark_folds", "inner_folds", "glmnet_maxit",
                     "glmnet_tolerance", "benchmark_jobs", "correction_min_gain")
stopifnot(all(vapply(config[numeric_scalars], function(x) is.numeric(x) && length(x)==1L && is.finite(x), logical(1))),
  all(vapply(config[c("run_benchmark","outlier_sensitivity")],
             function(x) is.logical(x) && length(x)==1L && !is.na(x), logical(1))),
  all(vapply(config[c("benchmark_folds","inner_folds","glmnet_maxit","benchmark_jobs")],
             function(x) x==as.integer(x), logical(1))),
  is.numeric(config$rf_trees), length(config$rf_trees)>0, all(is.finite(config$rf_trees)),
  all(config$rf_trees==as.integer(config$rf_trees)), config$correction_min_gain>=0)
stopifnot(is.finite(config$min_precision), config$min_precision > 0, config$min_precision <= 1,
  config$selection_objective %in% c("accuracy", "f2_precision"), config$benchmark_folds >= 3L,
  config$inner_folds >= 3L, config$glmnet_maxit >= 1000L, config$glmnet_tolerance > 0,
  all(config$rf_trees >= 10L), config$benchmark_jobs >= 1L)
stopifnot(is.numeric(config$seed), length(config$seed) == 1L, is.finite(config$seed),
  is.numeric(config$train_fraction), length(config$train_fraction) == 1L,
  is.finite(config$train_fraction), config$train_fraction > 0, config$train_fraction < 1,
  is.numeric(config$threshold), length(config$threshold) == 1L,
  is.finite(config$threshold), config$threshold > 0, config$threshold < 1,
  is.numeric(config$cv_folds), length(config$cv_folds) == 1L, config$cv_folds >= 3L,
  config$cv_folds == as.integer(config$cv_folds),
  is.logical(config$run_lasso), length(config$run_lasso) == 1L, !is.na(config$run_lasso),
  is.numeric(config$bootstrap_reps), length(config$bootstrap_reps) == 1L,
  is.finite(config$bootstrap_reps), config$bootstrap_reps >= 20L)
if (isTRUE(config$run_correction) && !isTRUE(config$run_lasso)) {
  stop("run_correction requires run_lasso = TRUE.")
}

required_packages <- function() {
  c("ggplot2", if (isTRUE(config$run_benchmark)) "jsonlite",
    if (isTRUE(config$run_lasso) || isTRUE(config$run_benchmark)) "glmnet")
}

invalidate_validation <- function() {
  unlink(project_file("outputs", "validation.txt"))
}

check_dependencies <- function() {
  missing <- required_packages()[!vapply(required_packages(), requireNamespace,
                                         logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop("Missing packages: ", paste(missing, collapse = ", "),
         ". Run Rscript scripts/setup.R first.", call. = FALSE)
  }
}

project_file <- function(...) {
  file.path(getwd(), ...)
}

ensure_directories <- function() {
  paths <- c("data/raw", "data/processed", "outputs/figures",
             "outputs/tables", "outputs/models")
  invisible(lapply(paths, function(path) {
    dir.create(project_file(path), recursive = TRUE, showWarnings = FALSE)
  }))
}

require_input <- function(path, hint = NULL) {
  if (!file.exists(path)) {
    stop(paste0("Missing file: ", path, if (!is.null(hint)) paste0("\n", hint)),
         call. = FALSE)
  }
  path
}

write_table <- function(data, name) {
  ensure_directories()
  utils::write.csv(data, project_file("outputs", "tables", name),
                   row.names = FALSE, na = "")
}

save_plot <- function(plot, name, width = 9, height = 5) {
  ensure_directories()
  ggplot2::ggsave(project_file("outputs", "figures", name), plot = plot,
                  width = width, height = height, dpi = 300, bg = "white")
}

analysis_theme <- function() {
  ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(plot.title = ggplot2::element_text(face = "bold", colour = "#193c35"),
      plot.title.position = "plot", panel.grid.minor = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(face = "bold"), legend.position = "bottom")
}

safe_mean <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  mean(x, na.rm = TRUE)
}

safe_sd <- function(x) {
  if (sum(!is.na(x)) < 2) return(NA_real_)
  stats::sd(x, na.rm = TRUE)
}

safe_median <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  stats::median(x, na.rm = TRUE)
}

scientific_p <- function(p) {
  if (is.na(p)) return("NA")
  if (p < 0.001) return("<0.001")
  sprintf("%.3f", p)
}

# Compare classes with confidence intervals and hypothesis tests.
source("R/utils.R", local = TRUE)
source("R/statistics.R", local = TRUE)
require_input(project_file("data", "processed", "analysis_unique.rds"),
              "Run scripts/01_prepare.R first.")
if (!requireNamespace("ggplot2", quietly = TRUE)) stop("Install ggplot2")
raw <- readRDS(project_file("data", "processed", "analysis_unique.rds"))
vars <- c("height", "width", "aratio")

normality <- list()
mean_ci <- list()
hypotheses <- list()

for (v in vars) {
  ad <- stats::na.omit(raw[[v]][raw$target == 1L])
  nonad <- stats::na.omit(raw[[v]][raw$target == 0L])

  for (group in c("ad", "nonad")) {
    x <- if (group == "ad") ad else nonad
    unique_n <- length(unique(x))
    sh <- if (length(x) >= 3L && length(x) <= 5000L && unique_n >= 3L) {
      tryCatch(stats::shapiro.test(x), error = function(e) NULL)
    } else NULL
    normality[[length(normality) + 1L]] <- data.frame(
      variable = v, class = group, n = length(x),
      shapiro_w = if (is.null(sh)) NA_real_ else unname(sh$statistic),
      shapiro_p = if (is.null(sh)) NA_real_ else sh$p.value,
      note = if (is.null(sh)) "Not testable (sample size / ties)" else
        "Interpret with Q-Q plot; large n can flag small deviations")

    # Use a t interval for the arithmetic mean.
    tt <- if (length(x) >= 2L && stats::sd(x) > 0) stats::t.test(x) else NULL
    mean_ci[[length(mean_ci) + 1L]] <- data.frame(
      variable = v, class = group, n = length(x), mean = safe_mean(x),
      ci_low = if (is.null(tt)) NA_real_ else tt$conf.int[1L],
      ci_high = if (is.null(tt)) NA_real_ else tt$conf.int[2L])
  }

  # Welch tests the mean difference: ad minus nonad.
  welch <- tryCatch(stats::t.test(ad, nonad, var.equal = FALSE),
                    error = function(e) NULL)
  if (is.null(welch)) {
    hypotheses[[length(hypotheses) + 1L]] <- data.frame(
      variable = v, test = "Welch t-test", statistic = NA_real_, df = NA_real_,
      p_value = NA_real_, diff_ad_minus_nonad = NA_real_,
      ci_low = NA_real_, ci_high = NA_real_, effect_size = NA_real_)
  } else {
    pooled <- sqrt(((length(ad) - 1) * stats::var(ad) +
                     (length(nonad) - 1) * stats::var(nonad)) /
                   (length(ad) + length(nonad) - 2))
    d <- if (is.finite(pooled) && pooled > 0) {
      (mean(ad) - mean(nonad)) / pooled
    } else NA_real_
    hypotheses[[length(hypotheses) + 1L]] <- data.frame(
      variable = v, test = "Welch t-test", statistic = unname(welch$statistic),
      df = unname(welch$parameter), p_value = welch$p.value,
      diff_ad_minus_nonad = mean(ad) - mean(nonad),
      ci_low = welch$conf.int[1L], ci_high = welch$conf.int[2L],
      effect_size = d)
  }
  # Wilcoxon checks ranks, not equality of means.
  wt <- tryCatch(stats::wilcox.test(ad, nonad, exact = FALSE),
                 error = function(e) NULL)
  hypotheses[[length(hypotheses) + 1L]] <- data.frame(
    variable = v, test = "Wilcoxon rank-sum (sensitivity)",
    statistic = if (is.null(wt)) NA_real_ else unname(wt$statistic),
    df = NA_real_, p_value = if (is.null(wt)) NA_real_ else wt$p.value,
    diff_ad_minus_nonad = NA_real_, ci_low = NA_real_, ci_high = NA_real_,
    effect_size = NA_real_)
}

write_table(do.call(rbind, normality), "03_shapiro_by_class.csv")
write_table(do.call(rbind, mean_ci), "03_mean_95ci_by_class.csv")

# Exclude missing local values from the 2x2 table.
assoc <- raw[!is.na(raw$local), c("target", "local")]
contingency <- table(target = factor(assoc$target, levels = c(1, 0),
                                    labels = c("ad", "nonad")),
                     local = factor(assoc$local, levels = c(0, 1)))
write_table(as.data.frame(contingency), "03_local_contingency.csv")

association <- binary_association(raw$target, raw$local)
chi <- list(expected = association$expected)
write_table(association$result, "03_local_effect.csv")
hypotheses[[length(hypotheses) + 1L]] <- data.frame(
  variable = "local", test = association$result$test, statistic = association$result$statistic,
  df = association$result$df, p_value = association$result$p_value,
  diff_ad_minus_nonad = NA_real_, ci_low = NA_real_, ci_high = NA_real_, effect_size = association$result$phi)
# Adjust the missingness tests as a separate exploratory family.
missingness <- do.call(rbind, lapply(vars, function(name) {
  data.frame(variable = name, binary_association(raw$target, as.integer(is.na(raw[[name]])))$result)
}))
missingness$p_adj_BH <- stats::p.adjust(missingness$p_value, method = "BH")
write_table(missingness, "03_missing_geometry_association.csv")

hypotheses <- do.call(rbind, hypotheses)
hypotheses$n_ad <- vapply(hypotheses$variable, function(v) {
  sum(raw$target == 1L & !is.na(raw[[v]]))
}, integer(1))
hypotheses$n_nonad <- vapply(hypotheses$variable, function(v) {
  sum(raw$target == 0L & !is.na(raw[[v]]))
}, integer(1))
primary_idx <- hypotheses$test %in% c("Welch t-test", "Fisher exact", "Pearson chi-square")
hypotheses$p_adj_BH <- NA_real_
hypotheses$p_adj_BH[primary_idx] <- stats::p.adjust(hypotheses$p_value[primary_idx],
                                                   method = "BH")
write_table(hypotheses, "03_hypothesis_results.csv")
write_table(data.frame(group = rownames(chi$expected),
                       expected_local_0 = chi$expected[, 1],
                       expected_local_1 = chi$expected[, 2]),
            "03_chi_square_expected.csv")

long <- do.call(rbind, lapply(vars, function(v) {
  data.frame(variable = v, value = raw[[v]], class = raw$class)
}))
long <- long[!is.na(long$value), , drop = FALSE]
p_qq <- ggplot2::ggplot(long, ggplot2::aes(sample = value)) +
  ggplot2::stat_qq(alpha = 0.35, size = 0.7) + ggplot2::stat_qq_line() +
  ggplot2::facet_grid(class ~ variable, scales = "free") +
  ggplot2::labs(title = "Normal Q-Q plots by class", x = "Theoretical quantiles",
                y = "Sample quantiles") + analysis_theme()
save_plot(p_qq, "03_qqplots.png", 11, 7)

ci <- do.call(rbind, mean_ci)
p_ci <- ggplot2::ggplot(ci, ggplot2::aes(x = class, y = mean, color = class)) +
  ggplot2::geom_pointrange(ggplot2::aes(ymin = ci_low, ymax = ci_high),
                           size = 0.5) +
  ggplot2::facet_wrap(~variable, scales = "free_y") +
  ggplot2::labs(title = "Class means with 95% confidence intervals", x = "Class",
                y = "Estimated class mean") +
  analysis_theme() + ggplot2::theme(legend.position = "none")
save_plot(p_ci, "03_mean_confidence_intervals.png", 10, 4)
message("Tests complete. Main comparisons are Welch + local association;",
        " Wilcoxon is a sensitivity analysis, not a test of mean equality.")

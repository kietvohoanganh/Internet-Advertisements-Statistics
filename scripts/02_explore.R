# Summarize the data and plot descriptive statistics.
source("R/utils.R", local = TRUE)
require_input(project_file("data", "processed", "analysis_raw.rds"),
              "Run Rscript scripts/01_prepare.R first.")
if (!requireNamespace("ggplot2", quietly = TRUE)) stop("Install ggplot2")
raw <- readRDS(project_file("data", "processed", "analysis_raw.rds"))
vars <- c("height", "width", "aratio")

summaries <- do.call(rbind, lapply(vars, function(v) {
  do.call(rbind, lapply(levels(raw$class), function(group) {
    x <- raw[[v]][raw$class == group]
    data.frame(variable = v, class = group, n = sum(!is.na(x)),
               missing = sum(is.na(x)), mean = safe_mean(x),
               median = safe_median(x), sd = safe_sd(x),
               min = if (all(is.na(x))) NA_real_ else min(x, na.rm = TRUE),
               max = if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE),
               q1 = if (all(is.na(x))) NA_real_ else unname(quantile(x, 0.25, na.rm = TRUE)),
               q3 = if (all(is.na(x))) NA_real_ else unname(quantile(x, 0.75, na.rm = TRUE)))
  }))
}))
write_table(summaries, "02_descriptive_by_class.csv")

local_distribution <- as.data.frame(table(class = raw$class, local = raw$local,
                                          useNA = "ifany"))
names(local_distribution)[3] <- "count"
write_table(local_distribution, "02_local_by_class.csv")

correlation <- stats::cor(raw[vars], use = "pairwise.complete.obs")
cor_long <- as.data.frame(as.table(correlation))
names(cor_long) <- c("variable_1", "variable_2", "correlation")
write_table(cor_long, "02_correlation.csv")

long <- do.call(rbind, lapply(vars, function(v) {
  data.frame(variable = v, value = raw[[v]], class = raw$class)
}))
long <- long[!is.na(long$value), , drop = FALSE]

p_hist <- ggplot2::ggplot(long, ggplot2::aes(x = value, fill = class)) +
  ggplot2::geom_histogram(bins = 32, alpha = 0.75, colour = "white") +
  ggplot2::facet_grid(class ~ variable, scales = "free") +
  ggplot2::scale_fill_manual(values = c(ad = "#227969", nonad = "#adbecb")) +
  ggplot2::labs(title = "Distribution of geometric features", x = "Observed value",
                y = "Number of images", fill = "Class") +
  analysis_theme()
save_plot(p_hist, "02_histograms.png", 12, 6)

p_box <- ggplot2::ggplot(long, ggplot2::aes(x = class, y = value, fill = class)) +
  ggplot2::geom_boxplot(outlier.alpha = 0.2) +
  ggplot2::facet_wrap(~variable, scales = "free_y", nrow = 1) +
  ggplot2::scale_fill_manual(values = c(ad = "#227969", nonad = "#adbecb")) +
  ggplot2::labs(title = "Geometric features by class", x = "Class", y = "Observed value") +
  analysis_theme() + ggplot2::theme(legend.position = "none")
save_plot(p_box, "02_boxplots.png", 11, 4)

p_scatter <- ggplot2::ggplot(raw, ggplot2::aes(x = width, y = height, color = class)) +
  ggplot2::geom_point(alpha = 0.28, size = 0.8, na.rm = TRUE) +
  ggplot2::facet_wrap(~class) +
  ggplot2::labs(title = "Image height and width (observed values)",
                x = "Width", y = "Height") +
  analysis_theme()
save_plot(p_scatter, "02_scatter_width_height.png")

p_cor <- ggplot2::ggplot(cor_long, ggplot2::aes(x = variable_1, y = variable_2,
                                                fill = correlation)) +
  ggplot2::geom_tile(color = "white") +
  ggplot2::geom_text(ggplot2::aes(label = sprintf("%.2f", correlation))) +
  ggplot2::scale_fill_gradient2(limits = c(-1, 1), low = "#607d97", mid = "white",
                                high = "#227969", midpoint = 0) +
  ggplot2::labs(title = "Correlation of geometric features", x = NULL, y = NULL, fill = "Pearson r") +
  analysis_theme()
save_plot(p_cor, "02_correlation_heatmap.png", 6, 5)

p_local <- ggplot2::ggplot(local_distribution,
                          ggplot2::aes(x = local, y = count, fill = class)) +
  ggplot2::geom_col(position = "dodge") +
  ggplot2::labs(title = "Local indicator by class", x = "Local (0/1)",
                y = "Number of images") + analysis_theme()
save_plot(p_local, "02_local_barplot.png", 7, 4.5)
message("EDA complete: outputs/tables/02_* and outputs/figures/02_*")

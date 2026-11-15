# Run without network access or optional packages.
source("R/utils.R", local = TRUE)
source("R/data_io.R", local = TRUE)
source("R/model_metrics.R", local = TRUE)
source("R/ad_classifier.R", local = TRUE)
separated <- suppressWarnings(stats::glm(c(0, 0, 0, 1, 1, 1) ~ seq_len(6L),
                                          family = stats::binomial()))
stopifnot(any(!extract_odds_ratios(separated)$stable_wald_interval))

set.seed(9)
n <- 80L
raw_matrix <- matrix(sample(0:1, n * 1558L, replace = TRUE, prob = c(0.95, 0.05)),
                     nrow = n, ncol = 1558L)
raw_matrix[, 1L] <- sample(30:600, n, replace = TRUE)
raw_matrix[, 2L] <- sample(40:900, n, replace = TRUE)
raw_matrix[, 3L] <- round(raw_matrix[, 2L] / raw_matrix[, 1L], 3L)
raw_matrix[, 4L] <- sample(0:1, n, replace = TRUE)
labels <- rep(c("ad.", "nonad."), each = n / 2L)
raw_matrix[c(2L, 4L, 6L), 1L] <- NA

fixture_dir <- tempfile(pattern = "ad_fixture_")
dir.create(fixture_dir)
write.table(cbind(raw_matrix, labels), file.path(fixture_dir, "ad.data"),
            sep = ",", quote = FALSE, row.names = FALSE, col.names = FALSE,
            na = "?")
parsed <- read_ads_data(fixture_dir)
stopifnot(nrow(parsed$data) == n,
          ncol(parsed$data) == 1560L,
          identical(as.integer(table(parsed$data$target)), c(40L, 40L)),
          sum(is.na(parsed$data$height)) == 3L,
          nrow(parsed$dictionary) == 1558L)

sp <- stratified_split(parsed$data$target, p = 0.8, seed = 123L)
stopifnot(length(intersect(sp$train, sp$test)) == 0L,
          length(sp$train) + length(sp$test) == n,
          sum(parsed$data$target[sp$train]) == 32L,
          sum(parsed$data$target[sp$test]) == 8L)

prep <- fit_preprocessor(parsed$data[sp$train, ])
xtr <- apply_preprocessor(parsed$data[sp$train, ], prep)
xts <- apply_preprocessor(parsed$data[sp$test, ], prep)
stopifnot(!anyNA(xtr[prep$columns]), !anyNA(xts[prep$columns]),
          all(c("height_missing", "width_missing", "aratio_missing") %in% names(xts)))

# Expected counts: tp=1, tn=2, fp=1, fn=0.
met <- classification_metrics(c(1L, 0L, 0L, 0L), c(0.9, 0.4, 0.3, 0.6))
stopifnot(met$tp == 1L, met$tn == 2L, met$fp == 1L, met$fn == 0L,
          identical(met$accuracy, 0.75), abs(met$roc_auc - 1) < 1e-10,
          abs(met$average_precision - 1) < 1e-10)

# Test values must not affect fitted preprocessing.
old <- prep$median
changed_test <- parsed$data[sp$test, ]
changed_test$height[] <- 999999
invisible(apply_preprocessor(changed_test, prep))
stopifnot(identical(old, prep$median))

# Kaggle exports may use numeric column names.
csv_dir <- tempfile(pattern = "ad_csv_")
dir.create(csv_dir)
colnames(raw_matrix) <- as.character(0:1557)
headered <- as.data.frame(cbind(raw_matrix, labels), check.names = FALSE)
colnames(headered) <- as.character(0:1558)
utils::write.csv(headered, file.path(csv_dir, "ads.csv"), row.names = FALSE, na = "?")
parsed_csv <- read_ads_data(csv_dir)
stopifnot(nrow(parsed_csv$data) == n, sum(is.na(parsed_csv$data$height)) == 3L)

# All-negative predictions have F1=0; tied scores have AUC=.5 and AP=.5.
zero <- classification_metrics(c(1L, 0L), c(0.1, 0.1))
stopifnot(zero$f1 == 0, is.na(zero$precision), zero$recall == 0,
          zero$roc_auc == 0.5, zero$average_precision == 0.5)

# Break accuracy ties by recall and include scores equal to the cutoff.
scores <- rank_classifier_settings(array(c(.25, .35, .01, .9), c(4, 1, 1)),
  c(1L, 0L, 0L, 1L), 1, .01, c(.25, .5))
stopifnot(scores$threshold[1L] == .25, scores$tp[1L] == 2L, all(scores$correct == 3L))

# Retain the first row of headerless CSVs.
utils::write.table(cbind(raw_matrix, labels), file.path(csv_dir, "ads.csv"),
  sep = ",", quote = FALSE, row.names = FALSE, col.names = FALSE, na = "?")
stopifnot(nrow(read_ads_data(csv_dir)$data) == n)

# Remove row indices only when they are sequential.
utils::write.csv(headered, file.path(csv_dir, "ads.csv"), row.names = TRUE, na = "?")
stopifnot(nrow(read_ads_data(csv_dir)$data) == n)

# Reject infinite geometry before preprocessing.
headered[[1L]][1L] <- "Inf"
utils::write.csv(headered, file.path(csv_dir, "ads.csv"), row.names = FALSE, na = "?")
bad <- tryCatch(read_ads_data(csv_dir), error = identity)
stopifnot(inherits(bad, "error"))

# Accept Latin-1 tokens in the UCI dictionary.
names_file <- file.path(fixture_dir, "legacy.names")
writeBin(c(charToRaw("height: continuous.\norigurl*"), as.raw(0xae),
           charToRaw("ion: 0,1.\n")), names_file)
feature_names <- read_feature_names(names_file)
stopifnot(length(feature_names) == 2L, all(validUTF8(feature_names)))

# Keep conflicting labels for one profile in the same group.
gy <- rep(c(0L, 1L), 20L)
gg <- rep(seq_len(20L), each = 2L)
gsp <- stratified_split(gy, groups = gg, seed = 7L)
stopifnot(!length(intersect(gg[gsp$train], gg[gsp$test])),
          all(c(0L, 1L) %in% gy[gsp$test]))
folds <- group_partitions(gy, gg, rep(1, 5), seed = 7L)
stopifnot(all(vapply(base::split(folds, gg), function(x) length(unique(x)) == 1L, logical(1))))
# Do not merge distinct continuous values through rounding.
profiles <- data.frame(height = c(1, 1 + 1e-12, 1), width = 2, aratio = 2, local = 0,
                       target = c(0L, 0L, 1L), class = c("nonad", "nonad", "ad"))
stopifnot(identical(predictor_groups(profiles), c(1L, 2L, 1L)))
ci <- bootstrap_metrics(gy, rep(.2, length(gy)), gg, reps = 30L, seed = 1L)
stopifnot(ci$ci_low[ci$metric == "roc_auc"] == .5,
          ci$ci_high[ci$metric == "roc_auc"] == .5,
          all(ci$valid_replicates <= 30L))

unlink(c(fixture_dir, csv_dir), recursive = TRUE)
cat("PASS: importer, CSV header/index, finite values, split, train-only imputation, F1, tied AUC/AP\n")

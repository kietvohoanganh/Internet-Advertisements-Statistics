# Validate inputs and create grouped partitions.
source("R/utils.R", local = TRUE)
source("R/data_io.R", local = TRUE)
ensure_directories()

imported <- read_ads_data()
raw_data <- imported$data
write.csv(imported$dictionary, project_file("data", "processed", "feature_dictionary.csv"),
          row.names = FALSE, na = "")
saveRDS(raw_data, project_file("data", "processed", "analysis_raw.rds"))
unique_data <- raw_data[!duplicated(raw_data), , drop = FALSE]
saveRDS(unique_data, project_file("data", "processed", "analysis_unique.rds"))
groups <- predictor_groups(raw_data)
saveRDS(groups, project_file("data", "processed", "group_ids.rds"))

split <- stratified_split(raw_data$target, p = config$train_fraction, seed = config$seed,
                           groups = groups)
split$train_fit <- split$train[!duplicated(raw_data[split$train, , drop = FALSE])]
split$test_unique <- which(!duplicated(raw_data[split$test, , drop = FALSE]))
stopifnot(!length(intersect(groups[split$train], groups[split$test])))
train_raw <- raw_data[split$train_fit, , drop = FALSE]
test_raw <- raw_data[split$test, , drop = FALSE]
prep <- fit_preprocessor(train_raw)
train <- apply_preprocessor(train_raw, prep)
test <- apply_preprocessor(test_raw, prep)

saveRDS(split, project_file("data", "processed", "split.rds"))
saveRDS(prep, project_file("data", "processed", "preprocessor.rds"))
saveRDS(train, project_file("data", "processed", "train_prepared.rds"))
saveRDS(test, project_file("data", "processed", "test_prepared.rds"))

missing_summary <- data.frame(
  feature = names(raw_data)[seq_len(1558L)],
  n_missing = colSums(is.na(raw_data[seq_len(1558L)])),
  missing_rate = colMeans(is.na(raw_data[seq_len(1558L)]))
)
write_table(missing_summary, "01_missing_values.csv")
write_table(data.frame(class = c("nonad", "ad"),
                       n = c(sum(raw_data$target == 0L), sum(raw_data$target == 1L))),
            "01_class_distribution.csv")
write_table(data.frame(stage = c("all", "train", "train_fit", "test", "test_unique"),
                       rows = c(nrow(raw_data), length(split$train), nrow(train),
                                nrow(test), length(split$test_unique)),
                       ad_count = c(sum(raw_data$target), sum(raw_data$target[split$train]),
                         sum(train$target), sum(test$target), sum(test$target[split$test_unique]))),
            "01_split_overview.csv")
write_table(data.frame(row_id = seq_len(nrow(raw_data)), group_id = groups,
  target = raw_data$target, partition = ifelse(seq_len(nrow(raw_data)) %in% split$train, "train", "test"),
  used_for_training = seq_len(nrow(raw_data)) %in% split$train_fit), "01_row_partitions.csv")

cat("Source:", imported$source_file, "\n")
cat("Observations:", nrow(raw_data), "| predictors: 1558\n")
cat("Train rows:", length(split$train), "| unique training rows:", nrow(train), "| test:", nrow(test), "\n")
cat("Duplicate complete rows:", sum(duplicated(raw_data)), "| overlapping train/test groups: 0\n")
cat("Important: analysis_raw.rds preserves NA; train/test use train-fitted imputation.\n")

write_table(data.frame(
  source_file = imported$source_file,
  source_md5 = unname(tools::md5sum(file.path("data/raw", imported$source_file))),
  rows = nrow(raw_data), predictors = 1558L,
  duplicate_rows = sum(duplicated(raw_data)),
  unique_rows = nrow(unique_data), unique_profiles = length(unique(groups)),
  conflicting_label_groups = sum(vapply(split(raw_data$target, groups), function(y) length(unique(y)) > 1L, logical(1))),
  overlapping_groups = length(intersect(groups[split$train], groups[split$test])),
  rows_missing_geometry = sum(!stats::complete.cases(raw_data[c("height", "width", "aratio")]))
), "01_data_quality.csv")
profile_counts <- data.frame(group_id = seq_len(max(groups)),
  ad = tabulate(groups[raw_data$target == 1], max(groups)),
  nonad = tabulate(groups[raw_data$target == 0], max(groups)))
write_table(profile_counts[profile_counts$ad > 0 & profile_counts$nonad > 0, ], "01_conflicting_profiles.csv")
missing_by_class <- do.call(rbind, lapply(c("height", "width", "aratio"), function(name) {
  do.call(rbind, lapply(c("ad", "nonad"), function(label) {
    observed <- raw_data[[name]][raw_data$class == label]
    data.frame(feature = name, class = label, n = length(observed), missing = sum(is.na(observed)),
               missing_rate = mean(is.na(observed)))
  }))
}))
write_table(missing_by_class, "01_missing_by_class.csv")
saveRDS(list(config = config, source_file = imported$source_file,
             source_md5 = unname(tools::md5sum(file.path("data/raw", imported$source_file))),
             created_at = format(Sys.time(), tz = "UTC", usetz = TRUE)),
        project_file("outputs", "run_info.rds"))
writeLines(capture.output(utils::sessionInfo()), project_file("outputs", "session_info.txt"))
sources <- c("config.R", list.files("R", "\\.R$", full.names = TRUE),
  list.files("scripts", "\\.(R|py)$", full.names = TRUE),
  list.files("tests", "\\.(R|py)$", full.names = TRUE),
  list.files("classifier", "\\.(R|py)$", full.names = TRUE),
  "requirements.txt", "Makefile")
environment_files <- c("renv.lock", ".Rprofile", "renv/activate.R")
sources <- c(sources, environment_files[file.exists(environment_files)])
utils::write.csv(data.frame(file = sources, md5 = unname(tools::md5sum(sources))),
  project_file("outputs", "source_checksums.csv"), row.names = FALSE)

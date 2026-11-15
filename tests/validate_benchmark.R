# Recompute metrics without refitting or changing model selection.
comparison <- read.csv("outputs/benchmark/comparison.csv", stringsAsFactors = FALSE)
oof_rows <- read.csv("outputs/benchmark/oof_predictions.csv", stringsAsFactors = FALSE)
outer <- read.csv("outputs/benchmark/outer_folds.csv")
stopifnot(all(vapply(base::split(outer$fold, outer$group_id), function(x) length(unique(x)) == 1L, logical(1))))
for (i in seq_len(nrow(comparison))) {
  result <- comparison[i, ]; rows <- oof_rows[oof_rows$model == result$model & oof_rows$objective == result$objective, ]
  stopifnot(!anyDuplicated(rows$row_id), setequal(rows$row_id, outer$row_id),
    !length(intersect(rows$group_id, groups[split$test])),
    identical(rows$target, raw$target[rows$row_id]), all(rows$prediction %in% 0:1))
  computed <- classification_metrics(rows$target, rows$prediction)
  fields <- c("accuracy", "precision", "recall", "f1", "f2", "balanced_accuracy", "mcc", "tp", "fp", "tn", "fn")
  stopifnot(isTRUE(all.equal(as.numeric(computed[1, fields]), as.numeric(result[1, fields]), tolerance = 1e-10)))
  if(result$model=="pattern_correction") {
    stopifnot(is.na(result$roc_auc),is.na(result$average_precision))
  } else {
    ranked <- classification_metrics(rows$target,rows$score)
    stopifnot(isTRUE(all.equal(as.numeric(ranked[1,c("roc_auc","average_precision")]),
                              as.numeric(result[1,c("roc_auc","average_precision")]),tolerance=1e-10)))
  }
}
selected <- read.csv("outputs/models/selected.csv", stringsAsFactors = FALSE)
stopifnot(nrow(selected) == 1L, identical(unname(tools::md5sum(selected$artifact)), selected$artifact_md5),
  selected$protocol == "nested_group_oof_train_only", selected$historical_test == "development_exposed")
for (fold in seq_len(info$model_config$benchmark_folds)) {
  part <- read.csv(file.path("outputs/benchmark", paste0("fold", fold), "partitions.csv"))
  fitting <- part[part$part == "train", ]; validation <- part[part$part == "validation", ]
  stopifnot(!length(intersect(fitting$group_id, validation$group_id)),
    all(vapply(base::split(fitting$fold, fitting$group_id), function(x) length(unique(x)) == 1L, logical(1))))
}
cat("PASS: matched nested predictions, all metrics, selected artifact and inner/outer group isolation\n")
# Replay the saved model through the public prediction interface.
verify_selected_predictions <- function() {
  source("scripts/predict.R", local = TRUE)
  input <- tempfile(fileext=".rds"); output <- tempfile(fileext=".csv")
  on.exit(unlink(c(input,output)),add=TRUE)
  dictionary <- read.csv("data/processed/feature_dictionary.csv")
  saveRDS(raw[split$test,dictionary$feature,drop=FALSE],input)
  replay <- predict_file(input,output)
  held <- read.csv("outputs/benchmark/development_test_predictions.csv",stringsAsFactors=FALSE)
  held <- held[held$model==selected$model & held$objective==selected$objective,]
  stopifnot(identical(held$row_id,split$test),identical(held$target,raw$target[split$test]),
            identical(as.integer(replay$predicted_class=="ad"),as.integer(held$prediction)))
  measured <- read.csv("outputs/benchmark/development_test_metrics.csv",stringsAsFactors=FALSE)
  measured <- measured[measured$model==selected$model & measured$objective==selected$objective,]
  counts <- classification_metrics(held$target,held$prediction)
  fields <- c("accuracy","precision","recall","f1","f2","tp","fp","tn","fn")
  stopifnot(nrow(measured)==1L,isTRUE(all.equal(as.numeric(counts[1,fields]),as.numeric(measured[1,fields]),tolerance=1e-10)))
  if(selected$model!="pattern_correction") {
    stopifnot(isTRUE(all.equal(replay$ad_score,held$score,tolerance=1e-12)))
    ranked <- classification_metrics(held$target,held$score)
    stopifnot(isTRUE(all.equal(as.numeric(ranked[1,c("roc_auc","average_precision")]),
                              as.numeric(measured[1,c("roc_auc","average_precision")]),tolerance=1e-10)))
  } else stopifnot(is.na(measured$roc_auc),is.na(measured$average_precision))
}
verify_selected_predictions()
cat("PASS: serialized selected model replays every historical-test class and available score\n")

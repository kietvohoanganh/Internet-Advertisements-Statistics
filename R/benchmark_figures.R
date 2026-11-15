benchmark_figures <- function() {
  comparison <- read.csv("outputs/benchmark/comparison.csv")
  result <- comparison[comparison$objective == "f2_precision", ]
  result$model <- factor(result$model, levels = result$model[order(result$f2)])
  long <- do.call(rbind, lapply(c("precision","recall","f2"), function(metric) {
    data.frame(model=result$model, metric=metric, value=result[[metric]])
  }))
  plot <- ggplot2::ggplot(long, ggplot2::aes(x=model,y=value,fill=metric)) +
    ggplot2::geom_col(position="dodge") + ggplot2::coord_flip() +
    ggplot2::labs(title="Nested grouped CV on training records",x=NULL,y="Score",fill="Metric") + analysis_theme()
  save_plot(plot,"05_model_comparison.png",10,5)
  importance <- read.csv("outputs/benchmark/permutation_importance.csv")
  means <- aggregate(ap_drop_mean~feature_family,importance,mean)
  plot <- ggplot2::ggplot(means,ggplot2::aes(x=reorder(feature_family,ap_drop_mean),y=ap_drop_mean)) +
    ggplot2::geom_col(fill="#227969") + ggplot2::coord_flip() +
    ggplot2::labs(title="Random Forest - validation group permutation importance",
      subtitle="Mean AP decrease across outer folds; diagnostic only",x=NULL,y="AP decrease") + analysis_theme()
  save_plot(plot,"05_permutation_importance.png",9,4.5)
  selected <- read.csv("outputs/models/selected.csv")
  predictions <- read.csv("outputs/benchmark/development_test_predictions.csv")
  predictions <- predictions[predictions$model==selected$model & predictions$objective==selected$objective, ]
  raw <- readRDS("data/processed/analysis_raw.rds")
  error <- predictions[predictions$prediction!=predictions$target, ]
  error$geometry_missing <- !complete.cases(raw[error$row_id,c("height","width","aratio")])
  error$active_text_terms <- rowSums(raw[error$row_id,5:1558,drop=FALSE]==1,na.rm=TRUE)
  write.csv(error,"outputs/benchmark/selected_test_errors.csv",row.names=FALSE)
  confusion <- as.data.frame(table(actual=factor(predictions$target,levels=0:1,labels=c("nonad","ad")),
                                   predicted=factor(predictions$prediction,levels=0:1,labels=c("nonad","ad"))))
  plot <- ggplot2::ggplot(confusion,ggplot2::aes(x=predicted,y=actual,fill=Freq)) +
    ggplot2::geom_tile(colour="white") + ggplot2::geom_text(ggplot2::aes(label=Freq),size=6) +
    ggplot2::scale_fill_gradient(low="#edf5f0",high="#6aa18a",guide="none") +
    ggplot2::labs(title=paste("Selected classifier:",selected$model),
      subtitle="Development-exposed historical test; not external validation",x="Predicted",y="Actual") + analysis_theme()
  save_plot(plot,"05_selected_confusion.png",7,4.5)
  thresholds <- read.csv("outputs/benchmark/full/threshold_curve.csv")
  thresholds <- thresholds[thresholds$objective=="f2_precision", ]
  long <- do.call(rbind,lapply(c("precision","recall","f2"),function(metric) {
    data.frame(model=thresholds$model,threshold=thresholds$threshold,metric=metric,value=thresholds[[metric]])
  }))
  plot <- ggplot2::ggplot(long,ggplot2::aes(x=threshold,y=value,colour=metric)) +
    ggplot2::geom_line(na.rm=TRUE) + ggplot2::facet_wrap(~model) +
    ggplot2::labs(title="Training-only threshold selection",subtitle="Inner OOF selection scores, not an independent performance estimate",
                  x="Threshold (SVM uses sigmoid margin)",y="Score",colour="Metric") + analysis_theme()
  save_plot(plot,"05_threshold_tradeoff.png",11,6)
  oof <- read.csv("outputs/benchmark/oof_predictions.csv")
  curves <- lapply(unique(oof$model), function(name) {
    rows <- oof[oof$model==name & oof$objective=="f2_precision", ]
    if (name=="pattern_correction") return(NULL)
    data <- probability_curves(rows$target,rows$score)
    list(roc=data.frame(model=name,data$roc),pr=data.frame(model=name,data$pr))
  })
  roc <- do.call(rbind,lapply(Filter(Negate(is.null),curves),`[[`,"roc"))
  pr <- do.call(rbind,lapply(Filter(Negate(is.null),curves),`[[`,"pr"))
  for (kind in c("roc","pr")) {
    data <- if(kind=="roc") roc else pr
    x <- if(kind=="roc") "fpr" else "recall";y <- if(kind=="roc") "tpr" else "precision"
    plot <- ggplot2::ggplot(data,ggplot2::aes(x=.data[[x]],y=.data[[y]],colour=model)) +
      ggplot2::geom_path() + ggplot2::labs(title=paste("Nested training OOF",toupper(kind)),
        subtitle="Component scores; correction has no standalone ranking score",x=x,y=y) + analysis_theme()
    save_plot(plot,paste0("05_nested_",kind,".png"),8,5)
  }
}

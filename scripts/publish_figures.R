# Draw the four public figures from validated results.
source("R/utils.R", local = TRUE)
source("R/model_metrics.R", local = TRUE)
arguments <- commandArgs(trailingOnly=TRUE)
if (length(arguments)!=1L) stop("Usage: Rscript scripts/publish_figures.R STAGING_RESULTS")
stage <- arguments[1L]
models <- read.csv(file.path(stage,"model_comparison.csv"),stringsAsFactors=FALSE)
labels <- c(logistic="Geometry logistic",lasso="LASSO",elastic_net="Elastic Net",random_forest="Random Forest",
            similarity_svm="Similarity SVM",pattern_correction="Pattern correction")
cv <- models[models$protocol=="nested_group_oof_train_only",]
selected <- cv$model[tolower(as.character(cv$selected))=="true"]
if(length(selected)!=1L) stop("Expected one selected CV model")
classes <- read.csv("outputs/tables/01_class_distribution.csv")
colours <- c(ad="#277565",nonad="#a9bac7")
theme <- analysis_theme() + ggplot2::theme(plot.title=ggplot2::element_text(size=16),legend.title=ggplot2::element_blank())
export <- function(plot,name,width=8,height=5) {
  ggplot2::ggsave(file.path(stage,"figures",name),plot=plot,width=width,height=height,dpi=180,bg="white")
}
plot <- ggplot2::ggplot(classes,ggplot2::aes(x=class,y=n,fill=class)) +
  ggplot2::geom_col(width=.55,show.legend=FALSE) +
  ggplot2::geom_text(ggplot2::aes(label=sprintf("%s (%.1f%%)",format(n,big.mark=","),100*n/sum(n))),vjust=-.5,size=4.2) +
  ggplot2::scale_fill_manual(values=colours) +
  ggplot2::scale_y_continuous(expand=ggplot2::expansion(mult=c(0,.15))) +
  ggplot2::labs(title="Class distribution",subtitle=sprintf("Original source records (n = %s)",format(sum(classes$n),big.mark=",")),x=NULL,y="Images") + theme
export(plot,"class_distribution.png",8,4)
cv$label <- factor(unname(labels[cv$model]),levels=unname(labels[cv$model[order(cv$f2)]]))
cv$is_selected <- cv$model==selected
plot <- ggplot2::ggplot(cv,ggplot2::aes(x=f2,y=label,colour=is_selected)) +
  ggplot2::geom_pointrange(ggplot2::aes(xmin=f2_ci_low,xmax=f2_ci_high),linewidth=.6,size=.8) +
  ggplot2::scale_colour_manual(values=c("FALSE"="#8295a6","TRUE"="#277565"),guide="none") +
  ggplot2::scale_x_continuous(limits=c(0,1),breaks=seq(0,1,.2)) +
  ggplot2::labs(title="Model comparison on matched training records",
    subtitle="Nested grouped CV; 95% group-bootstrap intervals (fixed fold predictions)",x="Advertisement F2",y=NULL) + theme
export(plot,"model_comparison.png",9,4.5)
held <- models[models$protocol=="development_exposed_historical_test",]
if(nrow(held)!=1L) stop("Expected selected historical-test counts")
confusion <- data.frame(actual=rep(c("ad","nonad"),each=2),predicted=rep(c("ad","nonad"),2),
  count=c(held$tp,held$fn,held$fp,held$tn))
confusion$percent <- 100*confusion$count/rep(c(held$tp+held$fn,held$fp+held$tn),each=2)
confusion$correct <- confusion$actual==confusion$predicted
confusion$actual <- factor(confusion$actual,levels=c("nonad","ad"))
plot <- ggplot2::ggplot(confusion,ggplot2::aes(x=predicted,y=actual,fill=correct)) +
  ggplot2::geom_tile(colour="white",linewidth=2) +
  ggplot2::geom_text(ggplot2::aes(label=sprintf("%d\n%.2f%%",count,percent)),size=5,lineheight=1.3) +
  ggplot2::scale_fill_manual(values=c("FALSE"="#f2e2db","TRUE"="#d9e9e2"),guide="none") +
  ggplot2::labs(title=paste(labels[selected],"- final decisions"),
    subtitle=sprintf("Development-exposed historical test (n = %d); percentages within actual class",held$n),
    x="Predicted class",y="Actual class") + theme + ggplot2::theme(panel.grid=ggplot2::element_blank())
export(plot,"confusion_matrix.png",8,5)
oof <- read.csv("outputs/benchmark/oof_predictions.csv",stringsAsFactors=FALSE)
curves <- do.call(rbind,lapply(setdiff(names(labels),"pattern_correction"),function(name) {
  rows <- oof[oof$model==name & oof$objective==cv$objective[1],]
  curve <- probability_curves(rows$target,rows$score)$pr
  data.frame(model=name,curve)
}))
ap <- setNames(cv$average_precision,cv$model)
legend_labels <- setNames(sprintf("%s (AP %.3f)",labels[names(ap)],ap),names(ap))
plot <- ggplot2::ggplot(curves,ggplot2::aes(x=recall,y=precision,colour=model)) +
  ggplot2::geom_step(direction="vh",linewidth=.7) +
  ggplot2::scale_colour_manual(values=c(logistic="#b87c62",lasso="#b09b58",elastic_net="#738ba4",random_forest="#7e769a",similarity_svm="#277565"),labels=legend_labels) +
  ggplot2::guides(colour=ggplot2::guide_legend(ncol=2,byrow=TRUE)) +
  ggplot2::labs(title="Precision-recall on nested CV scores",
    subtitle="SVM uses an uncalibrated margin score; correction has no standalone ranking score",x="Advertisement recall",y="Advertisement precision") + theme
export(plot,"precision_recall.png",9,6)
versions <- list(R=R.version.string,ggplot2=as.character(packageVersion("ggplot2")),
                 glmnet=as.character(packageVersion("glmnet")),jsonlite=as.character(packageVersion("jsonlite")))
jsonlite::write_json(versions,file.path(dirname(stage),"r_versions.json"),auto_unbox=TRUE)

source("R/utils.R")
source("R/data_io.R")
source("R/model_metrics.R")
source("R/ad_classifier.R")
source("R/benchmark.R")
# Sparse, correlated indicators can stall low-penalty cold starts.
set.seed(71)
x <- Matrix::rsparsematrix(120, 180, density = .03)
x@x <- abs(x@x)
y <- rep(0:1, 60)
x <- cbind(x, x[, 1:20])
penalties <- exp(seq(log(.05), log(.0005), length.out = 17))
warnings <- character()
fit <- withCallingHandlers(fit_penalty_path(x,y,.35,penalties),
                            warning = function(w) warnings <<- c(warnings, conditionMessage(w)))
stopifnot(!length(warnings), fit$jerr == 0, all(penalties %in% fit$lambda),
          all(is.finite(predict(fit,newx=x,s=penalties,type="response"))))
raw <- data.frame(height=c(1,2,3,4,NA,6),width=c(10,20,30,40,50,60),aratio=rep(10,6),local=rep(0:1,3),target=rep(0:1,3))
prep <- fit_preprocessor(raw)
for (bad in list(raw[0,],raw[setdiff(names(raw),"height")],transform(raw,local=2),transform(raw,width=Inf))) {
  stopifnot(inherits(tryCatch(apply_preprocessor(bad,prep),error=identity),"error"))
}
# Evaluation extremes must not affect fitted clipping bounds.
transformer <- fit_outlier_policy(raw,"clip")
before <- transformer
changed <- raw; changed$width <- 1e10
invisible(apply_outlier_policy(changed,transformer))
stopifnot(identical(before,transformer),nrow(apply_outlier_policy(changed,transformer))==nrow(changed))
met <- classification_metrics(c(1,0,0,0),c(.9,.6,.3,.1))
stopifnot(abs(met$f2-5/6)<1e-12,abs(met$balanced_accuracy-5/6)<1e-12,abs(met$mcc-2/sqrt(12))<1e-12)
cat("PASS: complete converged sparse paths, schema validation, training-only clipping, F2/balanced accuracy/MCC\n")

source("R/statistics.R")
stopifnot(!binary_association(c(0,1,0,1), rep(0,4))$result$estimable,
          binary_association(c(0,1,0,1), c(0,0,1,1))$result$test == "Fisher exact")

large <- classification_metrics(rep(c(1,0,0,0),1000),rep(c(.9,.6,.3,.1),1000))
stopifnot(is.finite(large$mcc),abs(large$mcc-met$mcc)<1e-12)

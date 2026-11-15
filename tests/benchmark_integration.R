# Run the R/Python benchmark in an isolated fixture.
benchmark_integration <- function() {
  root <- getwd()
  python <- Sys.getenv("PYTHON", unset = if (file.exists(".venv/bin/python")) file.path(root,".venv/bin/python") else Sys.which("python3"))
  if (!grepl("^/",python) && file.exists(python)) python <- file.path(root,python)
  previous_python <- Sys.getenv("PYTHON", unset = NA_character_)
  Sys.setenv(PYTHON = python)
  fixture <- tempfile("ads_nested_"); dir.create(fixture)
  on.exit({setwd(root);unlink(fixture,recursive=TRUE)
    if (is.na(previous_python)) Sys.unsetenv("PYTHON") else Sys.setenv(PYTHON=previous_python)},add=TRUE)
  stopifnot(all(file.copy(c("R","scripts","classifier","tests","docs","README.md","requirements.txt","Makefile"),fixture,recursive=TRUE)))
  setwd(fixture);dir.create("data/raw",recursive=TRUE)
  set.seed(37);n <- 200L
  x <- matrix(rbinom(n*1558,1,.01),n)
  y <- as.integer(seq_len(n) %% 4==0)
  x[,1] <- sample(20:150,n,TRUE);x[,2] <- sample(30:500,n,TRUE);x[,3] <- x[,2]/x[,1]
  x[,4] <- rbinom(n,1,.3);x[,5] <- y;x[seq(1,n,11),1] <- NA
  # Exercise both CV levels with duplicate and conflicting profiles.
  x <- rbind(x,x[1:6,],x[1,]);y <- c(y,y[1:6],1-y[1])
  write.table(cbind(x,ifelse(y==1,"ad.","nonad.")),"data/raw/ad.data",sep=",",quote=FALSE,
                row.names=FALSE,col.names=FALSE,na="?")
  writeLines(c("ad, nonad.","height: continuous.","width: continuous.","aratio: continuous.",
    "local: 0,1.","url*advert: 0,1.",paste0("url*term",6:1558,": 0,1.")),"data/raw/ad.names")
  writeLines(c("config <- list(seed=2026L,train_fraction=.8,threshold=.5,cv_folds=3L,",
    "run_lasso=TRUE,run_correction=FALSE,run_benchmark=TRUE,",
    "benchmark_folds=3L,inner_folds=3L,rf_trees=c(12L,16L),benchmark_jobs=1L,bootstrap_reps=20L,",
    "min_precision=.9,selection_objective='f2_precision',outlier_sensitivity=FALSE)"),"config.R")
  source("scripts/run.R",local=TRUE)
  stopifnot(all(read.csv("outputs/run_status.csv")$status=="passed"))
  source("scripts/predict.R",local=TRUE)
  raw <- readRDS("data/processed/analysis_raw.rds")
  dictionary <- read.csv("data/processed/feature_dictionary.csv")
  input <- raw[1:5,dictionary$feature];saveRDS(input,"features.rds")
  first <- predict_file("features.rds","first.csv")
  saveRDS(input[rev(names(input))],"reordered.rds")
  second <- predict_file("reordered.rds","second.csv")
  stopifnot(identical(first,second),all(first$predicted_class %in% c("ad","nonad")))
  # Check the Python model through the public R interface.
  manifest <- read.csv("outputs/models/selected.csv",stringsAsFactors=FALSE)
  original_manifest <- manifest
  manifest$engine <- "python";manifest$model <- "random_forest"
  manifest$artifact <- "outputs/benchmark/full/random_forest_f2_precision.joblib"
  manifest$artifact_md5 <- unname(tools::md5sum(manifest$artifact))
  write.csv(manifest,"outputs/models/selected.csv",row.names=FALSE)
  python_result <- predict_file("features.rds","forest.csv")
  stopifnot(nrow(python_result)==5L,all(is.finite(python_result$ad_score)))
  for (bad in list(input[0,],input[-1],transform(input,local=2))) {
    saveRDS(bad,"bad.rds")
    failure <- tryCatch(predict_file("bad.rds","bad.csv"),error=identity)
    stopifnot(inherits(failure,"error"),!file.exists("bad.csv"))
  }
  manifest$artifact_md5 <- "incorrect"
  write.csv(manifest,"outputs/models/selected.csv",row.names=FALSE)
  stopifnot(inherits(tryCatch(predict_file("features.rds","stale.csv"),error=identity),"error"),!file.exists("stale.csv"))
  write.csv(original_manifest,"outputs/models/selected.csv",row.names=FALSE)
  # Check repeatable publication and reject corrupted figures.
  status <- system2(python,c("scripts/publish_results.py"))
  stopifnot(status==0L,system2(python,c("scripts/publish_results.py","--check"))==0L)
  files <- list.files("results",recursive=TRUE,full.names=TRUE)
  stopifnot(length(files)==6L)
  before <- tools::md5sum(c(files,"README.md"))
  stopifnot(system2(python,c("scripts/publish_results.py"))==0L,
            identical(before,tools::md5sum(c(files,"README.md"))))
  original <- readBin("results/figures/confusion_matrix.png","raw",n=file.info("results/figures/confusion_matrix.png")$size)
  writeBin(as.raw(c(1,2,3)),"results/figures/confusion_matrix.png")
  bad <- suppressWarnings(system2(python,c("scripts/publish_results.py","--check"),stdout=FALSE,stderr=FALSE))
  stopifnot(bad!=0L)
  writeBin(original,"results/figures/confusion_matrix.png")
  cat("PASS: six-artifact export, byte-identical repeated publication, snapshot verification and tampering rejection\n")
  cat("PASS: nested six-model benchmark, R/Python deployment, serialization, deterministic scoring, invalid inputs, stale-artifact rejection\n")
}
benchmark_integration()

# Fit the legacy SVM and correction rule using training groups only.
source("R/utils.R", local = TRUE)
if (!isTRUE(config$run_lasso)) stop("The retained classifier requires run_lasso = TRUE.")
python <- Sys.getenv("PYTHON", unset = "")
if (!nzchar(python)) python <- if (file.exists(".venv/bin/python")) ".venv/bin/python" else "python3"
run_python <- function(arguments) {
  status <- system2(python, vapply(arguments, shQuote, character(1)))
  if (status != 0L) stop("Classifier step failed. Install requirements.txt and run again.")
}
unlink("outputs/classifier", recursive = TRUE)
unlink(c("outputs/models/similarity.joblib", "outputs/models/correction.joblib"))
source("classifier/folds.R", local = TRUE)
train_elastic_folds()
run_python(c("classifier/similarity.py", "run"))
source("classifier/evidence.R", local = TRUE)
export_evidence("outputs/training")
run_python(c("classifier/correction.py", "run", "outputs/training/contributions.csv"))

source("R/utils.R", local = TRUE)
lock <- renv::lockfile_read("renv.lock")
if (as.character(getRversion()) != lock$R$Version) {
  stop("Use R ", lock$R$Version, " to restore this environment.", call. = FALSE)
}
renv::restore(prompt = FALSE, retry = FALSE)
check_dependencies()
message("Dependencies ready. Next: Rscript scripts/run.R")

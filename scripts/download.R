# Download the original UCI dataset.
source("R/utils.R", local = TRUE)
ensure_directories()
raw_dir <- project_file("data", "raw")
if (file.exists(file.path(raw_dir, "ad.data"))) {
  message("data/raw/ad.data already exists; no download needed.")
} else {
  url <- "https://archive.ics.uci.edu/static/public/51/internet+advertisements.zip"
  zip_path <- tempfile(fileext = ".zip")
  temp_extract <- tempfile(pattern = "internet_ads_")
  dir.create(temp_extract)
  message("Downloading UCI dataset from: ", url)
  utils::download.file(url, zip_path, mode = "wb", quiet = FALSE)
  utils::unzip(zip_path, exdir = temp_extract)
  files <- list.files(temp_extract, recursive = TRUE, full.names = TRUE)
  for (name in c("ad.data", "ad.names", "ad.DOCUMENTATION")) {
    candidates <- files[basename(files) == name]
    if (length(candidates)) file.copy(candidates[1L], file.path(raw_dir, name),
                                      overwrite = FALSE)
  }
  unlink(c(zip_path, temp_extract), recursive = TRUE)
  if (!file.exists(file.path(raw_dir, "ad.data"))) {
    stop("Archive did not contain ad.data. Download manually from UCI/Kaggle.")
  }
  message("Ready: data/raw/ad.data")
}

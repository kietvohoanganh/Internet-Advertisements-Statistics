is_numeric_header <- function(fields) {
  if (length(fields) < 20L) return(FALSE)
  vals <- suppressWarnings(as.integer(fields))
  if (anyNA(vals)) return(FALSE)
  identical(vals, seq.int(0L, length(fields) - 1L)) ||
    identical(vals, seq_len(length(fields)))
}

has_csv_header <- function(path) {
  first <- readLines(path, n = 1L, warn = FALSE)
  if (!length(first)) stop("Empty input file: ", path)
  fields <- trimws(strsplit(first, ",", fixed = TRUE)[[1L]])
  fields <- sub('^"(.*)"$', '\\1', fields)
  if (is_numeric_header(fields)) return(TRUE)
  if (length(fields) < 5L) return(FALSE)
  # Data rows start with numbers or missing values and end with a label.
  numeric_head <- all(fields[1:4] %in% c("?", "", "NA") |
                        !is.na(suppressWarnings(as.numeric(fields[1:4]))))
  last <- tolower(gsub("[^a-z0-9]", "", tail(fields, 1L)))
  if (numeric_head && last %in% c("ad", "nonad", "0", "1")) return(FALSE)
  TRUE
}

read_feature_names <- function(names_path) {
  if (!file.exists(names_path)) return(character(0))
  lines <- readLines(names_path, warn = FALSE)
  # Some UCI URL tokens use Latin-1 encoding.
  invalid_utf8 <- !validUTF8(lines)
  lines[invalid_utf8] <- iconv(lines[invalid_utf8], from = "latin1", to = "UTF-8")
  lines <- trimws(lines)
  # UCI format: `height: continuous.` or `url*images+buttons: 0,1.`
  candidates <- lines[grepl("^[^|#[:space:]][^:]*:", lines)]
  trimws(sub(":.*$", "", candidates))
}

read_ads_data <- function(raw_dir = project_file("data", "raw")) {
  ad_path <- file.path(raw_dir, "ad.data")
  if (file.exists(ad_path)) {
    input_path <- ad_path
    header <- FALSE
  } else {
    csv_files <- list.files(raw_dir, pattern = "\\.csv$", full.names = TRUE,
                            ignore.case = TRUE)
    if (length(csv_files) != 1L) {
      stop("Place ad.data or exactly one Kaggle CSV in data/raw/. ",
           "Found ", length(csv_files), " CSV files. See README.md.", call. = FALSE)
    }
    input_path <- csv_files[1L]
    header <- has_csv_header(input_path)
  }

  raw <- utils::read.csv(input_path, header = header, check.names = FALSE,
                         stringsAsFactors = FALSE, strip.white = TRUE,
                         na.strings = c("?", "", "NA"), fill = FALSE)
  # Some Kaggle exports include a sequential row index.
  if (ncol(raw) == 1560L) {
    idx <- suppressWarnings(as.numeric(raw[[1L]]))
    if (!anyNA(idx) && (identical(idx, as.numeric(seq_len(nrow(raw)))) ||
                        identical(idx, as.numeric(seq_len(nrow(raw)) - 1L)))) {
      raw <- raw[-1L]
    }
  }
  if (ncol(raw) != 1559L) {
    stop("Expected 1558 predictors + 1 target (1559 columns); found ",
         ncol(raw), ". Check CSV header/index/format.", call. = FALSE)
  }

  raw_names <- read_feature_names(file.path(raw_dir, "ad.names"))
  original_names <- if (length(raw_names) == 1558L) {
    raw_names
  } else if (header) {
    names(raw)[seq_len(1558L)]
  } else {
    rep(NA_character_, 1558L)
  }
  clean_names <- c("height", "width", "aratio", "local",
                   sprintf("f%04d", seq.int(5L, 1558L)))
  dictionary <- data.frame(
    index = seq_len(1558L), feature = clean_names,
    original_name = original_names,
    feature_type = c(rep("continuous", 3L), rep("binary", 1555L)),
    stringsAsFactors = FALSE
  )

  raw_labels <- sub("[.]$", "", tolower(trimws(as.character(raw[[1559L]]))))
  unexpected <- setdiff(unique(raw_labels), c("ad", "nonad"))
  if (length(unexpected)) {
    stop("Unexpected target label(s): ", paste(unexpected, collapse = ", "),
         ". Expected ad./nonad.", call. = FALSE)
  }
  if (anyNA(raw_labels) || !all(c("ad", "nonad") %in% raw_labels)) {
    stop("Both ad. and nonad. classes must be present; no missing labels.")
  }

  x <- raw[seq_len(1558L)]
  for (j in seq_along(x)) {
    val <- trimws(as.character(x[[j]]))
    val[val %in% c("?", "", "NA", "NaN")] <- NA_character_
    numeric_val <- suppressWarnings(as.numeric(val))
    if (any(!is.na(val) & !is.finite(numeric_val))) {
      stop("Non-numeric predictor value in column ", j, call. = FALSE)
    }
    if (j >= 4L && any(!is.na(numeric_val) & !numeric_val %in% c(0, 1))) {
      stop("Non-binary predictor value in column ", j, call. = FALSE)
    }
    x[[j]] <- numeric_val
  }
  names(x) <- clean_names
  x$target <- as.integer(raw_labels == "ad")
  x$class <- factor(ifelse(x$target == 1L, "ad", "nonad"),
                    levels = c("ad", "nonad"))
  attr(x, "input_file") <- basename(input_path)
  list(data = x, dictionary = dictionary, source_file = basename(input_path))
}

predictor_groups <- function(data) {
  features <- data[setdiff(names(data), c("target", "class"))]
  # Preserve full precision when grouping continuous features.
  values <- lapply(seq_along(features), function(j) {
    x <- features[[j]]
    if (names(features)[j] %in% c("height", "width", "aratio")) sprintf("%.17g", x) else as.character(x)
  })
  keys <- do.call(paste, c(values, sep = "\r"))
  match(keys, unique(keys))
}

# Keep groups intact while balancing class counts; place large groups first.
# Use the seed only to break ties.
group_partitions <- function(y, groups, proportions, seed = 2026L) {
  stopifnot(length(y) == length(groups), !anyNA(groups), all(y %in% 0:1),
            length(proportions) >= 2L, all(proportions > 0))
  proportions <- proportions / sum(proportions)
  ids <- match(groups, unique(groups))
  k <- length(proportions)
  counts <- cbind(tabulate(ids[y == 0L], max(ids)), tabulate(ids[y == 1L], max(ids)))
  if (any(colSums(counts > 0) < k)) stop("Need at least k distinct groups per class")
  target <- outer(proportions, colSums(counts))
  allocated <- matrix(0, nrow = k, ncol = 2L)
  assignment <- integer(nrow(counts))
  set.seed(seed)
  for (g in order(-rowSums(counts), stats::runif(nrow(counts)))) {
    increment <- matrix(counts[g, ], nrow = k, ncol = 2L, byrow = TRUE)
    cost <- rowSums((2 * (allocated - target) * increment + increment^2) / pmax(target, 1))
    choices <- which(cost <= min(cost) + 1e-12)
    chosen <- choices[sample.int(length(choices), 1L)]
    assignment[g] <- chosen
    allocated[chosen, ] <- allocated[chosen, ] + counts[g, ]
  }
  if (any(allocated == 0)) stop("A partition lacks a class; reduce folds or review groups")
  assignment[ids]
}

stratified_split <- function(y, p = 0.80, seed = 2026L, groups = NULL) {
  stopifnot(all(y %in% c(0L, 1L)), p > 0, p < 1)
  if (!is.null(groups)) {
    partition <- group_partitions(y, groups, c(p, 1 - p), seed)
    return(list(train = which(partition == 1L), test = which(partition == 2L),
                seed = seed, train_fraction = p, method = "grouped_predictor_profiles"))
  }
  set.seed(seed)
  train <- unlist(lapply(split(seq_along(y), y), function(idx) {
    if (length(idx) < 2L) stop("Need >= 2 observations per class")
    sample(idx, max(1L, min(length(idx) - 1L, round(length(idx) * p))))
  }), use.names = FALSE)
  train <- sort(train)
  list(train = train, test = setdiff(seq_along(y), train), seed = seed,
       train_fraction = p)
}

fit_preprocessor <- function(train) {
  validate_predictors(train, setdiff(names(train), c("class", "target")))
  continuous <- c("height", "width", "aratio")
  binary <- setdiff(names(train)[vapply(train, is.numeric, logical(1))],
                    c(continuous, "target"))
  columns <- c(continuous, binary)
  medians <- vapply(train[continuous], function(x) {
    value <- stats::median(x, na.rm = TRUE)
    if (is.na(value)) stop("Train set has no observed values for a continuous variable")
    value
  }, numeric(1))
  modes <- vapply(train[binary], function(x) {
    nonmissing <- x[!is.na(x)]
    if (!length(nonmissing)) return(0)
    as.numeric(mean(nonmissing) >= 0.5)
  }, numeric(1))
  means <- vapply(continuous, function(name) {
    xx <- train[[name]]
    xx[is.na(xx)] <- medians[[name]]
    mean(xx)
  }, numeric(1))
  sds <- vapply(continuous, function(name) {
    xx <- train[[name]]
    xx[is.na(xx)] <- medians[[name]]
    z <- stats::sd(xx)
    if (!is.finite(z) || z == 0) 1 else z
  }, numeric(1))
  list(continuous = continuous, binary = binary, columns = columns,
       median = medians, binary_mode = modes, means = means, sds = sds)
}

apply_preprocessor <- function(data, prep) {
  validate_predictors(data, prep$columns)
  out <- data
  for (name in prep$continuous) {
    missing <- is.na(out[[name]])
    out[[paste0(name, "_missing")]] <- as.integer(missing)
    out[[name]][missing] <- prep$median[[name]]
    out[[name]] <- (out[[name]] - prep$means[[name]]) / prep$sds[[name]]
  }
  for (name in prep$binary) {
    missing <- is.na(out[[name]])
    out[[name]][missing] <- prep$binary_mode[[name]]
  }
  stopifnot(!anyNA(out[prep$columns]))
  out
}

# Allow missing values, but reject unknown binary categories.
validate_predictors <- function(data, columns) {
  if (!is.data.frame(data) || nrow(data) == 0L || anyDuplicated(names(data))) {
    stop("Predictors must be a non-empty data frame with unique column names.")
  }
  missing <- setdiff(columns, names(data))
  if (length(missing)) stop("Missing predictor: ", missing[1L])
  for (name in columns) {
    value <- data[[name]]
    if (!is.numeric(value) || any(!is.na(value) & !is.finite(value))) stop("Invalid numeric predictor: ", name)
    if (!name %in% c("height", "width", "aratio") && any(!is.na(value) & !value %in% 0:1)) {
      stop("Unknown binary category in predictor: ", name)
    }
  }
  invisible(TRUE)
}

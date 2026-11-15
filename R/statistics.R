binary_association <- function(y, x) {
  observed <- !is.na(y) & !is.na(x)
  tab <- table(target = factor(y[observed], levels = c(1,0), labels = c("ad","nonad")),
               feature = factor(x[observed], levels = 0:1))
  n <- sum(tab)
  expected <- if (n > 0) outer(rowSums(tab), colSums(tab)) / n else matrix(0,2,2)
  dimnames(expected) <- dimnames(tab)
  estimable <- n > 0 && all(rowSums(tab)>0) && all(colSums(tab)>0)
  result <- data.frame(test = "Not estimable", statistic = NA_real_, df = NA_real_,
    p_value = NA_real_, phi = NA_real_, conditional_odds_ratio = NA_real_,
    exact_ci_low = NA_real_, exact_ci_high = NA_real_, n = n, estimable = estimable)
  if (estimable) {
    exact <- stats::fisher.test(tab)
    chi_stat <- sum((tab-expected)^2 / expected)
    pearson <- all(expected >= 5)
    result$test <- if (pearson) "Pearson chi-square" else "Fisher exact"
    result$statistic <- if (pearson) chi_stat else NA_real_
    result$df <- if (pearson) 1 else NA_real_
    result$p_value <- if (pearson) stats::pchisq(chi_stat,1,lower.tail=FALSE) else exact$p.value
    result$phi <- sqrt(chi_stat/n)
    result$conditional_odds_ratio <- unname(exact$estimate)
    result$exact_ci_low <- exact$conf.int[1];result$exact_ci_high <- exact$conf.int[2]
  }
  list(table = tab, expected = expected, result = result)
}

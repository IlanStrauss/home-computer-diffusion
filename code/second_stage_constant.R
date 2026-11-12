# Check on the second stage of steps 1-4: the spending thresholds and the main price in constant dollars.
#
# The first-stage spending thresholds and the main price are in current dollars. This check deflates both by the BEA
# PCE price index for all items (DPCERG), built like ell_t in first_stage.py: interpolated monthly in logs between
# mid-year values, given to each unit for its inventory month, and averaged over the year's units with the first-stage
# weights. With pi_t the log PCE price index relative to 1994, the model, priors, and sampler are those of
# second_stage.R:
#   tau_hat_t - pi_t ~ Normal(tau_t, se_t^2),   tau_t = a + b (ell_t - pi_t) + d (t - 1994) + e_t,   e_t ~ Normal(0, omega^2)
# If pi_t rose at a constant rate, b would be unchanged and d would equal the current-dollar d minus (1 - b) times
# that rate. Each specification is also refitted in current dollars, which should reproduce second_stage.csv.
#
# Run from the project folder: Rscript code/second_stage_constant.R > output/logs/second_stage_constant.txt

suppressPackageStartupMessages({
  library(brms)
  library(posterior)
})
options(width = 200, digits = 4)

u <- read.csv("data/processed/ce_units.csv.gz")[, c("year", "ref_year", "ref_month", "w")]
pa <- read.csv("data/processed/prices_annual.csv")
tm <- u$ref_year + (u$ref_month - 0.5) / 12
ylog <- function(col) {
  ok <- !is.na(pa[[col]])
  v <- approx(pa$year[ok] + 0.5, log(pa[[col]][ok]), xout = tm, rule = 2)$y
  m <- tapply(v * u$w, u$year, sum) / tapply(u$w, u$year, sum)
  m - m[["1994"]]
}
pi_t <- ylog("pce")

fs <- read.csv("output/results/first_stage.csv")
fs$trend <- fs$year - 1994
s1 <- fs[fs$spec == "s1", ]
cat(sprintf("Largest gap between ell_t rebuilt here and first_stage.csv: %.1e\n",
            max(abs(ylog("pc")[as.character(s1$year)] - s1$ell_pc))))
cat("PCE price index, all items, log relative to 1994 (year means over units):\n")
print(round(pi_t, 4))
cat(sprintf("1994-2012: %.3f log points, %.4f a year\n\n", pi_t[["2012"]], pi_t[["2012"]] / 18))

priors <- c(prior(normal(10, 2), class = b, coef = Intercept),
            prior(normal(0.5, 1), class = b, coef = ell),
            prior(normal(0, 0.2), class = b, coef = trend),
            prior(normal(0, 0.2), class = sigma))
ctrl <- list(adapt_delta = 0.99)
prep <- function(spec, constant) {
  y <- fs[fs$spec == spec, ]
  p <- if (constant) pi_t[as.character(y$year)] else 0
  y$tau <- y$tau - p
  y$ell <- y$ell_pc - p
  y
}
fit0 <- brm(bf(tau | se(se_tau, sigma = TRUE) ~ 0 + Intercept + ell + trend), data = prep("s1", FALSE),
            family = gaussian(), prior = priors, chains = 4, iter = 4000, warmup = 2000, cores = 4, seed = 20261109,
            backend = "rstan", refresh = 0, control = ctrl)
refit <- function(data) update(fit0, newdata = data, recompile = FALSE, refresh = 0, seed = 20261109, control = ctrl)

ss <- read.csv("output/results/second_stage.csv")
rows <- list()
for (spec in c("s1", "s2", "s3", "s4")) {
  for (constant in c(FALSE, TRUE)) {
    fit <- if (spec == "s1" && !constant) fit0 else refit(prep(spec, constant))
    dr <- as_draws_df(fit)
    q <- function(x) unname(quantile(x, c(0.5, 0.05, 0.95)))
    sm <- summarise_draws(subset_draws(dr, variable = c("b_Intercept", "b_ell", "b_trend", "sigma")))
    rows[[length(rows) + 1]] <- data.frame(
      spec = spec, dollars = ifelse(constant, "constant", "current"),
      b = q(dr$b_ell)[1], b_lo = q(dr$b_ell)[2], b_hi = q(dr$b_ell)[3],
      d = q(dr$b_trend)[1], d_lo = q(dr$b_trend)[2], d_hi = q(dr$b_trend)[3],
      omega = median(dr$sigma), cor_bd = cor(dr$b_ell, dr$b_trend),
      rhat = max(sm$rhat), ess_bulk = min(sm$ess_bulk), ess_tail = min(sm$ess_tail),
      divergent = sum(nuts_params(fit)$Value[nuts_params(fit)$Parameter == "divergent__"]))
  }
}
res <- do.call(rbind, rows)
cat("Second stage, main price, steps 1-4: posterior medians with 90% intervals\n")
print(res, row.names = FALSE)

cat("\nCurrent-dollar refits against second_stage.csv (largest absolute gap in b, d and their bounds):\n")
for (spec in c("s1", "s2", "s3", "s4")) {
  x <- res[res$spec == spec & res$dollars == "current", ]
  y <- ss[ss$spec == spec & ss$concept == "pc" & ss$model == "independent errors", ]
  cat(sprintf("  %s: %.4f\n", spec, max(abs(c(x$b - y$b, x$b_lo - y$b_lo, x$b_hi - y$b_hi,
                                              x$d - y$d, x$d_lo - y$d_lo, x$d_hi - y$d_hi)))))
}
cat("\nConstant minus current dollars, and -(1 - b) times the mean inflation rate:\n")
for (spec in c("s1", "s2", "s3", "s4")) {
  x <- res[res$spec == spec & res$dollars == "current", ]
  y <- res[res$spec == spec & res$dollars == "constant", ]
  cat(sprintf("  %s: b %+.3f, d %+.4f (algebra %+.4f); d over 1994-2012 %+.3f log points\n", spec, y$b - x$b, y$d - x$d,
              -(1 - x$b) * pi_t[["2012"]] / 18, 18 * y$d))
}

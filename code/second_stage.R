# Second stage of steps 1-4 of the household PC model: yearly thresholds against the computer price, CE 1994-2013.
#
#   tau_hat_t ~ Normal(tau_t, se_t^2)                se_t = BRR standard error of the first-stage threshold
#   tau_t = a + b ell_t + d (t - 1994) + e_t,        e_t ~ Normal(0, omega^2)
#
# ell_t is the log computer price relative to 1994, so a is the 1994 threshold; pure affordability means b = 1, d = 0.
# Priors: a ~ Normal(10, 2); b ~ Normal(0.5, 1); d ~ Normal(0, 0.2); omega ~ Half-Normal(0, 0.2).
# brms with the rstan backend: NUTS, 4 chains of 2,000 warm-up and 2,000 kept draws, adapt_delta 0.99, seed 20261109.
#
# Fitted for every first-stage specification (output/results/first_stage.csv) with the main price (BEA PC index), for
# the main specifications of steps 1-4 with five other price concepts, and for two fake-data scenarios.
# Covariance check (steps 1, 3, 4): the sampling errors of the 20 thresholds are correlated, because all years share
# the survey's sampling areas. The check replaces the independent errors with
#   tau_hat ~ MultiNormal(a + b ell + d trend + e, V),   V = D R D
# D = diag(se_t); R_st = c phi^|s - t| for s != t, fitted by least squares to the replicate correlations by lag
# (a raw 20 x 20 replicate covariance from 44 replicates is too noisy to invert). brms: fcor(V) with sigma fixed at 1,
# and e_t as a year effect with sd omega.
#
# Run from the project folder: Rscript code/second_stage.R > output/logs/second_stage.txt
# Writes output/results/second_stage.csv, second_stage_fitted.csv (fitted paths, steps 1-4), and
# output/diagnostics/second_stage.pdf.

suppressPackageStartupMessages({
  library(brms)
  library(posterior)
})
options(width = 200, digits = 4)

fs <- read.csv("output/results/first_stage.csv")
reps <- read.csv("output/results/first_stage_replicates.csv.gz")
fk <- read.csv("output/results/first_stage_fake.csv")
fs$trend <- fs$year - 1994
main_specs <- c("s1", "s2", "s3", "s4")
check_specs <- c("s1_nogeo", "s1_logit", "s1_robit2", "s1_int2", "s1_nostudent", "s1_trim", "s1_incl", "s1_veh",
                 "s3_obs", "s3_robit7", "s4_logit", "s4_int2", "s4_incl", "s4_veh", "s4_cr", "s4_inc", "s4_cf")
concepts <- c(pc = "BEA PCE price index, PCs (main)", info = "BEA PCE price index, information processing equipment",
              business = "BEA investment price index, computers", cpi_pc = "BLS CPI-U, PCs (1999-2013)",
              paid_p50 = "CE price paid, weighted median", paid_p90 = "CE price paid, weighted 90th percentile")
pars <- c("b_Intercept", "b_ell", "b_trend", "sigma")
labels <- c(b_Intercept = "a (1994 threshold, log $)", b_ell = "b (threshold shift per log point of price)",
            b_trend = "d (leftover drift, log points a year)", sigma = "omega (year deviations, log points)")

priors <- c(prior(normal(10, 2), class = b, coef = Intercept),
            prior(normal(0.5, 1), class = b, coef = ell),
            prior(normal(0, 0.2), class = b, coef = trend),
            prior(normal(0, 0.2), class = sigma))
ctrl <- list(adapt_delta = 0.99)
pick <- function(spec, concept = "pc", src = fs) {
  y <- src[src$spec == spec, ]
  y$ell <- y[[paste0("ell_", concept)]]
  y[!is.na(y$ell), ]
}
fit0 <- brm(bf(tau | se(se_tau, sigma = TRUE) ~ 0 + Intercept + ell + trend), data = pick("s1"), family = gaussian(),
            prior = priors, chains = 4, iter = 4000, warmup = 2000, cores = 4, seed = 20261109, backend = "rstan",
            refresh = 0, control = ctrl)
refit <- function(data) update(fit0, newdata = data, recompile = FALSE, refresh = 0, seed = 20261109, control = ctrl)

summ <- function(fit, p = pars) {
  dr <- as.data.frame(as_draws_df(fit))[, p]
  t(sapply(dr, function(x) c(median = median(x), q05 = unname(quantile(x, 0.05)), q95 = unname(quantile(x, 0.95)))))
}
diag_vals <- function(fit, p = pars) {
  s <- summarise_draws(subset_draws(as_draws_df(fit), variable = p), "rhat", "ess_bulk", "ess_tail")
  np <- nuts_params(fit)
  c(max_rhat = max(s$rhat), min_ess_bulk = min(s$ess_bulk), min_ess_tail = min(s$ess_tail),
    divergent = sum(np$Value[np$Parameter == "divergent__"]))
}
fmt <- function(m) sprintf("  %-46s %8.3f  [%7.3f, %7.3f]", labels[rownames(m)], m[, 1], m[, 2], m[, 3])
q3 <- function(x) sprintf("%.3f [%.3f, %.3f]", median(x), quantile(x, 0.05), quantile(x, 0.95))

results <- list()
record <- function(fit, spec, concept, model, dk, p = pars) {
  m <- summ(fit, p)
  dg <- diag_vals(fit, p)
  dr <- as.data.frame(as_draws_df(fit))
  rownames(m)[rownames(m) == "sd_yearf__Intercept"] <- "sigma"
  span <- max(dk$year) - min(dk$year)
  dl <- dk$ell[dk$year == max(dk$year)] - dk$ell[dk$year == min(dk$year)]
  out <- data.frame(spec = spec, concept = concept, model = model, years = nrow(dk),
                    first = min(dk$year), last = max(dk$year),
                    a = m["b_Intercept", 1], a_lo = m["b_Intercept", 2], a_hi = m["b_Intercept", 3],
                    b = m["b_ell", 1], b_lo = m["b_ell", 2], b_hi = m["b_ell", 3],
                    d = m["b_trend", 1], d_lo = m["b_trend", 2], d_hi = m["b_trend", 3],
                    omega = m["sigma", 1], omega_lo = m["sigma", 2], omega_hi = m["sigma", 3],
                    cor_bd = cor(dr$b_ell, dr$b_trend),
                    dtau_obs = dk$tau[dk$year == max(dk$year)] - dk$tau[dk$year == min(dk$year)],
                    dell = dl, price_part = median(dr$b_ell * dl), drift_part = median(dr$b_trend * span),
                    t(dg))
  results[[length(results) + 1]] <<- out
  cat(sprintf("\n[%s] %s; %s; %d years (%d-%d)\n", spec, concepts[[concept]], model, nrow(dk), min(dk$year),
              max(dk$year)))
  cat(fmt(m), sep = "\n")
  cat(sprintf(paste("  Change in the threshold %d-%d: observed %.3f; from the price, b x %.3f = %s;",
                    "from drift, d x %d = %s\n"),
              min(dk$year), max(dk$year), out$dtau_obs, dl, q3(dr$b_ell * dl), span, q3(dr$b_trend * span)))
  cat(sprintf("  Posterior correlation of b and d: %+.2f\n", out$cor_bd))
  cat(sprintf("  max R-hat %.4f; min bulk ESS %.0f; min tail ESS %.0f; divergent transitions %d\n", dg[1], dg[2],
              dg[3], as.integer(dg[4])))
  invisible(out)
}

cat("Second stage: tau_hat_t ~ N(tau_t, se_t^2), tau_t = a + b ell_t + d (t - 1994) + e_t, e_t ~ N(0, omega^2)\n")
cat("Posterior medians with 90% intervals. Pure affordability: b = 1 and d = 0.\n")
cat("\n==== Main specifications, every price concept ====\n")
fits <- list()
for (s in main_specs) for (k in names(concepts)) {
  dk <- pick(s, k)
  f <- if (s == "s1" && k == "pc") fit0 else refit(dk)
  if (k == "pc") fits[[s]] <- f
  record(f, s, k, "independent errors", dk)
}
cat("\n==== Checks, main price ====\n")
for (s in check_specs) {
  dk <- pick(s)
  f <- refit(dk)
  fits[[s]] <- f
  record(f, s, "pc", "independent errors", dk)
}

cat("\n==== Covariance check: correlated sampling errors from the replicates ====\n")
lagfit <- function(s) {
  r <- reps[reps$spec == s, ]
  w <- reshape(r[, c("year", "rep", "tau")], idvar = "rep", timevar = "year", direction = "wide")
  full <- fs$tau[fs$spec == s]
  dev <- sweep(as.matrix(w[, -1]), 2, full)
  cr <- cor(dev)
  lag <- abs(outer(1:20, 1:20, "-"))
  emp <- sapply(1:19, function(k) mean(cr[lag == k]))
  obj <- function(p) sum((emp - p[1] * p[2]^(1:19))^2)
  g <- expand.grid(c = seq(0, 1, 0.01), phi = seq(0, 0.99, 0.01))    # grid start: (0, 0) is a stationary corner
  o <- optim(unlist(g[which.min(apply(g, 1, obj)), ]), obj, method = "L-BFGS-B", lower = c(0, 0), upper = c(1, 0.999))
  R <- o$par[1] * o$par[2]^lag
  diag(R) <- 1
  se <- fs$se_tau[fs$spec == s]
  list(V = diag(se) %*% R %*% diag(se), c = o$par[1], phi = o$par[2], emp = emp)
}
cov_priors <- c(prior(normal(10, 2), class = b, coef = Intercept), prior(normal(0.5, 1), class = b, coef = ell),
                prior(normal(0, 0.2), class = b, coef = trend), prior(constant(1), class = sigma),
                prior(normal(0, 0.2), class = sd))
fitC <- NULL
for (s in c("s1", "s3", "s4")) {
  lf <- lagfit(s)
  cat(sprintf("\n[%s] replicate correlation by lag 1-5, 10, 15: %s; fitted R_st = %.2f x %.2f^|s-t|\n", s,
              paste(sprintf("%+.2f", lf$emp[c(1:5, 10, 15)]), collapse = " "), lf$c, lf$phi))
  dk <- pick(s)
  dk$yearf <- factor(dk$year)
  if (is.null(fitC)) {
    fitC <- brm(bf(tau ~ 0 + Intercept + ell + trend + (1 | yearf) + fcor(V)), data = dk, data2 = list(V = lf$V),
                family = gaussian(), prior = cov_priors, chains = 4, iter = 4000, warmup = 2000, cores = 4,
                seed = 20261109, backend = "rstan", refresh = 0, control = ctrl)
    f <- fitC
  } else {
    f <- update(fitC, newdata = dk, data2 = list(V = lf$V), recompile = FALSE, refresh = 0, seed = 20261109,
                control = ctrl)
  }
  record(f, s, "pc", "correlated errors", dk, p = c("b_Intercept", "b_ell", "b_trend", "sd_yearf__Intercept"))
}

cat("\n==== Prior predictive check (step 1, main price): thresholds under the priors alone ====\n")
pri <- update(fit0, sample_prior = "only", recompile = FALSE, refresh = 0, seed = 20261109)
pp <- posterior_predict(pri)
d1 <- pick("s1")
qq <- function(x) paste(sprintf("%.2f", quantile(x, c(0.05, 0.5, 0.95))), collapse = " / ")
cat(sprintf("  tau_1994 5%% / 50%% / 95%%: %s (observed %.2f)\n", qq(pp[, 1]), d1$tau[1]))
cat(sprintf("  tau_2013 - tau_1994 5%% / 50%% / 95%%: %s (observed %.2f)\n", qq(pp[, 20] - pp[, 1]),
            d1$tau[20] - d1$tau[1]))

cat("\n==== Posterior predictive check, main price: observed thresholds inside 90% predictive intervals ====\n")
fitted_paths <- list()
for (s in main_specs) {
  yrep <- posterior_predict(fits[[s]])
  dk <- pick(s)
  lo <- apply(yrep, 2, quantile, 0.05)
  hi <- apply(yrep, 2, quantile, 0.95)
  inside <- dk$tau >= lo & dk$tau <= hi
  cat(sprintf("  [%s] %d of %d years inside; outside: %s\n", s, sum(inside), nrow(dk),
              if (all(inside)) "none" else paste(dk$year[!inside], collapse = ", ")))
  ft <- fitted(fits[[s]], robust = TRUE, probs = c(0.05, 0.95))
  fitted_paths[[s]] <- data.frame(spec = s, year = dk$year, tau = dk$tau, se_tau = dk$se_tau, ell = dk$ell,
                                  fit = ft[, 1], fit_lo = ft[, 3], fit_hi = ft[, 4], pred_lo = lo, pred_hi = hi)
}
write.csv(do.call(rbind, fitted_paths), "output/results/second_stage_fitted.csv", row.names = FALSE)

cat("\n==== Fake-data check: thresholds estimated from ownership simulated with known a, b, d, omega ====\n")
fake_fits <- list()
for (sc in unique(fk$scenario)) {
  ds <- fk[fk$scenario == sc, ]
  ds$ell <- ds$ell_pc
  fake_fits[[sc]] <- refit(ds)
  m <- summ(fake_fits[[sc]])
  truth <- c(b_Intercept = ds$a_true[1], b_ell = ds$b_true[1], b_trend = ds$d_true[1], sigma = ds$omega_true[1])
  cat(sprintf("\n%s\n", sc))
  cat(sprintf("%s  true %7.3f  %s", fmt(m), truth[rownames(m)],
              ifelse(truth[rownames(m)] >= m[, 2] & truth[rownames(m)] <= m[, 3], "inside", "OUTSIDE")), sep = "\n")
}

res <- do.call(rbind, results)
write.csv(res, "output/results/second_stage.csv", row.names = FALSE)
cat("\n==== Summary, main price (posterior medians and 90% intervals) ====\n")
sm <- res[res$concept == "pc", ]
print(data.frame(spec = sm$spec, model = sm$model, years = sm$years,
                 b = sprintf("%.2f [%.2f, %.2f]", sm$b, sm$b_lo, sm$b_hi),
                 d = sprintf("%+.3f [%+.3f, %+.3f]", sm$d, sm$d_lo, sm$d_hi),
                 omega = sprintf("%.3f", sm$omega), cor_bd = sprintf("%+.2f", sm$cor_bd),
                 dtau = sprintf("%.2f", sm$dtau_obs), rhat = sprintf("%.3f", sm$max_rhat),
                 div = sm$divergent), row.names = FALSE)

# ---------- plots ----------
ink <- "#222222"; accent <- "#0072B2"; accent2 <- "#D55E00"; grid_col <- "#DDDDDD"
pdf("output/diagnostics/second_stage.pdf", width = 11, height = 4.4)
par(mfrow = c(1, 2), mar = c(4.2, 4.6, 2.6, 1), cex = 0.8, las = 1, bty = "l")
for (s in c("s1", "s4")) {
  dk <- pick(s)
  yrep <- posterior_predict(fits[[s]])
  lo <- apply(yrep, 2, quantile, 0.05)
  hi <- apply(yrep, 2, quantile, 0.95)
  ft <- fitted(fits[[s]], robust = TRUE, probs = c(0.05, 0.95))
  plot(dk$year, dk$tau, type = "n", ylim = range(c(lo, hi, dk$tau)), xlab = "year",
       ylab = "threshold tau_t (log $ a year)", main = sprintf("[%s] fitted thresholds (90%%)", s))
  abline(h = pretty(range(c(lo, hi))), col = grid_col, lwd = 0.5)
  polygon(c(dk$year, rev(dk$year)), c(lo, rev(hi)), col = "#E8E8E8", border = NA)
  polygon(c(dk$year, rev(dk$year)), c(ft[, 3], rev(ft[, 4])), col = adjustcolor(accent, 0.25), border = NA)
  lines(dk$year, ft[, 1], col = accent, lwd = 1.5)
  arrows(dk$year, dk$tau - 1.645 * dk$se_tau, dk$year, dk$tau + 1.645 * dk$se_tau, length = 0, col = ink)
  points(dk$year, dk$tau, pch = 19, cex = 0.6, col = ink)
  legend("topright", c("estimated threshold (90%)", "fitted tau_t (90%)", "posterior predictive 90%"), bty = "n",
         col = c(ink, accent, "#C8C8C8"), lwd = c(1, 1.5, 8), pch = c(19, NA, NA), cex = 0.85)
}
par(mfrow = c(1, 2), mar = c(4.2, 9.5, 2.6, 1))
fr <- res[res$concept == "pc", ]
fr$lab <- paste(fr$spec, ifelse(fr$model == "correlated errors", "(corr.)", ""))
yy <- rev(seq_len(nrow(fr)))
for (p in c("b", "d")) {
  plot(fr[[p]], yy, xlim = range(c(fr[[paste0(p, "_lo")]], fr[[paste0(p, "_hi")]], if (p == "b") c(0, 1) else 0)),
       yaxt = "n", pch = 19, col = ink, xlab = if (p == "b") labels[["b_ell"]] else labels[["b_trend"]], ylab = "",
       main = paste("Main price, every specification:", p))
  abline(v = if (p == "b") c(0, 1) else 0, col = c("#888888", accent), lty = c(2, 1))
  segments(fr[[paste0(p, "_lo")]], yy, fr[[paste0(p, "_hi")]], yy, col = ink)
  axis(2, at = yy, labels = fr$lab, cex.axis = 0.75)
}
par(mfrow = c(1, 2), mar = c(4.2, 4.6, 2.6, 1))
for (p in c("b_ell", "b_trend")) {
  m <- t(sapply(fake_fits, function(f) summ(f)[p, ]))
  truth <- sapply(unique(fk$scenario), function(s) fk[fk$scenario == s, if (p == "b_ell") "b_true" else "d_true"][1])
  yy <- rev(seq_len(nrow(m)))
  plot(m[, 1], yy, xlim = range(c(m, truth)), ylim = c(0.5, nrow(m) + 0.5), yaxt = "n", pch = 19, col = ink,
       xlab = labels[[p]], ylab = "", main = "Fake data: posterior (90%) and truth (x)")
  segments(m[, 2], yy, m[, 3], yy, col = ink)
  points(truth, yy, pch = 4, col = accent2, cex = 1.4, lwd = 2)
  axis(2, at = yy, labels = substr(rownames(m), 1, 1), cex.axis = 0.9)
}
invisible(dev.off())
cat("\nWrote output/results/second_stage.csv, second_stage_fitted.csv, and output/diagnostics/second_stage.pdf\n")

# Step 6 of the household PC model: does the leftover drift change when ownership passes 20% (or 30%)?
#
# Thresholds from the CPS (1984-2003, imputed spending) and the CE (1994-2013), both from the step-5 specification
# (corrected or imputed long-run spending, geography, household traits, and education; data/processed/
# cps_thresholds.csv). Each threshold j has a date t_j (decimal year: the CPS survey month, or the CE year's mean
# inventory month) and the log PC price ell_j relative to the CE's 1994 average.
#
#   tau_hat_j ~ Normal(tau_j, se_j^2)
#   tau_j = a + o CPS_j + b ell_j + d1 min(t_j - t_s, 0) + d2 max(t_j - t_s, 0) + e_j,   e_j ~ Normal(0, omega^2)
#
# - o = the CPS offset (imputed against measured spending, households against consumer units), identified by the
#   years both surveys cover.
# - d1, d2 = the leftover drift before and after the switch date t_s, in log points a year. The hypothesis: after
#   the switch, affordability alone (b = 1, d2 = 0).
# - Models: one drift (d1 = d2), a switch when ownership passed 20%, and a switch when it passed 30%; dates from the
#   CPS by log-odds interpolation between surveys (output/logs/cps_extension.txt). Compared by leave-one-out cross-validation
#   (Vehtari, Gelman, and Gabry 2017), refitting the model without each point whose Pareto k exceeds 0.7.
# Priors: a ~ Normal(10, 2); o ~ Normal(0, 0.5); b ~ Normal(0.5, 1); d1, d2 ~ Normal(0, 0.2);
# omega ~ Half-Normal(0, 0.2).
# brms, rstan backend, 4 chains of 2,000 warm-up and 2,000 kept draws, adapt_delta 0.99, seed 20261109.
#
# Check: the same models on the CPS thresholds alone (no offset), since the step-5 validation found the CPS and CE
# threshold paths differ in the years both cover.
#
# Run from the project folder: Rscript code/cps_switch.R <date 20%> <date 30%> > output/logs/cps_switch.txt
# Writes output/results/cps_switch.csv and output/diagnostics/cps_switch.pdf.

suppressPackageStartupMessages({
  library(brms)
  library(posterior)
  library(loo)
})
options(width = 200, digits = 4)
args <- commandArgs(trailingOnly = TRUE)
t20 <- as.numeric(args[1])
t30 <- as.numeric(args[2])
stopifnot(!is.na(t20), !is.na(t30))

d <- read.csv("output/results/cps_thresholds.csv")
d$cps <- as.numeric(d$source == "CPS")
ctrl <- list(adapt_delta = 0.99)
base_priors <- c(prior(normal(10, 2), class = b, coef = Intercept), prior(normal(0, 0.5), class = b, coef = cps),
                 prior(normal(0.5, 1), class = b, coef = ell), prior(normal(0, 0.2), class = sigma))
setup <- function(ts, concept = "pc") {
  x <- d
  x$ell <- x[[paste0("ell_", concept)]]
  x$pre <- pmin(x$date - ts, 0)
  x$post <- pmax(x$date - ts, 0)
  x
}
fit_one <- setup(1994)
m0 <- brm(bf(tau | se(se_tau, sigma = TRUE) ~ 0 + Intercept + cps + ell + trend), data = fit_one, family = gaussian(),
          prior = c(base_priors, prior(normal(0, 0.2), class = b, coef = trend)), chains = 4, iter = 4000,
          warmup = 2000, cores = 4, seed = 20261109, backend = "rstan", refresh = 0, control = ctrl)
ms <- brm(bf(tau | se(se_tau, sigma = TRUE) ~ 0 + Intercept + cps + ell + pre + post), data = setup(t20),
          family = gaussian(), prior = c(base_priors, prior(normal(0, 0.2), class = b, coef = pre),
                                         prior(normal(0, 0.2), class = b, coef = post)),
          chains = 4, iter = 4000, warmup = 2000, cores = 4, seed = 20261109, backend = "rstan", refresh = 0,
          control = ctrl)
refit <- function(f, x) update(f, newdata = x, recompile = FALSE, refresh = 0, seed = 20261109, control = ctrl)
loo_exact <- function(f) {
  # PSIS-LOO; each point whose Pareto k exceeds 0.7 is left out and the model refitted (exact LOO for that point)
  raw <- loo(f)
  k <- pareto_k_values(raw)
  lo <- if (any(k > 0.7)) reloo(f, loo = raw, k_threshold = 0.7, refresh = 0, seed = 20261109) else raw
  attr(lo, "max_k") <- max(k)
  attr(lo, "n_refit") <- sum(k > 0.7)
  lo
}
k_note <- function(l) paste(sprintf("%s %.2f (%d refitted)", names(l), sapply(l, attr, "max_k"),
                                    sapply(l, attr, "n_refit")), collapse = "; ")

q <- function(x) sprintf("%7.3f [%7.3f, %7.3f]", median(x), quantile(x, 0.05), quantile(x, 0.95))
show <- function(f, name) {
  dr <- as.data.frame(as_draws_df(f))
  s <- summarise_draws(subset_draws(as_draws_df(f), variable = grep("^b_|^sigma$", variables(f), value = TRUE)),
                       "rhat", "ess_bulk", "ess_tail")
  np <- nuts_params(f)
  cat(sprintf("\n%s\n", name))
  for (v in grep("^b_|^sigma$", names(dr), value = TRUE)) cat(sprintf("  %-12s %s\n", sub("^b_", "", v), q(dr[[v]])))
  if ("b_post" %in% names(dr)) cat(sprintf("  %-12s %s\n", "d2 - d1", q(dr$b_post - dr$b_pre)))
  cat(sprintf("  max R-hat %.4f; min bulk ESS %.0f; min tail ESS %.0f; divergent transitions %d\n", max(s$rhat),
              min(s$ess_bulk), min(s$ess_tail), as.integer(sum(np$Value[np$Parameter == "divergent__"]))))
  dr
}

cat(sprintf("Step 6: %d thresholds (%d CPS, %d CE). Switch dates: 20%% at %.2f, 30%% at %.2f.\n", nrow(d), sum(d$cps),
            sum(1 - d$cps), t20, t30))
cat("Posterior medians [90% intervals]. pre = d1 (drift before the switch), post = d2 (after), cps = offset o,\n")
cat("ell = b, sigma = omega, trend = the single drift d.\n")
out <- list()
record_rows <- function(concept, drs, ls, cmp) {
  # one row per model: LOO (and its difference from the best model), b, offset o, omega, d1, d2, and d2 - d1
  cmp_name <- c(one = "one drift", "20" = "switch at 20%", "30" = "switch at 30%")
  q <- function(x) c(median(x), quantile(x, 0.05), quantile(x, 0.95))
  for (nm in c("one", "20", "30")) {
    dr <- drs[[nm]]
    lo <- ls[[nm]]
    d1 <- if (nm == "one") dr$b_trend else dr$b_pre
    d2 <- if (nm == "one") dr$b_trend else dr$b_post
    o <- if ("b_cps" %in% names(dr)) q(dr$b_cps) else rep(NA, 3)
    v <- rbind(b = q(dr$b_ell), o = o, omega = q(dr$sigma), d1 = q(d1), d2 = q(d2), dd = q(d2 - d1))
    row <- data.frame(concept = concept, model = nm, elpd = lo$estimates["elpd_loo", 1],
                      elpd_se = lo$estimates["elpd_loo", 2], elpd_diff = cmp[cmp_name[[nm]], "elpd_diff"],
                      elpd_diff_se = cmp[cmp_name[[nm]], "se_diff"], max_k = attr(lo, "max_k"),
                      n_refit = attr(lo, "n_refit"))
    for (p in rownames(v)) row[, paste0(p, c("", "_lo", "_hi"))] <- as.list(v[p, ])
    out[[length(out) + 1]] <<- row
  }
}
for (k in c("pc", "info", "business")) {
  cat(sprintf("\n======== Price: %s ========\n", k))
  f0 <- if (k == "pc") m0 else refit(m0, setup(1994, k))
  f20 <- if (k == "pc") ms else refit(ms, setup(t20, k))
  f30 <- refit(ms, setup(t30, k))
  dr0 <- show(f0, "One drift")
  dr20 <- show(f20, sprintf("Switch at 20%% ownership (%.2f)", t20))
  dr30 <- show(f30, sprintf("Switch at 30%% ownership (%.2f)", t30))
  l0 <- loo_exact(f0)
  l20 <- loo_exact(f20)
  l30 <- loo_exact(f30)
  cmp <- loo_compare(list("one drift" = l0, "switch at 20%" = l20, "switch at 30%" = l30))
  cat("\nLeave-one-out comparison (elpd difference from the best model, with its standard error)\n")
  print(cmp[, c("elpd_diff", "se_diff", "elpd_loo", "p_loo")])
  cat("Largest Pareto k before refitting (points above 0.7 refitted exactly):",
      k_note(list("one drift" = l0, "20%" = l20, "30%" = l30)), "\n")
  record_rows(k, list(one = dr0, "20" = dr20, "30" = dr30), list(one = l0, "20" = l20, "30" = l30), cmp)
  if (k == "pc") {
    main <- list(f0 = f0, f20 = f20, f30 = f30)
  }
}
cat("\n======== Check: CPS thresholds alone (one survey and one imputation, so no offset), main price ========\n")
cat("The step-5 validation found CPS thresholds falling more slowly than the CE's in the years both surveys cover.\n")
dc <- d[d$cps == 1, ]
setup_c <- function(ts) {
  x <- dc
  x$ell <- x$ell_pc
  x$pre <- pmin(x$date - ts, 0)
  x$post <- pmax(x$date - ts, 0)
  x
}
cps_priors <- c(prior(normal(10, 2), class = b, coef = Intercept), prior(normal(0.5, 1), class = b, coef = ell),
                prior(normal(0, 0.2), class = sigma))
c0 <- brm(bf(tau | se(se_tau, sigma = TRUE) ~ 0 + Intercept + ell + trend), data = setup_c(1994), family = gaussian(),
          prior = c(cps_priors, prior(normal(0, 0.2), class = b, coef = trend)), chains = 4, iter = 4000,
          warmup = 2000, cores = 4, seed = 20261109, backend = "rstan", refresh = 0, control = ctrl)
c20 <- brm(bf(tau | se(se_tau, sigma = TRUE) ~ 0 + Intercept + ell + pre + post), data = setup_c(t20),
           family = gaussian(), prior = c(cps_priors, prior(normal(0, 0.2), class = b, coef = pre),
                                          prior(normal(0, 0.2), class = b, coef = post)),
           chains = 4, iter = 4000, warmup = 2000, cores = 4, seed = 20261109, backend = "rstan", refresh = 0,
           control = ctrl)
c30 <- refit(c20, setup_c(t30))
drc <- list(one = show(c0, "CPS alone, one drift"), "20" = show(c20, sprintf("CPS alone, switch at 20%% (%.2f)", t20)),
            "30" = show(c30, sprintf("CPS alone, switch at 30%% (%.2f)", t30)))
lc <- list(one = loo_exact(c0), "20" = loo_exact(c20), "30" = loo_exact(c30))
cmp <- loo_compare(list("one drift" = lc$one, "switch at 20%" = lc[["20"]], "switch at 30%" = lc[["30"]]))
cat("\nLeave-one-out comparison, CPS alone\n")
print(cmp[, c("elpd_diff", "se_diff", "elpd_loo", "p_loo")])
cat("Largest Pareto k before refitting (points above 0.7 refitted exactly):",
    k_note(list("one drift" = lc$one, "20%" = lc[["20"]], "30%" = lc[["30"]])), "\n")
record_rows("pc, CPS alone", drc, lc, cmp)

res <- do.call(rbind, out)
rownames(res) <- NULL
write.csv(res, "output/results/cps_switch.csv", row.names = FALSE)

cat("\nPosterior predictive check (main price, switch at 20%): observed thresholds inside 90% predictive intervals\n")
yrep <- posterior_predict(main$f20)
lo <- apply(yrep, 2, quantile, 0.05)
hi <- apply(yrep, 2, quantile, 0.95)
inside <- d$tau >= lo & d$tau <= hi
cat(sprintf("  %d of %d inside; outside: %s\n", sum(inside), nrow(d),
            if (all(inside)) "none" else paste(d$source[!inside], d$year[!inside], collapse = ", ")))

ink <- "#222222"; accent <- "#0072B2"; accent2 <- "#D55E00"; grid_col <- "#DDDDDD"
pdf("output/diagnostics/cps_switch.pdf", width = 11, height = 4.6)
par(mfrow = c(1, 2), mar = c(4.2, 4.6, 2.6, 1), cex = 0.8, las = 1, bty = "l")
x <- setup(t20)
dr <- as.data.frame(as_draws_df(main$f20))
plot(d$date, d$tau - median(dr$b_cps) * d$cps, type = "n", xlab = "date", ylab = "threshold (log $ a year)",
     ylim = range(d$tau - median(dr$b_cps) * d$cps + c(-0.4, 0.4)),
     main = "Thresholds (CPS shifted by the offset) and pure-affordability path")
abline(h = pretty(range(d$tau)), col = grid_col, lwd = 0.5)
abline(v = c(t20, t30), col = "#888888", lty = 3)
for (src in c("CPS", "CE")) {
  k <- d$source == src
  col <- if (src == "CPS") accent2 else ink
  yv <- d$tau[k] - median(dr$b_cps) * d$cps[k]
  arrows(d$date[k], yv - 1.645 * d$se_tau[k], d$date[k], yv + 1.645 * d$se_tau[k], length = 0, col = col)
  points(d$date[k], yv, pch = if (src == "CPS") 17 else 19, cex = 0.7, col = col)
}
ce94 <- d$tau[d$source == "CE" & d$year == 1994]
lines(d$date[order(d$date)], (ce94 + d$ell_pc)[order(d$date)], col = accent, lwd = 1.3)
legend("topright", c("CPS (shifted by o)", "CE", "1994 threshold + price change (b = 1, no drift)"), bty = "n",
       col = c(accent2, ink, accent), pch = c(17, 19, NA), lwd = c(NA, NA, 1.3), cex = 0.85)
res_pc <- res[res$concept %in% c("pc", "pc, CPS alone"), ]
yy <- 6:1
par(mar = c(4.2, 8.5, 2.6, 1))
plot(res_pc$d1, yy + 0.12, xlim = range(c(res_pc$d1_lo, res_pc$d1_hi, res_pc$d2_lo, res_pc$d2_hi, 0)),
     ylim = c(0.5, 6.5), yaxt = "n", pch = 19, col = accent2, xlab = "leftover drift (log points a year)", ylab = "",
     main = "Drift before (orange) and after (blue) the switch, 90%")
segments(res_pc$d1_lo, yy + 0.12, res_pc$d1_hi, yy + 0.12, col = accent2)
points(res_pc$d2, yy - 0.12, pch = 19, col = accent)
segments(res_pc$d2_lo, yy - 0.12, res_pc$d2_hi, yy - 0.12, col = accent)
abline(v = 0, col = "#888888", lty = 2)
axis(2, at = yy, labels = c("one drift", "switch 20%", "switch 30%", "CPS: one drift", "CPS: switch 20%",
                           "CPS: switch 30%"), cex.axis = 0.8)
invisible(dev.off())
cat("\nWrote output/results/cps_switch.csv and output/diagnostics/cps_switch.pdf\n")

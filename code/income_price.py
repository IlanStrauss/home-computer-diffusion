"""Income, spending, and the price in household computer diffusion, CE 1994-2012 (the write-up's section on spending, the price,
and their interaction).

Uses the step-1 fits (observed long-run spending, geography only), whose spending axis ranks every unit, and the
step-1 second stage under the main price (its b). With x = log long-run spending, a unit in the pooled-average
geography owns with probability Pr = Phi(beta_t (x - tau_t)) (equation 2 of the write-up), and diffusion P_t is the
weighted mean of Pr over the year's units (equation 3).

A. Spending against current income as the measure of long-run resources: on complete income reporters with positive
   income, a weighted probit of ownership on log long-run spending alone and one on log current income alone, by year
   (McFadden R2 of each).
B. Where the spending threshold tau_t sits: the weighted share of the year's units whose long-run spending is below it,
   and weighted median long-run spending.
C. The interaction of price and income (equation 4): the rise in a unit's probability of owning from a 10% fall in the
   quality-adjusted price, Phi(beta_t (x - tau_t + b ln(1/0.9))) - Phi(beta_t (x - tau_t)), at each percentile of the
   year's spending, and averaged over units (with b at its median and its 90% interval); and the rise from a 10% rise
   in every unit's spending.
D. The balance of the two channels, 1994 to 2012: model ownership with the 1994 threshold and slope and the 1994 or
   2012 spending distribution, and with the 1994 threshold lowered by b times the price fall to 2012; Shapley shares
   of the combined rise (each channel's effect averaged over the two orders of adding them).
E. Does price responsiveness depend on income beyond spending relative to price? Weighted least squares of beta_t on
   ell_t, 1994-2012, steps 1-4: weights 1/se_beta^2, standard errors scaled up by the residual scatter when it exceeds
   the replicate errors.
F. D in constant dollars. Spending and the price are in current dollars, so D counts general inflation as spending
   growth. Here both are deflated by the PCE price index (all items), built like ell_t: interpolated monthly in logs
   between mid-year values and averaged over the year's units at their inventory months. The price channel is then the
   fall in the computer's price relative to all consumer prices, and the spending channel the growth of real spending.
G. F year by year, for the write-up's figure: model ownership with the 1994 threshold and slope when only the real
   price, only real spending, or both move from their 1994 values to year t's.

Run from the project folder: python3 code/income_price.py > output/logs/income_price.txt
Writes output/results/income_price_years.csv, income_price_percentiles.csv, and income_price_paths.csv.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

OUT = "data/processed"
RES = "output/results"
CUT10 = np.log(1 / 0.9)     # a 10% fall in the price, in log points
UP10 = np.log(1.1)          # a 10% rise in spending, in log points
PCTS = np.arange(1, 100)
pd.set_option("display.width", 200)


def wquantile(x, w, q):
    o = np.argsort(x)
    return x[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), q)]


def own_rate(d, tau, beta, shift=0.0):
    """Model ownership (%) of the units in d with threshold tau, slope beta, and every unit's spending raised by shift."""
    return 100 * np.average(norm.cdf(beta * (d.ln_s.to_numpy() + shift - tau)), weights=d.w)


if __name__ == "__main__":
    # w: the first stage's weights (each quarter an equal share of its year's weight, mean 1 within the year)
    u = pd.read_csv(f"{OUT}/ce_units.csv.gz", usecols=["year", "own", "ln_s", "inc_cur", "w", "ref_year", "ref_month"])
    u = u[u.year <= 2012].copy()
    pa = pd.read_csv(f"{OUT}/prices_annual.csv").dropna(subset=["pce"])
    u["ln_pce"] = np.interp(u.ref_year + (u.ref_month - 0.5) / 12, pa.year + 0.5, np.log(pa.pce))
    fs = pd.read_csv(f"{RES}/first_stage.csv")
    s1 = fs[fs.spec == "s1"].set_index("year")
    ss = pd.read_csv(f"{RES}/second_stage.csv")
    b = ss[(ss.spec == "s1") & (ss.concept == "pc") & (ss.model == "independent errors")].iloc[0]
    print(f"Step-1 second stage, main price: b = {b.b:.3f} [{b.b_lo:.3f}, {b.b_hi:.3f}]")

    # A. spending against current income
    probit = sm.families.Binomial(link=sm.families.links.Probit())
    rows = []
    for y, d in u[u.inc_cur > 0].groupby("year"):
        r = {"year": y, "n": len(d)}
        for v, col in [("spending", d.ln_s), ("income", np.log(d.inc_cur))]:
            f = sm.GLM(d.own, sm.add_constant(col.to_numpy()), family=probit, var_weights=d.w).fit()
            r[f"r2_{v}"] = 1 - f.llf / f.llnull
        rows.append(r)
    a = pd.DataFrame(rows)
    a["ratio"] = a.r2_income / a.r2_spending
    n_cr = int(a.n.sum())
    print(f"\nA. Complete income reporters with positive income, 1994-2012: {n_cr:,} of {len(u):,} units "
          f"({100 * (1 - n_cr / len(u)):.1f}% without). McFadden R2 of a probit on each measure alone:")
    print(a.round(3).to_string(index=False))
    print(f"income's R2 as a share of spending's: {a.ratio.min():.2f} to {a.ratio.max():.2f}; "
          f"spending fits better in {int((a.ratio < 1).sum())} of {len(a)} years")

    # B and C. where the threshold sits; the price-income interaction
    yrs, pct = [], []
    for y, d in u.groupby("year"):
        x, w = d.ln_s.to_numpy(), d.w.to_numpy()
        tau, beta = s1.loc[y, "tau"], s1.loc[y, "beta"]
        base = own_rate(d, tau, beta)
        yrs.append({"year": y, "own": 100 * np.average(d.own, weights=w), "own_model": base, "tau": tau,
                    "spend_tau": np.exp(tau), "beta": beta, "pct_below_tau": 100 * np.average(x < tau, weights=w),
                    "median_s": np.exp(wquantile(x, w, 0.5)),
                    "dP_price10": own_rate(d, tau - b.b * CUT10, beta) - base,
                    "dP_price10_lo": own_rate(d, tau - b.b_lo * CUT10, beta) - base,
                    "dP_price10_hi": own_rate(d, tau - b.b_hi * CUT10, beta) - base,
                    "dP_spend10": own_rate(d, tau, beta, UP10) - base})
        xq = np.array([wquantile(x, w, p / 100) for p in PCTS])
        pr = norm.cdf(beta * (xq - tau))
        pct.append(pd.DataFrame({"year": y, "pct": PCTS, "ln_s": xq, "pr": 100 * pr,
                                 "dpr_price10": 100 * (norm.cdf(beta * (xq - tau + b.b * CUT10)) - pr)}))
    yrs, pct = pd.DataFrame(yrs), pd.concat(pct, ignore_index=True)
    print("\nB and C. By year (step 1): ownership (actual and model), the spending threshold, the % of units spending "
          "less than it, median long-run spending, and the rise in ownership (percentage points) from a 10% fall in "
          "the price (b median and 90%) and from a 10% rise in every unit's spending")
    print(yrs.round(3).to_string(index=False))
    k = yrs.dP_price10.idxmax()
    print(f"10% price fall: {yrs.dP_price10.iloc[0]:.2f} pp in 1994, peak {yrs.dP_price10[k]:.2f} in "
          f"{yrs.year[k]}, {yrs.dP_price10.iloc[-1]:.2f} in 2012")
    print(f"ratio of spending to price effect, every year: {(yrs.dP_spend10 / yrs.dP_price10).min():.2f} to "
          f"{(yrs.dP_spend10 / yrs.dP_price10).max():.2f}")
    print("\nPeak of the unit-level response to a 10% price fall (percentile, pp):")
    for y in [1994, 1998, 2003, 2008, 2012]:
        d = pct[pct.year == y]
        j = d.dpr_price10.idxmax()
        print(f"  {y}: percentile {d.pct[j]}, {d.dpr_price10[j]:.2f} pp; at the 10th, 50th, 90th percentiles: "
              f"{d.dpr_price10[d.pct == 10].iloc[0]:.2f}, {d.dpr_price10[d.pct == 50].iloc[0]:.2f}, "
              f"{d.dpr_price10[d.pct == 90].iloc[0]:.2f}")

    # D. the balance of the two channels, 1994 to 2012
    d94, d12 = u[u.year == 1994], u[u.year == 2012]
    t94, b94 = s1.loc[1994, "tau"], s1.loc[1994, "beta"]
    dl = s1.loc[2012, "ell_pc"] - s1.loc[1994, "ell_pc"]
    start, end = own_rate(d94, t94, b94), own_rate(d12, s1.loc[2012, "tau"], s1.loc[2012, "beta"])
    spend = own_rate(d12, t94, b94)
    g = np.log(yrs.median_s.iloc[-1] / yrs.median_s.iloc[0])
    print(f"\nD. Balance, 1994 to 2012 (step 1). Price change {dl:.3f} log points; median long-run spending "
          f"${yrs.median_s.iloc[0]:,.0f} to ${yrs.median_s.iloc[-1]:,.0f} ({g:+.3f} log points). Actual ownership "
          f"{yrs.own.iloc[0]:.1f}% to {yrs.own.iloc[-1]:.1f}%; model {start:.1f}% to {end:.1f}% "
          f"(rise {end - start:.1f} points).")
    print(f"Spending growth alone (1994 threshold): {spend - start:+.1f} points")
    for lab, bb in [("median", b.b), ("5%", b.b_lo), ("95%", b.b_hi)]:
        price = own_rate(d94, t94 + bb * dl, b94)
        both = own_rate(d12, t94 + bb * dl, b94)
        sh_p = ((price - start) + (both - spend)) / 2
        sh_s = ((spend - start) + (both - price)) / 2
        print(f"b {lab} ({bb:.3f}): price alone (1994 spending) {price - start:+.1f}; price with 2012 spending "
              f"{both - spend:+.1f}; spending with the 1994 price {spend - start:+.1f}, with the 2012 price "
              f"{both - price:+.1f}; both {both - start:+.1f}; Shapley price {sh_p:.1f} ({100 * sh_p / (both - start):.0f}%), "
              f"spending {sh_s:.1f} ({100 * sh_s / (both - start):.0f}%); left for drift and slope "
              f"{end - both:+.1f}; affordability shift at the median: price {-bb * dl:.2f}, spending {g:.2f} log points "
              f"({100 * g / (g - bb * dl):.0f}% from spending)")

    # E. price responsiveness and income beyond the ratio
    print("\nE. WLS of beta_t on ell_t, 1994-2012 (main price)")
    for s in ["s1", "s2", "s3", "s4"]:
        f = fs[(fs.spec == s) & (fs.year <= 2012)].sort_values("year")
        m = sm.WLS(f.beta, sm.add_constant(f.ell_pc), weights=1 / f.se_beta ** 2).fit()
        se = m.bse["ell_pc"] / np.sqrt(m.scale) * np.sqrt(max(m.scale, 1))
        gam = m.params["ell_pc"]
        dlt = f.ell_pc.iloc[-1] - f.ell_pc.iloc[0]
        ch = 100 * gam * dlt / f.beta.mean()
        lo, hi = sorted([100 * (gam - 1.645 * se) * dlt / f.beta.mean(), 100 * (gam + 1.645 * se) * dlt / f.beta.mean()])
        print(f"{s}: gamma {gam:+.4f} [{gam - 1.645 * se:+.4f}, {gam + 1.645 * se:+.4f}] (residual scale {m.scale:.2f}); "
              f"change in beta over the price fall {ch:+.1f}% [{lo:+.1f}%, {hi:+.1f}%] of mean beta {f.beta.mean():.3f}")

    # F. the balance in constant dollars
    pce = u.groupby("year").apply(lambda d: np.average(d.ln_pce, weights=d.w), include_groups=False)
    infl = pce[2012] - pce[1994]
    dl_r, g_r = dl - infl, g - infl
    d12r = d12.assign(ln_s=d12.ln_s - infl)
    spend_r = own_rate(d12r, t94, b94)
    print(f"\nF. Balance in constant dollars. PCE prices rose {infl:.3f} log points, 1994 to 2012 (year means over units). "
          f"Computer price relative to all consumer prices {dl_r:.3f}; median long-run spending {g_r:+.3f} log points "
          f"(${yrs.median_s.iloc[-1] * np.exp(-infl):,.0f} in 1994 dollars). Real spending growth alone "
          f"{spend_r - start:+.1f} points")
    yrs["pce_rel"] = (pce - pce[1994]).to_numpy()
    for lab, bb in [("median", b.b), ("5%", b.b_lo), ("95%", b.b_hi)]:
        price = own_rate(d94, t94 + bb * dl_r, b94)
        both = own_rate(d12r, t94 + bb * dl_r, b94)
        sh_p = ((price - start) + (both - spend_r)) / 2
        sh_s = ((spend_r - start) + (both - price)) / 2
        print(f"b {lab} ({bb:.3f}): price alone {price - start:+.1f}; price with 2012 spending {both - spend_r:+.1f}; "
              f"spending with the 2012 price {both - price:+.1f}; both {both - start:+.1f}; Shapley price {sh_p:.1f} "
              f"({100 * sh_p / (both - start):.0f}%), spending {sh_s:.1f} ({100 * sh_s / (both - start):.0f}%); left "
              f"{end - both:+.1f}; shift at the median: price {-bb * dl_r:.2f}, spending {g_r:.2f} log points "
              f"({100 * g_r / (g_r - bb * dl_r):.0f}% from spending)")

    # G. F year by year
    paths = []
    for y, d in u.groupby("year"):
        infl_y = pce[y] - pce[1994]
        dl_y = s1.loc[y, "ell_pc"] - s1.loc[1994, "ell_pc"] - infl_y
        dr = d.assign(ln_s=d.ln_s - infl_y)
        paths.append({"year": y, "start": start, "price_only": own_rate(d94, t94 + b.b * dl_y, b94),
                      "spend_only": own_rate(dr, t94, b94), "both": own_rate(dr, t94 + b.b * dl_y, b94),
                      "own_model": own_rate(d, s1.loc[y, "tau"], s1.loc[y, "beta"]),
                      "own": 100 * np.average(d.own, weights=d.w), "real_price": dl_y, "infl": infl_y})
    paths = pd.DataFrame(paths)
    print("\nG. F year by year (b median): model ownership (%) with the 1994 threshold and slope when only the real price, "
          "only real spending, or both move to year t; the model's and the actual ownership")
    print(paths.round(3).to_string(index=False))

    yrs.to_csv(f"{RES}/income_price_years.csv", index=False)
    pct.to_csv(f"{RES}/income_price_percentiles.csv", index=False)
    paths.to_csv(f"{RES}/income_price_paths.csv", index=False)
    print(f"\nWrote {RES}/income_price_years.csv, income_price_percentiles.csv, and income_price_paths.csv")

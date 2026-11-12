"""Step 5 of the household PC model: thresholds from the CPS computer supplements, 1984-2003.

The CPS has bracketed family income, not spending. Long-run spending is imputed from income in two samples
(Skinner 1987; Blundell, Pistaferri, and Preston 2008):
1. In the CE (complete income reporters with positive income, 1994-1998), each unit's income is converted to the
   CPS year's dollars with the PCE price index, placed in that year's CPS bracket, and replaced by the bracket's
   value: the midpoint, or for the open top bracket the Pareto mean (Parker and Fenwick 1983), with the Pareto
   parameter from the CPS year's top two brackets. So the CE sees income exactly as the CPS reports it.
2. A weighted regression in the CE (1994 dollars) gives the mapping
       ln s_i = m0 + m1 ln y_i + m2 (ln y_i)^2 + z_i' m3 + region_i + u_i
   with z_i = the step-5 traits (age, children, adults, married, race, Hispanic origin, education).
3. Each CPS household gets ln s_hat = the mapping at its bracket value and traits, in the CPS year's dollars.
4. The CPS probit is the CE specification without the student-housing trait, which the CPS lacks:
       Pr(own_i = 1) = Phi(alpha_t + beta_t (ln s_hat_i - 10) + gamma_g(i),t + z_i' delta_t)
   with the CE geography groups (CPS states outside the CE's own-state groups go to the rest of their region) and
   the threshold read at the CE's pooled group shares and pooled traits.
   Because u_i is independent of the regressors, the imputation error rescales the probit's coefficients alike and
   leaves tau_t unchanged, if u_i is normal with constant variance (as in regression calibration).
Standard errors: the mapping is re-estimated with each of the 44 CE replicate weights (mapping uncertainty); the
CPS sampling error is the probit's sandwich standard error times sqrt(2), an assumed design effect of 2 (the public
CPS files of these years have no replicate weights or design codes). The two are added in quadrature.
Validation: CPS and CE thresholds from the same specification in 1997, 1998, 2000, 2001, and 2003, with the CE
threshold interpolated to the CPS survey month; and the imputation applied to CE units themselves.

Run from the project folder: python3 code/cps_extension.py > output/logs/cps_extension.txt
Writes output/results/cps_thresholds.csv and cps_validation.csv.
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

import first_stage as FS  # noqa: E402
import hhpc as H  # noqa: E402

T5 = [c for c in H.TRAITS if c != "student"] + H.EDUC
MAP_YEARS = range(1994, 1999)
CPS_MONTH = {1984: 10, 1989: 10, 1993: 10, 1997: 10, 1998: 12, 2000: 8, 2001: 9, 2003: 10}
DEFF = 2.0
REGION = {**{s: 1 for s in [9, 23, 25, 33, 34, 36, 42, 44, 50]},
          **{s: 2 for s in [17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55]},
          **{s: 3 for s in [1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54]},
          **{s: 4 for s in [2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56]}}
pd.set_option("display.width", 220)


def pareto_top(lo, hi, w):
    """Pareto mean of the open top bracket from the weighted counts in the top two brackets."""
    tops = sorted(set(lo[np.isinf(hi)]))
    assert len(tops) == 1
    L = tops[0]
    L1 = max(set(lo[lo < L]))
    f_top, f_next = w[lo == L].sum(), w[lo == L1].sum()
    alpha = np.log((f_next + f_top) / f_top) / np.log(L / L1)
    assert alpha > 1, alpha
    return L * alpha / (alpha - 1), alpha


def bracket_value(y, lo_edges, hi_edges, top_mean):
    """Places incomes y (current dollars) in brackets and returns each bracket's value (midpoint; top: Pareto mean)."""
    out = np.full(len(y), np.nan)
    for lo, hi in zip(lo_edges, hi_edges):
        k = (y >= lo) & (y < hi) if lo > 0 else (y < hi)
        out[k] = top_mean if np.isinf(hi) else (lo + hi) / 2
    return out


def mapping_matrix(d, lny):
    cols = [np.ones(len(d)), lny, lny ** 2] + [d[c].to_numpy(float) for c in T5] + \
           [(d.region5 == r).to_numpy(float) for r in [2, 3, 4]]
    return np.column_stack(cols)


def add_cps_traits(c):
    c = c.copy()
    c["fam_size"], c["tenure"] = c.persons, 0
    c["hisp"] = c.hisp.astype(float)
    c = H.add_traits(c)
    return c


def sandwich_tau(y, X, w, b, pen, design):
    eta = X @ b
    p, f = norm.cdf(eta).clip(1e-12, 1 - 1e-12), norm.pdf(eta)
    g = (w * (y - p) * f / (p * (1 - p)))[:, None] * X
    hinv = np.linalg.inv((X * (w * f ** 2 / (p * (1 - p)))[:, None]).T @ X + np.diag(pen))
    k = design.k_geo
    level = b[0] + b[2:2 + k] @ design.pi
    grad = np.zeros(len(b))
    grad[0], grad[1], grad[2:2 + k] = -1 / b[1], level / b[1] ** 2, -design.pi / b[1]
    return float(np.sqrt(grad @ hinv @ (g.T @ g) @ hinv @ grad))


if __name__ == "__main__":
    FS.worker_init()
    u = FS.U
    pa = pd.read_csv(f"{H.OUT}/prices_annual.csv").set_index("year")
    pce = pa.pce
    u["region5"] = u.region_filled
    own_states = sorted({int(g[1:]) for g in u.geo.unique() if g.startswith("s")})

    # ---------- CE side: the step-5 specification, with replicates ----------
    spec = {"label": "Step 5, CE: corrected spending, geography, traits without student housing, education",
            "rc": True, "traits": T5}
    FS.SPECS["s5_ce"] = spec
    _, ce_tab, _, _, secs = FS.do_spec("s5_ce")
    ce_tab = ce_tab.set_index("year")
    ce_tab["date"] = u.groupby("year").apply(lambda d: H.wmean(d.ref_year + (d.ref_month - 0.5) / 12, d.w),
                                             include_groups=False)
    print(f"CE thresholds, step-5 specification ({secs:.0f}s): tau_t, se, own %, mean reference date")
    print(ce_tab[["n", "own_pct", "tau", "se_tau", "beta", "date"]].round(3).to_string())

    # ---------- CPS ----------
    c = pd.read_csv(f"{H.OUT}/cps_households.csv.gz")
    n0 = c.groupby("year").size()
    c = c[c.owns.notna() & (c.hh_weight > 0)]
    ans = c.groupby("year").apply(lambda d: 100 * H.wmean(d.owns, d.hh_weight), include_groups=False)
    c = c[c.inc_lo.notna() & c[["age", "kids", "persons", "race3", "hisp", "married", "educ4", "state"]].notna().all(
        axis=1)].copy()
    c = add_cps_traits(c)
    c["own"] = c.owns.astype(int)
    c["state"] = c.state.astype(int)
    c["region5"] = c.state.map(REGION)
    assert c.region5.notna().all()
    c["geo"] = np.where(c.state.isin(own_states), "s" + c.state.astype(str).str.zfill(2),
                        "rest_r" + c.region5.astype(int).astype(str))
    c["w"] = c.hh_weight / c.groupby("year").hh_weight.transform("mean")
    kept = c.groupby("year").apply(lambda d: 100 * H.wmean(d.own, d.w), include_groups=False)
    print("\nCPS households: interviewed, answered the computer question (weighted ownership %), and kept "
          "(income bracket and traits known; weighted ownership %)")
    print(pd.DataFrame({"households": n0, "own_answering": ans.round(1), "kept": c.groupby("year").size(),
                        "own_kept": kept.round(1)}).to_string())

    # ownership crossing dates, from all answering households
    dates = pd.Series({t: t + (m - 0.5) / 12 for t, m in CPS_MONTH.items()})
    for level in [20, 30]:
        i = np.searchsorted(ans.to_numpy(), level)
        t0, t1, p0, p1 = dates.iloc[i - 1], dates.iloc[i], ans.iloc[i - 1], ans.iloc[i]
        lin = t0 + (t1 - t0) * (level - p0) / (p1 - p0)
        lg = lambda p: np.log(p / (100 - p))  # noqa: E731
        logi = t0 + (t1 - t0) * (lg(level) - lg(p0)) / (lg(p1) - lg(p0))
        print(f"Ownership passes {level}% (CPS, all answering households): {lin:.2f} by linear interpolation, "
              f"{logi:.2f} by log-odds interpolation, between surveys at {t0:.2f} ({p0:.1f}%) and {t1:.2f} ({p1:.1f}%)")

    # ---------- the mapping, estimated in the CE ----------
    donors = u[u.year.isin(MAP_YEARS) & u.ln_inc.notna() & u.region5.notna()].copy()
    defl = np.log(donors.year.map(pce) / pce.loc[1994])
    donors["y_real"] = np.exp(donors.ln_inc - defl)
    donors["lns_real"] = donors.ln_s - defl
    design = H.Design(u, "probit", geo=True, traits=T5, ratios=H.EDUC)
    print(f"\nMapping sample: {len(donors):,} CE complete income reporters with positive income and a known region, "
          f"{min(MAP_YEARS)}-{max(MAP_YEARS)}; 1994 dollars (PCE price index)")

    rows, cps_fit = [], {}
    for t in CPS_MONTH:
        d = c[c.year == t]
        infl = pce.loc[t] / pce.loc[1994]
        top_mean, alpha = pareto_top(d.inc_lo.to_numpy(), d.inc_hi.to_numpy(), d.w.to_numpy())
        edges = sorted(set(zip(d.inc_lo, d.inc_hi)))
        lo_e, hi_e = [e[0] for e in edges], [e[1] for e in edges]
        cps_val = bracket_value(d.inc_lo.to_numpy() + 1, lo_e, hi_e, top_mean)
        ce_val = bracket_value(donors.y_real.to_numpy() * infl, lo_e, hi_e, top_mean)
        lny_cps, lny_ce = np.log(cps_val / infl), np.log(ce_val / infl)
        M_ce, M_cps = mapping_matrix(donors, lny_ce), mapping_matrix(d, lny_cps)
        y_ce = donors.lns_real.to_numpy()
        coefs = [FS.wls(M_ce, y_ce, donors[wc].to_numpy()) for wc in ["w"] + H.REP]
        res = y_ce - M_ce @ coefs[0]
        r2 = 1 - H.wmean(res ** 2, donors.w.to_numpy()) / H.wmean((y_ce - H.wmean(y_ce, donors.w)) ** 2, donors.w)
        y = d.own.to_numpy(float)
        taus, b = [], None
        for k, m in enumerate(coefs):
            x = M_cps @ m + np.log(infl)
            X, pen = design.matrix(d, x)
            b = H.fit_binary(y, X, d.w.to_numpy(), "probit", pen, start=b)
            if k == 0:
                b_full, X_full, pen_full = b, X, pen
            taus.append(design.read(b)["tau"])
            b = b_full
        se_map = H.brr(taus[0], taus[1:])
        se_samp = np.sqrt(DEFF) * sandwich_tau(y, X_full, d.w.to_numpy(), b_full, pen_full, design)
        out = {"source": "CPS", "year": t, "date": dates[t], "n": len(d), "own_pct": 100 * H.wmean(y, d.w),
               **design.read(b_full), "se_map": se_map, "se_samp": se_samp,
               "se_tau": np.sqrt(se_map ** 2 + se_samp ** 2), "pareto_alpha": alpha, "top_mean": top_mean,
               "map_slope_at_mean": coefs[0][1] + 2 * coefs[0][2] * H.wmean(lny_ce, donors.w), "map_r2": r2,
               "map_resid_sd": np.sqrt(H.wmean(res ** 2, donors.w.to_numpy()))}
        rows.append(out)
        cps_fit[t] = (coefs[0], lo_e, hi_e, top_mean, infl)
    cps = pd.DataFrame(rows).set_index("year")
    print("\nCPS thresholds (imputed long-run spending; step-5 specification)")
    print(cps.drop(columns=["source"]).round(3).to_string())

    # ---------- validation ----------
    print("\nValidation: CPS against CE thresholds from the same specification (CE interpolated to the CPS month)")
    ce_at = np.interp(cps.date, ce_tab.date, ce_tab.tau)
    se_at = np.interp(cps.date, ce_tab.date, ce_tab.se_tau)
    val = pd.DataFrame({"cps_tau": cps.tau, "cps_se": cps.se_tau, "ce_tau": ce_at, "ce_se": se_at,
                        "gap": cps.tau - ce_at, "gap_in_se": (cps.tau - ce_at) / np.sqrt(cps.se_tau ** 2 + se_at ** 2)},
                       index=cps.index)
    val = val.loc[[1997, 1998, 2000, 2001, 2003]]
    print(val.round(3).to_string())
    print("The same imputation applied to CE units (complete reporters) in the CE year, against CE fits on the same "
          "units with corrected and with observed spending (point estimates)")
    for t in [1997, 1998, 2000, 2001, 2003]:
        m, lo_e, hi_e, top_mean, infl = cps_fit[t]
        d = u[(u.year == t) & u.ln_inc.notna()].copy()
        d["w"] = d.w / d.w.mean()
        yr_infl = pce.loc[t] / pce.loc[1994]
        val_ce = bracket_value(np.exp(d.ln_inc).to_numpy(), lo_e, hi_e, top_mean)
        x_imp = mapping_matrix(d, np.log(val_ce / yr_infl)) @ m + np.log(yr_infl)
        y, w = d.own.to_numpy(float), d.w.to_numpy()
        x_rc, _ = FS.calibrate(d, w, d.ln_s.to_numpy(), FS.covariates(d, design))
        fit = {}
        for name, x in [("imputed", x_imp), ("corrected", x_rc), ("observed", d.ln_s.to_numpy())]:
            X, pen = design.matrix(d, x)
            fit[name] = H.fit_binary(y, X, w, "probit", pen)
            val.loc[t, f"ce_units_{name}_tau"] = design.read(fit[name])["tau"]
            val.loc[t, f"ce_units_{name}_beta"] = fit[name][1]
        print(f"  {t}: " + "; ".join(f"{k} {design.read(b)['tau']:.3f} (beta {b[1]:.3f})" for k, b in fit.items()))
    val.to_csv(f"{H.RES}/cps_validation.csv")

    # ---------- output for step 6 ----------
    mp = pd.read_csv(f"{H.OUT}/prices_monthly.csv")
    lp94 = H.year_prices(u[u.year == 1994], ["pc", "info", "business"]).loc[1994]
    for k in ["pc", "info", "business"]:
        cps[f"ell_{k}"] = [mp.loc[(mp.year == t) & (mp.month == CPS_MONTH[t]), f"ln_{k}"].item() - lp94[k]
                           for t in cps.index]
    ce_tab["source"], ce_tab["se_map"], ce_tab["se_samp"] = "CE", np.nan, np.nan
    keep = ["source", "date", "n", "own_pct", "tau", "se_tau", "beta", "ell_pc", "ell_info", "ell_business"]
    out = pd.concat([cps[keep], ce_tab[keep]]).reset_index()
    out["trend"] = out.date - 1994
    out.to_csv(f"{H.RES}/cps_thresholds.csv", index=False)
    print("\nPrices at the CPS months (ell, log points relative to the CE's 1994 average)")
    print(cps[["date", "ell_pc", "ell_info", "ell_business"]].round(3).to_string())
    print("\nWrote output/results/cps_thresholds.csv and cps_validation.csv")

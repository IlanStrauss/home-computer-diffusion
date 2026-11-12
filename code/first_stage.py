"""First stage of steps 1-4 of the household PC model, CE 1994-2013: one fit per year under each specification.

Model in year t, fitted by weighted maximum likelihood (penalized for the geography effects; see hhpc.py):
    Pr(own_it = 1) = F(alpha_t + beta_t (x_it - 10) + gamma_g(i),t + z_it' delta_t)
    tau_t = 10 - (alpha_t + sum_g pi_g gamma_g,t) / beta_t
- own_it = 1 if unit i, observed in year t, has a computer; F = the probit link unless stated.
- x_it = the unit's log long-run spending, or a substitute (below). gamma_g,t = the effect of geography group g.
- z_it = household traits, centred at their pooled 1994-2013 means, so tau_t is the threshold of a unit with the
  pooled average traits, in the pooled average geography (fixed shares pi_g).
- theta_z,t = delta_z,t / beta_t = the shift in the threshold, in log points of spending, that trait z is worth
  (positive: a unit with the trait owns at lower spending).

Specifications:
- Step 1: x = ln s, geography. Checks: no geography (the first run), logit and robit links, units first seen at
  interview 2, no student housing, spending trimmed at the 0.5th and 99.5th percentiles, two other spending
  definitions, unweighted, and a heteroskedastic probit.
- Step 2: x = ln s corrected for noise between interviews by regression calibration (Carroll et al. 2006, ch. 4):
      ln s_i = x*_i + e_i,  Var(e_i) = s2u_t / n_i
      x*_i = c_i' pi_t + r_i,  Var(r_i) = v_t
      E[x*_i | ln s_i, c_i] = c_i' pi_t + lambda_i (ln s_i - c_i' pi_t),  lambda_i = v_t / (v_t + s2u_t / n_i)
  n_i = the unit's number of interviews; s2u_t = the pooled within-unit variance of log quarterly spending in year t;
  c_i = the fit's other covariates (geography, and traits from step 3). The probit on the calibrated value recovers
  tau_t; it understates beta_t by a factor sqrt(1 + beta_t^2 v_t (1 - lambda_i)), under 1.02 here.
- Step 3: step 2 plus household traits (hhpc.TRAITS). Check: robit with 7 degrees of freedom, the link the
  cross-validation rule of link_choice.py picks with traits.
- Step 4: step 3 plus education (hhpc.EDUC), with theta_e,t. Three income measures, on complete income reporters
  with positive income: current income; corrected spending; and observed spending with a control function for
  current income (Rivers and Vuong 1988): the residual of a regression of ln s on ln income and the other
  covariates enters the probit.

Standard errors: every step (including the calibration and the control function) is repeated with each of the 44
replicate weights. Quantities that compare years (L_t, S_t) are computed within each replicate, so the covariance
between years is kept.
- L_t = (tau_t - tau_1994) - ell_t: the leftover change in the threshold that the price does not account for.
- S_t = (P^A_t - Phat_1994) / (P_t - Phat_1994): the share of the rise in ownership since 1994 that the 1994 model
  predicts from year-t units and the year-t price; S_spend_t the same with the 1994 price.

Run from the project folder: python3 code/first_stage.py > output/logs/first_stage.txt
Writes output/results/first_stage.csv, output/results/first_stage_replicates.csv.gz, and
output/results/first_stage_fake.csv.
"""
import os
import time
from multiprocessing import Pool

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")   # one thread per worker: the matrices are small, and 8 workers run at once
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import norm  # noqa: E402

import hhpc as H  # noqa: E402

CONCEPTS = ["pc", "info", "business", "cpi_pc", "paid_p50", "paid_p90"]
T3, T4 = H.TRAITS, H.TRAITS + H.EDUC
CR = "complete reporters with positive income"
SPECS = {
    "s1": dict(label="Step 1: log spending, geography"),
    "s1_nogeo": dict(label="Step 1, no geography (first run)", geo=False),
    "s1_logit": dict(label="Step 1, logit", link="logit"),
    "s1_robit2": dict(label="Step 1, robit (2 df)", link="robit2"),
    "s1_int2": dict(label="Step 1, units first seen at interview 2", sample="int2"),
    "s1_nostudent": dict(label="Step 1, no student housing", sample="nostudent"),
    "s1_trim": dict(label="Step 1, spending trimmed at 0.5% and 99.5%", sample="trim"),
    "s1_incl": dict(label="Step 1, spending incl. insurance, pensions, contributions", x="ln_s_incl"),
    "s1_veh": dict(label="Step 1, spending incl. vehicles", x="ln_s_veh"),
    "s2": dict(label="Step 2: corrected spending, geography", rc=True),
    "s3_obs": dict(label="Step 3 with observed spending", traits=T3),
    "s3": dict(label="Step 3: corrected spending, geography, traits", rc=True, traits=T3),
    "s3_robit7": dict(label="Step 3, robit (7 df)", rc=True, traits=T3, link="robit7"),
    "s4": dict(label="Step 4: step 3 plus education", rc=True, traits=T4),
    "s4_logit": dict(label="Step 4, logit", rc=True, traits=T4, link="logit"),
    "s4_int2": dict(label="Step 4, units first seen at interview 2", rc=True, traits=T4, sample="int2"),
    "s4_incl": dict(label="Step 4, spending incl. insurance, pensions, contributions", rc=True, traits=T4,
                    x="ln_s_incl"),
    "s4_veh": dict(label="Step 4, spending incl. vehicles", rc=True, traits=T4, x="ln_s_veh"),
    "s4_cr": dict(label=f"Step 4, corrected spending, {CR}", rc=True, traits=T4, sample="cr"),
    "s4_inc": dict(label=f"Step 4, current income, {CR}", traits=T4, sample="cr", x="ln_inc"),
    "s4_cf": dict(label=f"Step 4, spending with a control function for income, {CR}", traits=T4, sample="cr",
                  cf=True),
}
UNWEIGHTED = ["s1", "s4"]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def wls(C, y, w):
    k = w > 0
    sw = np.sqrt(w[k])
    return np.linalg.lstsq(C[k] * sw[:, None], y[k] * sw, rcond=None)[0]


def calibrate(d, w, x, C):
    """Regression calibration of x = ln s given covariates C (see the module note). Returns values and diagnostics."""
    k = w > 0
    mu = C @ wls(C, x, w)
    r = x - mu
    n, sd = d.n_interviews.to_numpy(float), d.sd_log_cons.to_numpy()
    ok = k & np.isfinite(sd) & (n >= 2)
    s2u = np.sum(w[ok] * (n[ok] - 1) * sd[ok] ** 2) / np.sum(w[ok] * (n[ok] - 1))
    noise = s2u / n
    v = H.wmean(r[k] ** 2, w[k]) - H.wmean(noise[k], w[k])
    lam = v / (v + noise)
    return mu + lam * r, {"s2u": s2u, "v": v, "lam": H.wmean(lam[k], w[k])}


def covariates(d, design):
    """The calibration and control-function regressors: intercept, geography dummies (but one), and the traits."""
    cols = [np.ones(len(d))]
    if design.geo:
        cols += [(d.geo == g).to_numpy(float) for g in design.groups[1:]]
    cols += [d[c].to_numpy(float) for c in design.traits]
    return np.column_stack(cols)


def subsample(u, kind):
    if kind is None:
        return u
    keep = {"int2": u.interview == 2, "nostudent": u.tenure != 6, "cr": u.ln_inc.notna(),
            "trim": u.groupby("year").ln_s.transform(lambda s: s.between(s.quantile(0.005), s.quantile(0.995)))}[kind]
    d = u[keep].copy()
    for c in ["w"] + H.REP:
        d[c] = H.rescale(d[c], d.year, d.quarter)
    return d


def run_spec(name, spec, u, reps=True):
    """Fits every year with the full-sample and replicate weights. Returns the design, the full-sample results by
    year, the subsample used, and the stored fits: year -> (data, [(b, x, extra) for full sample and replicates])."""
    design = H.Design(u, spec.get("link", "probit"), geo=spec.get("geo", True), traits=spec.get("traits"),
                      ratios=[c for c in H.EDUC if c in spec.get("traits", [])])
    d_all = subsample(u, spec.get("sample"))
    xcol = spec.get("x", "ln_s")
    out, fits = [], {}
    for t in H.YEARS:
        d = d_all[d_all.year == t]
        y = d.own.to_numpy(float)
        C = covariates(d, design) if (spec.get("rc") or spec.get("cf")) else None
        x0 = d[xcol].to_numpy()
        stored, full_b = [], None
        for wc in ["w"] + (H.REP if reps else []):
            w = d[wc].to_numpy()
            x, extra, diag = x0, None, {}
            if spec.get("rc"):
                x, diag = calibrate(d, w, x0, C)
            if spec.get("cf"):
                Cz = np.column_stack([C, d.ln_inc.to_numpy()])
                extra = x0 - Cz @ wls(Cz, x0, w)
            X, pen = design.matrix(d, x, extra)
            try:
                b = H.fit_binary(y, X, w, design.link, pen, start=full_b)
            except RuntimeError as err:
                raise RuntimeError(f"{name}, {t}, weights {wc}: {err}") from None
            if wc == "w":
                full_b = b
                out.append({"spec": name, "year": t, "n": len(d), "own_pct": 100 * H.wmean(y, w), **design.read(b),
                            **({f"rc_{k}": v for k, v in diag.items()})})
            stored.append((b, x, extra))
        fits[t] = (d, stored)
    return design, pd.DataFrame(out).set_index("year"), d_all, fits


def rows_brr(full, reps):
    """BRR standard errors by row: full is a vector over years, reps a years x 44 matrix."""
    return np.sqrt(np.mean((np.asarray(reps) - np.asarray(full)[:, None]) ** 2, axis=1))


def year_ell(d_all):
    """ell_t under each concept, and ell_t (BEA PC price) under each replicate weight, for the units in d_all."""
    lp = H.year_prices(d_all, CONCEPTS)
    ell = lp - lp.loc[1994]
    ell["cpi_pc"] = lp.cpi_pc - lp.cpi_pc.loc[1999]
    lp_rep = pd.DataFrame({r: d_all.groupby("year").apply(lambda d, c=c: H.wmean(d.ln_pc, d[c]), include_groups=False)
                           for r, c in enumerate(H.REP, start=1)})
    return ell, lp_rep - lp_rep.loc[1994]


def replicate_table(name, design, fits, ell_rep):
    rows = []
    for t, (d, stored) in fits.items():
        for r, (b, x, extra) in enumerate(stored[1:], start=1):
            rows.append({"spec": name, "year": t, "rep": r, **design.read(b), "ell_pc": ell_rep.loc[t, r]})
    return pd.DataFrame(rows)


def shares(design, fits, ell, ell_rep):
    """S_t and S_spend_t with their joint-replicate standard errors."""
    F = H.LINKS[design.link][0]
    d94, st94 = fits[1994]
    res = {}
    for t in H.YEARS[1:]:
        d, st = fits[t]
        vals = {"S_pc": [], "S_spend": []}
        for r in range(len(st)):
            wc = "w" if r == 0 else H.REP[r - 1]
            b94, x94, e94 = st94[r]
            p94 = H.wmean(F(design.matrix(d94, x94, e94)[0] @ b94), d94[wc].to_numpy())
            _, xt, et = st[r]
            Xt = design.matrix(d, xt, et)[0]
            w = d[wc].to_numpy()
            actual = H.wmean(d.own.to_numpy(float), w)
            shift = ell.loc[t] if r == 0 else ell_rep.loc[t, r]
            for key, s in [("S_pc", shift), ("S_spend", 0.0)]:
                Xs = Xt.copy()
                Xs[:, 1] -= s
                vals[key].append((H.wmean(F(Xs @ b94), w) - p94) / (actual - p94))
        for key, v in vals.items():
            res[(t, key)] = v[0]
            res[(t, f"se_{key}")] = H.brr(v[0], v[1:]) if len(v) > 1 else np.nan
    return pd.Series(res).unstack()


def hetprobit(y, X, H_, w, pen, b0):
    """Probit with Pr = Phi(X b / exp(H k)), by L-BFGS with analytic gradients and |k| <= 2.5; returns b, k."""
    kx = X.shape[1]

    def f(par):
        b, k = par[:kx], par[kx:]
        s = np.exp(H_ @ k)
        z = X @ b / s
        p = norm.cdf(z).clip(1e-12, 1 - 1e-12)
        ll = np.sum(w * (y * np.log(p) + (1 - y) * np.log1p(-p))) - 0.5 * np.sum(pen * b ** 2)
        g = w * (y - p) * norm.pdf(z) / (p * (1 - p))
        return -ll, -np.r_[X.T @ (g / s) - pen * b, -(H_.T @ (g * z))]

    bounds = [(None, None)] * kx + [(-2.5, 2.5)] * H_.shape[1]
    r = minimize(f, np.r_[b0, np.zeros(H_.shape[1])], jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-8})
    return r.x[:kx], r.x[kx:], r.success


def checks(design, fits):
    """Unweighted fits and a heteroskedastic probit (point estimates), year by year."""
    rows = []
    for t in H.YEARS:
        d, stored = fits[t]
        b, x, extra = stored[0]
        y, w = d.own.to_numpy(float), d.w.to_numpy()
        X, pen = design.matrix(d, x, extra)
        bu = H.fit_binary(y, X, np.ones(len(d)), design.link, pen, start=b)
        Hm = np.column_stack([np.log(d.n_interviews / 4), (d.age < 25) | (d.tenure == 6)]).astype(float)
        bh, kh, ok = hetprobit(y, X, Hm, w, pen, b)
        rows.append({"year": t, "tau": design.read(b)["tau"], "tau_unweighted": design.read(bu)["tau"],
                     "tau_hetprobit": design.read(bh)["tau"], "beta": b[1], "beta_unweighted": bu[1],
                     "beta_hetprobit": bh[1], "scale_one_interview": np.exp(-kh[0] * np.log(4)),
                     "scale_young_or_student": np.exp(kh[1]), "converged": ok})
    return pd.DataFrame(rows).set_index("year")


def worker_init():
    global U, ELL_MAIN, ELL_REP_MAIN
    U = H.add_traits(H.load_units())
    U["ln_inc"] = np.log(U.inc_cur.where(U.inc_cur > 0))
    ELL_MAIN, ELL_REP_MAIN = year_ell(U)


def do_spec(name):
    spec = SPECS[name]
    t1 = time.time()
    design, tab, d_all, fits = run_spec(name, spec, U)
    ell, ell_rep = year_ell(d_all) if spec.get("sample") else (ELL_MAIN, ELL_REP_MAIN)
    reps = replicate_table(name, design, fits, ell_rep)
    rs = reps.pivot(index="year", columns="rep", values="tau")
    tab["se_tau"] = rows_brr(tab.tau, rs)
    tab["se_beta"] = rows_brr(tab.beta, reps.pivot(index="year", columns="rep", values="beta"))
    for c in design.ratios:
        tab[f"se_theta_{c}"] = rows_brr(tab[f"theta_{c}"], reps.pivot(index="year", columns="rep", values=f"theta_{c}"))
    tab["sigma"] = 1 / tab.beta
    for c in CONCEPTS:
        tab[f"ell_{c}"] = ell[c]
    tab["L_pc"] = (tab.tau - tab.tau.loc[1994]) - tab.ell_pc
    tab["se_L_pc"] = rows_brr(tab.L_pc, (rs - rs.loc[1994]) - ell_rep)
    tab = tab.join(shares(design, fits, ell.pc, ell_rep))
    chk = checks(design, fits) if name in UNWEIGHTED else None
    return name, tab.reset_index(), reps, chk, time.time() - t1


def do_fake(args):
    sname, a, b, dd, sigma0, seed = args
    rng = np.random.default_rng(seed)
    ell_pc = ELL_MAIN.pc
    true = pd.Series(a + b * ell_pc.to_numpy() + dd * np.arange(20) + rng.normal(0, 0.03, 20), index=H.YEARS)
    uf = U.copy()
    uf["own"] = ((uf.ln_s - uf.year.map(true)) / sigma0 + rng.standard_normal(len(uf)) > 0).astype(int)
    design, tab, _, fits = run_spec("fake", SPECS["s1"], uf)
    reps = replicate_table("fake", design, fits, ELL_REP_MAIN)
    tab["se_tau"] = rows_brr(tab.tau, reps.pivot(index="year", columns="rep", values="tau"))
    out = tab[["tau", "se_tau", "own_pct"]].assign(scenario=sname, tau_true=true, a_true=a, b_true=b, d_true=dd,
                                                   omega_true=0.03, ell_pc=ell_pc, trend=np.arange(20))
    return out, float(np.max(np.abs(tab.tau - true) / tab.se_tau))


if __name__ == "__main__":
    t0 = time.time()
    worker_init()
    u, ell_main, ell_rep_main = U, ELL_MAIN, ELL_REP_MAIN
    print(f"Sample: {len(u):,} consumer units, CE 1994-2013. Complete income reporters with positive income: "
          f"{u.ln_inc.notna().sum():,}.")
    print(f"Replicate standard deviation of ell_t (BEA PC price) from the weights alone: at most "
          f"{rows_brr(ell_main.pc.to_numpy(), ell_rep_main.to_numpy()).max():.4f} log points")
    order = sorted(SPECS, key=lambda k: (SPECS[k].get("link", "probit") == "probit", "traits" not in SPECS[k]))
    with Pool(8, initializer=worker_init) as pool:
        results = {r[0]: r[1:] for r in pool.imap_unordered(do_spec, order)}
    for name in SPECS:
        print(f"[{name}] {SPECS[name]['label']}: {results[name][3]:.0f}s")
    fs = pd.concat([results[k][0] for k in SPECS], ignore_index=True)
    fs.to_csv(f"{H.RES}/first_stage.csv", index=False)
    reps_all = pd.concat([results[k][1] for k in SPECS], ignore_index=True)
    reps_all.to_csv(f"{H.RES}/first_stage_replicates.csv.gz", index=False)

    show = ["n", "own_pct", "tau", "se_tau", "beta", "se_beta", "L_pc", "se_L_pc", "S_pc", "se_S_pc", "S_spend"]
    for name, spec in SPECS.items():
        tab = fs[fs.spec == name].set_index("year")
        print(f"\n=== [{name}] {spec['label']}")
        cols = show + [c for c in tab if c.startswith("rc_")] + [c for c in tab if c.startswith("theta_") or
                                                                  c.startswith("se_theta_")]
        print(tab[[c for c in cols if c in tab and tab[c].notna().any()]].round(3).to_string())

    print("\nChange in the threshold since 1994 under each specification (tau_t - tau_1994), with ell_t")
    chg = fs.pivot(index="year", columns="spec", values="tau")
    chg = (chg - chg.loc[1994])[list(SPECS)]
    print(pd.concat([chg, ell_main.pc.rename("ell_pc")], axis=1).round(3).to_string())

    print("\nCorrelation of replicate deviations of tau_t between years, by lag (mean over pairs)")
    for name in ["s1", "s3", "s4"]:
        rs = reps_all[reps_all.spec == name].pivot(index="year", columns="rep", values="tau")
        dev = rs.sub(fs[fs.spec == name].set_index("year").tau, axis=0)
        cor = np.corrcoef(dev.to_numpy())
        lags = {k: np.mean([cor[i, i + k] for i in range(20 - k)]) for k in [1, 2, 3, 5, 10, 15]}
        print(f"  {name}: " + ", ".join(f"lag {k}: {v:+.2f}" for k, v in lags.items()))

    print("\nChecks without replicates: unweighted fits, and a heteroskedastic probit whose scale is "
          "exp(k1 ln(n_i / 4) + k2 young_i), n_i = interviews, young_i = reference person under 25 or student housing, "
          "|k| <= 2.5; scales shown relative to a unit with four interviews (one interview) and to other units (young)")
    for name in UNWEIGHTED:
        chk = results[name][2]
        print(f"\n[{name}] {SPECS[name]['label']}")
        print(chk.round(3).to_string())
        for c in ["tau_unweighted", "tau_hetprobit"]:
            dd = (chk[c] - chk[c].loc[1994]) - (chk.tau - chk.tau.loc[1994])
            print(f"  {c}: largest gap in tau_t - tau_1994 from the main fit {dd.abs().max():.3f}")

    s1 = fs[fs.spec == "s1"].set_index("year")
    a0, b0 = np.linalg.lstsq(np.column_stack([np.ones(20), s1.ell_pc]), s1.tau.to_numpy(), rcond=None)[0]
    sigma0 = float(np.median(s1.sigma))
    scen = [("A: price only", a0, b0, 0.0, sigma0, 20261109),
            ("B: half the price effect plus drift", a0, b0 / 2, b0 / 2 * s1.ell_pc.loc[2013] / 19, sigma0, 20261110)]
    print(f"\nFake data: same units and weights; own = 1 if (ln s - tau_t)/sigma + z > 0, sigma = {sigma0:.3f}, "
          f"z ~ N(0,1); tau_t = a + b ell_t + d (t - 1994) + e_t, e_t ~ N(0, 0.03^2); fitted with the step-1 "
          "specification")
    with Pool(2, initializer=worker_init) as pool:
        fakes = pool.map(do_fake, scen)
    for (sname, a, b, dd, _, _), (_, z) in zip(scen, fakes):
        print(f"  {sname}: a = {a:.3f}, b = {b:.3f}, d = {dd:+.4f}; max |tau_hat - tau_true| / se = {z:.2f}")
    pd.concat([f[0] for f in fakes]).reset_index().to_csv(f"{H.RES}/first_stage_fake.csv", index=False)
    print(f"\nWrote output/results/first_stage.csv, first_stage_replicates.csv.gz, and first_stage_fake.csv "
          f"({time.time() - t0:.0f}s)")

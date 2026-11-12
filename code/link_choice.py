"""Step 1, choice of link: which distribution of thresholds fits the CE data best, 1994-2013?

Each year's model is Pr(own_i = 1) = F(alpha_t + beta_t (ln s_i - 10) + gamma_g(i),t [+ z_i' delta_t]), with the
geography groups of hhpc.py, fitted by weighted (penalized) maximum likelihood, under three sets of controls:
geography only (step 1), geography and household traits (step 3), and those plus education (step 4); all use the
observed ln s. F is one of five links:
- probit (normal thresholds, the approved specification), logit (logistic), and robit with 7, 4, and 2 degrees of
  freedom (Student t thresholds, with heavier tails; Liu 2004). A heavier tail lets a few units own (or not) far
  from the threshold without bending the fit.
All five are symmetric, so tau_t = -(alpha_t + sum pi_g gamma_g,t) / beta_t is the 50% point under each.

Comparison: 10-fold cross-validated log score within each year (folds drawn within geography groups, seed 1994),
weighted by the survey weights, summed over years; the difference from the probit gets a standard error from the
spread of the unit-level differences (as for elpd differences; Vehtari, Gelman, and Gabry 2017). Rule, set before
looking: switch from the probit only if another link beats it by more than four standard errors in total, and then
take the best link. Also binned residuals (Gelman and Hill 2007) against within-year spending percentiles.

Run from the project folder: python3 code/link_choice.py > output/logs/link_choice.txt
Writes output/results/link_choice.csv (thresholds by link), link_choice_cv.csv (the comparison), and
output/diagnostics/link_choice.pdf.
"""
import os
from multiprocessing import Pool

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")   # one thread per worker: the matrices are small
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import hhpc as H  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

LINK_LIST = ["probit", "logit", "robit7", "robit4", "robit2"]
K = 10
INK, ACCENT, ACCENT2, GRID = "#222222", "#0072B2", "#D55E00", "#DDDDDD"
pd.set_option("display.width", 200)

CONTROLS = {"step 1: geography": [], "step 3: + traits": H.TRAITS, "step 4: + traits, education": H.TRAITS + H.EDUC}


def load():
    global u
    u = H.add_traits(H.load_units())
    shuffled = u.sample(frac=1, random_state=1994)
    u["fold"] = (shuffled.groupby(["year", "geo"]).cumcount() % K).reindex(u.index)


def one(job):
    """Full-sample fit and 10-fold cross-validated log score for one set of controls, link, and year."""
    name, k, t = job
    des = H.Design(u, k, geo=True, traits=CONTROLS[name])
    d = u[u.year == t]
    y, w = d.own.to_numpy(float), d.w.to_numpy()
    X, pen = des.matrix(d, d.ln_s)
    F = H.LINKS[k][0]
    full = H.fit_binary(y, X, w, k, pen)
    lp = np.empty(len(d))
    for f in range(K):
        test = d.fold.to_numpy() == f
        b = H.fit_binary(y[~test], X[~test], w[~test], k, pen, start=full)
        q = F(X[test] @ b).clip(1e-12, 1 - 1e-12)
        lp[test] = y[test] * np.log(q) + (1 - y[test]) * np.log1p(-q)
    return (name, k, t), {"controls": name, "year": t, "link": k, **des.read(full)}, y - F(X @ full), w * lp


if __name__ == "__main__":
    load()
    jobs = [(name, k, t) for name in CONTROLS for k in LINK_LIST for t in H.YEARS]
    jobs.sort(key=lambda j: (j[1] == "probit", j[1] == "logit"))     # the slow robit fits first
    with Pool(8, initializer=load) as pool:
        out = {r[0]: r[1:] for r in pool.imap_unordered(one, jobs)}
    points, resid, summary, cv = [], {}, {}, []
    for name in CONTROLS:
        rows = []
        for t in H.YEARS:
            for k in LINK_LIST:
                pt, r, wl = out[(name, k, t)]
                points.append(pt)
                resid[(name, t, k)] = r
                rows.append(pd.DataFrame({"year": t, "link": k, "wl": wl}))
        score = pd.concat(rows)
        by_year = score.groupby(["year", "link"]).wl.sum().unstack("link")[LINK_LIST]
        base = score[score.link == "probit"].wl.to_numpy()
        res = []
        for k in LINK_LIST:
            diff = score[score.link == k].wl.to_numpy() - base
            res.append({"link": k, "cv_log_score": by_year[k].sum(), "diff_vs_probit": diff.sum(),
                        "se_diff": np.sqrt(len(diff) * diff.var()),
                        "years_better_than_probit": int((by_year[k] > by_year.probit).sum())})
        res = pd.DataFrame(res).set_index("link")
        cv.append(res.assign(controls=name).reset_index())
        best = res.cv_log_score.idxmax()
        choice = best if res.loc[best, "diff_vs_probit"] > 4 * res.loc[best, "se_diff"] else "probit"
        summary[name] = (best, choice)
        print(f"\n=== Controls, {name}: cross-validated log score (10-fold within each year, weighted), summed over "
              "1994-2013; higher is better")
        print(res.round(1).to_string())
        print(f"Best link: {best}. Rule (switch only if it beats the probit by more than 4 standard errors): {choice}.")
        print("By year, difference from the probit")
        print(by_year.sub(by_year.probit, axis=0).round(1).to_string())

    point = pd.DataFrame(points)
    point.to_csv(f"{H.RES}/link_choice.csv", index=False)
    pd.concat(cv).to_csv(f"{H.RES}/link_choice_cv.csv", index=False)
    for name in CONTROLS:
        p = point[point.controls == name]
        wide = p.pivot(index="year", columns="link", values="tau")[LINK_LIST]
        chg = wide - wide.loc[1994]
        print(f"\n=== Controls, {name}: threshold tau_t by link (log $ a year), and its change from 1994")
        print(pd.concat([wide.round(3), chg.add_suffix("_chg").round(3)], axis=1).to_string())
        print(f"Largest gap between links in tau_t - tau_1994: {(chg.max(axis=1) - chg.min(axis=1)).max():.3f}")

    with PdfPages("output/diagnostics/link_choice.pdf") as pdf:
        edges = np.r_[0, 0.005, 0.01, 0.02, 0.05, np.arange(0.1, 1.0, 0.1), 0.95, 0.98, 0.99, 0.995, 1.0]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
        for ax, name in zip(axes, CONTROLS):
            for k, c in zip(["probit", "logit", "robit2"], [INK, ACCENT, ACCENT2]):
                r = pd.concat([pd.DataFrame({"r": resid[(name, t, k)], "w": u.w[u.year == t].to_numpy(),
                                             "pct": u.ln_s[u.year == t].rank(pct=True).to_numpy()}) for t in H.YEARS])
                r["bin"] = pd.cut(r.pct, edges, include_lowest=True)
                g = r.groupby("bin", observed=True).apply(lambda b: pd.Series({"x": b.pct.mean(), "m": H.wmean(b.r, b.w)}),
                                                          include_groups=False)
                ax.plot(100 * g.x, 100 * g.m, "o-", ms=2.5, lw=0.9, color=c, label=k)
            ax.axhline(0, color="#888888", lw=0.6)
            ax.set_xlabel("within-year percentile of long-run spending")
            ax.set_ylabel("mean residual, own - fitted (points)")
            ax.set_title(f"Binned residuals, all years: {name}")
            ax.grid(color=GRID, lw=0.4)
        axes[0].legend(frameon=False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print("\nSummary (best link, choice under the rule):", summary)
    print("Wrote output/results/link_choice.csv, link_choice_cv.csv, and output/diagnostics/link_choice.pdf")

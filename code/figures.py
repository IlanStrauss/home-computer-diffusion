"""Figures for the results write-up (writeup/household_pc_results.tex), from the saved estimates.

1. ownership.pdf: ownership by year (CPS and CE) and raw ownership by long-run spending, no model.
2. prices.pdf: the log computer price under each concept.
3. thresholds.pdf: yearly spending thresholds in the baseline and extended regressions, with the fitted second stage and the
   pure-affordability path. thresholds_all.pdf (appendix) adds the two intermediate fits.
4. b_and_d.pdf: b and d in every specification (main price).
5. cps_ce.pdf: CPS and CE spending thresholds, 1984-2013, and the imputation check on CE units.
6. switch.pdf: the unexplained trend before and after the switch (CPS extension).
7. joint.pdf: spending thresholds from the joint model against the two-stage estimates.
8. income_price.pdf: where the spending threshold sits, and who a price fall reaches (code/income_price.py).
9. price_effect.pdf: b and d of the baseline and extended regressions in percentages: the fall in the spending threshold from a
   10% fall in the price, and the unexplained trend a year (every specification is in b_and_d.pdf).
10. paths.pdf: ownership, 1994-2012, when only the real price or only real spending moves from its 1994 value
   (code/income_price.py, part G).
11. fifths.pdf: raw ownership by income fifth (the CE's income rank, complete income reporters), and the gap
   between the richest and poorest fifths.
12. education.pdf: the spending threshold of each education group of the extended regression, relative to households headed by a
   high-school graduate, exp(-theta), by year.
13. ai_pc.pdf: home computers against generative AI, by years since each technology's first mass-market product,
   and use at work by education. The generative AI figures and the 1984 PC-use figures are published numbers (sources
   in the function); the household computer ownership rates are ours (CPS supplements and CE).
A figure whose inputs are missing is skipped with a message.

Run from the project folder: python3 code/figures.py
Writes writeup/figures/*.pdf.
"""
import os

import matplotlib
import numpy as np
import pandas as pd

import hhpc as H

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = "writeup/figures"
# Okabe-Ito
BLACK, ORANGE, SKY, GREEN, BLUE, VERMILLION, PURPLE = ("#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2",
                                                       "#D55E00", "#CC79A7")
INK, GRID, MUTED = "#222222", "#E3E3E3", "#777777"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.edgecolor": "#888888", "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.color": "#444444",
                     "ytick.color": "#444444", "axes.labelcolor": INK, "axes.titlesize": 8.5,
                     "axes.titleweight": "normal",
                     "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.2, "pdf.fonttype": 42})
Z90 = 1.645
SWITCH = {"20%": 1992.44, "30%": 1995.96}


def grid(ax):
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def have(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"  skipped: missing {', '.join(missing)}")
    return not missing


def fig1():
    print("ownership.pdf")
    if not have("output/results/first_stage.csv", "data/processed/cps_households.csv.gz"):
        return
    fs = pd.read_csv("output/results/first_stage.csv")
    ce = fs[fs.spec == "s1"].set_index("year").own_pct
    c = pd.read_csv("data/processed/cps_households.csv.gz")
    c = c[c.owns.notna() & (c.hh_weight > 0)]
    cps = c.groupby("year").apply(lambda d: 100 * H.wmean(d.owns, d.hh_weight), include_groups=False)
    months = {1984: 10, 1989: 10, 1993: 10, 1997: 10, 1998: 12, 2000: 8, 2001: 9, 2003: 10}
    cps.index = [t + (months[t] - 0.5) / 12 for t in cps.index]
    u = H.load_units()

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axes[0]
    ax.plot(cps.index, cps, "^-", color=VERMILLION, ms=4, lw=1)
    ax.plot(ce.index + 0.5, ce, "o-", color=BLACK, ms=3, lw=1)
    ax.text(1984.6, 46, "CPS (households,\nsurvey month)", color=VERMILLION, fontsize=6.8, va="center")
    ax.text(2004.0, 44, "CE (consumer units,\nyear average)", color=BLACK, fontsize=6.8, va="center")
    for lev, (name, t), ha, dx in zip([20, 30], SWITCH.items(), ["right", "left"], [-0.3, 0.3]):
        ax.axhline(lev, color=MUTED, lw=0.6, ls="--")
        ax.axvline(t, color=MUTED, lw=0.6, ls=":")
        ax.text(t + dx, 84, f"{name} in {t:.1f}", color=MUTED, fontsize=7, ha=ha)
    ax.set_xlabel("year")
    ax.set_ylabel("own a computer (%)")
    ax.set_ylim(0, 90)
    ax.set_title("(a) Ownership, 1984–2013")
    grid(ax)

    ax = axes[1]
    years = [1994, 1998, 2003, 2008, 2012]
    blues = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]
    for t, col in zip(years, blues):
        d = u[u.year == t].sort_values("ln_s")
        cw = d.w.cumsum() / d.w.sum()
        d = d.assign(bin=np.minimum((cw * 25).astype(int), 24))
        g = d.groupby("bin").apply(lambda b: pd.Series({"x": H.wmean(b.ln_s, b.w), "own": 100 * H.wmean(b.own, b.w)}),
                                   include_groups=False)
        ax.plot(g.x, g.own, "o-", color=col, ms=2.2, lw=1, label=str(t))
    ax.set_xlabel("log long-run spending (25 equal-weight bins)")
    ax.set_ylabel("own a computer (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(b) Ownership by spending, CE, no model")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=5, handlelength=1.4, columnspacing=0.9, fontsize=7)
    grid(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/ownership.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2():
    print("prices.pdf")
    if not have("data/processed/prices_monthly.csv", "output/results/first_stage.csv"):
        return
    mp = pd.read_csv("data/processed/prices_monthly.csv")
    mp["t"] = mp.year + (mp.month - 0.5) / 12
    mp = mp[(mp.year >= 1984) & (mp.year <= 2013)]
    base = mp[mp.year == 1994]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    series = [("pc", "BEA PCE: personal computers (main)", BLACK, "-"),
              ("info", "BEA PCE: information processing equipment", BLUE, "-"),
              ("business", "BEA investment: computers", SKY, "-"),
              ("cpi_pc", "BLS CPI-U: personal computers (from Dec 1997)", GREEN, "--"),
              ("paid_p50", "CE price paid: median (from 1994)", ORANGE, "-"),
              ("paid_p90", "CE price paid: 90th percentile (from 1994)", VERMILLION, "-")]
    for k, lab, col, ls in series:
        y = mp[f"ln_{k}"]
        if k == "cpi_pc":
            anchor = mp.loc[mp.year == 1999, "ln_pc"].mean() - base.ln_pc.mean()
            y = y - mp.loc[mp.year == 1999, "ln_cpi_pc"].mean() + anchor
        else:
            y = y - base[f"ln_{k}"].mean()
        if k.startswith("paid"):
            y = y.where(mp.year >= 1994)    # late-1993 months only carry the 1994 value
        ax.plot(mp.t, y, color=col, ls=ls, lw=1.1, label=lab)
    for name, t in SWITCH.items():
        ax.axvline(t, color=MUTED, lw=0.6, ls=":")
        ax.text(t + 0.3, 1.9, name, color=MUTED, fontsize=7)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("year")
    ax.set_ylabel(r"log price relative to the 1994 average, $\ell_t$")
    ax.set_title("Computer prices under each concept (monthly)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.8)
    grid(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/prices.pdf", bbox_inches="tight")
    plt.close(fig)


def fig3():
    print("thresholds.pdf, thresholds_all.pdf")
    if not have("output/results/second_stage_fitted.csv", "output/results/second_stage.csv"):
        return
    f = pd.read_csv("output/results/second_stage_fitted.csv")
    titles = {"s1": "Baseline: observed spending, geography", "s2": "+ spending corrected for noise",
              "s3": "+ household traits", "s4": "+ education: extended regression"}
    # the main text shows the baseline and extended regressions; the appendix adds the two intermediate fits
    for name, specs, shape, size in [("thresholds", ["s1", "s4"], (1, 2), (6.6, 2.9)),
                                     ("thresholds_all", ["s1", "s2", "s3", "s4"], (2, 2), (6.6, 5.0))]:
        fig, axes = plt.subplots(*shape, figsize=size, sharex=True, squeeze=False)
        for ax, s in zip(axes.flat, specs):
            d = f[f.spec == s]
            ax.fill_between(d.year, d.pred_lo, d.pred_hi, color="#EDEDED", lw=0, label="posterior predictive, 90%")
            ax.fill_between(d.year, d.fit_lo, d.fit_hi, color=BLUE, alpha=0.22, lw=0)
            ax.plot(d.year, d.fit, color=BLUE, lw=1.3, label=r"fitted $a + b\,\ell_t + d\,(t-1994)$, 90%")
            ax.plot(d.year, d.fit.iloc[0] + d.ell, color=VERMILLION, ls="--", lw=1.1,
                    label=r"pure affordability: 1994 fit $+\ \ell_t$")
            ax.errorbar(d.year, d.tau, yerr=Z90 * d.se_tau, fmt="o", color=BLACK, ms=2.6, lw=0.8, capsize=0,
                        label=r"estimated $\tau_t$, 90%")
            ax.set_title(titles[s])
            ax.set_ylabel(r"spending threshold $\tau_t$ (log \$ a year)")
            lo = min(d.tau.min(), d.pred_lo.min()) - 0.3
            ax.set_ylim(min(lo, (d.fit.iloc[0] + d.ell).min() - 0.1), max(d.tau.max(), d.pred_hi.max()) + 0.3)
            grid(ax)
        for ax in axes[-1]:
            ax.set_xlabel("year")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=7)
        fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
        plt.close(fig)


def fig4():
    print("b_and_d.pdf")
    if not have("output/results/second_stage.csv"):
        return
    r = pd.read_csv("output/results/second_stage.csv")
    r = r[r.concept == "pc"].copy()
    names = {"s1": "Baseline", "s2": "+ corrected spending", "s3": "+ household traits", "s4": "+ education (extended)",
             "s1_nogeo": "Baseline, no geography", "s1_logit": "Baseline, logit", "s1_robit2": "Baseline, robit (2 df)",
             "s1_int2": "Baseline, entered at interview 2",
             "s1_nostudent": "Baseline, no student housing", "s1_trim": "Baseline, trimmed 0.5%",
             "s1_incl": "Baseline, + insurance, pensions, contributions", "s1_veh": "Baseline, + vehicles",
             "s3_obs": "+ household traits, observed spending", "s3_robit7": "+ household traits, robit (7 df)",
             "s4_logit": "Extended, logit",
             "s4_int2": "Extended, entered at interview 2", "s4_incl": "Extended, + insurance, pensions, contributions",
             "s4_veh": "Extended, + vehicles", "s4_cr": "Extended, complete income reporters",
             "s4_inc": "Extended, current income", "s4_cf": "Extended, control function for income"}
    r["label"] = r.spec.map(names) + np.where(r.model == "correlated errors", ", correlated errors", "")
    order = ["s1", "s2", "s3", "s4"]
    main = r[r.spec.isin(order) & (r.model == "independent errors")].set_index("spec").loc[order].reset_index()
    corr = r[r.model == "correlated errors"]
    checks = r[~r.spec.isin(order) & (r.model == "independent errors")]
    r = pd.concat([main, corr, checks])
    y = np.arange(len(r))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.6), sharey=True)
    for ax, p, ref, lab in [(axes[0], "b", [0, 1], r"$b$ (1 under pure affordability)"),
                            (axes[1], "d", [0], r"$d$ (yearly change in logs)")]:
        col = np.where(r.spec.isin(order), BLACK, MUTED)
        ax.hlines(y, r[f"{p}_lo"], r[f"{p}_hi"], color=col, lw=1)
        ax.scatter(r[p], y, color=col, s=12, zorder=3)
        for v in ref:
            ax.axvline(v, color=VERMILLION if (p == "b" and v == 1) else MUTED, lw=1.0 if v == 1 else 0.8,
                       ls=":" if v == 1 else "--")
        ax.set_xlabel(lab)
        grid(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(r.label, fontsize=6.8)    # shared with axes[1], whose labels sharey hides
    axes[0].set_ylabel("specification (main price)")
    axes[0].set_title("(a) Price effect $b$")
    axes[1].set_title("(b) Unexplained trend $d$")
    fig.tight_layout()
    fig.savefig(f"{FIG}/b_and_d.pdf")
    plt.close(fig)


def fig5():
    print("cps_ce.pdf")
    if not have("output/results/cps_thresholds.csv", "output/results/cps_validation.csv"):
        return
    d = pd.read_csv("output/results/cps_thresholds.csv")
    v = pd.read_csv("output/results/cps_validation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axes[0]
    for src, col, mk in [("CPS", VERMILLION, "^"), ("CE", BLACK, "o")]:
        k = d[d.source == src]
        ax.errorbar(k.date, k.tau, yerr=Z90 * k.se_tau, fmt=mk, color=col, ms=3.2, lw=0.8, capsize=0,
                    label=f"{src}, 90%")
    for name, t in SWITCH.items():
        ax.axvline(t, color=MUTED, lw=0.6, ls=":")
        ax.text(t + 0.3, d.tau.min() - 0.05, name, color=MUTED, fontsize=7)
    ax.set_xlabel("year")
    ax.set_ylabel(r"spending threshold $\tau$ (log \$ a year)")
    ax.set_title("(a) Spending thresholds, CPS specification")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    grid(ax)
    ax = axes[1]
    ax.plot(v.year, v.ce_units_corrected_tau, "o-", color=BLACK, ms=3, label="CE units, corrected spending")
    ax.plot(v.year, v.ce_units_imputed_tau, "s--", color=BLUE, ms=3, label="CE units, spending imputed from income")
    ax.plot(v.year, v.cps_tau, "^-", color=VERMILLION, ms=3.2, label="CPS households, imputed")
    ax.set_xlabel("year")
    ax.set_ylabel(r"spending threshold $\tau$ (log \$ a year)")
    ax.set_title("(b) Imputation check, 1997–2003")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=6.8)
    grid(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/cps_ce.pdf", bbox_inches="tight")
    plt.close(fig)


def fig6():
    print("switch.pdf")
    if not have("output/results/cps_switch.csv"):
        return
    r = pd.read_csv("output/results/cps_switch.csv")
    r = r[r.concept.isin(["pc", "pc, CPS alone"])].reset_index(drop=True)
    lab = {"one": "one trend", "20": "switch at 20%", "30": "switch at 30%"}
    r["label"] = np.where(r.concept == "pc", "CPS + CE: ", "CPS alone: ") + r.model.astype(str).map(lab)
    y = np.arange(len(r))[::-1]
    one, sw = (r.model.astype(str) == "one").to_numpy(), (r.model.astype(str) != "one").to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    ax.hlines(y[one], r.d1_lo[one], r.d1_hi[one], color=BLACK, lw=1.1)
    ax.scatter(r.d1[one], y[one], color=BLACK, s=13, zorder=3, label=r"one trend for all years, $d$")
    ax.hlines(y[sw] + 0.13, r.d1_lo[sw], r.d1_hi[sw], color=ORANGE, lw=1.1)
    ax.scatter(r.d1[sw], y[sw] + 0.13, color=ORANGE, s=13, zorder=3, label=r"before the switch, $d_1$")
    ax.hlines(y[sw] - 0.13, r.d2_lo[sw], r.d2_hi[sw], color=BLUE, lw=1.1)
    ax.scatter(r.d2[sw], y[sw] - 0.13, color=BLUE, s=13, zorder=3, label=r"after the switch, $d_2$")
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(r.label, fontsize=7)
    ax.set_ylabel("model")
    ax.set_xlabel("unexplained trend (yearly change in logs), posterior median and 90% interval")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3)
    grid(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/switch.pdf", bbox_inches="tight")
    plt.close(fig)


def fig7():
    print("joint.pdf")
    if not have("output/results/joint_model.csv", "output/results/first_stage.csv"):
        return
    j = pd.read_csv("output/results/joint_model.csv")
    fs = pd.read_csv("output/results/first_stage.csv")
    fs = fs[fs.spec == "s4"].sort_values("year")
    tj = j[j.variable.str.startswith("tau[")]    # tau[1] is 1994
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.errorbar(fs.year - 0.15, fs.tau, yerr=Z90 * fs.se_tau, fmt="o", color=BLACK, ms=2.8, lw=0.8, capsize=0,
                label="two-stage: extended regression, 90%")
    ax.errorbar(fs.year.to_numpy() + 0.15, tj["median"], yerr=[tj["median"] - tj.q05, tj.q95 - tj["median"]], fmt="s",
                color=BLUE, ms=2.8, lw=0.8, capsize=0, label="joint model, 90%")
    ax.set_xticks([1995, 2000, 2005, 2010])
    ax.set_xlabel("year")
    ax.set_ylabel(r"spending threshold $\tau_t$ (log \$ a year)")
    ax.set_title("Joint model against the two-stage spending thresholds")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    grid(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/joint.pdf", bbox_inches="tight")
    plt.close(fig)


def fig8():
    print("income_price.pdf")
    if not have("output/results/income_price_years.csv", "output/results/income_price_percentiles.csv"):
        return
    y = pd.read_csv("output/results/income_price_years.csv")
    p = pd.read_csv("output/results/income_price_percentiles.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5))
    ax = axes[0]
    ax.axhline(50, color=MUTED, ls="--", lw=0.8)
    ax.text(2012.3, 52, "median", ha="right", va="bottom", color=MUTED, fontsize=6.5)
    ax.plot(y.year, y.pct_below_tau, "o-", color=BLACK, ms=2.4, lw=1.0)
    ax.set_ylim(0, 100)
    ax.set_xticks([1995, 2000, 2005, 2010])
    ax.set_xlabel("year")
    ax.set_ylabel("households spending less than\nthe spending threshold (%)")
    ax.set_title("(a) The spending threshold", loc="left")
    grid(ax)
    ax = axes[1]
    for yr, c, (dx, dy) in [(1994, VERMILLION, (-4, 0.06)), (2003, ORANGE, (4, 0.07)), (2012, BLUE, (6, 0.0))]:
        d = p[p.year == yr]
        ax.plot(d.pct, d.dpr_price10, color=c)
        k = d.dpr_price10.idxmax()
        ax.text(d.pct[k] + dx, d.dpr_price10[k] + dy, str(yr), color=INK, fontsize=6.5,
                ha="right" if dx < 0 else "left", va="bottom")
    ax.set_ylim(0, 2.25)
    ax.set_xlabel("household's percentile of long-run spending")
    ax.set_ylabel("rise in probability of owning\n(percentage points)")
    ax.set_title("(b) Who a 10% price fall helps", loc="left")
    grid(ax)
    ax = axes[2]
    ax.fill_between(y.year, y.dP_price10_lo, y.dP_price10_hi, color=BLACK, alpha=0.12, lw=0)
    ax.plot(y.year, y.dP_spend10, color=MUTED, ls="--")
    ax.plot(y.year, y.dP_price10, color=BLACK)
    ax.text(2012.6, y.dP_spend10.iloc[-1], "spending\nup 10%", color=MUTED, fontsize=6.5, va="center")
    ax.text(2012.6, y.dP_price10.iloc[-1], "price\ndown 10%", color=BLACK, fontsize=6.5, va="center")
    ax.set_ylim(0, 3.0)
    ax.set_xlim(1993.5, 2018.5)
    ax.set_xticks([1995, 2003, 2011])
    ax.set_xlabel("year")
    ax.set_ylabel("rise in ownership\n(percentage points)")
    ax.set_title("(c) All households", loc="left")
    grid(ax)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{FIG}/income_price.pdf")
    plt.close(fig)


def fig9():
    print("price_effect.pdf")
    if not have("output/results/second_stage.csv"):
        return
    r = pd.read_csv("output/results/second_stage.csv")
    r = r[(r.concept == "pc") & (r.model == "independent errors")].set_index("spec")
    cols = ["b", "b_lo", "b_hi", "d", "d_lo", "d_hi"]
    rows = [("Baseline regression", r.loc["s1", cols].to_numpy(float)),
            ("Extended regression", r.loc["s4", cols].to_numpy(float))]    # every specification: b_and_d.pdf
    fall = lambda b: 100 * (1 - 0.9 ** b)    # % fall in the spending threshold when the price falls 10%
    trend = lambda d: 100 * np.expm1(d)      # % change in the spending threshold a year
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.75), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    for (lab, v), yy in zip(rows, y):
        axes[0].hlines(yy, fall(v[1]), fall(v[2]), color=BLACK, lw=1.2)
        axes[0].scatter(fall(v[0]), yy, color=BLACK, s=14, zorder=3)
        axes[0].text(fall(v[2]) + 0.25, yy, f"{fall(v[0]):.1f}%", va="center", fontsize=6.8, color=INK)
        axes[1].hlines(yy, trend(v[4]), trend(v[5]), color=BLACK, lw=1.2)
        axes[1].scatter(trend(v[3]), yy, color=BLACK, s=14, zorder=3)
    axes[0].axvline(10, color=VERMILLION, lw=1.1, ls=":")
    axes[0].text(9.85, y[0] + 0.45, "pure affordability: 10%", color=VERMILLION, fontsize=6.8, ha="right", va="center")
    axes[0].set_xlim(0, 11)
    axes[0].set_ylim(-0.6, len(rows) - 0.2)
    axes[0].set_xlabel("fall in the spending threshold when\nthe quality-adjusted computer\nprice falls 10% (%)")
    axes[0].set_title("(a) Price effect", loc="left")
    axes[1].axvline(0, color=VERMILLION, lw=1.1, ls=":")
    axes[1].text(-0.25, y[0] + 0.45, "pure affordability: 0%", color=VERMILLION, fontsize=6.8, ha="right", va="center")
    axes[1].set_xlim(-5, 5)
    axes[1].set_xlabel("unexplained trend in\nthe spending threshold\n(% a year)")
    axes[1].set_title("(b) Unexplained trend", loc="left")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for lab, _ in rows], fontsize=7)
    for ax in axes:
        grid(ax)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(f"{FIG}/price_effect.pdf")
    plt.close(fig)


def fig10():
    print("paths.pdf")
    if not have("output/results/income_price_paths.csv"):
        return
    p = pd.read_csv("output/results/income_price_paths.csv")
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.plot(p.year, p.own, "o", color=BLACK, ms=3.2, zorder=3, label="actual computer ownership rate (CE)")
    ax.plot(p.year, p.price_only, color=BLUE, lw=1.6, label="computer price falls as it did; spending stays at 1994 level")
    ax.plot(p.year, p.spend_only, color=ORANGE, lw=1.6, label="spending rises as it did; computer price stays at 1994 level")
    ax.plot(p.year, p.both, color=MUTED, lw=1.1, ls="--", label="both change as they did")
    last = p.iloc[-1]
    ax.annotate(f"+{last.price_only - last.start:.0f} points", (2012, last.price_only), xytext=(2012.35, last.price_only),
                color=BLUE, fontsize=7, va="center")
    ax.annotate(f"+{last.spend_only - last.start:.0f} points", (2012, last.spend_only), xytext=(2012.35, last.spend_only),
                color=ORANGE, fontsize=7, va="center")
    ax.axhline(p.start.iloc[0], color=MUTED, lw=0.6, ls=":")
    ax.text(2009, p.start.iloc[0] - 4.5, f"1994 level: {p.start.iloc[0]:.0f}%", color=MUTED, fontsize=7)
    ax.set_xlim(1993.5, 2014)
    ax.set_xticks([1994, 1998, 2002, 2006, 2010, 2012])
    ax.set_ylim(0, 100)
    ax.set_xlabel("year")
    ax.set_ylabel("households that own a computer (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7)
    grid(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/paths.pdf", bbox_inches="tight")
    plt.close(fig)


def fig11():
    print("fifths.pdf")
    if not have("data/processed/ce_households.csv.gz"):
        return
    h = pd.read_csv("data/processed/ce_households.csv.gz",
                    usecols=["year", "inventory", "pc", "weight", "inc_rank", "income_complete"])
    h = h[h.inventory & h.income_complete].copy()
    h["fifth"] = pd.cut(h.inc_rank, [0, .2, .4, .6, .8, 1.0001], labels=[1, 2, 3, 4, 5], right=False)
    o = (100 * h.groupby(["year", "fifth"], observed=True).apply(lambda g: H.wmean(g.pc, g.weight), include_groups=False)
         .unstack()).reindex(range(1994, 2014))   # no income rank in the 2004 and 2005 files
    gap = o[5] - o[1]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    style = {5: (BLUE, "richest fifth", 1.6), 4: ("#7A7A7A", "fourth fifth", 1.0), 3: ("#9C9C9C", "middle fifth", 1.0),
             2: ("#BDBDBD", "second fifth", 1.0), 1: (VERMILLION, "poorest fifth", 1.6)}
    for k in [5, 4, 3, 2, 1]:
        col, lab, lw = style[k]
        ax.plot(o.index, o[k], "o-", color=col, lw=lw, ms=2.2, label=lab)
    ax.set_ylim(0, 100)
    ax.set_xticks([1994, 1998, 2002, 2006, 2010])
    ax.set_xlabel("year")
    ax.set_ylabel("households that own a computer (%)")
    ax.set_title("(a) By fifth of household income", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    grid(ax)
    ax = axes[1]
    ax.plot(gap.index, gap, "o-", color=BLACK, ms=2.6, lw=1.1)
    for yr, off in [(1994, (4, -11)), (2003, (0, 6)), (2012, (0, -11))]:
        ax.annotate(f"{gap[yr]:.1f}", (yr, gap[yr]), xytext=off, textcoords="offset points", ha="center",
                    fontsize=7, color=INK)
    ax.set_ylim(0, 65)
    ax.set_xticks([1994, 1998, 2002, 2006, 2010])
    ax.set_xlabel("year")
    ax.set_ylabel("richest fifth minus poorest fifth\n(percentage points)")
    ax.set_title("(b) Richest minus poorest fifth", loc="left")
    grid(ax)
    fig.tight_layout(w_pad=1.2)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=7)
    fig.savefig(f"{FIG}/fifths.pdf", bbox_inches="tight")
    plt.close(fig)


def fig12():
    print("education.pdf")
    if not have("output/results/first_stage.csv"):
        return
    f = pd.read_csv("output/results/first_stage.csv")
    f = f[(f.spec == "s4") & (f.year <= 2012)].sort_values("year")
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.axhline(1, color=MUTED, lw=1)
    ax.text(2012.4, 1.03, "high school = 1", color=MUTED, fontsize=7, va="bottom")
    for k, col, lab in [("lt_hs", VERMILLION, "less than high school"), ("some_col", SKY, "some college"),
                        ("col", BLUE, "college degree")]:
        th, se = f[f"theta_ed_{k}"], f[f"se_theta_ed_{k}"]
        ax.fill_between(f.year, np.exp(-th - Z90 * se), np.exp(-th + Z90 * se), color=col, alpha=0.18, lw=0)
        ax.plot(f.year, np.exp(-th), "o-", color=col, ms=2.6, lw=1.3)
        ax.text(2012.4, np.exp(-th.iloc[-1]), lab, color=col, fontsize=7, va="center")
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.3, 0.5, 1, 1.5, 2, 3])
    ax.set_yticklabels(["0.2", "0.3", "0.5", "1", "1.5", "2", "3"])
    ax.minorticks_off()
    ax.set_xlim(1993.5, 2016.5)
    ax.set_xticks([1994, 1998, 2002, 2006, 2010, 2012])
    ax.set_xlabel("year")
    ax.set_ylabel("spending threshold relative to\nhigh-school households (log scale)")
    grid(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/education.pdf")
    plt.close(fig)


def fig13():
    print("ai_pc.pdf")
    if not have("data/processed/cps_households.csv.gz", "output/results/income_price_paths.csv"):
        return
    # Published figures (Real-Time Population Survey and CPS, adults aged 18-64):
    # - Bick, Blandin, and Deming, "The State of Generative AI Adoption in 2025", Federal Reserve Bank of St. Louis,
    #   On the Economy, 13 November 2025: generative AI use 44.6% in August 2024 (after the authors' revision for
    #   question order) and 54.6% in August 2025; personal computer use 19.7% in 1984.
    # - Bick, Blandin, and Deming, NBER Working Paper 32966 (revised February 2025), Figure 5a: use at work among
    #   employed adults, generative AI (August and November 2024) and computers (CPS, October 1984).
    # Years count from the IBM PC (August 1981) and ChatGPT (30 November 2022), the launch dates those authors use.
    pc0, ai0 = 1981 + 7.5 / 12, 2022 + 11 / 12
    genai = pd.Series({2024 + 7.5 / 12 - ai0: 44.6, 2025 + 7.5 / 12 - ai0: 54.6})
    work = pd.DataFrame({"pc_1984": [20.2, 42.5, 42.3], "ai_2024": [18.1, 36.2, 41.1]},
                        index=["less than a\nbachelor's degree", "bachelor's degree", "graduate degree"])

    c = pd.read_csv("data/processed/cps_households.csv.gz")
    c = c[c.owns.notna() & (c.hh_weight > 0)]
    cps = c.groupby("year").apply(lambda d: 100 * H.wmean(d.owns, d.hh_weight), include_groups=False)
    months = {1984: 10, 1989: 10, 1993: 10, 1997: 10, 1998: 12, 2000: 8, 2001: 9, 2003: 10}
    cps.index = [t + (months[t] - 0.5) / 12 - pc0 for t in cps.index]
    ce = pd.read_csv("output/results/income_price_paths.csv").set_index("year").own
    ce.index = ce.index + 0.5 - pc0

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.plot(cps.index, cps, "^-", color=VERMILLION, ms=3.5, lw=1)
    ax.plot(ce.index, ce, "o-", color=BLACK, ms=2.6, lw=1)
    ax.plot([1984 + 9.5 / 12 - pc0], [19.7], "s", color=ORANGE, ms=5)
    ax.plot(genai.index, genai, "D-", color=BLUE, ms=4.5, lw=1.4)
    ax.text(31, 88, "households that own a computer, CE", color=BLACK, fontsize=6.8, ha="right", va="center")
    ax.text(9.5, 8, "households that own a computer, CPS", color=VERMILLION, fontsize=6.8, ha="left", va="center")
    ax.text(0.6, 27, "adults who used a\ncomputer, 1984", color=ORANGE, fontsize=6.8, ha="left", va="center")
    ax.text(3.8, 47, "adults who used\ngenerative AI", color=BLUE, fontsize=6.8, ha="left", va="center")
    ax.set_xlim(0, 36)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_ylim(0, 100)
    ax.set_xlabel("years since the first mass-market product\n(IBM PC, August 1981; ChatGPT, November 2022)")
    ax.set_ylabel("share (%)")
    ax.set_title("(a) Speed of adoption", loc="left")
    grid(ax)

    ax = axes[1]
    y = np.arange(len(work))[::-1]
    for yy, (_, r) in zip(y, work.iterrows()):
        ax.hlines(yy, min(r.pc_1984, r.ai_2024), max(r.pc_1984, r.ai_2024), color=GRID, lw=2.5, zorder=1)
    ax.scatter(work.pc_1984, y, marker="s", color=ORANGE, s=22, zorder=3, label="computer at work, 1984")
    ax.scatter(work.ai_2024, y, marker="D", color=BLUE, s=20, zorder=3, label="generative AI at work, 2024")
    ax.set_yticks(y)
    ax.set_yticklabels(work.index, fontsize=7)
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.7, len(work) - 0.3)
    ax.set_xlabel("employed adults who used it at work (%)")
    ax.set_title("(b) Use at work by education", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=1, fontsize=6.8)
    grid(ax)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(f"{FIG}/ai_pc.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13]:
        f()
    print(f"Wrote {FIG}/")

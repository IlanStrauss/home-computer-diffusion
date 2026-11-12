"""Step 1, before any fit: plots and outlier checks of the CE analysis sample, 1994-2013.

- the distribution of log long-run spending ln s by year, against a normal curve with the same mean and SD;
- scatterplots of ln s against log current income and against age, marking units in student housing;
- raw ownership by ln s (weighted shares in equal-count bins), no model;
- ownership by state against the state's mean ln s;
- outliers: units far out in ln s, who they are, and whether they own a computer;
- the log computer price under each concept.

Run from the project folder: python3 code/explore_data.py > output/logs/explore_data.txt
Writes output/diagnostics/explore_data.pdf.
"""
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

import hhpc as H

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

INK, ACCENT, ACCENT2, GRID = "#222222", "#0072B2", "#D55E00", "#DDDDDD"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#888888", "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.color": "#555555", "ytick.color": "#555555",
                     "axes.labelcolor": INK, "axes.titlesize": 8})
ABBR = {1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE", 11: "DC", 12: "FL", 13: "GA",
        15: "HI", 16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
        26: "MI", 27: "MN", 29: "MO", 31: "NE", 32: "NV", 33: "NH", 34: "NJ", 36: "NY", 37: "NC", 39: "OH", 40: "OK",
        41: "OR", 42: "PA", 44: "RI", 45: "SC", 47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV",
        55: "WI"}

u = H.load_units()
u["student"] = u.tenure == 6
u["z"] = u.groupby("year").ln_s.transform(lambda s: (s - s.mean()) / s.std())
print(f"Sample: {len(u):,} consumer units, CE 1994-2013")


def wmoments(x, w):
    m = H.wmean(x, w)
    c = x - m
    v = H.wmean(c ** 2, w)
    return m, np.sqrt(v), H.wmean(c ** 3, w) / v ** 1.5, H.wmean(c ** 4, w) / v ** 2 - 3


mom = pd.DataFrame([dict(zip(["mean", "sd", "skew", "excess_kurtosis"], wmoments(d.ln_s.to_numpy(), d.w.to_numpy())),
                         year=t, n=len(d), below_minus3sd=int((d.z < -3).sum()), below_minus4sd=int((d.z < -4).sum()),
                         above_3sd=int((d.z > 3).sum()), under_2000=int((d.s < 2000).sum()))
                    for t, d in u.groupby("year")]).set_index("year")
pd.set_option("display.width", 200)
print("\nLog long-run spending by year: weighted mean, SD, skewness, excess kurtosis; units beyond 3 and 4 SDs; "
      "units under $2,000 a year")
print(mom.round(3).to_string())

low = u[u.z < -3]
rest = u[u.z >= -3]
print(f"\nUnits more than 3 SDs below their year's mean ln s: {len(low):,} ({len(low) / len(u):.2%})")
for name, d in [("below -3 SD", low), ("everyone else", rest)]:
    print(f"  {name}: own {100 * H.wmean(d.own, d.w):.1f}%; one interview {100 * H.wmean(d.n_interviews == 1, d.w):.1f}%;"
          f" student housing {100 * H.wmean(d.student, d.w):.1f}%; age under 25 {100 * H.wmean(d.age < 25, d.w):.1f}%;"
          f" median income ${np.nanmedian(d.income[d.income_complete == 1]):,.0f} (complete reporters)")
cols = ["year", "s", "n_interviews", "age", "tenure", "fam_size", "educ4", "income", "own"]
print("\nThe 15 lowest and 10 highest long-run spenders (tenure 6 = student housing)")
print(u.nsmallest(15, "s")[cols].to_string(index=False))
print(u.nlargest(10, "s")[cols].to_string(index=False))

inc = u[(u.income_complete == 1) & (u.income > 0)]
r = inc.groupby("year").apply(lambda d: np.corrcoef(d.ln_s, np.log(d.income))[0, 1], include_groups=False)
print(f"\nCorrelation of ln s with log current income (complete reporters, positive income; none in 2004-05): "
      f"{r.min():.2f} to {r.max():.2f} by year")

with PdfPages("output/diagnostics/explore_data.pdf") as pdf:
    fig, axes = plt.subplots(4, 5, figsize=(11, 8.5), sharex=True)
    bins = np.linspace(5, 13.5, 60)
    for ax, t in zip(axes.flat, H.YEARS):
        d = u[u.year == t]
        m, sd = mom.loc[t, "mean"], mom.loc[t, "sd"]
        ax.hist(d.ln_s, bins=bins, weights=d.w / d.w.sum() / np.diff(bins)[0], color="#9DB9D5", lw=0)
        ax.plot(bins, norm.pdf(bins, m, sd), color=ACCENT2, lw=1)
        ax.set_title(f"{t}: skew {mom.loc[t, 'skew']:+.2f}, kurtosis {mom.loc[t, 'excess_kurtosis']:+.1f}")
        ax.set_ylabel("density")
        ax.grid(color=GRID, lw=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("log long-run spending")
    fig.suptitle("Log long-run spending by year (weighted histogram) against a normal curve with the same mean and SD",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)

    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for j, t in enumerate([1994, 2003, 2012]):
        d = inc[inc.year == t]
        d = d.iloc[rng.permutation(len(d))[:3000]]
        for ax, xv, xl in [(axes[0, j], np.log(d.income), "log current income (complete reporters)"),
                           (axes[1, j], d.age, "age of the reference person")]:
            ax.scatter(xv[~d.student], d.ln_s[~d.student], s=3, color="#7F7F7F", alpha=0.35, lw=0, label="others")
            ax.scatter(xv[d.student], d.ln_s[d.student], s=9, color=ACCENT2, lw=0, label="student housing")
            ax.set_xlabel(xl)
            ax.set_ylabel("log long-run spending")
            ax.set_title(f"{t} (3,000 random units)")
            ax.grid(color=GRID, lw=0.4)
    axes[0, 0].legend(frameon=False, markerscale=2)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    shades = plt.cm.Blues(np.linspace(0.4, 1.0, 5))
    for t, c in zip([1994, 1998, 2003, 2008, 2012], shades):
        d = u[u.year == t]
        q = pd.qcut(d.ln_s, 25, labels=False)
        b = d.assign(q=q).groupby("q").apply(lambda g: pd.Series({"x": H.wmean(g.ln_s, g.w), "own": H.wmean(g.own, g.w)}),
                                             include_groups=False)
        axes[0].plot(b.x, 100 * b.own, "o-", ms=2.5, lw=0.9, color=c, label=str(t))
    axes[0].set_xlabel("log long-run spending (25 equal-count bins)")
    axes[0].set_ylabel("own a computer (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, title="year")
    axes[0].set_title("Raw ownership by spending, no model")
    for t, c in zip([1994, 2003, 2012], [shades[0], shades[2], shades[4]]):
        d = u[u.year == t]
        b = d.assign(a=pd.cut(d.age, [0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 100])).groupby("a", observed=True)
        b = b.apply(lambda g: pd.Series({"x": H.wmean(g.age, g.w), "own": H.wmean(g.own, g.w)}), include_groups=False)
        axes[1].plot(b.x, 100 * b.own, "o-", ms=2.5, lw=0.9, color=c, label=str(t))
    axes[1].set_xlabel("age of the reference person")
    axes[1].set_ylabel("own a computer (%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False, title="year")
    axes[1].set_title("Raw ownership by age, no model")
    for ax in axes:
        ax.grid(color=GRID, lw=0.4)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    for ax, (a, b) in zip(axes, [(1994, 1996), (2002, 2004), (2010, 2012)]):
        d = u[u.year.between(a, b) & u.state.notna()]
        st = d.groupby("state").apply(lambda g: pd.Series({"x": H.wmean(g.ln_s, g.w), "own": H.wmean(g.own, g.w),
                                                           "n": len(g)}), include_groups=False)
        st = st[st.n >= 100]
        ax.scatter(st.x, 100 * st.own, s=np.sqrt(st.n) * 1.5, color=ACCENT, alpha=0.6, lw=0)
        for s_, row in st.iterrows():
            ax.annotate(ABBR.get(int(s_), str(int(s_))), (row.x, 100 * row.own), fontsize=6, xytext=(2, 2),
                        textcoords="offset points", color=INK)
        sup = d[d.state.isna()] if False else u[u.year.between(a, b) & u.state.isna()]
        ax.scatter([H.wmean(sup.ln_s, sup.w)], [100 * H.wmean(sup.own, sup.w)], marker="s", s=30, color=ACCENT2,
                   label="state suppressed")
        ax.set_xlabel("state's mean log long-run spending")
        ax.set_ylabel("own a computer (%)")
        ax.set_title(f"{a}-{b}: states with 100+ units (dot area = units)")
        ax.grid(color=GRID, lw=0.4)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for lab, d, c in [("everyone else", rest, "#7F7F7F"), ("more than 3 SDs below the year's mean", low, ACCENT2)]:
        g = d.groupby("year").apply(lambda x: 100 * H.wmean(x.own, x.w), include_groups=False)
        ax.plot(g.index, g, "o-", ms=3, lw=1, color=c, label=lab)
    ax.set_ylabel("own a computer (%)")
    ax.set_xlabel("year")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(1994, 2014, 2))
    ax.legend(frameon=False)
    ax.set_title("Ownership among very low spenders")
    ax.grid(color=GRID, lw=0.4)
    ax = axes[1]
    pa = pd.read_csv(f"{H.OUT}/prices_annual.csv")
    lab = {"pc": "BEA PCE, PCs (main)", "info": "BEA PCE, information processing", "business": "BEA investment, computers",
           "cpi_pc": "BLS CPI-U, PCs", "paid_p50": "CE price paid, median", "paid_p90": "CE price paid, 90th pct"}
    colors = ["#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#E69F00", "#D55E00"]
    lp = np.log(pa.set_index("year")[list(lab)]).loc[1984:2013]
    lp["cpi_pc"] = lp.cpi_pc - lp.cpi_pc.loc[1999] + (lp.pc.loc[1999] - lp.pc.loc[1994])
    for (c, l), col in zip(lab.items(), colors):
        s = lp[c].dropna()
        ax.plot(s.index, s - (lp[c].loc[1994] if c != "cpi_pc" else 0), lw=1.2, color=col, label=l)
    ax.set_ylabel("log price relative to 1994")
    ax.set_xlabel("year")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Computer prices (CPI-U placed on the BEA index in 1999)")
    ax.grid(color=GRID, lw=0.4)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
print("\nWrote output/diagnostics/explore_data.pdf")

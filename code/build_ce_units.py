"""Build the CE analysis sample for the household PC model: one row per consumer unit.

Input: data/processed/ce_households.csv.gz and data/processed/ce_replicate_weights.csv.gz (from build_ce.py).

Each unit enters once, at its first interview with the appliance inventory. The inventory is carried forward within
a unit, so later interviews add no ownership data; they are used only for long-run spending.
- s: long-run spending in current dollars a year, four times the unit's mean spending per three-month interview
  period over all its interviews, leaving out vehicle purchases, computer and internet spending, personal insurance
  and pensions, and cash contributions. Units whose s is not positive are dropped.
- s_incl and s_veh, two other definitions for checks: s with personal insurance, pensions, and cash contributions
  added back (the definition before 9 November 2026), and s with vehicle purchases added back.
- inc_cur: current income before taxes for complete income reporters (FINCBTAX; FINCBTXM in 2004-2005, when
  FINCBTAX is missing); missing for incomplete reporters.
- sd_log_cons: standard deviation of log spending across the unit's interviews (for the noise correction in step 2).
- ref_year, ref_month: the month the inventory was taken. The unpublished first (bounding) interview comes three
  months before interview 2, so the reference month is 3 x (interview - 1) months before the first record. Units
  whose first record is in 1994 at interviews 3-5 (mostly 1994Q1, when the public files begin) get that record's month.
- w: FINLWT21 of that interview, rescaled so each quarter carries an equal share of its year's weight and the weights
  average 1 within each year; wtrep01-wtrep44 are the 44 replicate weights treated the same way.

Run from the project folder: python3 code/build_ce_units.py > output/logs/build_ce_units.txt
Writes data/processed/ce_units.csv.gz.
"""
import numpy as np
import pandas as pd

OUT = "data/processed"
REP = [f"wtrep{i:02d}" for i in range(1, 45)]
TRAITS = ["age", "sex", "educ4", "race3", "hisp", "married", "kids", "fam_size", "earners", "region", "state", "urban",
          "tenure", "income", "income_imputed", "income_complete"]


def rescale(w, year, quarter):
    """Give each quarter an equal share of its year's total weight, then scale to a mean of 1 within each year."""
    share = w.groupby(year).transform("sum") / quarter.groupby(year).transform("nunique")
    w = w * share / w.groupby([year, quarter]).transform("sum")
    return w / w.groupby(year).transform("mean")


h = pd.read_csv(f"{OUT}/ce_households.csv.gz")
h["cons_q"] = h.spend_q - h.vehicles_q - h.comp_hw_q - h.comp_sw_q - h.internet_q - h.perins_q - h.cashco_q
h["cons_incl_q"] = h.cons_q + h.perins_q + h.cashco_q
h["cons_veh_q"] = h.cons_q + h.vehicles_q
lr = h.groupby("cu").agg(s=("cons_q", lambda c: 4 * c.mean()), n_interviews=("cons_q", "size"),
                         sd_log_cons=("cons_q", lambda c: np.log(c[c > 0]).std()),
                         s_incl=("cons_incl_q", lambda c: 4 * c.mean()), s_veh=("cons_veh_q", lambda c: 4 * c.mean()))

u = (h[h.inventory].sort_values(["year", "month", "interview"]).groupby("cu").head(1)
     .set_index("cu").join(lr).reset_index())
print(f"Units with an inventory record: {len(u):,}")
bad = ~(u.s > 0)
print(f"Dropped, long-run spending not positive: {bad.sum():,}")
u = u[~bad].copy()

back = (3 * (u.interview - 1)).where(~((u.year == 1994) & (u.interview >= 3)), 0)
t = u.year * 12 + u.month - 1 - back
u["ref_year"], u["ref_month"], u["months_back"] = t // 12, t % 12 + 1, back
u["quarter"] = (u.month - 1) // 3 + 1
u["own"] = u.pc.astype(int)
u["ln_s"] = np.log(u.s)
for c in ["s_incl", "s_veh"]:
    u[f"ln_{c}"] = np.log(u[c].where(u[c] > 0))
u["inc_cur"] = u.income.where(~u.year.isin([2004, 2005]), u.income_imputed).where(u.income_complete == 1)

rep = pd.read_csv(f"{OUT}/ce_replicate_weights.csv.gz")
u = u.merge(rep, on="newid", how="left", validate="one_to_one")
assert u[REP].notna().all().all()
u["w"] = rescale(u.weight, u.year, u.quarter)
for c in REP:
    u[c] = rescale(u[c], u.year, u.quarter)

cols = ["cu", "newid", "year", "month", "quarter", "interview", "ref_year", "ref_month", "months_back", "own", "s",
        "ln_s", "n_interviews", "sd_log_cons", "ln_s_incl", "ln_s_veh", "inc_cur", "weight", "w"] + TRAITS + REP
u[cols].to_csv(f"{OUT}/ce_units.csv.gz", index=False)

# ---------- checks ----------
pd.set_option("display.width", 200)


def wmedian(x, w):
    o = np.argsort(x.values)
    return x.values[o][np.searchsorted(np.cumsum(w.values[o]) / w.sum(), 0.5)]


g = u.groupby("year")
print(f"\nUnits in the sample: {len(u):,}")
print("\nBy year: units, weighted ownership (%), weighted median long-run spending ($), mean interviews per unit, "
      "share of each quarter's weight (Q1-Q4), share of zero replicate weights")
tab = pd.DataFrame({
    "units": g.size(),
    "own_pct": g.apply(lambda d: 100 * np.average(d.own, weights=d.w), include_groups=False).round(1),
    "median_s": g.apply(lambda d: wmedian(d.s, d.w), include_groups=False).round(0),
    "interviews": g.n_interviews.mean().round(2),
    **{f"q{q}": g.apply(lambda d, q=q: d.w[d.quarter == q].sum() / d.w.sum(), include_groups=False).round(3)
       for q in [1, 2, 3, 4]},
    "zero_rep": g.apply(lambda d: (d[REP] == 0).to_numpy().mean(), include_groups=False).round(3)})
print(tab.to_string())
print("\nMonths from the inventory to the first record, by first-record interview number")
print(pd.crosstab(u.interview, u.months_back).to_string())
print(f"\nUnits with a reference month before 1994: {(u.ref_year < 1994).sum():,}; latest reference month: "
      f"{u.ref_year.max():.0f}-{u.ref_month[u.ref_year == u.ref_year.max()].max():.0f}")
print("\nStandard deviation of log spending across a unit's interviews (units with 2+ positive interviews): "
      f"median {u.sd_log_cons.median():.3f}, mean {u.sd_log_cons.mean():.3f}")

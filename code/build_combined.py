"""Build one household file from both surveys, for readers who want the data without the model.

Inputs (from build_cps.py, build_ce_units.py, and build_prices.py):
- data/processed/ce_units.csv.gz: the CE analysis sample, 178,891 consumer units interviewed from January 1994 to
  March 2013, one row per unit at its first public interview with the appliance inventory.
- data/processed/cps_households.csv.gz: 425,627 households interviewed in the eight CPS Computer Use supplements,
  October 1984 to October 2003, one row per household.
- data/processed/prices_monthly.csv: the monthly computer price indexes.

Each row is one household. The file keeps the variables the two surveys share, under one set of names and codes, and
adds the BEA price index for personal computers in the month that the household's computer answer refers to (the
inventory month in the CE, the survey month in the CPS). Variables that only one survey records (long-run spending
and income in dollars in the CE, income brackets in the CPS) are blank for the other survey. in_paper is 1 for the
households that the paper's estimates use: every CE unit, and the 369,393 CPS households that answered the computer
question, have a positive weight, and have a known income bracket, state, and household traits.

weight is the survey weight that the paper uses, scaled to average 1 within each survey and year: for the CE, the
unit's FINLWT21 rescaled so that each quarter carries an equal share of its year's weight; for the CPS, the household
weight. Use it for weighted shares and means within a survey and year. The CE's 44 replicate weights are in
ce_units.csv.gz.

Run from the project folder: python3 code/build_combined.py > output/logs/build_combined.txt
Writes data/processed/households_combined.csv.gz.
"""
import numpy as np
import pandas as pd

OUT = "data/processed"
CPS_MONTH = {1984: 10, 1989: 10, 1993: 10, 1997: 10, 1998: 12, 2000: 8, 2001: 9, 2003: 10}
REGION = {**{s: 1 for s in [9, 23, 25, 33, 34, 36, 42, 44, 50]},
          **{s: 2 for s in [17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55]},
          **{s: 3 for s in [1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54]},
          **{s: 4 for s in [2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56]}}
TRAITS = ["age", "sex", "race3", "hisp", "married", "kids", "persons", "educ4", "state", "region"]
COLS = (["survey", "id", "year", "month", "ref_year", "ref_month", "owns_computer", "weight", "in_paper"] + TRAITS +
        ["spending", "income", "income_lo", "income_hi", "pc_price"])

u = pd.read_csv(f"{OUT}/ce_units.csv.gz", usecols=lambda c: not c.startswith("wtrep"))
ce = pd.DataFrame({"survey": "CE", "id": u.cu, "year": u.year, "month": u.month, "ref_year": u.ref_year,
                   "ref_month": u.ref_month, "owns_computer": u.own, "weight": u.w, "in_paper": 1,
                   **{k: u[k] for k in TRAITS if k != "persons"}, "persons": u.fam_size,
                   "spending": u.s, "income": u.inc_cur})

c = pd.read_csv(f"{OUT}/cps_households.csv.gz")
known = c[["age", "kids", "persons", "race3", "hisp", "married", "educ4", "state"]].notna().all(axis=1)
c["in_paper"] = (c.owns.notna() & (c.hh_weight > 0) & c.inc_lo.notna() & known).astype(int)
c["month"] = c.year.map(CPS_MONTH)
cps = pd.DataFrame({"survey": "CPS", "id": c.household, "year": c.year, "month": c.month, "ref_year": c.year,
                    "ref_month": c.month, "owns_computer": c.owns, "weight": c.hh_weight / c.groupby("year").hh_weight.transform("mean"),
                    "in_paper": c.in_paper, **{k: c[k] for k in TRAITS if k != "region"},
                    "region": c.state.map(REGION), "income_lo": c.inc_lo, "income_hi": c.inc_hi})

h = pd.concat([cps, ce], ignore_index=True)
pm = pd.read_csv(f"{OUT}/prices_monthly.csv").rename(columns={"year": "ref_year", "month": "ref_month"})
h = h.merge(pm[["ref_year", "ref_month", "ln_pc"]], on=["ref_year", "ref_month"], how="left", validate="many_to_one")
assert h.ln_pc.notna().all()
h["pc_price"] = np.exp(h.ln_pc).round(3)
for k in ["owns_computer", "hisp", "married", "kids", "persons", "state", "region", "age", "sex"]:
    h[k] = h[k].astype("Int64")
h = h.sort_values(["survey", "year", "id"], ascending=[False, True, True])[COLS]
assert not h.duplicated(["survey", "year", "id"]).any()
h.to_csv(f"{OUT}/households_combined.csv.gz", index=False)

# ---------- checks ----------
pd.set_option("display.width", 200)
print(f"Households: {len(h):,} ({(h.survey == 'CE').sum():,} CE, {(h.survey == 'CPS').sum():,} CPS); "
      f"in the paper's estimates: {h.in_paper.sum():,} ({h[h.survey == 'CPS'].in_paper.sum():,} CPS)")
ans = h[h.owns_computer.notna() & (h.weight > 0)]
own = ans.groupby(["survey", "year"]).apply(lambda d: 100 * np.average(d.owns_computer, weights=d.weight),
                                             include_groups=False).round(1)
print("\nComputer ownership rate (% of households, weighted), by survey and year")
print(own.unstack(0).to_string())
print("\nShare missing, by survey")
print(h.groupby("survey")[COLS[6:]].apply(lambda d: d.isna().mean()).round(3).T.to_string())

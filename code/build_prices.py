"""Build the computer price series for the household PC analysis.

Sources, saved unchanged in data/raw/prices/:
- NipaDataA.txt (annual values), the BEA NIPA flat file from
  https://apps.bea.gov/national/Release/TXT/, release of 4 Nov 2026, downloaded 9 Nov 2026:
  - DCPPRG: PCE price index, personal computers/tablets and peripheral equipment (Table 2.4.4U, line 49). Main price.
  - DIPERG: PCE price index, information processing equipment (Table 2.4.4U, line 48), which adds software and
    calculators, typewriters, and other information equipment.
  - B935RG: private fixed investment price index, computers and peripheral equipment (Table 5.3.4, line 11).
  - DPCERG: PCE price index, all items (Table 2.3.4, line 1), used to put incomes in constant dollars (step 5).
- CUUR0000SEEE01.csv: BLS CPI-U, personal computers and peripheral equipment, U.S. city average, not seasonally
  adjusted, monthly from December 1997, downloaded from FRED on 9 Nov 2026.
- The price paid comes from data/processed/ce_computer_purchases.csv.gz (built by build_ce.py): CE purchase records
  for code 640, bought for the unit's own use (GFTCMIN = 1), weighted by FINLWT21 of the interview.

Run from the project folder: python3 code/build_prices.py > output/logs/build_prices.txt
Writes data/processed/prices_annual.csv and data/processed/prices_monthly.csv. In the monthly file each series is in
natural logs; an annual value is placed at mid-year and months in between are interpolated linearly in logs.
"""
import numpy as np
import pandas as pd

RAW = "data/raw/prices"
OUT = "data/processed"
BEA = {"DCPPRG": "pc", "DIPERG": "info", "B935RG": "business", "DPCERG": "pce"}


def wquantile(x, w, q):
    """Weighted quantile: the smallest x whose cumulative weight share reaches q."""
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    return x[np.searchsorted(np.cumsum(w) / w.sum(), q)]


nipa = pd.read_csv(f"{RAW}/NipaDataA.txt", dtype=str)
nipa.columns = ["series", "period", "value"]
nipa = nipa[nipa.series.isin(BEA)]
annual = (nipa.assign(year=nipa.period.astype(int), value=nipa.value.str.replace(",", "").astype(float))
          .pivot(index="year", columns="series", values="value").rename(columns=BEA))

cpi = pd.read_csv(f"{RAW}/CUUR0000SEEE01.csv")
cpi["date"] = pd.to_datetime(cpi.observation_date)
cpi = pd.DataFrame({"year": cpi.date.dt.year, "month": cpi.date.dt.month,
                    "cpi_pc": pd.to_numeric(cpi.CUUR0000SEEE01, errors="coerce")}).dropna()
full_years = cpi.groupby("year").month.count().pipe(lambda n: n[n == 12].index)
annual["cpi_pc"] = cpi[cpi.year.isin(full_years)].groupby("year").cpi_pc.mean()

# price paid; build_ce.py has already removed first-quarter purchases repeated in two yearly releases
p = pd.read_csv(f"{OUT}/ce_computer_purchases.csv.gz")
assert p.groupby("newid").source.nunique().max() == 1
h = pd.read_csv(f"{OUT}/ce_households.csv.gz", usecols=["newid", "weight"])
p = p[(p.gift_code == 1) & (p.price > 0)].merge(h, on="newid", how="left", validate="many_to_one")
assert p.weight.notna().all()
paid = p.groupby("year").apply(lambda g: pd.Series({"paid_p50": wquantile(g.price, g.weight, 0.5),
                                                    "paid_p90": wquantile(g.price, g.weight, 0.9),
                                                    "n_paid": len(g)}), include_groups=False)
annual = annual.join(paid, how="left")
annual.index.name = "year"
annual.loc[1959:2025].to_csv(f"{OUT}/prices_annual.csv")

monthly = pd.DataFrame([(y, m) for y in range(1983, 2016) for m in range(1, 13)], columns=["year", "month"])
t = monthly.year + (monthly.month - 0.5) / 12
for c in ["pc", "info", "business", "paid_p50", "paid_p90"]:
    s = annual[c].dropna()
    monthly[f"ln_{c}"] = np.interp(t, s.index + 0.5, np.log(s.values))   # flat beyond the first and last mid-year
    if c.startswith("paid"):
        monthly.loc[monthly.year < 1993, f"ln_{c}"] = np.nan   # the price paid starts in 1994; only late 1993 is filled
monthly = monthly.merge(cpi.assign(ln_cpi_pc=np.log(cpi.cpi_pc))[["year", "month", "ln_cpi_pc"]], how="left")
monthly.to_csv(f"{OUT}/prices_monthly.csv", index=False)

# ---------- checks ----------
pd.set_option("display.width", 200)
la = np.log(annual[["pc", "info", "business", "cpi_pc", "paid_p50", "paid_p90"]])
print("\nLog price relative to 1994 (CPI-U relative to 1998, its first full year)")
rel = la - la.loc[1994]
rel["cpi_pc"] = la.cpi_pc - la.cpi_pc.loc[1998]
print(rel.loc[[1984, 1989, 1993, 1994, 1995, 1996, 1997, 1998, 2000, 2001, 2003, 2008, 2013]].round(3).to_string())
print("\nMean log change per year")
for a, b in [(1984, 1993), (1993, 2003), (1994, 2003), (2003, 2013), (1994, 2013), (1998, 2013)]:
    print(f"  {a}-{b}: " + ", ".join(f"{c} {(la.loc[b, c] - la.loc[a, c]) / (b - a):+.3f}" for c in la.columns
                                   if la[c].notna()[[a, b]].all()))
print("\nPrice paid for own use, weighted by FINLWT21 (current dollars) and number of purchases")
print(annual.loc[1994:2014, ["paid_p50", "paid_p90", "n_paid"]].astype(float).round(0).to_string())
check = monthly.assign(lp=np.log(annual.pc.reindex(monthly.year).values)).groupby("year")[["ln_pc", "lp"]].mean()
gap = (check.ln_pc - check.lp).loc[1994:2013]
print(f"\nInterpolation check: within each year 1994-2013 the monthly log PC index averages to the annual log value "
      f"within {gap.abs().max():.3f}")

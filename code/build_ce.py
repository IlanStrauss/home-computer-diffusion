"""Build the Consumer Expenditure Survey household file for home-computer ownership.

Source: BLS Consumer Expenditure Interview Survey public-use microdata (PUMD), comma-delimited
yearly files intrvwYY.zip from https://www.bls.gov/cex/pumd_data.htm, saved unchanged in data/raw/ce/.

One row per consumer-unit interview (NEWID; the last digit is the interview number, the rest
identifies the consumer unit, CU). Ownership comes from the major-appliance inventory (APL file,
MAJCODE 13: "Computers, not solely for games" 1994Q1-2001Q1, "Home Computer" 2001Q2-2013Q1).
Computer purchases come from the minor-appliance file (APB, MINAPPLY 640: "Computers, computer
systems and related hardware for non-business use") and from monthly spending (MTBI, UCC 690111
computers and hardware, 690112 software and accessories).

Run from the project folder: python3 code/build_ce.py > output/logs/build_ce.txt
Writes data/processed/ce_households.csv.gz, data/processed/ce_replicate_weights.csv.gz (the 44 BRR replicate
weights of each interview, blanks set to zero as BLS instructs), and data/processed/ce_computer_purchases.csv.gz.
"""
import glob
import re
import zipfile

import numpy as np
import pandas as pd

RAW = "data/raw/ce"
OUT = "data/processed"

FMLI_VARS = ["NEWID", "QINTRVYR", "QINTRVMO", "FINLWT21", "FINCBTAX", "FINCBTXM", "RESPSTAT", "INC_RANK", "INC_RNKM",
             "TOTEXPPQ", "TOTEXPCQ", "FOODPQ", "FOODCQ", "CARTKNPQ", "CARTKNCQ", "CARTKUPQ", "CARTKUCQ",
             "OTHVEHPQ", "OTHVEHCQ", "AGE_REF", "SEX_REF", "EDUC_REF", "EDUCA2", "REF_RACE", "ORIGIN1", "HORREF1",
             "HISP_REF", "FAM_SIZE", "FAM_TYPE", "PERSLT18", "PERSOT64", "NO_EARNR", "REGION", "STATE",
             "BLS_URBN", "MARITAL1", "CUTENURE", "PERINSPQ", "PERINSCQ", "CASHCOPQ", "CASHCOCQ"]
REPLICATES = [f"WTREP{i:02d}" for i in range(1, 45)]
# UCC lists in the zips (ucciYY.txt): from 2011Q2, 690112 is split into 690119 software and 690120 accessories;
# 690310 is installation for computers; 690114 computer information services and 690116 internet services away from home
COMPUTER_UCC = {"690111": "comp_hw", "690310": "comp_hw", "690112": "comp_sw", "690119": "comp_sw", "690120": "comp_sw",
                "690114": "internet", "690116": "internet"}

# EDUC_REF uses 1-7 up to 1995 and 0/10-17 from 1996 (BLS PUMD dictionary); four common groups
EDUC4 = {1: "lt_hs", 2: "lt_hs", 7: "lt_hs", 3: "hs", 4: "some_col", 5: "col", 6: "col",
         0: "lt_hs", 10: "lt_hs", 11: "lt_hs", 12: "hs", 13: "some_col", 14: "some_col", 15: "col", 16: "col",
         17: "col"}


def read_csv(z, name, cols=None):
    head = pd.read_csv(z.open(name), nrows=0).columns
    use = [c for c in head if c.upper() in cols] if cols else None
    d = pd.read_csv(z.open(name), usecols=use, dtype=str)
    d.columns = [c.upper() for c in d.columns]
    return d


def num(s):
    return pd.to_numeric(s, errors="coerce")


fam, purchases = [], []
for path in sorted(glob.glob(f"{RAW}/intrvw*.zip")):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    pick = lambda pat: [n for n in names if re.search(pat, n.lower())]

    f = pd.concat([read_csv(z, n, FMLI_VARS + REPLICATES).assign(source=n.split("/")[-1])
                   for n in pick(r"fmli\d{3}x?\.csv$")])
    f["NEWID"] = num(f.NEWID).astype("int64")

    apl_files = pick(r"apl\d{2}\.csv$")   # the inventory ends with 2013Q1, in intrvw12.zip; intrvw13.zip has no APL file
    apl = (pd.concat([read_csv(z, n, ["NEWID", "MAJCODE", "MAJAPPLQ"]) for n in apl_files]) if apl_files
           else pd.DataFrame(columns=["NEWID", "MAJCODE", "MAJAPPLQ"]))
    apl["NEWID"] = num(apl.NEWID).astype("int64")
    pc = apl[num(apl.MAJCODE) == 13].groupby("NEWID").MAJAPPLQ.apply(lambda q: num(q).sum(min_count=1))
    f["inventory"] = f.NEWID.isin(apl.NEWID)
    f["pc"] = f.NEWID.isin(pc.index)
    f["pc_qty"] = f.NEWID.map(pc)

    apb_cols = ["NEWID", "QYEAR", "MINAPPLY", "MINPURX", "MIN_MO", "GFTCMIN"]
    apb_files = pick(r"apb\d{2}\.csv$")   # intrvw00.zip has no APB file
    apb = pd.concat([read_csv(z, n, apb_cols) for n in apb_files]) if apb_files else pd.DataFrame(columns=apb_cols)
    apb = apb[num(apb.MINAPPLY) == 640].copy()
    apb["NEWID"] = num(apb.NEWID).astype("int64")
    purchases.append(apb.assign(source=path.split("/")[-1]))

    spend = []
    for n in pick(r"mtbi\d{3}x?\.csv$"):
        m = read_csv(z, n, ["NEWID", "UCC", "COST", "GIFT"])
        m = m[m.UCC.isin(COMPUTER_UCC)]
        spend.append(m.assign(NEWID=num(m.NEWID).astype("int64"), COST=num(m.COST), item=m.UCC.map(COMPUTER_UCC)))
    spend = pd.concat(spend)
    tot = spend.pivot_table(index="NEWID", columns="item", values="COST", aggfunc="sum")
    own = spend[spend.GIFT != "1"].pivot_table(index="NEWID", columns="item", values="COST", aggfunc="sum")
    for c in sorted(set(COMPUTER_UCC.values())):
        f[f"{c}_q"] = f.NEWID.map(tot[c] if c in tot else pd.Series(dtype=float)).fillna(0)
        f[f"{c}_own_q"] = f.NEWID.map(own[c] if c in own else pd.Series(dtype=float)).fillna(0)
    fam.append(f)
    print(f"{path}: {len(f):,} interviews, {f.inventory.mean():.1%} with inventory, "
          f"{len(apb):,} computer purchase records")

d = pd.concat(fam)
# a first quarter is released twice, as fmliYY1 at the end of one yearly zip and as fmliYY1x at the start of the next;
# keep the copy that has the appliance inventory and, of two such copies, the later (x) release
d["x_file"] = d.source.str.contains(r"x\.csv$", case=False)
d = d.sort_values(["inventory", "x_file"], ascending=False, kind="stable")
dups = d.NEWID.duplicated()
print(f"\nInterviews read: {len(d):,}; duplicate NEWIDs across files dropped: {dups.sum():,}")
d = d[~dups].sort_values("NEWID").copy()

year = num(d.QINTRVYR)
year = year.where(year >= 100, year + 1900)   # the 1994-95 files give the interview year in two digits
# Hispanic origin: ORIGIN1 = 2 ("Spanish") to 2003Q1; from 2003Q2 HORREF1 gives the Hispanic group (1-6) and is blank
# for everyone else (BLS PUMD dictionary). REF_RACE: 1 white, 2 black, 3-6 other groups (6 multi-race from 2003Q2).
yq = d.source.str.extract(r"fmli(\d\d)(\d)").astype(int)
quarter = (yq[0] + np.where(yq[0] >= 80, 1900, 2000)) * 10 + yq[1]
origin, race = num(d.ORIGIN1), num(d.REF_RACE)
horref = num(d.get("HORREF1", pd.Series(np.nan, index=d.index)))
hisp = np.where(quarter >= 20032, horref.between(1, 6), np.where(origin.notna(), origin == 2, np.nan))
out = pd.DataFrame({
    "newid": d.NEWID, "cu": d.NEWID // 10, "interview": d.NEWID % 10,
    "year": year, "month": num(d.QINTRVMO), "weight": num(d.FINLWT21),
    "inventory": d.inventory, "pc": d.pc, "pc_qty": d.pc_qty,
    "income": num(d.FINCBTAX), "income_imputed": num(d.get("FINCBTXM")), "income_complete": num(d.RESPSTAT) == 1,
    "inc_rank": num(d.INC_RANK), "inc_rank_imputed": num(d.get("INC_RNKM")),
    "spend_q": num(d.TOTEXPPQ).fillna(0) + num(d.TOTEXPCQ).fillna(0),
    "food_q": num(d.FOODPQ).fillna(0) + num(d.FOODCQ).fillna(0),
    "vehicles_q": sum(num(d[c]).fillna(0) for c in ["CARTKNPQ", "CARTKNCQ", "CARTKUPQ", "CARTKUCQ", "OTHVEHPQ", "OTHVEHCQ"]),
    "perins_q": num(d.PERINSPQ).fillna(0) + num(d.PERINSCQ).fillna(0),   # personal insurance and pensions
    "cashco_q": num(d.CASHCOPQ).fillna(0) + num(d.CASHCOCQ).fillna(0),   # cash contributions
    "comp_hw_q": d.comp_hw_q, "comp_sw_q": d.comp_sw_q, "comp_hw_own_q": d.comp_hw_own_q, "internet_q": d.internet_q,
    "age": num(d.AGE_REF), "sex": num(d.SEX_REF), "educ_raw": num(d.EDUC_REF),
    "educ4": num(d.EDUC_REF).map(EDUC4), "educ_spouse_raw": num(d.EDUCA2),
    "race": race, "race3": race.map(lambda r: "white" if r == 1 else "black" if r == 2 else "other" if r >= 3 else np.nan),
    "origin": origin, "horref1": horref, "hispanic_ref": num(d.get("HISP_REF")), "hisp": hisp,
    "fam_size": num(d.FAM_SIZE), "fam_type": num(d.FAM_TYPE), "kids": num(d.PERSLT18), "over64": num(d.PERSOT64),
    "earners": num(d.NO_EARNR), "region": num(d.REGION), "state": num(d.STATE), "urban": num(d.BLS_URBN),
    "marital": num(d.MARITAL1), "married": num(d.MARITAL1).eq(1).astype(float).where(num(d.MARITAL1).notna()),
    "tenure": num(d.CUTENURE), "source": d.source})
out.to_csv(f"{OUT}/ce_households.csv.gz", index=False)
# in the public files about half the replicate weights are blank; BLS says to read a blank as zero
rep = pd.DataFrame({"newid": d.NEWID, **{c.lower(): num(d[c]).fillna(0).round(3) for c in REPLICATES}})
rep.to_csv(f"{OUT}/ce_replicate_weights.csv.gz", index=False)

p = pd.concat(purchases, ignore_index=True)
# a first-quarter interview's purchases appear in two yearly releases; keep the later release, as for the household file
p["release"] = p.source.str.extract(r"intrvw(\d\d)")[0].astype(int).pipe(lambda y: np.where(y >= 80, 1900 + y, 2000 + y))
repeat = p.release < p.groupby("NEWID").release.transform("max")
print(f"Computer purchase records: {len(p):,}; repeats from an earlier release dropped: {repeat.sum():,}")
p = p[~repeat]
p = pd.DataFrame({"newid": p.NEWID, "qyear": num(p.QYEAR), "month": num(p.MIN_MO), "price": num(p.MINPURX),
                  "gift_code": num(p.GFTCMIN), "source": p.source})
# two purchases at the same price in the same month are separate records in APB, so identical rows are kept
print(f"Identical purchase rows within a release, kept as separate purchases: {p.duplicated().sum():,}")
p["year"] = out.set_index("newid").year.reindex(p.newid).values
p.to_csv(f"{OUT}/ce_computer_purchases.csv.gz", index=False)

# ---------- checks ----------
pd.set_option("display.width", 200)
inv = out[out.inventory]
w = lambda g, x: np.average(g[x], weights=g.weight)
cps = {1997: 36.6, 1998: 42.1, 2000: 51.0, 2001: 56.3, 2003: 61.8}   # Census P23-208, Figure 1
print("\nBy interview year: interviews, CUs, share with inventory, weighted PC ownership (CPS benchmark), "
      "complete income reporters")
for y, g in out.groupby("year"):
    gi = g[g.inventory]
    own = f"{100 * w(gi, 'pc'):5.1f}%" if len(gi) else "    - "
    print(f"  {y:.0f}: {len(g):6,} {g.cu.nunique():6,} {g.inventory.mean():6.1%} "
          f"{own} ({cps.get(int(y), '  -  ')}) {g.income_complete.mean():6.1%}")

inv = inv.assign(fifth=pd.cut(inv.inc_rank, [0, .2, .4, .6, .8, 1.0001], labels=[1, 2, 3, 4, 5], right=False))
print("\nWeighted PC ownership (%) by income fifth (INC_RANK), complete income reporters")
print((100 * inv[inv.income_complete].groupby(["year", "fifth"], observed=True)[["pc", "weight"]].apply(lambda g: w(g, "pc"))).unstack().round(1)
      .to_string())
print("\nWeighted PC ownership (%) by education of reference person")
print((100 * inv.groupby(["year", "educ4"])[["pc", "weight"]].apply(lambda g: w(g, "pc"))).unstack()
      [["lt_hs", "hs", "some_col", "col"]].round(1).to_string())
print("\nWeighted shares of interviews by year: Hispanic, black, and married reference person")
print(out.assign(black=(out.race3 == "black").astype(float)).groupby("year")[["hisp", "black", "married", "weight"]].apply(
    lambda g: pd.Series({k: np.average(g[k].dropna(), weights=g.weight[g[k].notna()]) for k in ["hisp", "black", "married"]}))
      .round(3).to_string())
print("\nBy year: share of replicate-weight cells equal to zero, and personal insurance and pensions and cash "
      "contributions as weighted shares of total spending")
chk = out[["year", "weight", "spend_q", "perins_q", "cashco_q"]].assign(zero=(rep.iloc[:, 1:] == 0).mean(axis=1).values)
print(chk.groupby("year").apply(lambda g: pd.Series({
    "zero_share": g.zero.mean(), "perins": np.average(g.perins_q, weights=g.weight) / np.average(g.spend_q, weights=g.weight),
    "cashco": np.average(g.cashco_q, weights=g.weight) / np.average(g.spend_q, weights=g.weight)}),
    include_groups=False).round(3).to_string())
both = out[out.hispanic_ref.isin([1, 2]) & out.hisp.notna()]
print(f"hisp agrees with HISP_REF (2009Q2 on) in {(both.hisp == (both.hispanic_ref == 1)).mean():.2%} of {len(both):,} interviews")
own = p[(p.gift_code == 1) & (p.price > 0)]
print("\nComputer purchases for own use (APB 640), price paid in current dollars, by interview year")
print(own.groupby("year").price.describe(percentiles=[.25, .5, .75, .9])[["count", "25%", "50%", "75%", "90%", "mean"]]
      .round(0).to_string())

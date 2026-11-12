"""Build the CPS household file for home-computer ownership from the Computer Use supplements, 1984-2003.

Sources, saved unchanged in data/raw/cps/:
- October 1984, 1989, 1993, 1997 and December 1998: NBER copies of the Census public-use files (data.nber.org/cps/).
- August 2000, September 2001, October 2003: Census public-use files
  (www2.census.gov/programs-surveys/cps/datasets/YYYY/supp/).
- Record layouts, 1997-2003: NBER Stata dictionaries (data.nber.org/progs/cps/*.dct), checked against the
  Census technical documentation (the cps*.pdf files in the same folder).
- Record layouts, 1984-1993: typed from the scanned Census technical documentation (cpsoct84/89/93.pdf);
  the pdf pages are listed next to each layout.

The files hold one record per person. The computer question, family income and the household weight are
household items. Each household row takes the person characteristics of the reference person
(relationship code 1 or 2: the reference person with or without relatives), which is also the universe of
the household computer question from 1997.

Run from the project folder: python3 code/build_cps.py > output/logs/build_cps.txt
Writes data/processed/cps_households.csv.gz.
"""
import re
import subprocess

import numpy as np
import pandas as pd

RAW = "data/raw/cps"
OUT = "data/processed"

# 1997-2003: canonical name -> dictionary variable, per supplement
COMMON = {"hhid": "hrhhid", "sample": "hrsample", "sersuf": "hrsersuf", "hhnum": "huhhnum", "faminc": "hufaminc",
          "hh_weight": "hwhhwgt", "state": "gestfips", "relation": "perrp", "sex": "pesex",
          "educ": "peeduca", "marital": "pemaritl", "fam_type": "prfamtyp"}
FILES = {
    1997: ("cpsoct97.zip", "cpsoct97.dct", {"persons": "rnumhou", "age": "peage", "race": "perace", "hispanic": "prhspnon",
                                            "computer": "pescu1", "n_computers": "pescu2"}),
    1998: ("cpsdec98.zip", "cpsdec1998.dct", {"persons": "hrnumhou", "age": "peage", "race": "perace", "hispanic": "prhspnon",
                                              "computer": "hescu1a", "n_computers": "hescu3"}),
    2000: ("aug00supp.zip", "cpsaug00.dct", {"persons": "hrnumhou", "age": "peage", "race": "perace", "hispanic": "prhspnon",
                                             "computer": "hescu1a", "n_computers": "hescu3"}),
    2001: ("sep01supp.zip", "cpssep01.dct", {"persons": "hrnumhou", "age": "peage", "race": "perace", "hispanic": "prhspnon",
                                             "computer": "hesc1", "n_computers": "hesc2"}),
    2003: ("oct03supp.zip", "cpsoct03.dct", {"persons": "hrnumhou", "age": "prtage", "race": "ptdtrace", "hispanic": "pehspnon",
                                             "computer": "hesc1", "n_computers": "hesc2"}),
}

# 1984-1993: name -> (first column, width, implied decimals). The 1984 documentation gives word and character
# (six characters to a word), so column = 6 * (word - 1) + character.
OLD = {
    # Attachment 6 pp. 1-2, 18-21, 24, 47, 56-57 (pdf pp. 34-35, 51-54, 56, 78, 87-88), Attachment 11 (pdf pp. 160, 164).
    # Relationship to the reference person, age, marital status and sex come from the "new demographic edit"
    # (words 65-66), the only edit filled on Armed Forces records. Race, education and Hispanic origin come from
    # the basic edit (words 17-18, 26) on adult records, where the new edit leaves them blank, and from the new
    # edit (word 67) on Armed Forces records. The basic edit's age and relationship to the head (words 16-17)
    # are read only to check agreement.
    1984: ("cpsoct84.zip", {"rectype": (1, 1, 0), "hhid": (4, 12, 0), "state60": (17, 2, 0), "relation": (387, 1, 0),
                            "age": (391, 2, 0), "marital": (393, 1, 0), "sex": (396, 1, 0), "race": (100, 1, 0),
                            "grade": (103, 2, 0), "grade_done": (105, 1, 0), "race_af": (401, 1, 0),
                            "grade_af": (398, 2, 0), "grade_done_af": (400, 1, 0), "principal": (111, 1, 0),
                            "weight": (121, 12, 2), "hispanic": (155, 1, 0), "faminc": (350, 1, 0),
                            "computer": (487, 1, 0), "supp_weight": (517, 12, 2), "age_basic": (97, 2, 0),
                            "head_basic": (96, 1, 0)}),
    # pdf pp. 35-39, 43, 68
    1989: ("cpsoct89.zip", {"rectype": (101, 1, 0), "hhid": (102, 12, 0), "hhnum": (6, 1, 0), "faminc": (26, 2, 0),
                            "persons": (36, 2, 0), "hh_weight": (49, 9, 2), "state": (83, 2, 0), "relation": (116, 2, 0),
                            "age": (120, 2, 0), "marital": (122, 1, 0), "sex": (125, 1, 0), "grade": (127, 2, 0),
                            "grade_done": (129, 1, 0), "race": (130, 1, 0), "hispanic": (194, 2, 0),
                            "computer": (377, 1, 0)}),
    # pdf pp. 31-36, 39, 65
    1993: ("cpsoct93.zip", {"rectype": (101, 1, 0), "hhid": (102, 12, 0), "hhnum": (6, 1, 0), "faminc": (26, 2, 0),
                            "persons": (36, 2, 0), "hh_weight": (49, 9, 2), "state": (83, 2, 0), "relation": (116, 2, 0),
                            "age": (120, 2, 0), "marital": (122, 1, 0), "sex": (125, 1, 0), "educ": (127, 2, 0),
                            "race": (130, 1, 0), "hispanic": (194, 2, 0), "computer": (379, 1, 0),
                            "n_computers": (380, 1, 0)}),
}
# 1984 family income codes 0-9 and A-D; "-" is no entry (pdf p. 78)
INC84 = {**{str(i): i for i in range(10)}, "A": 10, "B": 11, "C": 12, "D": 13}
# 1960 Census state code (1984 file) -> FIPS; checked against the 1989 file, which carries both
FIPS60 = {11: 23, 12: 33, 13: 50, 14: 25, 15: 44, 16: 9, 21: 36, 22: 34, 23: 42, 31: 39, 32: 18, 33: 17, 34: 26,
          35: 55, 41: 27, 42: 19, 43: 29, 44: 38, 45: 46, 46: 31, 47: 20, 51: 10, 52: 24, 53: 11, 54: 51, 55: 54,
          56: 37, 57: 45, 58: 13, 59: 12, 61: 21, 62: 47, 63: 1, 64: 28, 71: 5, 72: 22, 73: 40, 74: 48, 81: 30,
          82: 16, 83: 56, 84: 8, 85: 35, 86: 4, 87: 49, 88: 32, 91: 53, 92: 41, 93: 6, 94: 2, 95: 15}
# family income brackets, lower bounds in current dollars (the top bracket is open)
CUTS84 = [0, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000, 30000, 35000, 40000, 50000, 75000]
CUTS = [0, 5000, 7500, 10000, 12500, 15000, 20000, 25000, 30000, 35000, 40000, 50000, 60000, 75000]
CUTS03 = CUTS + [100000, 150000]
CENSUS = {1984: 8.2, 1989: 15.0, 1993: 22.8, 1997: 36.6, 1998: 42.1, 2000: 51.0, 2001: 56.3, 2003: 61.8}  # P23-208, Fig. 1


def parse_dct(path):
    """Variable name -> (start, end, implied decimals) from a Stata dictionary's _column() lines."""
    out = {}
    for line in open(path, encoding="latin-1"):
        m = re.match(r"\s*_column\(\s*(\d+)\s*\)\s+\w+\s+(\w+)\s+%(\d+)(?:\.(\d+))?[fs]", line)
        if m:
            start, name, width, dec = int(m[1]), m[2].lower(), int(m[3]), int(m[4] or 0)
            out[name] = (start - 1, start - 1 + width, dec)
    return out


def stream(zname):
    # the NBER 1997 zip has an extra field Python's zipfile rejects, so the unzip utility streams each file
    member = subprocess.run(["unzip", "-Z1", f"{RAW}/{zname}"], capture_output=True, text=True).stdout.split()[0]
    return member, subprocess.Popen(["unzip", "-p", f"{RAW}/{zname}", member], stdout=subprocess.PIPE).stdout


def new_households(year, zname, dct, extra):
    layout = parse_dct(f"{RAW}/{dct}")
    names = {**COMMON, **extra}
    specs = [layout[v] for v in names.values()]
    member, s = stream(zname)
    raw = pd.read_fwf(s, colspecs=[x[:2] for x in specs], names=list(names), dtype=str, header=None)
    d = pd.DataFrame({k: pd.to_numeric(raw[k], errors="coerce") for k in names if k not in ("hhid", "sample", "sersuf")})
    for k, (_, _, dec) in zip(names, specs):
        if dec and k in d:
            d[k] = d[k] / 10 ** dec
    d["household"] = raw.hhid.str.strip() + raw["sample"].fillna("").str.strip() + raw.sersuf.fillna("").str.strip() \
        + raw.hhnum.fillna("").str.strip()
    persons = d.groupby("household").size()
    kids = d[d.age < 18].groupby("household").size()
    ref = d[d.relation.isin([1, 2])].copy()
    ref["kids"] = ref.household.map(kids).fillna(0)
    ref["records"] = ref.household.map(persons)
    print(f"{year} [{member}]: {len(d):,} person records, {len(ref):,} reference persons, "
          f"{ref.household.duplicated().sum()} duplicate households")
    return ref.drop(columns=["relation", "hhnum"])


def old_households(year, zname, layout):
    member, s = stream(zname)
    lines = [x for x in s.read().decode("latin-1").replace("\r", "").split("\n") if len(x) > 100]  # 1989 ends in a blank line
    raw = pd.DataFrame({k: [x[a - 1:a - 1 + w].strip() for x in lines] for k, (a, w, _) in layout.items()})
    if year == 1984:
        raw["faminc"] = raw.faminc.map(INC84)
    d = pd.DataFrame({k: pd.to_numeric(raw[k], errors="coerce") / 10 ** dec
                      for k, (_, _, dec) in layout.items() if k not in ("hhid", "hhnum")})
    d["household"] = raw.hhid + raw.get("hhnum", "")
    if year == 1984:
        d["grade"] = d.grade - 1          # basic-edit codes run 01 (none) to 19 (college 6+); the 1989 codes 00 to 18
        af = d.rectype == 4
        for k in ["race", "grade", "grade_done"]:
            d.loc[af, k] = d.loc[af, f"{k}_af"]
    # record types: 1 interviewed adult (14+ in 1984, 15+ later), 2-3 noninterviews, 4 Armed Forces, 5 children
    adults = d[d.rectype == 1]
    # the computer item is a household item on every adult's record; some adults are coded no response (9)
    yes, no = adults.computer.eq(1).groupby(adults.household).any(), adults.computer.eq(2).groupby(adults.household).any()
    computer = pd.Series(np.where(yes, 1, np.where(no, 2, np.nan)), index=yes.index)
    both = (yes & no).sum()
    people = d[d.rectype.isin([1, 4, 5])]
    kids = people[(people.rectype == 5) | (people.age < 18)].groupby("household").size()
    ref = people[people.rectype.isin([1, 4]) & people.relation.isin([1, 2]) & people.household.isin(computer.index)].copy()
    ref["computer"] = ref.household.map(computer)
    ref["kids"] = ref.household.map(kids).fillna(0)
    ref["records"] = ref.household.map(people.groupby("household").size())
    print(f"{year} [{member}]: {len(d):,} records, {len(computer):,} interviewed households, {len(ref):,} reference "
          f"persons ({(ref.rectype == 4).sum()} in the Armed Forces), {ref.household.duplicated().sum()} duplicates; "
          f"households answering both yes and no: {both}")
    if year == 1984:
        # no household weight on non-March files: the documentation (pdf p. 22) says to use the weight of the
        # principal person (the wife in married-couple households, otherwise the reference person)
        pp = adults[adults.principal == 1].set_index("household")
        print(f"  principal persons: {len(pp):,}, households with more than one: {pp.index.duplicated().sum()}")
        ref["hh_weight"] = ref.household.map(pp.weight)
        ref["persons"] = ref.records
        a = ref[ref.computer.notna()]
        for w in ["weight", "supp_weight"]:
            print(f"  ownership with the principal person's {w}: "
                  f"{100 * np.average(a.computer == 1, weights=a.household.map(pp[w])):.2f}%; "
                  f"with the reference person's {w}: {100 * np.average(a.computer == 1, weights=a[w]):.2f}%")
        ref["state"] = ref.state60.map(FIPS60)
        print(f"  adults whose age agrees between the two edits: {(adults.age == adults.age_basic).mean():.2%}; "
              f"reference persons who are the head in the basic edit: {ref.head_basic.isin([1, 2]).mean():.2%}")
    else:
        print(f"  household size (H-NUMPER) equals the number of person records in "
              f"{(ref.persons == ref.records).mean():.1%} of households")
    return ref.drop(columns=["rectype", "relation", "principal", "weight", "supp_weight", "state60", "age_basic",
                             "head_basic", "race_af", "grade_af", "grade_done_af"], errors="ignore")


frames = [new_households(y, *spec) for y, spec in FILES.items()] + [old_households(y, *spec) for y, spec in OLD.items()]
for f, y in zip(frames, list(FILES) + list(OLD)):
    f["year"] = y
    f["household"] = f"{y}-" + f.household     # keeps the identifier text, so leading zeros survive a CSV read
h = pd.concat(frames, ignore_index=True).sort_values(["year", "household"], ignore_index=True)

# ---------- codes made common across years ----------
h["owns"] = h.computer.map({1: 1.0, 2: 0.0})
old_scheme = h.year <= 1989     # 1984, 1989: highest grade attended and whether completed; 1993 on: highest degree
years_done = h.grade - (h.grade_done == 2)       # grade codes 00 (none), 01-08 elementary, 09-12 high school, 13-18 college
h["educ4"] = np.where(old_scheme,
                      pd.cut(years_done, [-np.inf, 11, 12, 15, 99], labels=["lt_hs", "hs", "some_col", "col"]).astype(object),
                      pd.cut(h.educ, [30, 38, 39, 42, 46], labels=["lt_hs", "hs", "some_col", "col"]).astype(object))
h["race3"] = h.race.map(lambda r: "white" if r == 1 else "black" if r == 2 else "other" if r >= 3 else np.nan)
h["hisp"] = np.where(h.year <= 1993, h.hispanic.map(lambda c: 1.0 if 1 <= c <= 7 else 0.0 if c == 8 else np.nan),
                     h.hispanic.map({1: 1.0, 2: 0.0}))
h["married"] = np.where(h.year <= 1993, h.marital.isin([1, 2]), h.marital == 1).astype(float)
h.loc[h.marital.isna(), "married"] = np.nan


def bracket(row):
    if pd.isna(row.faminc):
        return np.nan, np.nan
    cuts, i = (CUTS84, row.faminc) if row.year == 1984 else (CUTS, row.faminc) if row.year <= 1993 \
        else (CUTS03 if row.year == 2003 else CUTS, row.faminc - 1)
    if not 0 <= i < len(cuts):          # missing codes: 19 in 1989-93; -1 to -3 from 1997
        return np.nan, np.nan
    i = int(i)
    return cuts[i], cuts[i + 1] if i + 1 < len(cuts) else np.inf


h[["inc_lo", "inc_hi"]] = pd.DataFrame(h[["year", "faminc"]].apply(bracket, axis=1).tolist(), index=h.index)
cols = ["year", "household", "hh_weight", "owns", "computer", "n_computers", "faminc", "inc_lo", "inc_hi", "persons",
        "records", "kids", "age", "sex", "race", "race3", "hispanic", "hisp", "marital", "married", "educ", "grade",
        "grade_done", "educ4", "fam_type", "state"]
h[cols].to_csv(f"{OUT}/cps_households.csv.gz", index=False)

# ---------- checks ----------
pd.set_option("display.width", 200)
print("\nYear: households with positive weight, weighted number of households (millions), answered the computer "
      "question, weighted ownership (Census P23-208)")
for y, g in h.groupby("year"):
    g = g[g.hh_weight > 0]
    a = g[g.owns.notna()]
    print(f"  {y}: {len(g):6,} {g.hh_weight.sum() / 1e6:6.1f} {len(a) / len(g):6.1%} "
          f"{100 * np.average(a.owns, weights=a.hh_weight):5.1f}% ({CENSUS[y]})  "
          f"computer codes: {g.computer.value_counts(dropna=False).sort_index().to_dict()}")
a = h[(h.hh_weight > 0) & h.owns.notna()].copy()


def w(g, x):
    """Weighted mean of column x over the rows where it is known."""
    g = g[g[x].notna()]
    return np.average(g[x].astype(float), weights=g.hh_weight)


print("\nFamily income codes among answering households, by year (1984: 0-13 = codes 0-9, A-D)")
print(a.groupby("year").faminc.value_counts(dropna=False).unstack().fillna(0).astype(int).to_string())
a["top"] = np.where(a.inc_lo.notna(), np.isinf(a.inc_hi), np.nan)
a["white"], a["black"], a["with_kids"] = a.race3.eq("white"), a.race3.eq("black"), a.kids > 0
print("\nAnswering households by year: share with an income bracket; weighted shares in the open top bracket, white,")
print("black, Hispanic, married (spouse present), with children under 18; weighted mean household size and age; states")
rows = {y: {"bracket": g.inc_lo.notna().mean(), **{x: w(g, x) for x in ["top", "white", "black", "hisp", "married",
                                                                         "with_kids", "persons", "age"]},
            "states": g.state.nunique()} for y, g in a.groupby("year")}
print(pd.DataFrame(rows).T.round(3).to_string())
share = a.groupby(["year", "educ4"]).hh_weight.sum()
print("\nReference person's education, weighted shares (%); 1984-89 from grade attended/completed, 1993 on from degree")
print((100 * share / share.groupby(level=0).transform("sum")).unstack()[["lt_hs", "hs", "some_col", "col"]].round(1)
      .to_string())
print("\nWeighted ownership (%) by education of the reference person")
print((100 * a.groupby(["year", "educ4"])[["owns", "hh_weight"]].apply(lambda g: w(g, "owns"))).unstack()
      [["lt_hs", "hs", "some_col", "col"]].round(1).to_string())

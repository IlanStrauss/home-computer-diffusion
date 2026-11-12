"""Raw computer ownership by education of the household head, CE 1994-2013.

Weighted by w, the quarter-equalized first-stage weight. Prints each group's yearly
computer ownership rate (%) and the college minus high-school gap in percentage points.
Usage: python3 code/education_raw_gap.py > output/logs/education_raw_gap.txt
"""
import numpy as np
import pandas as pd

u = pd.read_csv("data/processed/ce_units.csv.gz", usecols=["year", "own", "w", "educ4"]).dropna(subset=["educ4"])
rate = (u.assign(ow=u.own * u.w).groupby(["year", "educ4"])[["ow", "w"]].sum()
        .pipe(lambda g: 100 * g.ow / g.w).unstack()[["lt_hs", "hs", "some_col", "col"]])
rate["col_minus_hs"] = rate.col - rate.hs
rate["col_minus_lt_hs"] = rate.col - rate.lt_hs
print(rate.round(1).to_string())

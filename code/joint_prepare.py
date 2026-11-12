"""Step 7, data for the joint model: every CE unit with the step-4 variables, 1994-2013.

Writes, sorted by year:
- output/results/joint_units.csv.gz: year index, ownership, x = corrected log long-run spending minus 10 (the
  step-4 regression calibration with the full-sample weights), geography group index, the step-4 traits centred at
  their pooled means, and the weight (mean 1 within each year);
- output/results/joint_years.csv: ell_t (BEA PC price), the trend, and each year's first row and number of units;
- output/results/joint_groups.csv: geography groups and their pooled shares (0 for units with no geography).

Run from the project folder: python3 code/joint_prepare.py > output/logs/joint_prepare.txt
"""
import numpy as np
import pandas as pd

import first_stage as FS
import hhpc as H

if __name__ == "__main__":
    FS.worker_init()
    u = FS.U
    spec = FS.SPECS["s4"]
    design = H.Design(u, "probit", geo=True, traits=spec["traits"], ratios=H.EDUC)
    parts = []
    for t in H.YEARS:
        d = u[u.year == t].copy()
        x, diag = FS.calibrate(d, d.w.to_numpy(), d.ln_s.to_numpy(), FS.covariates(d, design))
        d["x"] = x - H.CENTER
        parts.append(d)
        print(f"{t}: {len(d):,} units; calibration s2u {diag['s2u']:.3f}, v {diag['v']:.3f}, mean lambda "
              f"{diag['lam']:.3f}")
    d = pd.concat(parts)
    gidx = {g: i + 1 for i, g in enumerate(design.groups)}
    out = pd.DataFrame({"yr": d.year - 1993, "own": d.own, "x": d.x, "geo": d.geo.map(gidx), "w": d.w})
    for c in design.traits:
        out[c] = d[c] - design.zbar[c]
    out.to_csv(f"{H.RES}/joint_units.csv.gz", index=False)
    ell, _ = FS.year_ell(u)
    yrs = pd.DataFrame({"year": H.YEARS, "ell": ell.pc.to_numpy(), "trend": np.arange(20),
                        "start": np.r_[0, np.cumsum(out.groupby("yr").size().to_numpy())[:-1]] + 1,
                        "n": out.groupby("yr").size().to_numpy()})
    yrs.to_csv(f"{H.RES}/joint_years.csv", index=False)
    pd.DataFrame({"group": design.groups, "pi": design.pi}).to_csv(f"{H.RES}/joint_groups.csv", index=False)
    print(f"Wrote {len(out):,} units, {len(design.groups)} geography groups, {len(design.traits)} traits")

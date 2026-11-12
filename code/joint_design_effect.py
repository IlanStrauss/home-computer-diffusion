"""Step 7 check: how far the joint model's pseudo-likelihood understates the sampling error of the thresholds.

Each year's penalized fit on its own (the joint model's coordinate origin, code/joint_model.py) gives two standard
errors of tau_t, by the delta method:
- the pseudo-likelihood's own, from the inverse Hessian: units independent, weights scaled to mean 1 within the year,
  the joint model's priors on the geography and trait effects;
- a sandwich, inverse Hessian x (sum of squared weighted scores) x inverse Hessian, which allows for unequal weights
  but not for the survey's strata and clusters.
Both are set against the replicate (BRR) standard errors of the step-4 first stage. Then, from the joint model's
posterior (output/results/joint_model.csv), 1994-2012: the width of its 90% intervals for tau_t against the replicate
90% intervals, and the correlation of its move from each first-stage threshold with that threshold's distance from the
joint model's fitted line a + b ell_t + d (t - 1994) (posterior medians).

Run from the project folder: python3 code/joint_design_effect.py > output/logs/joint_design_effect.txt
"""
import numpy as np
import pandas as pd
from scipy.special import expit

from joint_model import design, year_fit

OUT = "output/results"
A, B = 0.07056, 1.5976    # Phi_approx: Pr = inv_logit(A eta^3 + B eta)
pd.set_option("display.width", 200)

if __name__ == "__main__":
    u = pd.read_csv(f"{OUT}/joint_units.csv.gz").sort_values("yr", kind="stable")
    grp = pd.read_csv(f"{OUT}/joint_groups.csv")
    traits = [c for c in u.columns if c not in ["yr", "own", "x", "geo", "w"]]
    fs = pd.read_csv(f"{OUT}/first_stage.csv")
    fs = fs[fs.spec == "s4"].sort_values("year").reset_index(drop=True)
    G, ref = len(grp), int(grp.pi.idxmax()) + 1
    X, Lp = design(u, traits, G, ref, grp.pi.to_numpy())
    P = X.shape[1]
    rows = []
    for t in range(20):
        k = (u.yr == t + 1).to_numpy()
        Xt, y, w = X[k], u.own.to_numpy(float)[k], u.w.to_numpy()[k]
        start = np.r_[fs.beta[t] * (10 - fs.tau[t]), fs.beta[t], np.zeros(P - 2)]
        coef = year_fit(Xt, y, w, Lp, start)
        eta, s = Xt @ coef, 2 * y - 1
        z = A * eta ** 3 + B * eta
        d1, d2 = 3 * A * eta ** 2 + B, 6 * A * eta
        q = expit(-s * z)
        score = -s * q * d1                                   # d/d eta of -log Pr(y)
        curv = q * (1 - q) * d1 ** 2 - s * q * d2              # d2/d eta2 of -log Pr(y)
        Hinv = np.linalg.inv(Xt.T @ (Xt * (w * curv)[:, None]) + Lp.T @ Lp)
        J = Xt.T @ (Xt * (w * score)[:, None] ** 2)
        c, beta = coef[0], coef[1]
        g = np.zeros(P)
        g[0], g[1] = -1 / beta, c / beta ** 2                  # tau = 10 - c / beta
        rows.append({"year": 1994 + t, "n": int(k.sum()), "tau": 10 - c / beta, "se_brr": fs.se_tau[t],
                     "se_model": np.sqrt(g @ Hinv @ g), "se_sandwich": np.sqrt(g @ Hinv @ J @ Hinv @ g)})
    r = pd.DataFrame(rows)
    r["ratio_model"] = r.se_model / r.se_brr
    r["ratio_sandwich"] = r.se_sandwich / r.se_brr
    print("Standard errors of tau_t: replicate (BRR, step-4 first stage), the pseudo-likelihood's own (inverse "
          "Hessian), and a sandwich that allows for unequal weights")
    print(r.round(3).to_string(index=False))
    e = r[r.year <= 2012]
    for c in ["ratio_model", "ratio_sandwich"]:
        print(f"{c}, 1994-2012: {e[c].min():.2f} to {e[c].max():.2f}, median {e[c].median():.2f}")

    j = pd.read_csv(f"{OUT}/joint_model.csv").set_index("variable")
    f = fs[fs.year <= 2012]
    tau_j = j.loc[[f"tau[{i}]" for i in range(1, len(f) + 1)]]
    width = (tau_j.q95 - tau_j.q05).to_numpy() / (2 * 1.645 * f.se_tau.to_numpy())
    line = j.loc["a", "median"] + j.loc["b", "median"] * f.ell_pc.to_numpy() + j.loc["d", "median"] * (f.year - 1994)
    move, dist = tau_j["median"].to_numpy() - f.tau.to_numpy(), line.to_numpy() - f.tau.to_numpy()
    print(f"\nJoint model, 1994-2012: 90% interval width for tau_t over the replicate 90% width, {width.min():.2f} to "
          f"{width.max():.2f}; correlation of (joint - first stage) with (joint line - first stage), "
          f"{np.corrcoef(move, dist)[0, 1]:.2f}")

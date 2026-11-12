"""Step 7 of the household PC model: one joint Bayesian probit on all CE units, 1994-2013 (code/joint_model.stan).

The second stage goes inside the probit: tau_t ~ Normal(a + b ell_t + d (t - 1994), omega^2), with yearly slopes
beta_t and yearly geography and trait effects as in the step-4 first stage. It should match the two-step estimates
(the step-4 first stage with the second stage, output/results/second_stage.csv).
Priors: a ~ N(10, 2); b ~ N(0.5, 1); d ~ N(0, 0.2); omega ~ Half-N(0, 0.2); beta_t ~ N+(0.8, 0.5); geography
effects ~ N(0, 1) and trait effects ~ N(0, 2.5), the ridge penalties of the first stage.
The sampler works in QR coordinates centred at each year's penalized fit on its own, which is the step-4 first stage
with this link (see the .stan file); this changes the geometry the sampler sees, not the posterior.
Sampling: CmdStan NUTS, 4 chains of 1,000 warm-up and 1,000 kept draws, 2 threads each, adapt_delta 0.9, seed
20261109; the chains start at the two-step solution, jittered. (Not at the joint mode: in a hierarchical model the
joint density grows without bound as omega goes to 0, and L-BFGS goes there.)
The weighted likelihood treats units as independent and ignores the survey's clustering, so its intervals are
narrower than the two-step ones, which use BRR standard errors: the comparison is about the point estimates.

Run from the project folder: python3 code/joint_model.py > output/logs/joint_model.txt
(python3 code/joint_model.py 0.05 is a pilot on every 20th unit, 1 for all units: one chain of 300 + 100 iterations,
4 threads; it writes nothing.)
Writes output/results/joint_model.csv.
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from scipy.optimize import minimize
from scipy.special import expit, log_expit

OUT = "output/results"
pd.set_option("display.width", 200)


def design(u, traits, G, ref, pi):
    """The design of the .stan file: [1, x, geography under sum_g pi_g gamma_g = 0, traits], and the square root of
    the prior precision of the geography and trait effects (gamma ~ N(0, 1), delta ~ N(0, 2.5))."""
    free = [g for g in range(1, G + 1) if g != ref]
    X = np.zeros((len(u), 1 + G + len(traits)))
    X[:, 0] = 1
    X[:, 1] = u.x
    geo = u.geo.to_numpy()
    for j, g in enumerate(free):
        X[geo == g, 2 + j] = 1
    X[np.ix_(geo == ref, range(2, G + 1))] = -pi[np.array(free) - 1] / pi[ref - 1]
    X[:, G + 1:] = u[traits].to_numpy()
    Lp = np.zeros((G + len(traits), X.shape[1]))
    Lp[np.arange(G - 1), 2 + np.arange(G - 1)] = 1
    Lp[G - 1, 2:G + 1] = -pi[np.array(free) - 1] / pi[ref - 1]
    Lp[G - 1 + 1 + np.arange(len(traits)), G + 1 + np.arange(len(traits))] = 1 / 2.5
    return X, Lp


def year_fit(X, y, w, Lp, start):
    """Penalized weighted fit of one year with the model's link, Pr = inv_logit(0.07056 eta^3 + 1.5976 eta)."""
    sg, LL = 2 * y - 1, Lp.T @ Lp

    def f(c):
        eta = X @ c
        u = 0.07056 * eta ** 3 + 1.5976 * eta
        g = -X.T @ (w * sg * expit(-sg * u) * (3 * 0.07056 * eta ** 2 + 1.5976)) + LL @ c
        return -w @ log_expit(sg * u) + 0.5 * c @ LL @ c, g

    r = minimize(f, start, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-7, "ftol": 1e-14})
    assert np.abs(f(r.x)[1]).max() < 1e-3, (r.message, np.abs(f(r.x)[1]).max())    # weights sum to ~9,000 a year
    return r.x

if __name__ == "__main__":
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    pilot = len(sys.argv) > 1
    u = pd.read_csv(f"{OUT}/joint_units.csv.gz")
    yrs = pd.read_csv(f"{OUT}/joint_years.csv")
    grp = pd.read_csv(f"{OUT}/joint_groups.csv")
    traits = [c for c in u.columns if c not in ["yr", "own", "x", "geo", "w"]]
    if frac < 1:
        u = u.iloc[::round(1 / frac)]
    u = u.sort_values("yr", kind="stable")
    n = u.groupby("yr").size().reindex(range(1, 21)).to_numpy()

    fs = pd.read_csv(f"{OUT}/first_stage.csv")
    fs = fs[fs.spec == "s4"].sort_values("year")
    two = pd.read_csv(f"{OUT}/second_stage.csv")
    two = two[(two.spec == "s4") & (two.concept == "pc") & (two.model == "independent errors")].iloc[0]

    G, ref = len(grp), int(grp.pi.idxmax()) + 1
    X, Lp = design(u, traits, G, ref, grp.pi.to_numpy())
    P = X.shape[1]
    coef0 = np.zeros((20, P))
    for t in range(20):
        k = (u.yr == t + 1).to_numpy()
        start = np.r_[fs.beta.iloc[t] * (10 - fs.tau.iloc[t]), fs.beta.iloc[t], np.zeros(P - 2)]
        coef0[t] = year_fit(X[k], u.own.to_numpy(float)[k], u.w.to_numpy()[k], Lp, start)
    tau_own = 10 - coef0[:, 0] / coef0[:, 1]
    print(f"Each year on its own (the coordinate origin): largest |tau_t - step-4 first stage| "
          f"{np.abs(tau_own - fs.tau.to_numpy()).max():.4f}; largest |beta_t - step 4| "
          f"{np.abs(coef0[:, 1] - fs.beta.to_numpy()).max():.4f}")
    data = {"N": len(u), "T": 20, "G": G, "K": len(traits), "start": np.r_[1, 1 + np.cumsum(n)[:-1]],
            "len": n, "geo": u.geo.to_numpy(), "x": u.x.to_numpy(), "Z": u[traits].to_numpy(),
            "y": u.own.to_numpy(float), "w": u.w.to_numpy(), "ell": yrs.ell.to_numpy(),
            "trend": yrs.trend.to_numpy(float), "pi": grp.pi.to_numpy(), "ref": ref, "coef0": coef0,
            "abd0": [two.a, two.b, two.d], "grainsize": 1}
    print(f"Joint model: {data['N']:,} units, 20 years, {data['G']} geography groups (constraint group: "
          f"{grp.group[data['ref'] - 1]}), {data['K']} traits; {20 * P + 4} parameters")
    rng = np.random.default_rng(20261109)
    chains = 1 if pilot else 4

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy("code/joint_model.stan", tmp)
        model = CmdStanModel(stan_file=os.path.join(tmp, "joint_model.stan"), cpp_options={"STAN_THREADS": "TRUE"})

        def init():
            return {"theta_H": rng.normal(0, 0.01, 3), "omega": two.omega * np.exp(rng.normal(0, 0.2)),
                    "theta": rng.normal(0, 0.3, (20, P))}

        t0 = time.time()
        fit = model.sample(data=data, chains=chains, parallel_chains=chains, threads_per_chain=4 if pilot else 2,
                           iter_warmup=300 if pilot else 1000, iter_sampling=100 if pilot else 1000, adapt_delta=0.9,
                           seed=20261109, inits=[init() for _ in range(chains)], output_dir=tmp, show_progress=False)
        secs = time.time() - t0
        mv = fit.method_variables()
        print(f"Sampling took {secs / 60:.1f} minutes; mean leapfrog steps per iteration "
              f"{mv['n_leapfrog__'].mean():.0f}; max treedepth {int(mv['treedepth__'].max())}; divergent transitions "
              f"{int(mv['divergent__'].sum())}; step size {mv['stepsize__'][0, :].round(4).tolist()}")
        s = fit.summary(percentiles=(5, 50, 95))
        print(f"Convergence over all {len(s) - 1} parameters: max R-hat {s.R_hat.max():.4f}; min bulk ESS "
              f"{s.ESS_bulk.min():.0f}; min tail ESS {s.ESS_tail.min():.0f}")
        if pilot:
            keep = ["a", "b", "d", "omega"] + [f"tau[{t}]" for t in range(1, 21)]
            print(s.loc[keep, ["Mean", "StdDev", "5%", "95%", "R_hat", "ESS_bulk"]].round(3))
            sys.exit(0)
        print(fit.diagnose())
        draws = fit.draws_pd(vars=["a", "b", "d", "omega"])

    keep = ["a", "b", "d", "omega"] + [f"tau[{t}]" for t in range(1, 21)] + [f"beta[{t}]" for t in range(1, 21)]
    s = s.loc[keep]
    print("\nJoint model: posterior medians [90% intervals]")
    for v in ["a", "b", "d", "omega"]:
        print(f"  {v:6s} {s.loc[v, '50%']:7.3f} [{s.loc[v, '5%']:7.3f}, {s.loc[v, '95%']:7.3f}]")
    print(f"  Posterior correlation of b and d: {np.corrcoef(draws.b, draws.d)[0, 1]:+.2f}")
    print(f"\nTwo-step (step 4, BEA PC price): a {two.a:.3f} [{two.a_lo:.3f}, {two.a_hi:.3f}]; b {two.b:.3f} "
          f"[{two.b_lo:.3f}, {two.b_hi:.3f}]; d {two.d:.4f} [{two.d_lo:.4f}, {two.d_hi:.4f}]; omega {two.omega:.3f}")
    cmp = pd.DataFrame({"year": fs.year.to_numpy(), "tau_joint": s.loc[keep[4:24], "50%"].to_numpy(),
                        "tau_two_step": fs.tau.to_numpy(), "se_two_step": fs.se_tau.to_numpy(),
                        "beta_joint": s.loc[keep[24:], "50%"].to_numpy(), "beta_two_step": fs.beta.to_numpy()})
    cmp["gap"] = cmp.tau_joint - cmp.tau_two_step
    print("\nThresholds and slopes: joint posterior medians against the step-4 first stage")
    print(cmp.round(3).to_string(index=False))
    print(f"Largest |gap| in tau_t: {cmp.gap.abs().max():.3f}; in first-stage standard errors: "
          f"{(cmp.gap.abs() / cmp.se_two_step).max():.2f}")
    out = s[["50%", "5%", "95%", "R_hat", "ESS_bulk", "ESS_tail"]].rename(
        columns={"50%": "median", "5%": "q05", "95%": "q95", "R_hat": "rhat", "ESS_bulk": "ess_bulk",
                 "ESS_tail": "ess_tail"})
    out.index.name = "variable"
    out.to_csv(f"{OUT}/joint_model.csv")
    print(f"\nWrote {OUT}/joint_model.csv")

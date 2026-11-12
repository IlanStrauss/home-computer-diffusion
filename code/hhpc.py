"""Shared pieces of the household PC model (CE 1994-2013): the data, geography groups, the yearly threshold fit, and
balanced repeated replication (BRR). The step scripts in this folder import it.

The yearly fit is a binary regression of ownership on log long-run spending, geography groups, and optional traits:
    Pr(own_i = 1) = F(alpha + beta (x_i - CENTER) + gamma_g(i) + z_i' delta)
- F is the link: the standard normal (probit), logistic (logit), or Student t with nu degrees of freedom (robit).
- Every geography group g has its own effect gamma_g, with a weak Normal(0, k^2) prior (a ridge penalty, so the fit is
  penalized maximum likelihood). The prior keeps a small group's effect finite when all its units own or none do.
  k = 1 for the probit and k = phi(0) / f(0) for other links, which puts the prior on the same scale.
- Traits z enter centred at their pooled 1994-2013 means, with a weaker Normal(0, (2.5 k)^2) prior, which only
  matters when a trait separates owners from non-owners in a small subsample or replicate.
- The threshold is read at fixed pooled group shares pi_g:  tau = CENTER - (alpha + sum_g pi_g gamma_g) / beta.
  With traits centred at their pooled means, tau is the threshold of a unit with the pooled average traits.
Standard errors: every fit is repeated with each of the 44 replicate weights; the variance is the mean squared
deviation of the replicate estimates from the full-sample estimate (BLS, for CE data before 2025).
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, t as student_t

OUT = "data/processed"
RES = "output/results"
REP = [f"wtrep{i:02d}" for i in range(1, 45)]
YEARS = list(range(1994, 2014))
CENTER = 10.0   # log spending is centred at 10 (about $22,000 a year) inside the fits
EPS = 1e-12


def _t_link(nu):
    dist = student_t(nu)
    return dist.cdf, dist.pdf, dist.ppf


LINKS = {"probit": (norm.cdf, norm.pdf, norm.ppf),
         "logit": (expit, lambda z: expit(z) * expit(-z), lambda q: np.log(q / (1 - q))),
         "robit7": _t_link(7), "robit4": _t_link(4), "robit2": _t_link(2)}
PRIOR_SD = {k: norm.pdf(0) / f(np.array([0.0]))[0] for k, (F, f, Q) in LINKS.items()}


def _derivs(link, y, eta):
    """Log-likelihood of each unit, its first derivative in eta, and the weight for the Hessian (minus the second
    derivative: observed for probit and logit, expected (Fisher scoring) for robit links)."""
    if link == "probit":
        l1, l0 = norm.logcdf(eta), norm.logcdf(-eta)
        lam1, lam0 = np.exp(norm.logpdf(eta) - l1), np.exp(norm.logpdf(eta) - l0)
        ll = y * l1 + (1 - y) * l0
        g = y * lam1 - (1 - y) * lam0
        h = y * lam1 * (eta + lam1) + (1 - y) * lam0 * (lam0 - eta)
        return ll, g, h
    if link == "logit":
        p = expit(eta)
        ll = y * np.log(p.clip(EPS)) + (1 - y) * np.log((1 - p).clip(EPS))
        return ll, y - p, p * (1 - p)
    F, f, _ = LINKS[link]
    p, d = F(eta).clip(EPS, 1 - EPS), f(eta)
    v = p * (1 - p)
    return y * np.log(p) + (1 - y) * np.log1p(-p), (y - p) * d / v, d ** 2 / v


def fit_binary(y, X, w, link="probit", pen=None, start=None, tol=1e-8, maxiter=200):
    """Penalized weighted maximum likelihood for Pr(y = 1) = F(X b): Newton steps with the observed Hessian for the
    probit and logit (both log-concave), Fisher scoring for robit links, with step halving.
    pen holds each coefficient's prior precision (0 = unpenalized). Rows with zero weight drop out."""
    k = w > 0
    y, X, w = y[k], X[k], w[k]
    pen = np.zeros(X.shape[1]) if pen is None else pen

    def objective(b):
        return np.sum(w * _derivs(link, y, X @ b)[0]) - 0.5 * np.sum(pen * b ** 2)

    if start is None:
        b = np.zeros(X.shape[1])
        b[0] = LINKS[link][2](np.average(y, weights=w))
    else:
        b = np.array(start, float)
    obj = objective(b)
    for it in range(maxiter):
        _, g, h = _derivs(link, y, X @ b)
        score = X.T @ (w * g) - pen * b
        info = (X * (w * h)[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(info, score)
        for _ in range(40):
            new = objective(b + step)
            if new >= obj - 1e-10 * abs(obj):
                break
            step = step / 2
        b, obj = b + step, new
        if np.max(np.abs(step)) < tol:
            return b
    raise RuntimeError(f"fit_binary did not converge ({link}, {maxiter} iterations)")


def brr(full, reps):
    """BRR standard error: root mean squared deviation of the replicate estimates from the full-sample estimate."""
    return np.sqrt(np.mean((np.asarray(reps) - np.asarray(full)) ** 2, axis=0))


def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def wquantile(x, w, q):
    x, w = np.asarray(x), np.asarray(w)
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return x[o][np.searchsorted(c, q)]


def rescale(w, year, quarter):
    """Give each quarter an equal share of its year's total weight, then scale to a mean of 1 within each year (as in
    build_ce_units.py; applied again within a subsample)."""
    share = w.groupby(year).transform("sum") / quarter.groupby(year).transform("nunique")
    w = w * share / w.groupby([year, quarter]).transform("sum")
    return w / w.groupby(year).transform("mean")


def load_units(path=f"{OUT}/ce_units.csv.gz"):
    """The CE analysis sample with each unit's log prices for the month its inventory was taken, and geography."""
    u = pd.read_csv(path)
    pm = pd.read_csv(f"{OUT}/prices_monthly.csv").rename(columns={"year": "ref_year", "month": "ref_month"})
    u = u.merge(pm, on=["ref_year", "ref_month"], how="left", validate="many_to_one")
    assert u.ln_pc.notna().all()
    return add_geography(u)


def add_geography(u):
    """Geography groups for the fits.
    - A state is its own group if, in every year 1994-2013, it has units with positive weight in all 44 replicates
      (otherwise its effect could not be estimated in some replicates).
    - Every other unit goes to a group for the rest of its region: units whose state is suppressed, and units in
      states that miss a year or a replicate. The region is the survey's REGION or, where that is blank, the
      state's usual region.
    - Units with neither state nor region form one more group. It is fitted, but left out of the threshold.
    Adds columns geo (the group label) and region_filled."""
    pos = (u.groupby(["state", "year"])[REP].sum() > 0).all(axis=1).unstack("year").reindex(columns=YEARS)
    own = set(pos.index[pos.eq(True).all(axis=1)])
    usual = u.dropna(subset=["state", "region"]).groupby("state").region.agg(lambda r: r.mode().iloc[0])
    u["region_filled"] = u.region.fillna(u.state.map(usual))
    u["geo"] = np.where(u.state.isin(own), "s" + u.state.fillna(0).astype(int).astype(str).str.zfill(2),
                        np.where(u.region_filled.notna(), "rest_r" + u.region_filled.fillna(0).astype(int).astype(str),
                                 "none"))
    groups = sorted(set(u.geo) - {"none"})
    tot = u[u.geo != "none"].groupby(["geo", "year"])[REP].sum()
    assert (tot > 0).all(axis=None) and len(tot) == len(groups) * len(YEARS), "a threshold group misses a replicate"
    return u


TRAITS = ["age_lt25", "age_25_34", "age_45_54", "age_55_64", "age_65_74", "age_75p", "kids1", "kids2", "kids3p",
          "adults2", "adults3p", "married", "black", "other_race", "hispanic", "student"]
EDUC = ["ed_lt_hs", "ed_some_col", "ed_col"]


def add_traits(u):
    """Household traits for steps 3-5 as 0/1 columns, with reference groups in brackets: age of the reference person
    under 25, 25-34, 45-54, 55-64, 65-74, 75 or over [35-44]; 1, 2, or 3+ children under 18 [none]; 2 or 3+ adults
    [one]; married; black, other race [white]; Hispanic; student housing (CUTENURE 6); and education of the
    reference person: less than high school, some college, college degree [high school]."""
    age = pd.cut(u.age, [0, 24, 34, 44, 54, 64, 74, 200], labels=["lt25", "25_34", "35_44", "45_54", "55_64",
                                                               "65_74", "75p"])
    for a in ["lt25", "25_34", "45_54", "55_64", "65_74", "75p"]:
        u[f"age_{a}"] = (age == a).astype(float)
    adults = (u.fam_size - u.kids).clip(lower=1)
    new = {"kids1": u.kids == 1, "kids2": u.kids == 2, "kids3p": u.kids >= 3, "adults2": adults == 2,
           "adults3p": adults >= 3, "black": u.race3 == "black", "other_race": u.race3 == "other",
           "hispanic": u.hisp == 1, "student": u.tenure == 6, "ed_lt_hs": u.educ4 == "lt_hs",
           "ed_some_col": u.educ4 == "some_col", "ed_col": u.educ4 == "col"}
    for k, v in new.items():
        u[k] = v.astype(float)
    u["married"] = u.married.astype(float)
    assert u[TRAITS + EDUC].notna().all().all()
    return u


def group_shares(u, groups):
    """Pooled shares of the threshold groups: each year's weighted share, averaged over the 20 years."""
    d = u[u.geo.isin(groups)]
    yearly = d.groupby(["year", "geo"]).w.sum() / d.groupby("year").w.sum()
    return yearly.unstack("geo").fillna(0).mean().reindex(groups).to_numpy()


class Design:
    """Builds the design matrix of a yearly fit and reads the threshold off the coefficients.
    Columns: intercept, x - CENTER, one dummy per geography group (all groups, every year), the traits (centred at
    their pooled means), then any extra columns (already centred, such as a control-function residual)."""

    def __init__(self, u, link="probit", geo=True, traits=None, ratios=None):
        self.link, self.geo, self.traits = link, geo, list(traits or [])
        self.groups = sorted(set(u.geo) - {"none"}) + (["none"] if (u.geo == "none").any() else [])
        self.avg_groups = [g for g in self.groups if g != "none"]
        pi = group_shares(u, self.avg_groups)
        self.pi = np.append(pi, [0.0] * (len(self.groups) - len(self.avg_groups)))
        # traits are centred at their pooled means: the average over years of each year's weighted mean
        self.zbar = (u.groupby("year").apply(lambda d: pd.Series({c: wmean(d[c], d.w) for c in self.traits}),
                                             include_groups=False).mean() if self.traits else pd.Series(dtype=float))
        self.ratios = list(ratios or [])     # traits whose coefficient is also reported as gamma_z / beta

    @property
    def k_geo(self):
        return len(self.groups) if self.geo else 0

    def matrix(self, d, x, extra=None):
        cols = [np.ones(len(d)), np.asarray(x) - CENTER]
        if self.geo:
            cols += [(d.geo == g).to_numpy(float) for g in self.groups]
        cols += [d[c].to_numpy(float) - self.zbar[c] for c in self.traits]
        X = np.column_stack(cols + ([np.asarray(extra).reshape(len(d), -1)] if extra is not None else []))
        pen = np.zeros(X.shape[1])
        pen[2:2 + self.k_geo] = 1 / PRIOR_SD[self.link] ** 2
        pen[2 + self.k_geo:2 + self.k_geo + len(self.traits)] = 1 / (2.5 * PRIOR_SD[self.link]) ** 2
        return X, pen

    def read(self, b):
        """tau, beta, and the requested trait ratios gamma_z / beta from one coefficient vector."""
        k = self.k_geo
        level = b[0] + (b[2:2 + k] @ self.pi if self.geo else 0.0)
        out = {"tau": CENTER - level / b[1], "beta": b[1]}
        for c in self.ratios:
            out[f"theta_{c}"] = b[2 + k + self.traits.index(c)] / b[1]
        return out


def year_prices(u, concepts, weights="w"):
    """Each year's weighted mean log price under each concept (NaN if any unit lacks it)."""
    return pd.DataFrame({c: u.groupby("year").apply(
        lambda d, c=c: wmean(d[f"ln_{c}"], d[weights]) if d[f"ln_{c}"].notna().all() else np.nan,
        include_groups=False) for c in concepts})

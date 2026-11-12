"""LaTeX tables for the results write-up (writeup/household_pc_results.tex), from the saved estimates.

Every table shows posterior medians (second stage, switch test, joint model) or point estimates (first stage) with 90% intervals:
Bayesian intervals from the 5th and 95th posterior percentiles, and +/- 1.645 BRR standard errors for first-stage
quantities. A table whose inputs are missing is skipped with a message.

Run from the project folder: python3 code/tables.py
Writes writeup/tables/*.tex.
"""
import os

import numpy as np
import pandas as pd

import hhpc as H

TAB = "writeup/tables"
P = "output/results"
Z90 = 1.645
STEP = {"s1": "Baseline", "s2": "+ corrected spending", "s3": "+ household traits", "s4": "+ education (extended)"}
CHECK = {"s1_nogeo": "Baseline, no geography", "s1_logit": "Baseline, logit", "s1_robit2": "Baseline, robit (2 df)",
         "s1_int2": "Baseline, households entering at interview 2", "s1_nostudent": "Baseline, no student housing",
         "s1_trim": "Baseline, spending trimmed at 0.5\\% and 99.5\\%",
         "s1_incl": "Baseline, spending with insurance, pensions, contributions",
         "s1_veh": "Baseline, spending with vehicles",
         "s3_obs": "+ household traits, observed spending", "s3_robit7": "+ household traits, robit (7 df)",
         "s4_logit": "Extended, logit", "s4_int2": "Extended, households entering at interview 2",
         "s4_incl": "Extended, spending with insurance, pensions, contributions",
         "s4_veh": "Extended, spending with vehicles", "s4_cr": "Extended, complete income reporters",
         "s4_inc": "Extended, current income", "s4_cf": "Extended, control function for income"}
CONCEPT = {"pc": "BEA PCE, personal computers (main)", "info": "BEA PCE, information processing equipment",
           "business": "BEA investment, computers", "cpi_pc": "BLS CPI-U, personal computers",
           "paid_p50": "CE price paid, median", "paid_p90": "CE price paid, 90th percentile"}


def n(x, nd=2, sign=False):
    """A number for LaTeX, with a proper minus sign."""
    if pd.isna(x):
        return ""
    if abs(x) < 0.5 * 10 ** -nd:
        return f"{0:.{nd}f}"    # no signed or negative zero
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$")


def ci(m, lo, hi, nd=2, sign=False):
    return f"{n(m, nd, sign)} [{n(lo, nd, sign)}, {n(hi, nd, sign)}]"


def have(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"  skipped: missing {', '.join(missing)}")
    return not missing


def write(name, body):
    with open(f"{TAB}/{name}", "w") as f:
        f.write(body)
    print(f"  wrote {TAB}/{name}")


def tabular(cols, header, rows, size="\\footnotesize"):
    lines = [size, f"\\begin{{tabular}}{{{cols}}}", "\\toprule", header + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(r if r.startswith("\\") else r + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def second_stage():
    print("tab_second_stage.tex, tab_concepts.tex, tab_checks.tex")
    if not have(f"{P}/second_stage.csv", f"{P}/first_stage.csv"):
        return
    r = pd.read_csv(f"{P}/second_stage.csv")
    pc = r[r.concept == "pc"]
    fs = pd.read_csv(f"{P}/first_stage.csv").set_index(["spec", "year"])
    dell = fs.loc[("s1", 2012), "ell_pc"] - fs.loc[("s1", 1994), "ell_pc"]    # 2013 rests on one quarter
    rows = []
    for model, specs in [("independent errors", list(STEP)), ("correlated errors", ["s1", "s3", "s4"])]:
        if model == "correlated errors":
            rows.append("\\midrule")
            rows.append("\\multicolumn{8}{l}{\\textit{Sampling errors correlated across years (covariance check)}} \\\\")
        for s in specs:
            x = pc[(pc.spec == s) & (pc.model == model)].iloc[0]
            span = 2012 - 1994
            price = sorted([x.b * dell, x.b_lo * dell, x.b_hi * dell])
            drift = sorted([x.d * span, x.d_lo * span, x.d_hi * span])
            dtau = fs.loc[(s, 2012), "tau"] - fs.loc[(s, 1994), "tau"]
            rows.append(" & ".join([STEP[s], n(x.a, 2), n(x.b, 2), n(x.d, 3, True), n(x.omega, 3), n(dtau, 2),
                                    n(price[1], 2), n(drift[1], 2, True)]))
            rows.append(" & ".join(["", f"[{n(x.a_lo)}, {n(x.a_hi)}]", f"[{n(x.b_lo)}, {n(x.b_hi)}]",
                                    f"[{n(x.d_lo, 3, True)}, {n(x.d_hi, 3, True)}]",
                                    f"[{n(x.omega_lo, 3)}, {n(x.omega_hi, 3)}]", "",
                                    f"[{n(price[0])}, {n(price[2])}]", f"[{n(drift[0], 2, True)}, {n(drift[2], 2, True)}]"]))
    header = ("Specification & $a$ & $b$ & $d$ & $\\omega$ & $\\hat\\tau_{2012}-\\hat\\tau_{1994}$ & "
              "$b\\,(\\ell_{2012}-\\ell_{1994})$ & $18\\,d$")
    write("tab_second_stage.tex", tabular("lccccccc", header, rows))

    rows = []
    for k, lab in CONCEPT.items():
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{lab}}}}} \\\\")
        for s in STEP:
            x = r[(r.spec == s) & (r.concept == k) & (r.model == "independent errors")].iloc[0]
            dl = fs.loc[(s, 2012), f"ell_{k}"] - fs.loc[(s, int(x["first"])), f"ell_{k}"]
            rows.append(" & ".join([STEP[s], f"{int(x['first'])}--{int(x['last'])}", n(dl, 2),
                                    ci(x.b, x.b_lo, x.b_hi), ci(x.d, x.d_lo, x.d_hi, 3, True), n(x.omega, 3)]))
        if k != "paid_p90":
            rows.append("\\addlinespace")
    header = ("Specification & Years & $\\Delta\\ell$ to 2012 & $b$ & $d$ & $\\omega$")
    write("tab_concepts.tex", tabular("lccccc", header, rows))

    rows = []
    for s, lab in CHECK.items():
        x = pc[(pc.spec == s) & (pc.model == "independent errors")].iloc[0]
        rows.append(" & ".join([lab, ci(x.b, x.b_lo, x.b_hi), ci(x.d, x.d_lo, x.d_hi, 3, True), n(x.omega, 3),
                                n(fs.loc[(s, 2012), "tau"] - fs.loc[(s, 1994), "tau"], 2), n(x.max_rhat, 3)]))
    header = ("Check & $b$ & $d$ & $\\omega$ & $\\hat\\tau_{2012}-\\hat\\tau_{1994}$ & max $\\hat R$")
    write("tab_checks.tex", tabular("lccccc", header, rows))


def thresholds():
    print("tab_thresholds.tex")
    if not have(f"{P}/first_stage.csv"):
        return
    fs = pd.read_csv(f"{P}/first_stage.csv")
    base = fs[fs.spec == "s1"].set_index("year")
    rows = []
    for t in H.YEARS:
        cells = [str(t), f"{int(base.n[t]):,}", n(base.own_pct[t], 1), n(base.ell_pc[t], 2)]
        for s in STEP:
            x = fs[(fs.spec == s) & (fs.year == t)].iloc[0]
            cells.append(f"{n(x.tau, 2)} ({n(x.se_tau, 2)})")
            cells.append(n(x.beta, 2))
        rows.append(" & ".join(cells))
    header = ("Year & Households & Own (\\%) & $\\ell_t$ & \\multicolumn{2}{c}{Baseline} & \\multicolumn{2}{c}{+ corrected spending} & "
              "\\multicolumn{2}{c}{+ household traits} & \\multicolumn{2}{c}{+ education (extended)} \\\\\n"
              "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}\n"
              " & & & & $\\hat\\tau_t$ (SE) & $\\hat\\beta_t$ & $\\hat\\tau_t$ (SE) & $\\hat\\beta_t$ & "
              "$\\hat\\tau_t$ (SE) & $\\hat\\beta_t$ & $\\hat\\tau_t$ (SE) & $\\hat\\beta_t$")
    write("tab_thresholds.tex", tabular("rrrrcccccccc", header, rows, size="\\scriptsize"))


def education():
    print("tab_education.tex")
    if not have(f"{P}/first_stage.csv", f"{P}/first_stage_replicates.csv.gz"):
        return
    fs = pd.read_csv(f"{P}/first_stage.csv")
    rp = pd.read_csv(f"{P}/first_stage_replicates.csv.gz")
    yrs = list(range(1994, 2013))
    specs = {"s4": "Corrected spending, all households", "s4_logit": "Corrected spending, logit",
             "s4_cr": "Corrected spending, complete income reporters",
             "s4_cf": "Spending with a control function for income", "s4_inc": "Current income"}
    rows = []
    for s, lab in specs.items():
        f = fs[(fs.spec == s) & fs.year.isin(yrs)]
        rr = rp[(rp.spec == s) & rp.year.isin(yrs)]
        cells = [lab]
        for c in ["theta_ed_lt_hs", "theta_ed_some_col", "theta_ed_col"]:
            full = f[c].mean()
            se = H.brr(full, rr.groupby("rep")[c].mean().to_numpy())
            cells.append(ci(full, full - Z90 * se, full + Z90 * se, 2, True))
        cells.append(n(f.beta.mean(), 2))
        rows.append(" & ".join(cells))
    header = ("Spending or income measure & Less than high school & Some college & College degree & "
              "Mean $\\hat\\beta_t$")
    write("tab_education.tex", tabular("lcccc", header, rows))


def cps():
    print("tab_cps.tex, tab_validation.tex")
    if not have(f"{P}/cps_thresholds.csv", f"{P}/cps_validation.csv"):
        return
    d = pd.read_csv(f"{P}/cps_thresholds.csv")
    c = d[d.source == "CPS"]
    rows = [" & ".join([str(int(x.year)), n(x.date, 2), f"{int(x.n):,}", n(x.own_pct, 1), n(x.tau, 2),
                        n(x.se_tau, 3), n(x.beta, 2), n(x.ell_pc, 2)]) for x in c.itertuples()]
    header = "Year & Date & Households & Own (\\%) & $\\hat\\tau$ & SE & $\\hat\\beta$ & $\\ell$"
    write("tab_cps.tex", tabular("rcrrcccc", header, rows))
    v = pd.read_csv(f"{P}/cps_validation.csv")
    rows = [" & ".join([str(int(x.year)), n(x.cps_tau, 2), n(x.ce_tau, 2), n(x.gap, 2, True), n(x.gap_in_se, 1, True),
                        n(x.ce_units_imputed_tau, 2), n(x.ce_units_corrected_tau, 2), n(x.ce_units_observed_tau, 2),
                        n(x.ce_units_imputed_beta, 2), n(x.ce_units_corrected_beta, 2)]) for x in v.itertuples()]
    header = ("& \\multicolumn{4}{c}{CPS against CE} & \\multicolumn{5}{c}{CE households, same year} \\\\\n"
              "\\cmidrule(lr){2-5}\\cmidrule(lr){6-10}\n"
              "Year & CPS $\\hat\\tau$ & CE $\\hat\\tau$ & Gap & Gap/SE & $\\hat\\tau$ imputed & $\\hat\\tau$ corrected & "
              "$\\hat\\tau$ observed & $\\hat\\beta$ imputed & $\\hat\\beta$ corrected")
    write("tab_validation.tex", tabular("rccccccccc", header, rows))


def switch():
    print("tab_switch.tex")
    if not have(f"{P}/cps_switch.csv"):
        return
    r = pd.read_csv(f"{P}/cps_switch.csv")
    if "dd" not in r.columns:
        print("  skipped: cps_switch.csv predates the d2 - d1 columns")
        return
    lab = {"one": "One trend", "20": "Switch at 20\\%", "30": "Switch at 30\\%"}
    rows = []
    for conc, title in [("pc", "CPS and CE spending thresholds, with a CPS offset (the approved model)"),
                        ("pc, CPS alone", "CPS spending thresholds alone (check)")]:
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for x in r[r.concept == conc].itertuples():
            dd = "" if x.model == "one" else ci(x.dd, x.dd_lo, x.dd_hi, 3, True)
            d1 = ci(x.d1, x.d1_lo, x.d1_hi, 3, True)
            d2 = "" if x.model == "one" else ci(x.d2, x.d2_lo, x.d2_hi, 3, True)
            o = "" if pd.isna(x.o) else ci(x.o, x.o_lo, x.o_hi, 2, True)
            elpd = "0 (best)" if abs(x.elpd_diff) < 1e-9 else f"{n(x.elpd_diff, 1, True)} ({n(x.elpd_diff_se, 1)})"
            rows.append(" & ".join([lab[str(x.model)], ci(x.b, x.b_lo, x.b_hi), d1, d2, dd, o, n(x.omega, 3), elpd]))
        if conc == "pc":
            rows.append("\\addlinespace")
    header = ("Model & $b$ & $d_1$ (one trend: $d$) & $d_2$ & $d_2-d_1$ & $o$ & $\\omega$ & "
              "$\\Delta\\widehat{\\mathrm{elpd}}$ (SE)")
    write("tab_switch.tex", tabular("lccccccc", header, rows, size="\\scriptsize"))


def links():
    print("tab_links.tex")
    if not have(f"{P}/link_choice_cv.csv"):
        return
    r = pd.read_csv(f"{P}/link_choice_cv.csv")
    names = {"probit": "Probit", "logit": "Logit", "robit7": "Robit, 7 df", "robit4": "Robit, 4 df",
             "robit2": "Robit, 2 df"}
    ctrl = list(dict.fromkeys(r.controls))
    rows = []
    for k, lab in names.items():
        cells = [lab]
        for c in ctrl:
            x = r[(r.controls == c) & (r.link == k)].iloc[0]
            cells.append("0" if k == "probit" else f"{n(x.diff_vs_probit, 1, True)} ({n(x.se_diff, 1)})")
        rows.append(" & ".join(cells))
    header = "Link & Baseline (geography) & + household traits & + education (extended)"
    write("tab_links.tex", tabular("lccc", header, rows))


def joint():
    print("tab_joint.tex")
    if not have(f"{P}/joint_model.csv", f"{P}/second_stage.csv"):
        return
    j = pd.read_csv(f"{P}/joint_model.csv").set_index("variable")
    two = pd.read_csv(f"{P}/second_stage.csv")
    two = two[(two.spec == "s4") & (two.concept == "pc") & (two.model == "independent errors")].iloc[0]
    rows = []
    for v, lab, nd in [("a", "$a$", 2), ("b", "$b$", 2), ("d", "$d$", 3), ("omega", "$\\omega$", 3)]:
        rows.append(" & ".join([lab, ci(j.loc[v, "median"], j.loc[v, "q05"], j.loc[v, "q95"], nd, v == "d"),
                                ci(two[v], two[f"{v}_lo"], two[f"{v}_hi"], nd, v == "d")]))
    header = "Parameter & Joint model & Two-stage (extended regression)"
    write("tab_joint.tex", tabular("lcc", header, rows))


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    for f in [second_stage, thresholds, education, cps, switch, links, joint]:
        f()

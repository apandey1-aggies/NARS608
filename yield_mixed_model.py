"""
Linear mixed model for cumulative yield (Python port of the R / lme4 / lmerTest analysis).

Model
-----
    Y_ijk = b0 + b_j(Crop) + b_i(AmarVar within Amaranth) + u_k(Irrigation) + e_ijk

    Y_ijk = cumulative yield (lbs) of one plot
    b0    = intercept (Eggplant, reference crop)
    b_j   = fixed effect of crop (j = 1-5)
    b_i   = fixed effect of Amaranth variety (reference = Golden giant)
    u_k   = random effect of irrigation (k = 1-3),     u_k  ~ N(0, s2_k)
    e_ijk = residual error,                            e_ijk ~ N(0, s2_e)

R equivalent
------------
    lmer(CumYield ~ Crop + AmarVar + (1 | Irrigation), data = cum)

Usage
-----
    pip install pandas numpy scipy statsmodels openpyxl matplotlib
    python yield_mixed_model.py "C:/Users/ASUS/OneDrive/Desktop/Datafinal.xlsx"

Notes on differences from R
---------------------------
* statsmodels' MixedLM fits crossed random effects by putting every plot in a
  single group and declaring each random term as a variance component.  The
  fit is REML, as in lmer().
* lmerTest reports Satterthwaite degrees of freedom.  statsmodels has no
  Satterthwaite option, so t / F tests here use the residual degrees of
  freedom (n - number of fixed parameters).  When the irrigation variance
  components are estimated at 0 (as in your results) the two are identical.
* ranova() is reproduced by refitting the model without each random term and
  running a REML likelihood-ratio test (chi-square, 1 df).
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # write plots to files; no display needed
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

AMARANTH_VARIETIES = ["Golden giant", "Green callaloo"]
REF_CROP = "Egg plant"
IRRIGATION_LABELS = {"T1": "Control", "T2": "Moderate", "T3": "High"}
IRRIGATION_ORDER = ["Control", "Moderate", "High"]


# ---------------------------------------------------------------- data ----
def load_data(path, sheet="Sheet1"):
    data = pd.read_excel(path, sheet_name=sheet)
    data["Variety"] = data["Variety"].astype(str).str.strip()

    # wide (T1, T2, T3) -> long
    id_cols = [c for c in data.columns if c not in ("T1", "T2", "T3")]
    long = data.melt(id_vars=id_cols, value_vars=["T1", "T2", "T3"],
                     var_name="Irrigation", value_name="Yield")

    # cumulative yield per plot; NA if every harvest of that plot is missing
    cum = (long.groupby(["replication", "Variety", "Irrigation"], as_index=False)["Yield"]
               .sum(min_count=1)
               .rename(columns={"Yield": "CumYield"})
               .dropna(subset=["CumYield"])
               .reset_index(drop=True))

    crop = np.where(cum["Variety"].isin(AMARANTH_VARIETIES), "Amaranth", cum["Variety"])
    crop_levels = [REF_CROP] + sorted(set(crop) - {REF_CROP})
    cum["Crop"] = pd.Categorical(crop, categories=crop_levels)
    cum["AmarVar"] = (cum["Variety"] == "Green callaloo").astype(int)
    cum["Irrigation"] = pd.Categorical(cum["Irrigation"].map(IRRIGATION_LABELS),
                                       categories=IRRIGATION_ORDER)
    return cum


def pretty_term(name):
    """'C(Crop)[T.Amaranth]' -> 'CropAmaranth' (R-style coefficient names)."""
    if name == "Intercept":
        return "(Intercept)"
    if name.startswith("C(") and "[T." in name:
        var = name[2:name.index(")")]
        level = name[name.index("[T.") + 3:-1]
        return f"{var}{level}"
    return name


# ------------------------------------------------------- mixed model -----
FIXED_FORMULA = "CumYield ~ C(Crop) + AmarVar"
VC_TERMS = {"Irrigation": "0 + C(Irrigation)"}


def fit_mixed(cum, vc_terms=VC_TERMS):
    """Crossed random intercepts via variance components in one dummy group.

    A single optimizer can stop at a poor local optimum (especially when a
    variance is near 0), so the model is fitted with several and the fit with
    the highest REML log-likelihood is kept.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # boundary (singular) fit warnings
        model = smf.mixedlm(FIXED_FORMULA, data=cum,
                            groups=np.ones(len(cum)),
                            re_formula="0",
                            vc_formula=vc_terms)
        best = None
        for method in ("lbfgs", "bfgs", "powell", "nm"):
            try:
                res = model.fit(reml=True, method=method)
            except Exception:
                continue
            if np.isfinite(res.llf) and (best is None or res.llf > best.llf):
                best = res
        return best


def fixed_effects_table(res, df_resid):
    est = res.fe_params
    se = res.bse_fe
    t = est / se
    p = 2 * stats.t.sf(np.abs(t), df_resid)
    return pd.DataFrame({
        "Term": [pretty_term(n) for n in est.index],
        "Estimate": est.values.round(3),
        "SE": se.values.round(3),
        "df": df_resid,
        "t_value": t.values.round(3),
        "p_value": [float(f"{v:.3g}") for v in p],
    })


def type3_ftests(res, df_resid):
    """Wald F-tests per fixed term (equivalent of anova(model_rand))."""
    names = list(res.fe_params.index)
    beta = res.fe_params.values
    vcov = res.cov_params().loc[names, names].values
    terms = {"Crop": [i for i, n in enumerate(names) if n.startswith("C(Crop)")],
             "AmarVar": [names.index("AmarVar")]}
    rows = []
    for term, idx in terms.items():
        L = np.zeros((len(idx), len(names)))
        for r, c in enumerate(idx):
            L[r, c] = 1.0
        Lb = L @ beta
        q = len(idx)
        F = float(Lb @ np.linalg.solve(L @ vcov @ L.T, Lb)) / q
        rows.append({"Term": term, "NumDF": q, "DenDF": df_resid,
                     "F_value": round(F, 3),
                     "p_value": float(f"{stats.f.sf(F, q, df_resid):.3g}")})
    return pd.DataFrame(rows)


def variance_table(res):
    vc = pd.Series(res.vcomp, index=res.model.exog_vc.names)
    vc = pd.concat([vc, pd.Series({"Residual": res.scale})])
    return pd.DataFrame({"Component": vc.index,
                         "Variance": vc.values.round(3),
                         "Std_Dev": np.sqrt(vc.values).round(3)})


def ols_reml_llf(cum):
    """REML log-likelihood of the fixed-effects-only model (no random terms)."""
    ols = smf.ols(FIXED_FORMULA, data=cum).fit()
    X = ols.model.exog
    n, k = X.shape
    s2 = ols.ssr / (n - k)
    return -0.5 * ((n - k) * np.log(2 * np.pi * s2) + (n - k)
                   + np.linalg.slogdet(X.T @ X)[1])


def ranova(cum, full):
    """REML likelihood-ratio test for dropping each random term."""
    rows = [{"Model": "<none>", "npar": len(full.params), "logLik": round(full.llf, 3),
             "LRT": np.nan, "Df": np.nan, "p_value": np.nan}]
    for term in VC_TERMS:
        reduced_terms = {k: v for k, v in VC_TERMS.items() if k != term}
        if reduced_terms:
            red_llf = fit_mixed(cum, reduced_terms).llf
        else:
            # no random effects left: REML log-likelihood of the plain linear
            # model, as logLik(lm(...), REML = TRUE) in R
            red_llf = ols_reml_llf(cum)
        lrt = max(2 * (full.llf - red_llf), 0.0)
        rows.append({"Model": f"-(1 | {term})", "npar": len(full.params) - 1,
                     "logLik": round(red_llf, 3), "LRT": round(lrt, 4), "Df": 1,
                     "p_value": float(f"{stats.chi2.sf(lrt, 1):.3g}")})
    return pd.DataFrame(rows)


def blups(res):
    """Random-effect predictions (equivalent of ranef(model_rand))."""
    re = pd.Series(res.random_effects[1.0])
    out = {}
    for term in VC_TERMS:
        sub = re[re.index.str.startswith(term + "[")]
        # index looks like 'Irrigation[C(Irrigation)[Control]]'
        sub.index = [s.split("[")[-1].rstrip("]") for s in sub.index]
        out[term] = sub.round(4).to_frame("(Intercept)")
    return out


# ------------------------------------------------------------- main ------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="Datafinal.xlsx", help="Excel data file")
    ap.add_argument("--sheet", default="Sheet1")
    args = ap.parse_args()

    pd.set_option("display.width", 120)
    cum = load_data(args.path, args.sheet)

    print("\n==================== PLOT COUNTS (Crop x Irrigation) ====================")
    print(pd.crosstab(cum["Crop"], cum["Irrigation"]))

    # ---------------- Fixed-effects linear model (lm + car::Anova type 2) ----
    print("\n==================== FIXED-EFFECTS MODEL (lm) ====================")
    ols = smf.ols("CumYield ~ C(Crop) + AmarVar + C(Irrigation) + C(Crop):C(Irrigation)",
                  data=cum).fit()
    print(ols.summary())
    print("\n---- ANOVA (Type II) ----")
    print(sm.stats.anova_lm(ols, typ=2))

    # ---------------- Mixed model ----------------
    res = fit_mixed(cum)
    n, p = len(cum), res.model.k_fe
    df_resid = n - p

    print("\n==================== MODEL EQUATION ====================")
    print("Y_ijk = b0 + b_j(Crop) + b_i(AmarVar within Amaranth) + u_k(Irrigation)"
          " + e_ijk\n")
    print("Dictionary:")
    print("  Y_ijk   = cumulative yield (lbs) of one plot")
    print("  b0      = intercept (Eggplant, reference crop)")
    print("  b_j     = fixed effect of crop (j = 1-5)")
    print("  b_i     = fixed effect of Amaranth variety (reference = Golden giant)")
    print("  u_k     = random effect of irrigation (k = 1-3), u_k ~ N(0, s2_k)")
    print("  e_ijk   = residual error, e_ijk ~ N(0, s2_e)")

    print("\n==================== MIXED MODEL SUMMARY ====================")
    print(res.summary())

    print("\n==================== FIXED EFFECTS ====================")
    print(fixed_effects_table(res, df_resid).to_string(index=False))
    print(f"(t-tests use {df_resid} residual df)")

    print("\n---- Overall F-tests (Type III) ----")
    print(type3_ftests(res, df_resid).to_string(index=False))

    print("\n==================== RANDOM EFFECTS ====================")
    print(variance_table(res).to_string(index=False))

    print("\n---- Likelihood ratio tests for random effects ----")
    print(ranova(cum, res).to_string(index=False))

    # ---------------- Model error: RMSE and %CV ----------------
    print("\n==================== MODEL ERROR ====================")
    cum["Fitted"] = res.fittedvalues  # includes the random effects, like fitted(lmer)
    cum["Residual"] = cum["CumYield"] - cum["Fitted"]

    rmse_all = np.sqrt(np.mean(cum["Residual"] ** 2))
    mean_all = cum["CumYield"].mean()
    print(f"Overall: RMSE = {rmse_all:.3f} lbs, Mean yield = {mean_all:.3f} lbs, "
          f"CV = {rmse_all / mean_all * 100:.1f}%\n")

    cv_table = (cum.groupby("Irrigation", observed=True)
                   .apply(lambda g: pd.Series({
                       "RMSE": np.sqrt(np.mean(g["Residual"] ** 2)),
                       "MeanYield": g["CumYield"].mean()}), include_groups=False)
                   .reset_index())
    cv_table["RMSE"] = cv_table["RMSE"].round(3)
    cv_table["MeanYield"] = cv_table["MeanYield"].round(3)
    cv_table["CV_percent"] = (cv_table["RMSE"] / cv_table["MeanYield"] * 100).round(1)
    cv_table["Adequacy"] = np.where(cv_table["CV_percent"] < 20,
                                    "Acceptable (<20%)", "Not acceptable (>=20%)")
    print(cv_table.to_string(index=False))

    # ---------------- Model formula (hierarchy check) ----------------
    print("\n==================== MODEL FORMULA ====================")
    print(f"Fixed:  {FIXED_FORMULA}")
    print(f"Random: {VC_TERMS}")
    print("Fixed single effects: Crop, AmarVar (Amaranth variety nested in crop)")
    print("Random single effect: Irrigation")

    # ---------------- Predicted cumulative yield by crop/variety ----------------
    print("\n==================== PREDICTED CUMULATIVE YIELD (lbs) ====================")
    b = res.fe_params
    b0 = b["Intercept"]
    rows = [("Eggplant (reference)", b0)]
    for crop in cum["Crop"].cat.categories[1:]:
        bj = b[f"C(Crop)[T.{crop}]"]
        if crop == "Amaranth":
            rows.append(("Golden giant", b0 + bj))
            rows.append(("Green callaloo", b0 + bj + b["AmarVar"]))
        else:
            rows.append((crop, b0 + bj))
    pred = pd.DataFrame(rows, columns=["Crop_Variety", "Predicted_lbs"]).round(2)
    print(pred.to_string(index=False))

    # ---------------- Irrigation effects (BLUPs) ----------------
    print("\n==================== ESTIMATED IRRIGATION EFFECTS (BLUPs) ====================")
    re = blups(res)
    print(re["Irrigation"])

    print("\nObserved mean cumulative yield by irrigation:")
    print(cum.groupby("Irrigation", observed=True)["CumYield"].mean().round(2)
             .reset_index(name="MeanYield").to_string(index=False))

    # ---------------- Diagnostics ----------------
    print("\n==================== RESIDUAL NORMALITY ====================")
    w, p_sw = stats.shapiro(cum["Residual"])
    print(f"Shapiro-Wilk normality test: W = {w:.4f}, p-value = {p_sw:.4g}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(cum["Fitted"], cum["Residual"], s=18)
    ax[0].axhline(0, color="grey", lw=1)
    ax[0].set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
    sm.qqplot(cum["Residual"], line="s", ax=ax[1])
    ax[1].set_title("Normal Q-Q plot of residuals")
    fig.tight_layout()
    fig.savefig("diagnostics.png", dpi=150)

    # ---------------- Interaction plot ----------------
    means = (cum.groupby(["Irrigation", "Variety"], observed=True)["CumYield"]
                .mean().unstack("Variety"))
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ["o", "^", "+", "x", "D", "v"]
    for i, variety in enumerate(means.columns):
        ax.plot(means.index.astype(str), means[variety],
                marker=markers[i % len(markers)], label=variety)
    ax.set(xlabel="Irrigation", ylabel="Mean cumulative yield (lbs)")
    ax.legend(title="Variety", fontsize=8)
    fig.tight_layout()
    fig.savefig("interaction_plot.png", dpi=150)

    print("\nPlots saved: diagnostics.png, interaction_plot.png")


if __name__ == "__main__":
    main()

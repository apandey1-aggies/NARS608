"""
Linear model for cumulative yield with crop, Amaranth variety and irrigation
all as fixed effects (Python port of the R lm() analysis).

Model
-----
    Y_ijk = b0 + b_j(Crop) + b_i(AmarVar within Amaranth) + b_k(Irrigation) + e_ijk

    Y_ijk = cumulative yield (lbs) of one plot
    b0    = intercept (Eggplant under Control irrigation)
    b_j   = fixed effect of crop (j = 1-5, reference = Eggplant)
    b_i   = fixed effect of Amaranth variety (reference = Golden giant)
    b_k   = fixed effect of irrigation (k = 1-3, reference = Control)
    e_ijk = residual error, e_ijk ~ N(0, s2_e)

R equivalent
------------
    lm(CumYield ~ Crop + AmarVar + Irrigation, data = cum)
    Anova(model, type = 2)

With no random effects left this is an ordinary least-squares model, so the
t-tests, F-tests and p-values match R's lm() / car::Anova() exactly.

Usage
-----
    pip install pandas numpy scipy statsmodels openpyxl matplotlib
    python yield_fixed_model.py "C:/Users/ASUS/OneDrive/Desktop/Datafinal.xlsx"
"""

import argparse

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

FORMULA = "CumYield ~ C(Crop) + AmarVar + C(Irrigation)"


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


# ------------------------------------------------------------ model ------
def fit_model(cum):
    return smf.ols(FORMULA, data=cum).fit()


def coefficient_table(res):
    return pd.DataFrame({
        "Term": [pretty_term(n) for n in res.params.index],
        "Estimate": res.params.values.round(3),
        "SE": res.bse.values.round(3),
        "t_value": res.tvalues.values.round(3),
        "p_value": [float(f"{v:.3g}") for v in res.pvalues.values],
    })


def anova_table(res):
    """Type II ANOVA (car::Anova(model, type = 2)); equal to Type III here
    because the model has no interaction terms."""
    a = sm.stats.anova_lm(res, typ=2)
    a.index = [i.replace("C(", "").replace(")", "") for i in a.index]
    a = a.rename(columns={"sum_sq": "Sum_Sq", "df": "Df", "F": "F_value",
                          "PR(>F)": "p_value"})
    a["Sum_Sq"] = a["Sum_Sq"].round(3)
    a["F_value"] = a["F_value"].round(3)
    a["p_value"] = [np.nan if pd.isna(v) else float(f"{v:.3g}") for v in a["p_value"]]
    return a


def predicted_table(res, cum):
    """Predicted cumulative yield for every crop/variety x irrigation level."""
    grid = (cum[["Variety", "Crop", "AmarVar"]].drop_duplicates()
               .merge(pd.DataFrame({"Irrigation": IRRIGATION_ORDER}), how="cross"))
    grid["Crop"] = pd.Categorical(grid["Crop"], categories=cum["Crop"].cat.categories)
    grid["Irrigation"] = pd.Categorical(grid["Irrigation"], categories=IRRIGATION_ORDER)
    pred = res.get_prediction(grid).summary_frame(alpha=0.05)
    grid["Predicted_lbs"] = pred["mean"].values
    grid["CI_low"] = pred["mean_ci_lower"].values
    grid["CI_high"] = pred["mean_ci_upper"].values
    wide = grid.pivot(index="Variety", columns="Irrigation", values="Predicted_lbs")
    return wide[IRRIGATION_ORDER].round(2), grid


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

    res = fit_model(cum)

    print("\n==================== MODEL EQUATION ====================")
    print("Y_ijk = b0 + b_j(Crop) + b_i(AmarVar within Amaranth) + b_k(Irrigation) + e_ijk\n")
    print("Dictionary:")
    print("  Y_ijk   = cumulative yield (lbs) of one plot")
    print("  b0      = intercept (Eggplant under Control irrigation)")
    print("  b_j     = fixed effect of crop (j = 1-5, reference = Eggplant)")
    print("  b_i     = fixed effect of Amaranth variety (reference = Golden giant)")
    print("  b_k     = fixed effect of irrigation (k = 1-3, reference = Control)")
    print("  e_ijk   = residual error, e_ijk ~ N(0, s2_e)")

    print("\n==================== MODEL SUMMARY ====================")
    print(res.summary())

    print("\n==================== FIXED EFFECTS ====================")
    print(coefficient_table(res).to_string(index=False))
    print(f"(t-tests use {int(res.df_resid)} residual df)")

    print("\n---- ANOVA (Type II = Type III, no interactions) ----")
    print(anova_table(res).to_string())

    print("\n==================== RESIDUAL VARIANCE ====================")
    print(f"Residual variance = {res.scale:.3f}, residual SE = {np.sqrt(res.scale):.3f} lbs")
    print(f"R-squared = {res.rsquared:.3f}, adjusted R-squared = {res.rsquared_adj:.3f}")

    # ---------------- Model error: RMSE and %CV ----------------
    print("\n==================== MODEL ERROR ====================")
    cum["Fitted"] = res.fittedvalues
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
    print(f"Fixed: {FORMULA}")
    print("Fixed single effects: Crop, AmarVar (Amaranth variety nested in crop), Irrigation")
    print("No random effects")

    # ---------------- Predicted cumulative yield ----------------
    print("\n==================== PREDICTED CUMULATIVE YIELD (lbs) ====================")
    wide, long_pred = predicted_table(res, cum)
    print(wide.to_string())
    print("\nWith 95% confidence intervals:")
    print(long_pred[["Variety", "Irrigation", "Predicted_lbs", "CI_low", "CI_high"]]
          .round(2).to_string(index=False))

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

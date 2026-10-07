"""
Residual vs predicted cumulative yield, one colour/marker per crop.

Uses the same data preparation and mixed model as yield_mixed_model.py:
    Y_ijk = b0 + b_j(Crop) + b_i(AmarVar) + u_k(Irrigation) + u_jk(Irrigation x Crop) + e_ijk

Residual = actual cumulative yield - predicted cumulative yield (lbs).

Usage
-----
    python residual_plot.py "C:/Users/ASUS/OneDrive/Desktop/Datafinal.xlsx"

Saves residual_vs_predicted.png in the folder you run it from.
"""

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yield_mixed_model import load_data, fit_mixed

FONT_SIZE = 40

# Legend label, colour and marker for each crop. Amaranth is split by variety
# because the model gives each variety its own predicted yield (AmarVar).
# Keys must match the Variety names in the Excel file (after trimming spaces).
CROP_STYLE = {
    "Egg plant":        ("Eggplant",                 "#2a78d6", "o"),
    "Golden giant":     ("Amaranth – Golden Giant",  "#eb6834", "s"),
    "Green callaloo":   ("Amaranth – Green Callaloo", "#1baf7a", "^"),
    "Georgia southern": ("Georgia Southern",         "#eda100", "D"),
    "Huckberry":        ("Huckleberry",              "#e87ba4", "v"),
    "Water leaf":       ("Waterleaf",                "#008300", "P"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="Datafinal.xlsx", help="Excel data file")
    ap.add_argument("--sheet", default="Sheet1")
    ap.add_argument("--out", default="residual_vs_predicted.png")
    args = ap.parse_args()

    cum = load_data(args.path, args.sheet)
    res = fit_mixed(cum)
    cum["Predicted"] = res.fittedvalues
    cum["Residual"] = cum["CumYield"] - cum["Predicted"]

    # one font size for every text element in the figure
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "legend.title_fontsize": FONT_SIZE,
    })

    fig, ax = plt.subplots(figsize=(30, 16))

    # unknown variety names (e.g. a spelling difference) still get plotted
    extra = [v for v in cum["Variety"].unique() if v not in CROP_STYLE]
    fallback_markers = iter(["X", "*", "h", "<", ">"])
    styles = dict(CROP_STYLE)
    for v in extra:
        styles[v] = (v, "#6b6b6b", next(fallback_markers, "o"))

    for variety, (label, colour, marker) in styles.items():
        sub = cum[cum["Variety"] == variety]
        if sub.empty:
            continue
        rmse = np.sqrt(np.mean(sub["Residual"] ** 2))
        ax.scatter(sub["Predicted"], sub["Residual"],
                   s=900, c=colour, marker=marker, alpha=0.85,
                   edgecolors="#1a1a19", linewidths=2,
                   label=f"{label} (RMSE = {rmse:.2f})")

    ax.axhline(0, color="#4a4a48", linestyle="--", linewidth=3)
    ax.set_xlabel("Predicted cumulative yield (lbs)")
    ax.set_ylabel("Residual (lbs)\n(actual − predicted)")
    ax.set_title("Residual vs predicted cumulative yield by crop")

    # symmetric y-axis so over- and under-prediction look equally large
    lim = np.ceil(np.abs(cum["Residual"]).max() * 1.1)
    ax.set_ylim(-lim, lim)
    ax.grid(True, color="#d9d9d6", linewidth=1.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # legend outside the plot so it never covers points at this font size
    ax.legend(title="Crop", loc="upper left", bbox_to_anchor=(1.02, 1),
              frameon=False, markerscale=0.8, borderaxespad=0)

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")
    print("\nResidual summary by crop:")
    summary = (cum.groupby("Variety")
                  .agg(n=("Residual", "size"),
                       Predicted=("Predicted", "mean"),
                       RMSE=("Residual", lambda r: np.sqrt(np.mean(r ** 2))),
                       MeanYield=("CumYield", "mean")))
    summary["CV_percent"] = summary["RMSE"] / summary["MeanYield"] * 100
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()

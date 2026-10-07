"""
Residual vs predicted cumulative yield plots.

Figure 1: all crops, one colour/marker per crop (both Amaranth varieties combined).
Figure 2: Amaranth plots only, one colour/marker per variety.

Uses the same data preparation and mixed model as yield_mixed_model.py:
    Y_ijk = b0 + b_j(Crop) + b_i(AmarVar) + u_k(Irrigation) + u_jk(Irrigation x Crop) + e_ijk

Residual = actual cumulative yield - predicted cumulative yield (lbs).

Usage
-----
    python residual_plot.py "C:/Users/ASUS/OneDrive/Desktop/Datafinal.xlsx"

Saves residual_by_crop.png and residual_by_amaranth_variety.png in the folder
you run it from.
"""

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yield_mixed_model import load_data, fit_mixed

FONT_SIZE = 40

# Figure 1: one legend entry per crop (both Amaranth varieties combined).
# Keys must match the Crop names built from the Excel Variety column.
CROP_STYLE = {
    "Egg plant":        ("Eggplant",         "#2a78d6", "o"),
    "Amaranth":         ("Amaranth",         "#eb6834", "s"),
    "Georgia southern": ("Georgia Southern", "#eda100", "D"),
    "Huckberry":        ("Huckleberry",      "#e87ba4", "v"),
    "Water leaf":       ("Waterleaf",        "#008300", "P"),
}

# Figure 2: Amaranth plots only, one legend entry per variety.
# Keys must match the Variety names in the Excel file (after trimming spaces).
AMARANTH_STYLE = {
    "Golden giant":   ("Golden Giant",   "#eb6834", "s"),
    "Green callaloo": ("Green Callaloo", "#1baf7a", "^"),
}


def set_font_size(size):
    """One font size for every text element in the figure."""
    plt.rcParams.update({
        "font.size": size,
        "axes.titlesize": size,
        "axes.labelsize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size,
        "legend.title_fontsize": size,
    })


def residual_plot(df, group_col, styles, title, legend_title, out):
    fig, ax = plt.subplots(figsize=(30, 16))

    # unknown group names (e.g. a spelling difference) still get plotted
    styles = dict(styles)
    fallback_markers = iter(["X", "*", "h", "<", ">"])
    for g in df[group_col].astype(str).unique():
        if g not in styles:
            styles[g] = (g, "#6b6b6b", next(fallback_markers, "o"))

    for group, (label, colour, marker) in styles.items():
        sub = df[df[group_col].astype(str) == group]
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
    ax.set_title(title)

    # symmetric y-axis so over- and under-prediction look equally large
    lim = np.ceil(np.abs(df["Residual"]).max() * 1.1)
    ax.set_ylim(-lim, lim)
    ax.grid(True, color="#d9d9d6", linewidth=1.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # legend outside the plot so it never covers points at this font size
    ax.legend(title=legend_title, loc="upper left", bbox_to_anchor=(1.02, 1),
              frameon=False, markerscale=0.8, borderaxespad=0)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def residual_summary(df, group_col):
    summary = (df.groupby(group_col, observed=True)
                 .agg(n=("Residual", "size"),
                      Predicted=("Predicted", "mean"),
                      RMSE=("Residual", lambda r: np.sqrt(np.mean(r ** 2))),
                      MeanYield=("CumYield", "mean")))
    summary["CV_percent"] = summary["RMSE"] / summary["MeanYield"] * 100
    return summary.round(3).to_string()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="Datafinal.xlsx", help="Excel data file")
    ap.add_argument("--sheet", default="Sheet1")
    args = ap.parse_args()

    cum = load_data(args.path, args.sheet)
    res = fit_mixed(cum)
    cum["Predicted"] = res.fittedvalues
    cum["Residual"] = cum["CumYield"] - cum["Predicted"]

    set_font_size(FONT_SIZE)

    # Figure 1: all crops, Amaranth varieties combined
    residual_plot(cum, "Crop", CROP_STYLE,
                  title="Residual vs predicted cumulative yield by crop",
                  legend_title="Crop", out="residual_by_crop.png")

    # Figure 2: Amaranth only, varieties separated
    amaranth = cum[cum["Crop"] == "Amaranth"]
    residual_plot(amaranth, "Variety", AMARANTH_STYLE,
                  title="Residual vs predicted cumulative yield of Amaranth varieties",
                  legend_title="Amaranth variety", out="residual_by_amaranth_variety.png")

    print("\nResidual summary by crop:")
    print(residual_summary(cum, "Crop"))
    print("\nResidual summary by Amaranth variety:")
    print(residual_summary(amaranth, "Variety"))


if __name__ == "__main__":
    main()

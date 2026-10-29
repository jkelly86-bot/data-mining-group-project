"""
Week 2 - Data Audit & EDA (Wellness Grocery Inventory Analysis)

Run from the repository root:
    python code/week2_data_audit.py
Reads  data/food.csv
Writes outputs/week2/  (figures, audit tables, interactive dashboard)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

DATA = Path("data/food.csv")
OUT = Path("outputs/week2")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- load
df = pd.read_csv(DATA)
NUM = [c for c in df.columns if c.startswith("Data.")]
LABEL = {c: c.replace("Data.Fat.", "").replace("Data.", "") for c in NUM}
UNIT = {"Kilocalories": "kcal"}
nice = lambda c: f"{LABEL[c]} ({UNIT.get(LABEL[c], 'g')} per 100 g)"

# ---------------------------------------------------------------- 1. data audit
audit = pd.DataFrame({
    "mean": df[NUM].mean(), "median": df[NUM].median(), "std": df[NUM].std(),
    "min": df[NUM].min(), "max": df[NUM].max(),
    "skewness": df[NUM].skew(),
    "pct_zero": (df[NUM] == 0).mean() * 100,
    "missing": df[NUM].isna().sum(),
})
q1, q3 = df[NUM].quantile(0.25), df[NUM].quantile(0.75)
iqr = q3 - q1
audit["iqr_outliers"] = ((df[NUM] < q1 - 1.5 * iqr) | (df[NUM] > q3 + 1.5 * iqr)).sum()
audit.index = [LABEL[c] for c in audit.index]
audit.round(2).to_csv(OUT / "audit_summary.csv")

# Data-quality flags (rows that break nutritional logic)
est_kcal = 4 * df["Data.Protein"] + 4 * df["Data.Carbohydrate"] + 9 * df["Data.Fat.Total Lipid"]
flags = pd.DataFrame({
    "exact_duplicate": df.duplicated(keep=False),
    "sugar_gt_carb": df["Data.Sugar Total"] > df["Data.Carbohydrate"] + 0.01,
    "fiber_gt_carb": df["Data.Fiber"] > df["Data.Carbohydrate"] + 0.01,
    "kcal_mismatch_gt100": (df["Data.Kilocalories"] - est_kcal).abs() > 100,
})
flagged = pd.concat([df[["Category", "Description"] + NUM], flags], axis=1)[flags.any(axis=1)]
flagged.to_csv(OUT / "data_quality_flags.csv", index=False)

cat_counts = df["Category"].value_counts()
cat_counts.to_csv(OUT / "category_counts.csv", header=["n_items"])

summary = {
    "rows": len(df), "columns": df.shape[1], "numeric_features": len(NUM),
    "missing_values": int(df.isna().sum().sum()),
    "unique_categories": df["Category"].nunique(),
    "categories_with_one_item": int((cat_counts == 1).sum()),
    "no_category_items": int((df["Category"] == "No Category").sum()),
    **{k: int(v.sum()) for k, v in flags.items()},
}
pd.Series(summary).to_csv(OUT / "audit_overview.csv", header=["value"])

# ---------------------------------------------------------------- 2. histograms
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, c in zip(axes.ravel(), NUM):
    ax.hist(df[c], bins=40, color="#2f6f9f", edgecolor="white", linewidth=0.4)
    ax.axvline(df[c].median(), color="#d9822b", ls="--", lw=1.2, label="median")
    ax.set_title(LABEL[c], fontsize=11)
    ax.set_xlabel(f"{UNIT.get(LABEL[c], 'g')} per 100 g", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axes[0, 0].set_ylabel("Number of items"); axes[1, 0].set_ylabel("Number of items")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("Distribution audit: histograms of nutritional features (n = %d items)" % len(df), fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "fig1_histograms.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 3. box plots
fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
for ax, c in zip(axes.ravel(), NUM):
    ax.boxplot(df[c], orientation="horizontal", widths=0.6, patch_artist=True,
               boxprops=dict(facecolor="#dce8f7", color="#2f6f9f"),
               medianprops=dict(color="#d9822b", lw=2),
               flierprops=dict(marker="o", ms=2, alpha=0.25, markeredgecolor="#2f6f9f"))
    ax.set_title(f"{LABEL[c]}  ({int(audit.loc[LABEL[c], 'iqr_outliers'])} IQR outliers)", fontsize=10.5)
    ax.set_yticks([]); ax.set_xlabel(f"{UNIT.get(LABEL[c], 'g')} per 100 g", fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
fig.suptitle("Distribution audit: box plots (points = values beyond 1.5 x IQR)", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "fig2_boxplots.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 4. correlation matrices
def heatmap(corr, title, path):
    lab = [LABEL[c] for c in corr.columns]
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(lab)), lab, rotation=40, ha="right"); ax.set_yticks(range(len(lab)), lab)
    for i in range(len(lab)):
        for j in range(len(lab)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title(title, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)

pearson = df[NUM].corr(method="pearson")
spearman = df[NUM].corr(method="spearman")
pearson.rename(index=LABEL, columns=LABEL).round(3).to_csv(OUT / "corr_pearson.csv")
spearman.rename(index=LABEL, columns=LABEL).round(3).to_csv(OUT / "corr_spearman.csv")
heatmap(pearson, "Pearson correlation (linear)", OUT / "fig3_corr_pearson.png")
heatmap(spearman, "Spearman correlation (rank-based, robust to skew)", OUT / "fig4_corr_spearman.png")

# Ranked list of nutrient pairs
pairs = []
for i, a in enumerate(NUM):
    for b in NUM[i + 1:]:
        pairs.append((LABEL[a], LABEL[b], pearson.loc[a, b], spearman.loc[a, b]))
pairs = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "pearson", "spearman"])
pairs["abs_spearman"] = pairs["spearman"].abs()
pairs.sort_values("abs_spearman", ascending=False).drop(columns="abs_spearman").round(3) \
    .to_csv(OUT / "corr_pairs_ranked.csv", index=False)

# ---------------------------------------------------------------- 5. interactive dashboard
# Dropdowns toggle the feature shown (preview of the "dynamic visualization" requirement).
dash = go.Figure()
for k, c in enumerate(NUM):
    dash.add_trace(go.Histogram(x=df[c], nbinsx=40, name=LABEL[c], visible=(k == 0),
                                marker_color="#2f6f9f", xaxis="x", yaxis="y"))
for k, c in enumerate(NUM):
    dash.add_trace(go.Box(x=df[c], name=LABEL[c], visible=(k == 0), boxpoints="outliers",
                          marker=dict(size=3, opacity=0.4), line_color="#2f6f9f",
                          text=df["Description"], xaxis="x2", yaxis="y2"))
buttons = []
for k, c in enumerate(NUM):
    vis = [False] * (2 * len(NUM)); vis[k] = True; vis[len(NUM) + k] = True
    buttons.append(dict(label=LABEL[c], method="update",
                        args=[{"visible": vis},
                              {"xaxis.title.text": nice(c), "xaxis2.title.text": nice(c)}]))
dash.update_layout(
    title="Week 2 Data Audit Dashboard: wellness grocery inventory (choose a nutrient)",
    updatemenus=[dict(buttons=buttons, x=0, y=1.13, xanchor="left", showactive=True)],
    xaxis=dict(domain=[0, 1], anchor="y", title=nice(NUM[0])), yaxis=dict(domain=[0.4, 1], title="Items"),
    xaxis2=dict(domain=[0, 1], anchor="y2", title=nice(NUM[0])), yaxis2=dict(domain=[0, 0.25], showticklabels=False),
    showlegend=False, height=720, template="plotly_white", margin=dict(t=110))
dash.write_html(OUT / "data_audit_dashboard.html", include_plotlyjs="cdn")

# Interactive scatter: pick any two nutrients
sc = go.Figure(go.Scattergl(x=df[NUM[0]], y=df[NUM[1]], mode="markers",
                            marker=dict(size=4, opacity=0.35, color="#2f6f9f"),
                            text=df["Category"] + " | " + df["Description"], hoverinfo="text+x+y"))
xbtn = [dict(label=LABEL[c], method="update", args=[{"x": [df[c]]}, {"xaxis.title.text": nice(c)}]) for c in NUM]
ybtn = [dict(label=LABEL[c], method="update", args=[{"y": [df[c]]}, {"yaxis.title.text": nice(c)}]) for c in NUM]
sc.update_layout(title="Nutrient vs. nutrient explorer (X and Y menus)", template="plotly_white", height=650,
                 xaxis_title=nice(NUM[0]), yaxis_title=nice(NUM[1]), margin=dict(t=110),
                 updatemenus=[dict(buttons=xbtn, x=0, y=1.13, xanchor="left", active=0),
                              dict(buttons=ybtn, x=0.3, y=1.13, xanchor="left", active=1)],
                 annotations=[dict(text="X:", x=-0.03, y=1.11, xref="paper", yref="paper", showarrow=False),
                              dict(text="Y:", x=0.27, y=1.11, xref="paper", yref="paper", showarrow=False)])
sc.write_html(OUT / "scatter_explorer.html", include_plotlyjs="cdn")

print(pd.Series(summary).to_string())
print("\nAudit summary:\n", audit.round(2).to_string())
print("\nTop correlated pairs:\n", pairs.sort_values("abs_spearman", ascending=False).head(8).round(2).to_string(index=False))
print(f"\nSaved outputs to {OUT.resolve()}")

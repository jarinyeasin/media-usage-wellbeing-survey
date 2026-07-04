"""
Generates 8 publication-quality charts saved as PNG:
  fig1_demographics.png        — gender + screentime + level
  fig2_likert_means.png        — all 11 Likert dimensions
  fig3_screentime_impact.png   — screentime group × neg. impact
  fig4_gender_comparison.png   — gender × all Likert items
  fig5_correlation_heatmap.png — Pearson r between all items
  fig6_elbow_silhouette.png    — cluster selection
  fig7_cluster_radar.png       — radar / spider chart of 3 clusters
  fig8_pca_scatter.png         — 2D PCA coloured by cluster

"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

matplotlib.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi":      120,
    "savefig.dpi":     150,
    "savefig.bbox":    "tight",
})

os.makedirs("outputs", exist_ok=True)

# colour palette
TEAL   = "#1D9E75"
BLUE   = "#378ADD"
CORAL  = "#D85A30"
PURPLE = "#7F77DD"
AMBER  = "#BA7517"
GRAY   = "#888780"

CLUSTER_COLORS = [TEAL, BLUE, CORAL]
CLUSTER_NAMES  = ["Low-Impact Users", "Moderate Impact", "High-Risk Users"]

# load data
df  = pd.read_csv("data/survey_with_clusters.csv")
df2 = pd.read_csv("data/survey_clean.csv")          # fallback if needed

LIKERT = [
    "q1_self_esteem","q2_comparison","q3_relaxation","q4_concentration",
    "q5_overthinking","q6_emotional_drain","q7_fomo","q8_sleep",
    "q9_info_overwhelm","q10_satisfaction","q11_mental_wellbeing"
]
SHORT_LABELS = [
    "Self-esteem\nfrom posts","Social\ncomparison","Media\nhelps relax",
    "Difficulty\nconcentrating","Overthinking\noffline","Emotional\ndrain",
    "FOMO\nmanagement","Sleep\ndisruption","Info\noverload",
    "Satisfied\nw/ usage","Wellbeing\nimpact"
]
ST_ORDER = ["0 to 1 hour","1 to 2 hour","2 to 3 hours",
            "3 to 4 hours","4 to 5 hours","5+ hours"]
ST_LABELS = ["0-1h","1-2h","2-3h","3-4h","4-5h","5+h"]

print("Generating visualisations...")

# FIG 1 — DEMOGRAPHICS
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Survey Sample Demographics  (n=56)", fontsize=14, fontweight="bold", y=1.02)

# 1a: Gender donut
ax = axes[0]
sizes  = [df["gender"].value_counts()["Female"],
          df["gender"].value_counts()["Male"]]
labels = [f"Female\n{sizes[0]} ({sizes[0]/56*100:.0f}%)",
          f"Male\n{sizes[1]} ({sizes[1]/56*100:.0f}%)"]
wedges, _ = ax.pie(sizes, colors=[BLUE, PURPLE],
                   startangle=90, wedgeprops=dict(width=0.5))
ax.legend(wedges, labels, loc="center", fontsize=10)
ax.set_title("Gender", fontweight="bold")

# 1b: Screentime bar
ax = axes[1]
counts = [df[df["screentime"]==s].shape[0] for s in ST_ORDER]
bars = ax.bar(ST_LABELS, counts, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
ax.set_ylim(0, max(counts) * 1.25)
ax.set_xlabel("Daily screentime on social media")
ax.set_ylabel("Number of students")
ax.set_title("Screentime Distribution", fontweight="bold")

# 1c: Level of study
ax = axes[2]
level_vc = df["level"].value_counts()
level_vc = level_vc[level_vc.index.isin(["Bachelor","HSC","Masters","MBBS"])]
colors_l  = [TEAL, BLUE, PURPLE, CORAL][:len(level_vc)]
bars = ax.barh(level_vc.index, level_vc.values,
               color=colors_l, edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
ax.set_xlabel("Number of students")
ax.set_title("Level of Study", fontweight="bold")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("outputs/fig1_demographics.png")
plt.close()
print("  ✓ fig1_demographics.png")

# FIG 2 — LIKERT MEANS (horizontal bar + colour coding)
fig, ax = plt.subplots(figsize=(10, 7))
means = [df[c].mean() for c in LIKERT]
colors = [CORAL if m >= 5 else BLUE if m >= 4 else TEAL for m in means]
bars = ax.barh(SHORT_LABELS, means, color=colors,
               edgecolor="white", linewidth=0.8, height=0.65)

# value labels
for bar, mean in zip(bars, means):
    ax.text(mean + 0.05, bar.get_y() + bar.get_height()/2,
            f"{mean:.2f}", va="center", ha="left", fontsize=10)

ax.axvline(x=4, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Scale midpoint (4)")
ax.set_xlim(1, 8)
ax.set_xlabel("Mean score (1 = Strongly Disagree, 7 = Strongly Agree)")
ax.set_title("Mean Likert Scores Across All Survey Dimensions",
             fontweight="bold", fontsize=13)

patches = [
    mpatches.Patch(color=CORAL, label="High concern (≥5.0)"),
    mpatches.Patch(color=BLUE,  label="Moderate (4.0–4.9)"),
    mpatches.Patch(color=TEAL,  label="Low (< 4.0)"),
]
ax.legend(handles=patches, loc="lower right", fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/fig2_likert_means.png")
plt.close()
print("  ✓ fig2_likert_means.png")

# FIG 3 — SCREENTIME × NEG. IMPACT (line + bars overlay)
fig, ax1 = plt.subplots(figsize=(10, 5))

st_means = df.groupby("screentime")[["neg_impact","wellbeing_impact"]].mean()
st_means = st_means.reindex([s for s in ST_ORDER if s in st_means.index])
x_pos    = range(len(st_means))

# bar: negative impact
ax1.bar(x_pos, st_means["neg_impact"], color=CORAL, alpha=0.6,
        label="Negative impact score", width=0.5)
# line: wellbeing
ax2 = ax1.twinx()
ax2.plot(x_pos, st_means["wellbeing_impact"], color=BLUE, marker="o",
         linewidth=2, markersize=7, label="Wellbeing impact (q11)")

ax1.set_xticks(list(x_pos))
ax1.set_xticklabels(ST_LABELS)
ax1.set_xlabel("Daily screentime on social media")
ax1.set_ylabel("Negative impact score (mean)", color=CORAL)
ax2.set_ylabel("Wellbeing impact — q11 mean", color=BLUE)
ax1.set_ylim(0, 8)
ax2.set_ylim(0, 8)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
ax1.set_title("Screentime Group vs. Wellbeing & Negative Impact Scores",
              fontweight="bold", fontsize=13)

# annotate ANOVA result
ax1.text(0.98, 0.05, "ANOVA: F=1.26, p=0.30 (n.s.)",
         transform=ax1.transAxes, ha="right", fontsize=9,
         color="gray", style="italic")

plt.tight_layout()
plt.savefig("outputs/fig3_screentime_impact.png")
plt.close()
print("  ✓ fig3_screentime_impact.png")

# FIG 4 — GENDER COMPARISON (grouped bar chart)
fig, ax = plt.subplots(figsize=(12, 6))

male_means   = df[df["gender"]=="Male"][LIKERT].mean()
female_means = df[df["gender"]=="Female"][LIKERT].mean()
x = np.arange(len(LIKERT))
w = 0.38

ax.bar(x - w/2, male_means,   w, label="Male",   color=PURPLE, alpha=0.85)
ax.bar(x + w/2, female_means, w, label="Female", color=BLUE,   alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(SHORT_LABELS, fontsize=9.5)
ax.axhline(4, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_ylim(1, 8)
ax.set_ylabel("Mean Likert score (1–7)")
ax.set_title("Likert Mean Scores by Gender  (t-test: p=0.76, n.s.)",
             fontweight="bold", fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("outputs/fig4_gender_comparison.png")
plt.close()
print("  ✓ fig4_gender_comparison.png")

# FIG 5 — CORRELATION HEATMAP
from scipy import stats as scipy_stats

fig, ax = plt.subplots(figsize=(12, 10))

n = len(LIKERT)
corr_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        r, _ = scipy_stats.pearsonr(df[LIKERT[i]].dropna(), df[LIKERT[j]].dropna())
        corr_matrix[i, j] = r

# use simple label names
simple_labels = ["Self-esteem","Comparison","Relaxation","Concentration",
                 "Overthinking","Emo. Drain","FOMO","Sleep","Info Overload",
                 "Satisfaction","Wellbeing"]

im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(simple_labels, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(simple_labels, fontsize=10)

# annotate each cell
for i in range(n):
    for j in range(n):
        val = corr_matrix[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8.5, color=color)

ax.set_title("Pearson Correlation Matrix — All Likert Variables",
             fontweight="bold", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig("outputs/fig5_correlation_heatmap.png")
plt.close()
print("  ✓ fig5_correlation_heatmap.png")

# FIG 6 — ELBOW + SILHOUETTE
X = df[LIKERT].values
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

K_RANGE   = range(2, 9)
inertias  = []
sil_scores= []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_sc)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_sc, km.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Choosing Optimal k for K-Means Clustering", fontweight="bold", fontsize=13)

# Elbow
ax1.plot(list(K_RANGE), inertias, "o-", color=BLUE, linewidth=2, markersize=7)
ax1.axvline(x=3, color=CORAL, linestyle="--", linewidth=1.5, label="k=3 (chosen)")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Within-cluster sum of squares (inertia)")
ax1.set_title("Elbow Method")
ax1.legend(fontsize=10)

# Silhouette
ax2.plot(list(K_RANGE), sil_scores, "s-", color=TEAL, linewidth=2, markersize=7)
ax2.axvline(x=3, color=CORAL, linestyle="--", linewidth=1.5, label="k=3 (chosen)")
best_k = list(K_RANGE)[int(np.argmax(sil_scores))]
ax2.axvline(x=best_k, color=GRAY, linestyle=":", linewidth=1.2,
            label=f"Best silhouette k={best_k}")
ax2.set_xlabel("Number of clusters (k)")
ax2.set_ylabel("Average silhouette score")
ax2.set_title("Silhouette Score")
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig("outputs/fig6_elbow_silhouette.png")
plt.close()
print("  ✓ fig6_elbow_silhouette.png")

# FIG 7 — RADAR CHART (cluster profiles)
from matplotlib.patches import FancyArrowPatch

# 5 key dimensions for radar (reduce to avoid clutter)
RADAR_COLS   = ["q1_self_esteem","q4_concentration","q6_emotional_drain",
                "q8_sleep","q9_info_overwhelm","q10_satisfaction","q11_mental_wellbeing"]
RADAR_LABELS = ["Self-esteem","Concentration\nDifficulty","Emotional\nDrain",
                "Sleep\nDisruption","Info\nOverload","Satisfaction","Wellbeing\nImpact"]

angles = np.linspace(0, 2 * np.pi, len(RADAR_COLS), endpoint=False).tolist()
angles += angles[:1]  # close polygon

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

for c_id, (name, color) in enumerate(zip(CLUSTER_NAMES, CLUSTER_COLORS)):
    values = df[df["cluster"]==c_id][RADAR_COLS].mean().tolist()
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, color=color, label=name)
    ax.fill(angles, values, alpha=0.12, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(RADAR_LABELS, fontsize=10.5)
ax.set_ylim(1, 7)
ax.set_yticks([2, 3, 4, 5, 6, 7])
ax.set_yticklabels(["2","3","4","5","6","7"], fontsize=8, color="gray")
ax.grid(color="gray", alpha=0.3)
ax.set_title("Cluster Behavioural Profiles — Radar Chart",
             fontweight="bold", fontsize=13, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

plt.tight_layout()
plt.savefig("outputs/fig7_cluster_radar.png")
plt.close()
print("  ✓ fig7_cluster_radar.png")

# FIG 8 — PCA SCATTER (coloured by cluster)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("PCA Scatter — Respondents in Latent Space (54.5% variance)",
             fontweight="bold", fontsize=13)

# 8a: Coloured by cluster
ax = axes[0]
for c_id, (name, color) in enumerate(zip(CLUSTER_NAMES, CLUSTER_COLORS)):
    mask = df["cluster"] == c_id
    ax.scatter(df.loc[mask,"pca1"], df.loc[mask,"pca2"],
               c=color, label=f"{name} (n={mask.sum()})",
               s=60, alpha=0.85, edgecolors="white", linewidth=0.5)
ax.set_xlabel("PC1 — 'harm axis'  (39.1% variance)")
ax.set_ylabel("PC2 — 'social comparison axis'  (15.4%)")
ax.set_title("Coloured by cluster", fontweight="bold")
ax.legend(fontsize=9)
ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
ax.axvline(0, color="gray", linewidth=0.5, alpha=0.4)

# 8b: Coloured by gender
ax = axes[1]
for gender, color, marker in [("Female", BLUE, "o"), ("Male", PURPLE, "s")]:
    mask = df["gender"] == gender
    ax.scatter(df.loc[mask,"pca1"], df.loc[mask,"pca2"],
               c=color, label=gender, s=55, marker=marker,
               alpha=0.75, edgecolors="white", linewidth=0.5)
ax.set_xlabel("PC1 — 'harm axis'  (39.1% variance)")
ax.set_ylabel("PC2 — 'social comparison axis'  (15.4%)")
ax.set_title("Coloured by gender", fontweight="bold")
ax.legend(fontsize=10)
ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
ax.axvline(0, color="gray", linewidth=0.5, alpha=0.4)

plt.tight_layout()
plt.savefig("outputs/fig8_pca_scatter.png")
plt.close()
print("  ✓ fig8_pca_scatter.png")

print("\n✓ All 8 figures saved in outputs/")

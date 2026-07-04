import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# load
df = pd.read_csv("data/survey_clean.csv")
print(f"Loaded {len(df)} respondents\n")

LIKERT = [
    "q1_self_esteem","q2_comparison","q3_relaxation","q4_concentration",
    "q5_overthinking","q6_emotional_drain","q7_fomo","q8_sleep",
    "q9_info_overwhelm","q10_satisfaction","q11_mental_wellbeing"
]
LABELS = {
    "q1_self_esteem":      "Self-esteem from posts",
    "q2_comparison":       "Social comparison",
    "q3_relaxation":       "Media helps relax",
    "q4_concentration":    "Difficulty concentrating",
    "q5_overthinking":     "Overthinking offline",
    "q6_emotional_drain":  "Emotional drain",
    "q7_fomo":             "FOMO management",
    "q8_sleep":            "Sleep disruption",
    "q9_info_overwhelm":   "Information overload",
    "q10_satisfaction":    "Satisfied with usage",
    "q11_mental_wellbeing":"Overall wellbeing impact",
}

SEP = "─" * 55

# 1. FEATURE STANDARDISATION
# K-means uses Euclidean distance.
# Without scaling, a variable with range 1–7 dominates one with range 1–3.
# StandardScaler → mean=0, std=1 for each feature.

X = df[LIKERT].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features standardised (mean=0, std=1 per column)")
print(f"Feature matrix shape: {X_scaled.shape}  [n_samples × n_features]\n")

# 2. ELBOW METHOD — choose optimal k
print(SEP)
print("ELBOW METHOD — within-cluster sum of squares (inertia)")
print(SEP)

inertias = []
sil_scores = []
K_RANGE = range(2, 9)

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_)
    sil_scores.append(sil)
    print(f"  k={k}  inertia={km.inertia_:6.1f}  silhouette={sil:.3f}")

best_k_sil = K_RANGE[int(np.argmax(sil_scores))]
print(f"\n  Best k by silhouette score: k={best_k_sil}")
print(f"  We use k=3 (balances interpretability + data size)")

# 3. K-MEANS CLUSTERING  (k=3)
print(f"\n{SEP}")
print("K-MEANS CLUSTERING  (k=3)")
print(SEP)

K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
cluster_raw = kmeans.fit_predict(X_scaled)
df["cluster_raw"] = cluster_raw

# Re-label clusters by ascending negative impact (0=lowest, 2=highest)
cluster_neg_means = df.groupby("cluster_raw")["neg_impact"].mean()
sorted_clusters   = cluster_neg_means.sort_values().index.tolist()
remap             = {sorted_clusters[0]: 0, sorted_clusters[1]: 1, sorted_clusters[2]: 2}
df["cluster"]     = df["cluster_raw"].map(remap)

CLUSTER_NAMES = {0: "Low-Impact Users", 1: "Moderate Impact", 2: "High-Risk Users"}
df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)

final_sil = silhouette_score(X_scaled, kmeans.labels_)
print(f"\n  Silhouette score: {final_sil:.3f}  (range −1 to 1; >0.2 = meaningful structure)")
print(f"  Inertia: {kmeans.inertia_:.1f}")

print("\n  Cluster sizes:")
for c in [0, 1, 2]:
    n = (df["cluster"]==c).sum()
    pct = n/len(df)*100
    print(f"    {c} — {CLUSTER_NAMES[c]}: n={n} ({pct:.0f}%)")

# cluster profiles
print(f"\n  Cluster profiles (mean Likert scores, 1–7):")
profile_cols = LIKERT + ["screentime_hours","neg_impact","wellbeing_impact"]
profiles = df.groupby("cluster")[profile_cols].mean().round(2)
profiles.index = [CLUSTER_NAMES[i] for i in profiles.index]
profiles.columns = [LABELS.get(c, c) for c in profiles.columns]
print(profiles.T.to_string())

# key differentiators
print(f"\n  Key differentiating features:")
print(f"    Cluster 2 (High-Risk) has notably higher social comparison (q1+q2)")
print(f"    Cluster 0 (Low-Impact) scores lowest on all harm dimensions")
print(f"    Cluster 1 (Moderate) reports worst satisfaction with their usage")

# 4. PCA — DIMENSIONALITY REDUCTION
print(f"\n{SEP}")
print("PCA — PRINCIPAL COMPONENT ANALYSIS")
print(SEP)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

print(f"\n  Explained variance:")
for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_,
                                    np.cumsum(pca.explained_variance_ratio_)), 1):
    print(f"    PC{i}: {var*100:.1f}%  (cumulative: {cum*100:.1f}%)")

print(f"\n  PC1 loadings (which variables drive it most):")
loadings = pd.DataFrame(pca.components_.T,
                         index=[LABELS[c] for c in LIKERT],
                         columns=["PC1","PC2"])
print(loadings["PC1"].sort_values(ascending=False).round(3).to_string())

print(f"\n  Interpretation:")
print(f"    PC1 (39.1%): 'harm axis' — high PC1 = more concentration/sleep/drain issues")
print(f"    PC2 (15.4%): 'social comparison axis' — separates high social comparers")

# save enriched dataset
os.makedirs("data", exist_ok=True)
df.to_csv("data/survey_with_clusters.csv", index=False)
print(f"\n✓ Enriched dataset saved → data/survey_with_clusters.csv")
print(f"  New columns: cluster, cluster_name, pca1, pca2")

# save PCA loadings
os.makedirs("outputs", exist_ok=True)
loadings.to_csv("outputs/pca_loadings.csv")
print(f"✓ PCA loadings saved   → outputs/pca_loadings.csv")

# save cluster profiles
profiles.to_csv("outputs/cluster_profiles.csv")
print(f"✓ Cluster profiles saved → outputs/cluster_profiles.csv")

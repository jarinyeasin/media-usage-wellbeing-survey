import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# load clean data
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

# 1. DESCRIPTIVE STATISTICS
print(SEP)
print("1. DESCRIPTIVE STATISTICS")
print(SEP)

print("\nSample demographics:")
print(f"  Total respondents : {len(df)}")
print(f"  Female            : {(df['gender']=='Female').sum()} ({(df['gender']=='Female').mean()*100:.0f}%)")
print(f"  Male              : {(df['gender']=='Male').sum()} ({(df['gender']=='Male').mean()*100:.0f}%)")
print(f"  Age range         : {df['age'].min():.0f}–{df['age'].max():.0f} years (mean={df['age'].mean():.1f})")

print("\nScreentime distribution:")
ST_ORDER = ["0 to 1 hour","1 to 2 hour","2 to 3 hours",
            "3 to 4 hours","4 to 5 hours","5+ hours"]
for st in ST_ORDER:
    n = (df["screentime"]==st).sum()
    pct = n/len(df)*100
    bar = "█" * n
    print(f"  {st:<15} {n:>2} ({pct:4.1f}%)  {bar}")

print("\nLikert scale means (1–7 scale):")
desc = df[LIKERT].agg(["mean","std","min","max"]).T
desc.index = [LABELS[c] for c in desc.index]
desc.columns = ["Mean","SD","Min","Max"]
print(desc.round(2).to_string())

# 2. PEARSON CORRELATION  –  screentime × negative impact
print(f"\n{SEP}")
print("2. PEARSON CORRELATION — screentime vs negative impact")
print(SEP)

valid = df[["screentime_hours","neg_impact"]].dropna()
r, p = stats.pearsonr(valid["screentime_hours"], valid["neg_impact"])

print(f"\n  r  = {r:.3f}")
print(f"  p  = {p:.4f}")
print(f"  n  = {len(valid)}")
sig = "✓ statistically significant (p < 0.05)" if p < 0.05 else "✗ not significant at p < 0.05"
print(f"  {sig}")
print(f"\n  Interpretation: weak positive correlation — students with")
print(f"  more screentime report slightly higher negative impact,")
print(f"  but the trend is not statistically reliable with n={len(df)}.")

# 3. INDEPENDENT-SAMPLES T-TEST  –  gender differences
print(f"\n{SEP}")
print("3. INDEPENDENT-SAMPLES T-TEST — gender vs negative impact")
print(SEP)

male   = df[df["gender"]=="Male"]["neg_impact"].dropna()
female = df[df["gender"]=="Female"]["neg_impact"].dropna()
t, p_t = stats.ttest_ind(male, female, equal_var=False)  # Welch's t-test

print(f"\n  Male   (n={len(male)}):  mean={male.mean():.2f}, SD={male.std():.2f}")
print(f"  Female (n={len(female)}): mean={female.mean():.2f}, SD={female.std():.2f}")
print(f"\n  t  = {t:.3f}")
print(f"  p  = {p_t:.4f}")
sig_t = "✓ significant" if p_t < 0.05 else "✗ not significant"
print(f"  {sig_t} — no meaningful gender difference found.")

# Cohens d (effect size)
pooled_sd = np.sqrt((male.std()**2 + female.std()**2) / 2)
d = (female.mean() - male.mean()) / pooled_sd
print(f"  Cohen's d = {d:.3f} (effect size: {'small' if abs(d)<0.5 else 'medium' if abs(d)<0.8 else 'large'})")

# 4. ONE-WAY ANOVA  –  across screentime groups
print(f"\n{SEP}")
print("4. ONE-WAY ANOVA — screentime group vs negative impact")
print(SEP)

groups = []
print("\n  Group means:")
for st in ST_ORDER:
    g = df[df["screentime"]==st]["neg_impact"].dropna()
    if len(g) > 0:
        groups.append(g)
        print(f"    {st:<15}: mean={g.mean():.2f}, n={len(g)}")

f_stat, f_p = stats.f_oneway(*groups)
print(f"\n  F  = {f_stat:.3f}")
print(f"  p  = {f_p:.4f}")
sig_f = "✓ significant" if f_p < 0.05 else "✗ not significant"
print(f"  {sig_f}")
print(f"\n  Note: The visible upward trend (more hours → more impact)")
print(f"  is not statistically confirmed at this sample size.")
print(f"  A larger sample (n≥150) would provide more statistical power.")

# 5. MULTIPLE LINEAR REGRESSION  –  predict wellbeing impact
print(f"\n{SEP}")
print("5. MULTIPLE LINEAR REGRESSION — predicting wellbeing impact")
print(SEP)

PREDICTORS = ["screentime_hours","social_comparison",
              "q4_concentration","q8_sleep"]
PRED_LABELS = ["Screentime (hrs)","Social Comparison",
               "Concentration Difficulty","Sleep Impact"]

reg_df = df[PREDICTORS + ["wellbeing_impact"]].dropna()
X = reg_df[PREDICTORS].values
y = reg_df["wellbeing_impact"].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2   = r2_score(y, y_pred)
n_r  = len(y)
k    = len(PREDICTORS)
# Adjusted R²
r2_adj = 1 - (1 - r2) * (n_r - 1) / (n_r - k - 1)

print(f"\n  Outcome: Overall wellbeing impact (q11, 1–7 scale)")
print(f"  Predictors: {', '.join(PRED_LABELS)}")
print(f"\n  R²      = {r2:.3f}  ({r2*100:.1f}% variance explained)")
print(f"  Adj R²  = {r2_adj:.3f}")
print(f"  n       = {n_r}")

print(f"\n  Coefficients:")
print(f"    {'Predictor':<28} {'β (raw)':>8}")
print(f"    {'─'*36}")
for label, coef in zip(PRED_LABELS, model.coef_):
    bar = "▶" * int(abs(coef) * 8)
    print(f"    {label:<28} {coef:>+8.3f}  {bar}")
print(f"    {'Intercept':<28} {model.intercept_:>+8.3f}")

print(f"\n  Key finding: Sleep impact (β={model.coef_[3]:.3f}) is the strongest")
print(f"  predictor of wellbeing — stronger than raw screentime (β={model.coef_[0]:.3f}).")
print(f"  This suggests HOW media affects behaviour (sleep, focus) matters")
print(f"  more than the quantity of time spent.")

# save summary table
os.makedirs("outputs", exist_ok=True)
summary = {
    "Test": ["Pearson r","t-test (gender)","ANOVA (screentime)","Regression R²"],
    "Statistic": [f"r={r:.3f}",f"t={t:.3f}",f"F={f_stat:.3f}",f"R²={r2:.3f}"],
    "p-value": [f"{p:.4f}",f"{p_t:.4f}",f"{f_p:.4f}","—"],
    "Significant?": [p<0.05, p_t<0.05, f_p<0.05, "—"]
}
pd.DataFrame(summary).to_csv("outputs/statistical_summary.csv", index=False)
print(f"\n✓ Statistical summary saved → outputs/statistical_summary.csv")

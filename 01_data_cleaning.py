import pandas as pd
import numpy as np
import os

# paths
RAW_FILE = "data/survey_raw.xlsx"   # put your .xlsx here
OUTPUT    = "data/survey_clean.csv"

# load
df = pd.read_excel(RAW_FILE)
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

# rename columns positionally (order matches the survey form)
new_names = [
    "timestamp", "age_raw", "gender", "discipline", "level",
    "platforms", "screentime",
    "q1_self_esteem",       # Seeing others' posts online motivates my self-esteem
    "q2_comparison",        # I compare myself to others when browsing social media
    "q3_relaxation",        # I feel that media consumption always helps me relax
    "q4_concentration",     # I experience difficulty concentrating after using media
    "q5_overthinking",      # I often overthink my offline daily activities
    "q6_emotional_drain",   # I feel emotionally drained after prolonged usage
    "q7_fomo",              # I can manage my media consumption without FOMO
    "q8_sleep",             # My media usage negatively affects my sleep schedule
    "q9_info_overwhelm",    # I often feel overwhelmed by the amount of info online
    "q10_satisfaction",     # I am satisfied with how I manage my media consumption
    "q11_mental_wellbeing", # I feel my media usage affects my overall mental wellbeing
    "open_response"
]
df.columns = new_names
print("Columns renamed.")

# clean: age
# handles Bengali digits (১৮+ → 18), "21 years", "21 year", etc.
BENGALI_MAP = {"০":"0","১":"1","২":"2","৩":"3","৪":"4",
               "৫":"5","৬":"6","৭":"7","৮":"8","৯":"9"}

def clean_age(x):
    s = str(x).strip()
    for bn, en in BENGALI_MAP.items():
        s = s.replace(bn, en)
    s = s.replace("years","").replace("year","").replace("+","").strip()
    try:
        return int(float(s))
    except ValueError:
        return np.nan

df["age"] = df["age_raw"].apply(clean_age)
print(f"Age cleaned. Nulls: {df['age'].isna().sum()}")

# clean: academic discipline 
def clean_discipline(x):
    x = str(x).strip()
    medical = {"MBBS", "Medicine", "Medical Science", " Medical Science",
               "Doctor", "mbbs", "MD"}
    return "Medical Science" if x in medical else x

df["discipline_clean"] = df["discipline"].apply(clean_discipline)

# clean: level of study
df["level"] = df["level"].str.strip().str.upper().replace({"MBBS":"MBBS", "MD":"MBBS"})
# normalise to title case except known acronyms
level_map = {"BACHELOR":"Bachelor","HSC":"HSC","MASTERS":"Masters",
             "MBBS":"MBBS","SSC":"SSC"}
df["level"] = df["level"].map(level_map).fillna(df["level"].str.title())

# clean: screentime → numeric midpoint (hours)
SCREENTIME_MAP = {
    "0 to 1 hour":  0.5,
    "1 to 2 hour":  1.5,
    "2 to 3 hours": 2.5,
    "3 to 4 hours": 3.5,
    "4 to 5 hours": 4.5,
    "5+ hours":     6.0,
}
df["screentime_hours"] = df["screentime"].map(SCREENTIME_MAP)
print(f"Screentime mapped. Unmapped: {df['screentime_hours'].isna().sum()}")

# Likert columns
LIKERT = [
    "q1_self_esteem","q2_comparison","q3_relaxation","q4_concentration",
    "q5_overthinking","q6_emotional_drain","q7_fomo","q8_sleep",
    "q9_info_overwhelm","q10_satisfaction","q11_mental_wellbeing"
]

# verify all are numeric
for col in LIKERT:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nLikert scale ranges:")
print(df[LIKERT].agg(["min","max","mean"]).round(2))

# composite scores 
# Negative Impact: average of 5 harm-related items (higher = more harm)
NEGATIVE_COLS = ["q4_concentration","q5_overthinking",
                 "q6_emotional_drain","q8_sleep","q9_info_overwhelm"]
df["neg_impact"]       = df[NEGATIVE_COLS].mean(axis=1).round(3)

# Social Comparison: average of self-esteem + comparison items
df["social_comparison"] = df[["q1_self_esteem","q2_comparison"]].mean(axis=1).round(3)

# Overall Wellbeing Impact (direct from q11)
df["wellbeing_impact"]  = df["q11_mental_wellbeing"]

print("\nComposite score means:")
print(df[["neg_impact","social_comparison","wellbeing_impact"]].mean().round(3))

# save clean file
os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT, index=False)
print(f"\n✓ Clean data saved → {OUTPUT}")
print(f"  Final shape: {df.shape[0]} rows × {df.shape[1]} columns")

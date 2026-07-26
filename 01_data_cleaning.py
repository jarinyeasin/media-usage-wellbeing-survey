import pandas as pd
import numpy as np
import os

GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTQVPOI0xiU8tKewFgAqRfSmOdXcE7crT0TcSsWXRg9QIYe45hJklceMFstL7QpxtYY3NQVvjXid4s1/pub?output=csv"

USE_LOCAL = False
LOCAL_FILE = "data/survey_raw.xlsx"

OUTPUT = "data/survey_clean.csv"

# LOAD DATA — from Google Sheets or local file
if USE_LOCAL:
    print("Loading from local file...")
    df = pd.read_excel(LOCAL_FILE)
else:
    print("Fetching latest responses from Google Sheets...")
    try:
        df = pd.read_csv(GOOGLE_SHEET_URL)
        print(f"✓ Fetched {len(df)} responses from Google Sheets")

        # Save a local backup with timestamp
        os.makedirs("data", exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        backup_path = f"data/backup_{timestamp}.csv"
        df.to_csv(backup_path, index=False)
        print(f"✓ Backup saved → {backup_path}")

    except Exception as e:
        print(f"✗ Could not fetch from Google Sheets: {e}")
        print("  Falling back to local file...")
        df = pd.read_excel(LOCAL_FILE)

print(f"  Total responses loaded: {len(df)}")

# RENAME COLUMNS
new_names = [
    "timestamp", "age_raw", "gender", "discipline", "level",
    "platforms", "screentime",
    "q1_self_esteem",
    "q2_comparison",
    "q3_relaxation",
    "q4_concentration",
    "q5_overthinking",
    "q6_emotional_drain",
    "q7_fomo",
    "q8_sleep",
    "q9_info_overwhelm",
    "q10_satisfaction",
    "q11_mental_wellbeing",
    "open_response"
]

if df.shape[1] != len(new_names):
    print(f"\n⚠ WARNING: Expected {len(new_names)} columns but got {df.shape[1]}")
    print("  Your form may have changed. Current column names:")
    for i, col in enumerate(df.columns):
        print(f"  [{i}] {col}")
    print("\n  Update new_names list in this script to match.")
    raise SystemExit(1)

df.columns = new_names
print("✓ Columns renamed")

# CLEAN AGE
BENGALI_MAP = {
    "০":"0","১":"1","২":"2","৩":"3","৪":"4",
    "৫":"5","৬":"6","৭":"7","৮":"8","৯":"9"
}

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
print(f"✓ Age cleaned. Missing: {df['age'].isna().sum()}")

# CLEAN DISCIPLINE
def clean_discipline(x):
    x = str(x).strip()
    medical = {"MBBS","Medicine","Medical Science"," Medical Science",
               "Doctor","mbbs","MD"}
    return "Medical Science" if x in medical else x

df["discipline_clean"] = df["discipline"].apply(clean_discipline)

# CLEAN LEVEL
df["level"] = df["level"].astype(str).str.strip()

# SCREENTIME → NUMERIC
SCREENTIME_MAP = {
    "0 to 1 hour":  0.5,
    "1 to 2 hour":  1.5,
    "2 to 3 hours": 2.5,
    "3 to 4 hours": 3.5,
    "4 to 5 hours": 4.5,
    "5+ hours":     6.0,
}
df["screentime_hours"] = df["screentime"].map(SCREENTIME_MAP)
unmapped = df["screentime_hours"].isna().sum()
if unmapped > 0:
    print(f"⚠ {unmapped} screentime entries could not be mapped:")
    print(df[df["screentime_hours"].isna()]["screentime"].value_counts())
else:
    print(f"✓ Screentime mapped. No unmapped entries.")

# LIKERT COLUMNS → NUMERIC
LIKERT = [
    "q1_self_esteem","q2_comparison","q3_relaxation","q4_concentration",
    "q5_overthinking","q6_emotional_drain","q7_fomo","q8_sleep",
    "q9_info_overwhelm","q10_satisfaction","q11_mental_wellbeing"
]
for col in LIKERT:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# COMPOSITE SCORES
NEGATIVE_COLS = [
    "q4_concentration","q5_overthinking",
    "q6_emotional_drain","q8_sleep","q9_info_overwhelm"
]
df["neg_impact"]        = df[NEGATIVE_COLS].mean(axis=1).round(3)
df["social_comparison"] = df[["q1_self_esteem","q2_comparison"]].mean(axis=1).round(3)
df["wellbeing_impact"]  = df["q11_mental_wellbeing"]

# SUMMARY
print(f"\n{'─'*40}")
print(f"DATASET SUMMARY")
print(f"{'─'*40}")
print(f"Total responses   : {len(df)}")
print(f"Female            : {(df['gender']=='Female').sum()}")
print(f"Male              : {(df['gender']=='Male').sum()}")
print(f"Neg. impact mean  : {df['neg_impact'].mean():.2f} / 7")
print(f"Wellbeing mean    : {df['wellbeing_impact'].mean():.2f} / 7")
print(f"{'─'*40}")

# SAVE
os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT, index=False)
print(f"\n✓ Clean data saved → {OUTPUT}")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n--- SCRIPT 1 COMPLETE. Run 02_statistical_analysis.py next ---")

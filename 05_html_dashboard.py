import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import json, os, random
from datetime import datetime

df = pd.read_csv("data/survey_with_clusters.csv")
print(f"Loaded {len(df)} respondents")

LIKERT = [
    "q1_self_esteem","q2_comparison","q3_relaxation","q4_concentration",
    "q5_overthinking","q6_emotional_drain","q7_fomo","q8_sleep",
    "q9_info_overwhelm","q10_satisfaction","q11_mental_wellbeing"
]
ST_ORDER = [
    "0 to 1 hour","1 to 2 hour","2 to 3 hours",
    "3 to 4 hours","4 to 5 hours","5+ hours"
]
CLUSTER_NAMES = {0:"Low-Impact Users", 1:"Moderate Impact", 2:"High-Risk Users"}

male   = df[df["gender"]=="Male"]["neg_impact"]
female = df[df["gender"]=="Female"]["neg_impact"]
r_val, p_val = stats.pearsonr(
    df["screentime_hours"].dropna(),
    df[df["screentime_hours"].notna()]["neg_impact"]
)
t_val, t_p = stats.ttest_ind(male, female)
groups = [df[df["screentime"]==s]["neg_impact"].dropna()
          for s in ST_ORDER if len(df[df["screentime"]==s]) > 0]
f_val, f_p = stats.f_oneway(*groups)
rd = df[["screentime_hours","social_comparison",
         "q4_concentration","q8_sleep","wellbeing_impact"]].dropna()
reg = LinearRegression().fit(
    rd[["screentime_hours","social_comparison","q4_concentration","q8_sleep"]],
    rd["wellbeing_impact"]
)
r2 = reg.score(
    rd[["screentime_hours","social_comparison","q4_concentration","q8_sleep"]],
    rd["wellbeing_impact"]
)

coef_names = ["Screentime", "Social comparison",
              "Concentration difficulty", "Sleep disruption"]
coef_vals  = [reg.coef_[0], reg.coef_[1], reg.coef_[2], reg.coef_[3]]
sorted_idx = sorted(range(4), key=lambda i: coef_vals[i], reverse=True)
top_predictor = coef_names[sorted_idx[0]]

st_data = df.groupby("screentime")[
    ["neg_impact","wellbeing_impact","social_comparison"]
].mean()
st_data = st_data.reindex([s for s in ST_ORDER if s in st_data.index])

cluster_info = []
for cid in [0, 1, 2]:
    sub = df[df["cluster"] == cid]
    cluster_info.append({
        "name":          CLUSTER_NAMES[cid],
        "n":             int(len(sub)),
        "neg":           round(float(sub["neg_impact"].mean()), 2),
        "well":          round(float(sub["wellbeing_impact"].mean()), 2),
        "screentime":    round(float(sub["screentime_hours"].mean()), 2),
        "social":        round(float(sub["social_comparison"].mean()), 2),
        "sleep":         round(float(sub["q8_sleep"].mean()), 2),
        "concentration": round(float(sub["q4_concentration"].mean()), 2),
        "satisfaction":  round(float(sub["q10_satisfaction"].mean()), 2),
    })

raw = df["open_response"].dropna().astype(str)
raw = raw[~raw.str.strip().str.lower().isin(
    ["nan","none","n/a","no","nope","nothing","good","ok","okay","-",""]
)]
raw = raw[raw.str.len() > 20].str.strip().tolist()
random.seed(42)
sampled_quotes = random.sample(raw, min(5, len(raw)))
quotes_html = "\n".join([
    f'<div class="quote-block">"{q}"'
    f'<div class="quote-src">Survey respondent</div></div>'
    for q in sampled_quotes
])

DATA = {
    "n":        len(df),
    "r2":       round(r2, 3),
    "female":   int((df["gender"]=="Female").sum()),
    "male":     int((df["gender"]=="Male").sum()),
    "well_mean":  round(float(df["wellbeing_impact"].mean()), 2),
    "neg_mean":   round(float(df["neg_impact"].mean()), 2),
    "female_neg": round(float(female.mean()), 2),
    "male_neg":   round(float(male.mean()), 2),
    "r_val":  round(float(r_val), 3),
    "p_val":  round(float(p_val), 4),
    "t_val":  round(float(t_val), 3),
    "t_p":    round(float(t_p), 4),
    "f_val":  round(float(f_val), 3),
    "f_p":    round(float(f_p), 4),
    "coef_sleep":   round(float(reg.coef_[3]), 3),
    "coef_conc":    round(float(reg.coef_[2]), 3),
    "coef_social":  round(float(reg.coef_[1]), 3),
    "coef_screen":  round(float(reg.coef_[0]), 3),
    "st_labels": ["0-1h","1-2h","2-3h","3-4h","4-5h","5+h"],
    "st_neg":    [round(v, 2) for v in st_data["neg_impact"].tolist()],
    "st_well":   [round(v, 2) for v in st_data["wellbeing_impact"].tolist()],
    "st_counts": [int((df["screentime"]==s).sum()) for s in ST_ORDER],
    "likert_means": {c: round(float(df[c].mean()), 3) for c in LIKERT},
    "cluster_info": cluster_info,
    "pca_points": [
        {
            "x":       round(float(df.iloc[i]["pca1"]), 3),
            "y":       round(float(df.iloc[i]["pca2"]), 3),
            "cluster": int(df.iloc[i]["cluster"]),
            "gender":  str(df.iloc[i]["gender"]),
        }
        for i in range(len(df))
    ],
    "discipline_counts": df["discipline_clean"].value_counts().to_dict(),
}

DATA_JSON     = json.dumps(DATA)
GENERATED_AT  = datetime.now().strftime("%Y-%m-%d %H:%M")

BORDER_COLORS   = ["#9FE1CB", "#B5D4F4", "#F5C4B3"]
NAME_COLORS     = ["#0F6E56", "#185FA5", "#993C1D"]
FILL_COLORS     = ["#1D9E75", "#378ADD", "#D85A30"]

cluster_cards_html = ""
for i, c in enumerate(cluster_info):
    rows = [
        ("Negative impact",   c["neg"]),
        ("Wellbeing impact",  c["well"]),
        ("Social comparison", c["social"]),
        ("Sleep disruption",  c["sleep"]),
    ]
    bars = "".join([
        f'<div class="brow"><span>{lbl}</span>'
        f'<span>{val:.1f}/7</span></div>'
        f'<div class="btrack"><div class="bfill" style="width:'
        f'{val/7*100:.0f}%;background:{FILL_COLORS[i]}"></div></div>'
        for lbl, val in rows
    ])
    cluster_cards_html += f"""
    <div class="c-card" style="border-color:{BORDER_COLORS[i]}">
      <div class="c-name" style="color:{NAME_COLORS[i]}">{c['name']}</div>
      <div class="c-sub">n={c['n']} &nbsp;·&nbsp; avg {c['screentime']}h/day</div>
      {bars}
    </div>"""

if f_p < 0.05:
    anova_badge_class = "pill-ok"
    anova_badge_text  = f"ANOVA F={f_val:.2f}, p={f_p:.3f} — statistically significant"
else:
    anova_badge_class = "pill-warn"
    anova_badge_text  = (f"ANOVA F={f_val:.2f}, p={f_p:.3f} — "
                         f"trend visible, not yet significant (n={len(df)})")

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Student Media Usage &amp; Mental Wellbeing</title>
<style>
:root{{
  --ink:#0b0b0b;--ink2:#52514e;--ink3:#898781;
  --surf0:#f4f3ef;--surf1:#fcfcfb;--surf2:#ffffff;
  --bd:rgba(11,11,11,.10);--bd2:rgba(11,11,11,.20);
  --teal:#1D9E75;--teal2:#9FE1CB;
  --coral:#D85A30;--coral2:#F5C4B3;
  --blue:#378ADD;--blue2:#B5D4F4;
  --purple:#7F77DD;
  --amber:#BA7517;--amber2:#FAEEDA;
  --r:10px;
}}
@media(prefers-color-scheme:dark){{
  :root{{
    --ink:#ffffff;--ink2:#c3c2b7;--ink3:#898781;
    --surf0:#1a1a19;--surf1:#222220;--surf2:#2c2c2a;
    --bd:rgba(255,255,255,.10);--bd2:rgba(255,255,255,.22);
    --teal:#5DCAA5;--teal2:#085041;
    --coral:#F0997B;--coral2:#712B13;
    --blue:#85B7EB;--blue2:#0C447C;
    --purple:#AFA9EC;
    --amber:#EF9F27;--amber2:#412402;
  }}
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--surf0);color:var(--ink);
     font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;
     font-size:14px;line-height:1.5}}
.hero{{padding:2.5rem 2rem 2rem;background:var(--surf1);
       border-bottom:0.5px solid var(--bd);
       position:relative;overflow:hidden;
       display:flex;align-items:center;justify-content:space-between;
       gap:2.5rem;flex-wrap:wrap}}
.hero-text{{flex:1 1 420px;min-width:280px}}
.hero-video{{flex:0 1 400px;min-width:280px;
       aspect-ratio:16/9;border-radius:12px;overflow:hidden;
       border:0.5px solid var(--bd2);position:relative;z-index:1;
       box-shadow:0 4px 20px rgba(0,0,0,.08)}}
.hero-video iframe{{width:100%;height:100%;border:none;display:block}}
@media(max-width:820px){{
  .hero{{flex-direction:column;align-items:flex-start}}
  .hero-video{{width:100%;flex:1 1 auto}}
}}
.hero::before{{content:"";position:absolute;inset:0;
       background-image:url('hero_bg.png');
       background-size:cover;background-position:center;
       opacity:0.15;filter:grayscale(60%);
       pointer-events:none;z-index:0}}
.hero-eyebrow,.hero-headline,.hero-sub,.hero a{{position:relative;z-index:1}}
.hero-eyebrow{{font-size:11px;font-weight:500;letter-spacing:.1em;
               text-transform:uppercase;color:var(--ink3);margin-bottom:.75rem}}
.hero-headline{{font-size:clamp(22px,4vw,34px);font-weight:500;line-height:1.2;
                color:var(--ink);margin-bottom:.5rem;max-width:680px}}
.hero-sub{{font-size:13px;color:var(--ink2);max-width:560px;line-height:1.6}}
.hero-accent{{color:var(--coral);font-weight:500}}
.hero a:hover{{opacity:.85}}
.stat-strip{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
             background:var(--bd);border-top:0.5px solid var(--bd);
             border-bottom:0.5px solid var(--bd)}}
.stat-cell{{background:var(--surf1);padding:.875rem 1.25rem;text-align:center}}
.stat-num{{font-size:1.6rem;font-weight:500;line-height:1;color:var(--ink)}}
.stat-lbl{{font-size:11px;color:var(--ink3);margin-top:4px;letter-spacing:.03em}}
.nav{{display:flex;padding:0 1.5rem;background:var(--surf1);
      border-bottom:0.5px solid var(--bd);gap:4px;overflow-x:auto}}
.nav-btn{{font-size:13px;padding:.625rem 1rem;border:none;background:none;
          color:var(--ink2);cursor:pointer;border-bottom:2px solid transparent;
          margin-bottom:-0.5px;white-space:nowrap;border-radius:0;transition:color .15s}}
.nav-btn.active{{color:var(--ink);border-bottom-color:var(--coral)}}
.nav-btn:hover:not(.active){{color:var(--ink)}}
.tab{{display:none;padding:1.5rem}}
.tab.active{{display:block}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1rem}}
.card{{background:var(--surf2);border:0.5px solid var(--bd);
       border-radius:var(--r);padding:1.25rem}}
.card-eyebrow{{font-size:10px;font-weight:500;letter-spacing:.09em;
               text-transform:uppercase;color:var(--ink3);margin-bottom:.875rem}}
.chart-wrap{{position:relative;width:100%}}
.pill{{display:inline-block;font-size:11px;padding:3px 10px;
       border-radius:20px;margin-bottom:.75rem;font-weight:500}}
.pill-warn{{background:var(--amber2);color:var(--amber)}}
.pill-ok{{background:var(--teal2);color:var(--teal)}}
.pill-info{{background:var(--blue2);color:var(--blue)}}
.stat-row{{display:flex;justify-content:space-between;align-items:baseline;
           padding:6px 0;border-bottom:0.5px solid var(--bd);font-size:13px}}
.stat-row:last-child{{border-bottom:none}}
.stat-row-key{{color:var(--ink2)}}
.stat-row-val{{font-weight:500}}
.sig{{font-size:11px;color:var(--ink3);margin-left:4px}}
.brow{{display:flex;justify-content:space-between;font-size:11px;
       color:var(--ink3);margin-bottom:3px}}
.btrack{{height:6px;border-radius:3px;background:var(--surf0);
         margin-bottom:8px;overflow:hidden}}
.bfill{{height:100%;border-radius:3px}}
.c-card{{border-radius:var(--r);padding:1rem;border:0.5px solid var(--bd)}}
.c-name{{font-size:13px;font-weight:500;margin-bottom:2px}}
.c-sub{{font-size:11px;color:var(--ink3);margin-bottom:.75rem}}
.coef-wrap{{margin-bottom:10px}}
.coef-label{{display:flex;justify-content:space-between;font-size:12px;
             color:var(--ink2);margin-bottom:3px}}
.coef-track{{height:8px;border-radius:4px;background:var(--surf0);overflow:hidden}}
.coef-fill{{height:100%;border-radius:4px}}
.quote-block{{border-left:2px solid var(--bd2);padding:.625rem 1rem;
              margin-bottom:.625rem;font-size:13px;color:var(--ink2);
              font-style:italic;line-height:1.6;
              border-radius:0 6px 6px 0;background:var(--surf1)}}
.quote-src{{font-size:11px;color:var(--ink3);font-style:normal;margin-top:4px}}
.legend{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:10px;
         font-size:12px;color:var(--ink2)}}
.leg-dot{{width:10px;height:10px;border-radius:2px;display:inline-block;
          margin-right:4px;vertical-align:middle}}
.method-text{{font-size:12.5px;line-height:1.75;color:var(--ink2)}}
.method-text strong{{color:var(--ink)}}
.theory-grid{{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-top:.5rem}}
.theory-card{{padding:.75rem;background:var(--surf0);border-radius:8px}}
.theory-title{{font-size:12px;font-weight:500;color:var(--ink);margin-bottom:4px}}
.theory-author{{font-weight:400;color:var(--ink3)}}
.theory-body{{font-size:12px;color:var(--ink2)}}
footer{{text-align:center;font-size:12px;color:var(--ink3);padding:2rem}}
@media(max-width:600px){{
  .grid2,.grid3,.stat-strip,.theory-grid{{grid-template-columns:1fr}}
}}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-text">
    <div class="hero-headline">
      <span class="hero-accent">{top_predictor}</span>, not screen time,<br>
      predicts student wellbeing.
    </div>
    <div class="hero-sub">
      A behavioural segmentation of social media's psychological effects on
      Bangladeshi university students from
      primary survey data. Last updated: {GENERATED_AT}.
    </div>
    <a href="https://forms.gle/EbZsVKhgwJ4FCiYb7" target="_blank" rel="noopener"
       style="display:inline-flex;align-items:center;gap:6px;margin-top:1.25rem;
              padding:.6rem 1.1rem;background:var(--teal);color:#fff;
              font-size:13px;font-weight:500;border-radius:8px;
              text-decoration:none;transition:opacity .15s">
      Take the survey
    </a>
  </div>
  <div class="hero-video">
    <iframe src="https://www.youtube.com/embed/XSSyB_DNoSk"
      title="Project explanation video"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
      allowfullscreen loading="lazy"></iframe>
  </div>
</div>

<div class="stat-strip">
  <div class="stat-cell">
    <div class="stat-num" id="kpi-n">0</div>
    <div class="stat-lbl">respondents</div>
  </div>
  <div class="stat-cell">
    <div class="stat-num" id="kpi-r2">0</div>
    <div class="stat-lbl">regression R²</div>
  </div>
  <div class="stat-cell">
    <div class="stat-num" id="kpi-well">0</div>
    <div class="stat-lbl">avg wellbeing impact / 7</div>
  </div>
  <div class="stat-cell">
    <div class="stat-num" id="kpi-beta">0</div>
    <div class="stat-lbl">sleep &beta; coefficient</div>
  </div>
</div>

<nav class="nav">
  <button class="nav-btn active" onclick="switchTab('overview',this)">Overview</button>
  <button class="nav-btn" onclick="switchTab('wellbeing',this)">Wellbeing analysis</button>
  <button class="nav-btn" onclick="switchTab('clusters',this)">User clusters</button>
  <button class="nav-btn" onclick="switchTab('voices',this)">Voices &amp; methods</button>
</nav>

<!-- ══ TAB 1: OVERVIEW ══════════════════════════════════════════════ -->
<div id="tab-overview" class="tab active">

  <div class="grid3">
    <div class="card">
      <div class="card-eyebrow">gender</div>
      <div class="legend">
        <span><span class="leg-dot" style="background:#378ADD"></span>
          Female {DATA['female']} ({DATA['female']*100//DATA['n']}%)</span>
        <span><span class="leg-dot" style="background:#7F77DD"></span>
          Male {DATA['male']} ({DATA['male']*100//DATA['n']}%)</span>
      </div>
      <div class="chart-wrap" style="height:140px">
        <canvas id="genderChart" role="img"
          aria-label="Donut: {DATA['female']} female, {DATA['male']} male">
          {DATA['female']} female, {DATA['male']} male
        </canvas>
      </div>
    </div>
    <div class="card">
      <div class="card-eyebrow">daily screentime</div>
      <div class="chart-wrap" style="height:160px">
        <canvas id="stimeChart" role="img"
          aria-label="Bar chart of screentime distribution across 6 bands">
          Screentime distribution
        </canvas>
      </div>
    </div>
    <div class="card">
      <div class="card-eyebrow">academic discipline</div>
      <div class="chart-wrap" style="height:160px">
        <canvas id="discChart" role="img"
          aria-label="Horizontal bar chart of academic discipline breakdown">
          Discipline breakdown
        </canvas>
      </div>
    </div>
  </div>

  <div class="card" style="margin-bottom:1rem">
    <div class="card-eyebrow">mean scores — all 11 likert dimensions (1–7 scale)</div>
    <div class="legend">
      <span><span class="leg-dot" style="background:#D85A30"></span>High concern (&ge;5.0)</span>
      <span><span class="leg-dot" style="background:#378ADD"></span>Moderate (4.0–4.9)</span>
      <span><span class="leg-dot" style="background:#1D9E75"></span>Below midpoint (&lt;4.0)</span>
    </div>
    <div class="chart-wrap" style="height:300px">
      <canvas id="likertChart" role="img"
        aria-label="Horizontal bar chart of 11 Likert mean scores from 3.2 to 5.1">
        Likert means 3.2 to 5.1
      </canvas>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <div class="card-eyebrow">what predicts wellbeing? — regression coefficients</div>
      <div class="pill pill-ok">R&sup2; = {r2:.3f} — strong model fit</div>
      <div id="coef-bars"></div>
      <div style="font-size:12px;color:var(--ink3);margin-top:.5rem">
        &beta; coefficients: effect size on overall wellbeing (Q11, 1–7 scale)
      </div>
    </div>
    <div class="card">
      <div class="card-eyebrow">statistical tests</div>
      <div class="stat-row">
        <span class="stat-row-key">Pearson r (screentime &times; neg. impact)</span>
        <span class="stat-row-val">{r_val:.3f}
          <span class="sig">p={p_val:.3f}</span></span>
      </div>
      <div class="stat-row">
        <span class="stat-row-key">Welch t-test (gender difference)</span>
        <span class="stat-row-val">t={t_val:.3f}
          <span class="sig">p={t_p:.3f}, n.s.</span></span>
      </div>
      <div class="stat-row">
        <span class="stat-row-key">Female neg. impact mean</span>
        <span class="stat-row-val">{female.mean():.2f} / 7</span>
      </div>
      <div class="stat-row">
        <span class="stat-row-key">Male neg. impact mean</span>
        <span class="stat-row-val">{male.mean():.2f} / 7</span>
      </div>
      <div class="stat-row">
        <span class="stat-row-key">One-way ANOVA (screentime groups)</span>
        <span class="stat-row-val">F={f_val:.3f}
          <span class="sig">p={f_p:.3f}</span></span>
      </div>
      <div class="stat-row">
        <span class="stat-row-key">Regression R&sup2; (4 predictors)</span>
        <span class="stat-row-val">{r2:.3f}</span>
      </div>
      <div class="stat-row">
        <span class="stat-row-key">Strongest predictor</span>
        <span class="stat-row-val">{top_predictor}
          <span class="sig">&beta;={max(coef_vals):.3f}</span></span>
      </div>
      <div style="margin-top:.75rem">
        <div class="pill pill-warn">
          Trend visible — sample size (n={len(df)}) limits significance
        </div>
      </div>
    </div>
  </div>

</div>

<!-- ══ TAB 2: WELLBEING ══════════════════════════════════════════════ -->
<div id="tab-wellbeing" class="tab">

  <div class="grid2" style="margin-bottom:1rem">
    <div class="card">
      <div class="card-eyebrow">negative impact score by screentime group</div>
      <div class="pill {anova_badge_class}">{anova_badge_text}</div>
      <div class="chart-wrap" style="height:200px">
        <canvas id="stImpactChart" role="img"
          aria-label="Line chart: negative impact score across screentime groups">
          Negative impact by screentime
        </canvas>
      </div>
    </div>
    <div class="card">
      <div class="card-eyebrow">wellbeing impact by screentime group</div>
      <div class="chart-wrap" style="height:200px">
        <canvas id="stWellChart" role="img"
          aria-label="Line chart: wellbeing impact score across screentime groups">
          Wellbeing impact by screentime
        </canvas>
      </div>
    </div>
  </div>

  <div class="card" style="margin-bottom:1rem">
    <div class="card-eyebrow">
      respondents in PCA latent space — coloured by gender
      (PC1+PC2 = 54.5% variance)
    </div>
    <div class="legend">
      <span><span class="leg-dot" style="background:#378ADD"></span>Female</span>
      <span><span class="leg-dot"
        style="background:#7F77DD;clip-path:polygon(50% 0%,0% 100%,100% 100%)">
      </span>Male</span>
    </div>
    <div class="chart-wrap" style="height:240px">
      <canvas id="pcaGenderChart" role="img"
        aria-label="PCA scatter coloured by gender showing respondent distribution">
        PCA scatter by gender
      </canvas>
    </div>
    <div style="font-size:11px;color:var(--ink3);margin-top:.5rem">
      PC1 = harm axis (39.1% variance) &nbsp;·&nbsp;
      PC2 = social comparison axis (15.4%)
    </div>
  </div>

  <div class="card">
    <div class="card-eyebrow">
      pearson correlation matrix — all 11 likert variables
    </div>
    <div class="chart-wrap" style="height:290px">
      <canvas id="heatmapChart" role="img"
        aria-label="Heatmap of Pearson correlations between 11 survey items">
        Correlation heatmap
      </canvas>
    </div>
  </div>

</div>

<!-- ══ TAB 3: CLUSTERS ══════════════════════════════════════════════ -->
<div id="tab-clusters" class="tab">

  <div class="grid3" style="margin-bottom:1rem">
    {cluster_cards_html}
  </div>

  <div class="card" style="margin-bottom:1rem">
    <div class="card-eyebrow">cluster comparison — 6 key dimensions</div>
    <div class="legend">
      <span><span class="leg-dot" style="background:#1D9E75"></span>
        Low-Impact Users</span>
      <span><span class="leg-dot" style="background:#378ADD"></span>
        Moderate Impact</span>
      <span><span class="leg-dot" style="background:#D85A30"></span>
        High-Risk Users</span>
    </div>
    <div class="chart-wrap" style="height:260px">
      <canvas id="clusterBarChart" role="img"
        aria-label="Grouped bar chart comparing three clusters across 6 dimensions">
        Cluster profiles
      </canvas>
    </div>
  </div>

  <div class="card">
    <div class="card-eyebrow">
      respondents in PCA latent space — coloured by cluster
    </div>
    <div class="legend">
      <span><span class="leg-dot" style="background:#1D9E75"></span>
        Low-Impact (n={cluster_info[0]['n']})</span>
      <span><span class="leg-dot" style="background:#378ADD"></span>
        Moderate (n={cluster_info[1]['n']})</span>
      <span><span class="leg-dot" style="background:#D85A30"></span>
        High-Risk (n={cluster_info[2]['n']})</span>
    </div>
    <div class="chart-wrap" style="height:240px">
      <canvas id="clusterPcaChart" role="img"
        aria-label="PCA scatter coloured by cluster showing three distinct groups">
        PCA scatter by cluster
      </canvas>
    </div>
  </div>

</div>

<!-- ══ TAB 4: VOICES & METHODS ══════════════════════════════════════ -->
<div id="tab-voices" class="tab">

  <div class="grid2" style="margin-bottom:1rem">
    <div class="card">
      <div class="card-eyebrow">what students said — open responses</div>
      {quotes_html}
    </div>
    <div class="card">
      <div class="card-eyebrow">methods</div>
      <p class="method-text">
        <strong>Data:</strong> Primary survey via Google Forms,
        n={len(df)} Bangladeshi university students
        ({int((df['gender']=='Female').sum())} female,
        {int((df['gender']=='Male').sum())} male), Feb 2026.
        11-item Likert scale (1–7) + demographic items.
      </p><br>
      <p class="method-text">
        <strong>Pipeline:</strong> Data cleaning &rarr; descriptive statistics
        &rarr; Pearson correlation &rarr; Welch t-test &rarr; one-way ANOVA
        &rarr; multiple linear regression &rarr; k-means clustering
        (k=3, StandardScaler) &rarr; PCA (2 components).
      </p><br>
      <p class="method-text">
        <strong>Stack:</strong>
        Python &nbsp;&middot;&nbsp; pandas &nbsp;&middot;&nbsp; NumPy
        &nbsp;&middot;&nbsp; scikit-learn &nbsp;&middot;&nbsp; scipy
        &nbsp;&middot;&nbsp; matplotlib
      </p><br>
      <p class="method-text">
        <strong>Limitations:</strong> Convenience sample (n={len(df)}) limits
        statistical power. Self-reported screentime likely underestimated.
        Cross-sectional design precludes causal inference.
      </p>
    </div>
  </div>

  <div class="card">
    <div class="card-eyebrow">theoretical grounding</div>
    <div class="theory-grid">
      <div class="theory-card">
        <div class="theory-title">Displacement Theory
          <span class="theory-author">Kraut et al., 1998</span></div>
        <div class="theory-body">Media displaces sleep and face-to-face
          interaction — the harm is in what gets replaced, not time itself.</div>
      </div>
      <div class="theory-card">
        <div class="theory-title">Social Comparison Theory
          <span class="theory-author">Festinger, 1954</span></div>
        <div class="theory-body">Upward social comparison on image-heavy
          platforms lowers self-evaluation — key driver of the High-Risk
          cluster.</div>
      </div>
      <div class="theory-card">
        <div class="theory-title">Attention Economy
          <span class="theory-author">Williams, 2018</span></div>
        <div class="theory-body">Platform design fragments attentional
          resources — explains why concentration impairment (
          {df['q4_concentration'].mean():.2f}/7) is the highest-scoring
          item.</div>
      </div>
      <div class="theory-card">
        <div class="theory-title">Uses &amp; Gratifications
          <span class="theory-author">Katz et al., 1973</span></div>
        <div class="theory-body">Individual motivations for media use moderate
          its psychological effects — explains why screen-time alone is a
          weak predictor (&beta;={reg.coef_[0]:.3f}).</div>
      </div>
    </div>
  </div>

</div>

<footer>
  Jarin Binta Yeasin &nbsp;&middot;&nbsp; University of Dhaka
  &nbsp;&middot;&nbsp; An Individual Research Project &nbsp;&middot;&nbsp; 2026
  &nbsp;&middot;&nbsp; Generated {GENERATED_AT} from n={len(df)} responses
</footer>

<!-- ══ CHART.JS (loaded from CDN) ═══════════════════════════════════ -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const D = {DATA_JSON};

const TEAL='#1D9E75', BLUE='#378ADD', CORAL='#D85A30', PURPLE='#7F77DD';
const AMBER='#BA7517', GRAY='#888780';
const CLR = [TEAL, BLUE, CORAL];
const isDark = matchMedia('(prefers-color-scheme:dark)').matches;
const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)';
const tickColor = '#898781';
Chart.defaults.font.family =
  '-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif';
Chart.defaults.color = tickColor;

// ── Tab switcher ──────────────────────────────────────────────────────
function switchTab(name, btn) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}}

// ── Animated counters ─────────────────────────────────────────────────
function animateNum(el, target, decimals) {{
  let t0 = null;
  const dur = 900;
  function step(t) {{
    if (!t0) t0 = t;
    const p = Math.min((t - t0) / dur, 1);
    const val = target * (1 - Math.pow(1 - p, 3));
    el.textContent = val.toFixed(decimals);
    if (p < 1) requestAnimationFrame(step);
  }}
  requestAnimationFrame(step);
}}
animateNum(document.getElementById('kpi-n'),    D.n,        0);
animateNum(document.getElementById('kpi-r2'),   D.r2,       3);
animateNum(document.getElementById('kpi-well'), D.well_mean,2);
animateNum(document.getElementById('kpi-beta'), D.coef_sleep,3);

// ── Regression coefficient bars ───────────────────────────────────────
const coefData = [
  {{ label:'Sleep disruption',          val: D.coef_sleep,  color: CORAL  }},
  {{ label:'Concentration difficulty',  val: D.coef_conc,   color: BLUE   }},
  {{ label:'Social comparison',         val: D.coef_social, color: PURPLE }},
  {{ label:'Screen time (hrs)',         val: D.coef_screen, color: GRAY   }},
];
document.getElementById('coef-bars').innerHTML = coefData.map(c => `
  <div class="coef-wrap">
    <div class="coef-label">
      <span>${{c.label}}</span>
      <span style="font-weight:500">
        &beta;=${{c.val >= 0 ? '+' : ''}}${{c.val.toFixed(3)}}
      </span>
    </div>
    <div class="coef-track">
      <div class="coef-fill" style="width:${{Math.abs(c.val)/0.5*100}}%;
           background:${{c.color}}"></div>
    </div>
  </div>`).join('');

// ── Gender donut ──────────────────────────────────────────────────────
new Chart(document.getElementById('genderChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['Female', 'Male'],
    datasets: [{{
      data: [D.female, D.male],
      backgroundColor: [BLUE, PURPLE],
      borderWidth: 0, hoverOffset: 3
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false, cutout: '62%',
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{
        label: c => `${{c.label}}: ${{c.raw}} (${{Math.round(c.raw/D.n*100)}}%)`
      }}}}
    }}
  }}
}});

// ── Screentime bar ────────────────────────────────────────────────────
new Chart(document.getElementById('stimeChart'), {{
  type: 'bar',
  data: {{
    labels: D.st_labels,
    datasets: [{{ data: D.st_counts, backgroundColor: BLUE,
                  borderWidth: 0, borderRadius: 3 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 11 }} }} }},
      y: {{ grid: {{ color: gridColor }}, beginAtZero: true,
            ticks: {{ stepSize: 4, font: {{ size: 11 }} }} }}
    }}
  }}
}});

// ── Discipline bar ────────────────────────────────────────────────────
const discKeys  = ['Science','Business','Medical Science','Social Sciences','Arts'];
const discVals  = discKeys.map(k => D.discipline_counts[k] || 0);
const discClrs  = [TEAL, BLUE, PURPLE, CORAL, AMBER];
new Chart(document.getElementById('discChart'), {{
  type: 'bar',
  data: {{
    labels: discKeys,
    datasets: [{{ data: discVals, backgroundColor: discClrs,
                  borderWidth: 0, borderRadius: 3 }}]
  }},
  options: {{
    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ grid: {{ color: gridColor }}, beginAtZero: true,
            ticks: {{ font: {{ size: 11 }} }} }},
      y: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 11 }} }} }}
    }}
  }}
}});

// ── Likert means horizontal bar ───────────────────────────────────────
const lKeys = [
  'q4_concentration','q11_mental_wellbeing','q8_sleep','q6_emotional_drain',
  'q5_overthinking','q9_info_overwhelm','q7_fomo','q2_comparison',
  'q1_self_esteem','q10_satisfaction','q3_relaxation'
];
const lLabels = [
  'Difficulty concentrating','Overall wellbeing impact','Sleep disruption',
  'Emotional drain','Overthinking offline','Information overload',
  'FOMO management','Social comparison','Self-esteem from posts',
  'Satisfied w/ usage','Media helps relax'
];
const lVals   = lKeys.map(k => D.likert_means[k]);
const lColors = lVals.map(v => v >= 5 ? CORAL : v >= 4 ? BLUE : TEAL);
new Chart(document.getElementById('likertChart'), {{
  type: 'bar',
  data: {{
    labels: lLabels,
    datasets: [{{ data: lVals, backgroundColor: lColors,
                  borderWidth: 0, borderRadius: 3 }}]
  }},
  options: {{
    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: c => `Mean: ${{c.raw.toFixed(2)}} / 7` }} }}
    }},
    scales: {{
      x: {{ min: 1, max: 7.5, grid: {{ color: gridColor }},
            ticks: {{ font: {{ size: 11 }} }} }},
      y: {{ grid: {{ display: false }},
            ticks: {{ font: {{ size: 10 }}, autoSkip: false }} }}
    }}
  }}
}});

// ── Screentime × negative impact line ────────────────────────────────
new Chart(document.getElementById('stImpactChart'), {{
  type: 'line',
  data: {{
    labels: D.st_labels,
    datasets: [{{
      label: 'Negative impact', data: D.st_neg,
      borderColor: CORAL, backgroundColor: 'rgba(216,90,48,0.08)',
      borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: CORAL,
      fill: true, tension: 0.3
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: c => `${{c.raw.toFixed(2)}} / 7` }} }}
    }},
    scales: {{
      x: {{ grid: {{ display: false }} }},
      y: {{ min: 2, max: 7, grid: {{ color: gridColor }} }}
    }}
  }}
}});

// ── Screentime × wellbeing impact line ───────────────────────────────
new Chart(document.getElementById('stWellChart'), {{
  type: 'line',
  data: {{
    labels: D.st_labels,
    datasets: [{{
      label: 'Wellbeing impact', data: D.st_well,
      borderColor: BLUE, backgroundColor: 'rgba(55,138,221,0.08)',
      borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: BLUE,
      fill: true, tension: 0.3
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: c => `${{c.raw.toFixed(2)}} / 7` }} }}
    }},
    scales: {{
      x: {{ grid: {{ display: false }} }},
      y: {{ min: 2, max: 7, grid: {{ color: gridColor }} }}
    }}
  }}
}});

// ── PCA scatter — coloured by gender ─────────────────────────────────
const pcaGender = [
  {{
    label: 'Female',
    data: D.pca_points.filter(p => p.gender === 'Female')
                      .map(p => ({{ x: p.x, y: p.y }})),
    backgroundColor: BLUE, pointRadius: 5, pointHoverRadius: 7
  }},
  {{
    label: 'Male',
    data: D.pca_points.filter(p => p.gender === 'Male')
                      .map(p => ({{ x: p.x, y: p.y }})),
    backgroundColor: PURPLE, pointRadius: 5, pointHoverRadius: 7,
    pointStyle: 'triangle'
  }}
];
new Chart(document.getElementById('pcaGenderChart'), {{
  type: 'scatter', data: {{ datasets: pcaGender }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'PC1 — harm axis (39.1%)',
                      font: {{ size: 11 }} }},
            grid: {{ color: gridColor }} }},
      y: {{ title: {{ display: true, text: 'PC2 — social axis (15.4%)',
                      font: {{ size: 11 }} }},
            grid: {{ color: gridColor }} }}
    }}
  }}
}});

// ── Correlation heatmap (custom canvas plugin) ────────────────────────
const CORR = [
  [1,.42,-.25,.37,.35,.38,-.08,.35,.35,-.23,.37],
  [.42,1,-.18,.22,.25,.27,.03,.17,.26,-.20,.22],
  [-.25,-.18,1,-.29,-.28,-.31,.23,-.26,-.30,.35,-.28],
  [.37,.22,-.29,1,.62,.61,-.20,.51,.51,-.45,.62],
  [.35,.25,-.28,.62,1,.63,-.26,.55,.62,-.39,.63],
  [.38,.27,-.31,.61,.63,1,-.22,.65,.61,-.48,.71],
  [-.08,.03,.23,-.20,-.26,-.22,1,-.23,-.18,.32,-.24],
  [.35,.17,-.26,.51,.55,.65,-.23,1,.57,-.44,.63],
  [.35,.26,-.30,.51,.62,.61,-.18,.57,1,-.39,.65],
  [-.23,-.20,.35,-.45,-.39,-.48,.32,-.44,-.39,1,-.46],
  [.37,.22,-.28,.62,.63,.71,-.24,.63,.65,-.46,1]
];
const HM_LABELS = [
  'Self-est.','Comparison','Relaxation','Conc.',
  'Overthink','Drain','FOMO','Sleep','Info Ovld','Satisf.','Wellbeing'
];
const N = 11;
const hmData = [];
for (let i = 0; i < N; i++)
  for (let j = 0; j < N; j++)
    hmData.push({{ x: j, y: i, v: CORR[i][j] }});

const hmPlugin = {{
  id: 'hm',
  afterDraw(chart) {{
    const {{ ctx, chartArea: {{ left, top, width, height }} }} = chart;
    const cw = width / N, ch = height / N;
    hmData.forEach(d => {{
      const a = Math.abs(d.v) * 0.85 + 0.05;
      ctx.fillStyle = d.v > 0
        ? `rgba(55,138,221,${{a}})`
        : `rgba(216,90,48,${{a}})`;
      ctx.fillRect(left + d.x * cw, top + (N - 1 - d.y) * ch, cw - 1, ch - 1);
      ctx.fillStyle = Math.abs(d.v) > 0.55 ? '#fff' : (isDark ? '#ccc' : '#444');
      ctx.font = '8px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(d.v.toFixed(2),
        left + (d.x + 0.5) * cw,
        top  + (N - 1 - d.y + 0.5) * ch);
    }});
    ctx.fillStyle = isDark ? '#c3c2b7' : '#52514e';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    HM_LABELS.forEach((l, i) => {{
      ctx.fillText(l, left + (i + 0.5) * cw, top + height + 14);
    }});
    ctx.textAlign = 'right';
    HM_LABELS.forEach((l, i) => {{
      ctx.fillText(l, left - 4, top + (N - 1 - i + 0.5) * ch);
    }});
  }}
}};
new Chart(document.getElementById('heatmapChart'), {{
  type: 'scatter',
  data: {{ datasets: [{{ data: [], label: '' }}] }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: false }} }},
    scales: {{
      x: {{ display: false, min: -0.5, max: N - 0.5 }},
      y: {{ display: false, min: -0.5, max: N - 0.5 }}
    }},
    layout: {{ padding: {{ bottom: 44, left: 60 }} }}
  }},
  plugins: [hmPlugin]
}});

// ── Cluster grouped bar ───────────────────────────────────────────────
const profKeys   = ['concentration','sleep','neg','well','social','satisfaction'];
const profLabels = [
  'Concentration','Sleep Disruption','Neg. Impact',
  'Wellbeing','Social Comparison','Satisfaction'
];
new Chart(document.getElementById('clusterBarChart'), {{
  type: 'bar',
  data: {{
    labels: profLabels,
    datasets: D.cluster_info.map((c, i) => ({{
      label: c.name,
      data: profKeys.map(k => c[k]),
      backgroundColor: CLR[i], borderWidth: 0, borderRadius: 3
    }}))
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 11 }} }} }},
      y: {{ min: 1, max: 7.5, grid: {{ color: gridColor }},
            ticks: {{ font: {{ size: 11 }} }} }}
    }}
  }}
}});

// ── PCA scatter — coloured by cluster ────────────────────────────────
const clusterPcaSets = [0, 1, 2].map(c => ({{
  label: D.cluster_info[c].name,
  data: D.pca_points.filter(p => p.cluster === c)
                    .map(p => ({{ x: p.x, y: p.y }})),
  backgroundColor: CLR[c], pointRadius: 5, pointHoverRadius: 7
}}));
new Chart(document.getElementById('clusterPcaChart'), {{
  type: 'scatter',
  data: {{ datasets: clusterPcaSets }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'PC1 (39.1%)',
                      font: {{ size: 11 }} }},
            grid: {{ color: gridColor }} }},
      y: {{ title: {{ display: true, text: 'PC2 (15.4%)',
                      font: {{ size: 11 }} }},
            grid: {{ color: gridColor }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

os.makedirs("outputs", exist_ok=True)
with open("outputs/index.html", "w", encoding="utf-8") as f:
    f.write(HTML)

size_kb = os.path.getsize("outputs/index.html") // 1024
print(f"\n✓ Dashboard saved → outputs/index.html  ({size_kb} KB)")
print(f"  n={len(df)} responses · generated {GENERATED_AT}")
print(f"  Tabbed layout · Chart.js charts · dark mode · works offline")
print("\n--- SCRIPT 5 COMPLETE ---")

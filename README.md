# Student Media Usage and Mental Wellbeing Analysis

**Jarin Binta Yeasin**
| Department of Mass Communication and Journalism | University of Dhaka

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas)](https://pandas.pydata.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![scipy](https://img.shields.io/badge/scipy-1.x-8CAAE6)](https://scipy.org)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.x-11557c)](https://matplotlib.org)

> **🔗 [View Live Dashboard](https://jarinyeasin.github.io/media-usage-wellbeing-project/)**

---

## Abstract

This study presents a mixed-methods computational analysis of a primary survey (n=56) administered to Bangladeshi students across disciplines in February 2026. Using an 11-item 7-point Likert instrument, I measured dimensions including social comparison, emotional drain, sleep disruption, concentration difficulty, and overall perceived impact on mental wellbeing.

Applied 4 statistical tests — Pearson correlation, Welch's independent-samples t-test, one-way ANOVA, and multiple linear regression, alongside two unsupervised machine learning techniques: k-means clustering (k=3) and Principal Component Analysis (PCA). The regression model (R²=0.518) identified **sleep disruption** as the dominant predictor of wellbeing impact (β=0.449), significantly outweighing raw screentime (β=−0.045). Clustering revealed 3 behavioural profiles — Low-Impact Users (32%), Moderate Impact (43%), and High-Risk Users (25%) — with the High-Risk segment distinguished by markedly elevated social comparison scores (mean=5.9/7). These findings carry implications for digital literacy education and student mental health policy in resource-constrained university settings.

---

## Table of Contents

- [Motivation](#motivation)
- [Research Questions](#research-questions)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Reproducing This Study](#reproducing-this-study)
- [Visualisations](#visualisations)
- [Limitations](#limitations)
- [Academic Context](#academic-context)
- [Author](#author)

---

## Motivation

University students navigating academic pressure, identity crisis, and economic uncertainty simultaneously represent a particularly vulnerable demographic for social media's psychological effects. Most existing research on social media and mental health originates from Western populations, limiting the generalisability of findings to South Asian contexts.

This project addresses that gap through a small data-driven attempt, combining primary survey design with a full computational analysis pipeline from raw data cleaning through unsupervised behavioural profiling. It is also a demonstration of applied data science on a real-world social science problem, built entirely with open-source Python tools.

---

## Research Questions

1. Is there a statistically significant relationship between daily social media screentime and negative psychological impact among young adult students?
2. Do gender differences exist in how students experience media-related psychological harm?
3. Which behavioural and psychological variables are the strongest predictors of overall perceived wellbeing impact?
4. Can unsupervised clustering identify meaningful, interpretable user profiles with distinct risk characteristics?

---

## Dataset

| Attribute | Details |
|---|---|
| **Collection method** | Primary survey via Google Forms |
| **Population** | Young Adult students living in Dhaka city |
| **Sample size** | n = 56 |
| **Collection period** | February 10–13, 2026 |
| **Instrument** | 11-item Likert scale (1–7) + demographic items |
| **Gender breakdown** | Female: 34 (61%), Male: 22 (39%) |
| **Disciplines** | Science, Business, Medical Science, Social Sciences, Arts |
| **Study levels** | Bachelor, HSC, Masters, MBBS |

### Survey Instrument — Likert Items (1 = Strongly Disagree, 7 = Strongly Agree)

| Code | Question |
|---|---|
| Q1 | Seeing others' posts online motivates my self-esteem |
| Q2 | I compare myself to others when browsing social media |
| Q3 | I feel that media consumption always helps me relax |
| Q4 | I experience difficulty concentrating after using media platforms |
| Q5 | I often overthink my offline daily activities |
| Q6 | I feel emotionally drained after prolonged usage of social media |
| Q7 | I can manage my media consumption without having FOMO |
| Q8 | My media usage negatively affects my sleep schedule |
| Q9 | I often feel overwhelmed by the amount of information I consume online |
| Q10 | I am satisfied with how I manage my media consumption |
| Q11 | I feel my media usage affects my overall mental wellbeing |

### Composite Scores Derived

- **Negative Impact Score** — mean of Q4, Q5, Q6, Q8, Q9 (five harm-related dimensions)
- **Social Comparison Score** — mean of Q1, Q2
- **Wellbeing Impact** — direct from Q11 (outcome variable in regression)

---

## Methodology

The analysis pipeline consists of 5 sequential stages, each implemented in a dedicated Python script.

### 1. Data Cleaning & Preprocessing (`01_data_cleaning.py`)

- Parsed heterogeneous age entries including Bengali numeral variants (e.g. ১৮ → 18) and free-text formats
- Standardised academic discipline categories (MBBS / Medicine / Medical Science → Medical Science)
- Mapped ordinal screentime bands to numeric midpoint values (e.g. "2 to 3 hours" → 2.5h)
- Validated all Likert items for numeric type and range
- Derived three composite scores for use in downstream analysis

### 2. Statistical Analysis (`02_statistical_analysis.py`)

| Test | Variables | Purpose |
|---|---|---|
| Pearson correlation | Screentime × Negative Impact | Association strength |
| Welch's t-test | Gender × Negative Impact | Group difference |
| One-way ANOVA | Screentime group × Negative Impact | Multi-group comparison |
| Multiple linear regression | 4 predictors → Wellbeing Impact | Prediction & effect sizes |

### 3. Machine Learning (`03_ml_clustering_pca.py`)

**K-Means Clustering**
- Features: all 11 Likert items, standardised via `StandardScaler` (mean=0, std=1)
- Optimal k selected using the Elbow Method and Silhouette Score across k=2 to k=8
- Final model: k=3, validated with silhouette score
- Clusters re-labelled by ascending Negative Impact mean for interpretability

**Principal Component Analysis**
- Input: 11 standardised Likert variables
- Output: 2 components explaining 54.5% of total variance (PC1=39.1%, PC2=15.4%)
- PC1 interpreted as a "harm axis" (high loadings on concentration, sleep, drain)
- PC2 interpreted as a "social comparison axis"

### 4. Visualisation (`04_visualisations.py`)

Eight publication-quality PNG figures generated via matplotlib: demographics, Likert means, screentime–impact relationship, gender comparison, Pearson correlation matrix, elbow/silhouette diagnostics, radar chart of cluster profiles, and PCA scatter.

### 5. Interactive Dashboard (`05_html_dashboard.py`)

Fully self-contained HTML report with all charts embedded as base64-encoded PNG images. Zero external dependencies, opens in any browser, works completely offline.

---

## Results

### Descriptive Statistics

| Dimension | Mean (1–7) | SD | Interpretation |
|---|---|---|---|
| Difficulty concentrating (Q4) | **5.11** | 1.64 | High concern |
| Overall wellbeing impact (Q11) | **5.04** | 1.73 | High concern |
| Sleep disruption (Q8) | **4.91** | 1.84 | Moderate–high |
| Emotional drain (Q6) | **4.88** | 1.77 | Moderate–high |
| Overthinking offline (Q5) | **4.70** | 1.79 | Moderate |
| Information overload (Q9) | **4.57** | 1.78 | Moderate |
| FOMO management (Q7) | **4.54** | 1.83 | Moderate |
| Self-esteem from posts (Q1) | 3.88 | 1.85 | Below midpoint |
| Social comparison (Q2) | 3.91 | 1.80 | Below midpoint |
| Satisfied with usage (Q10) | 3.54 | 1.79 | Below midpoint |
| Media helps relax (Q3) | 3.21 | 1.63 | Below midpoint |

### Inferential Statistics

| Test | Statistic | p-value | Interpretation |
|---|---|---|---|
| Pearson r (screentime × neg. impact) | r = 0.151 | 0.267 | Weak positive trend, not significant |
| Welch t-test (gender) | t = −0.305 | 0.761 | No significant gender difference |
| One-way ANOVA (screentime groups) | F = 1.256 | 0.298 | Trend visible, not significant at n=56 |
| Regression R² | 0.518 | — | Strong model fit |

### Regression Coefficients

| Predictor | β | Interpretation |
|---|---|---|
| Sleep disruption (Q8) | **+0.449** | Strongest predictor |
| Concentration difficulty (Q4) | **+0.250** | Second strongest |
| Social comparison score | +0.228 | Moderate effect |
| Daily screentime (hours) | −0.045 | Weakest — near zero |

### Cluster Profiles

| Profile | n (%) | Neg. Impact | Wellbeing | Avg Screentime | Defining Feature |
|---|---|---|---|---|---|
| Low-Impact Users | 18 (32%) | 3.04 / 7 | 3.2 / 7 | 3.7h | Low harm across all dimensions |
| Moderate Impact | 24 (43%) | 5.55 / 7 | 5.8 / 7 | 3.5h | High harm, low satisfaction |
| High-Risk Users | 14 (25%) | 5.90 / 7 | 6.1 / 7 | 4.0h | Very high social comparison (5.9/7) |

---

## Key Findings

**1. Sleep, not screentime, drives wellbeing outcomes.**
Raw daily screentime contributes almost nothing to wellbeing impact (β=−0.045) once behavioural mediators are controlled. Sleep disruption (β=0.449) and concentration difficulty (β=0.250) are the primary mechanisms through which social media harms student wellbeing. This aligns with "displacement theory" in media psychology, the problem is not time spent but vital functions displaced.

**2. One in four students is at high psychological risk.**
The High-Risk cluster (25%) scores near the ceiling on nearly every harm dimension and is uniquely characterised by elevated social comparison, suggesting that upward social comparison on social media is a distinct risk factor, not merely a correlate of heavy use.

**3. Statistical non-significance is itself a finding.**
The absence of a significant gender difference (p=0.761) and the non-significant ANOVA (p=0.298) are meaningful at this sample size. They suggest that psychological harm from social media is broadly distributed across gender and usage levels in this population, a pattern that warrants investigation with a larger sample.

**4. Concentration impairment is the most prevalent single symptom.**
Q4 (difficulty concentrating after media use) has the highest mean of all 11 items (5.11/7), above even overall wellbeing impact (5.04/7). This has direct implications for academic performance and points to attention as the primary resource depleted by social media use.

---

## Project Structure

```
media-wellbeing-bangladesh/
│
├── data/
│   ├── survey_raw.xlsx              # raw survey responses (not tracked in git)
│   ├── survey_clean.csv             # output of script 01
│   └── survey_with_clusters.csv    # output of script 03 (adds cluster + PCA cols)
│
├── outputs/
│   ├── index.html               # self-contained interactive report
│   ├── fig1_demographics.png
│   ├── fig2_likert_means.png
│   ├── fig3_screentime_impact.png
│   ├── fig4_gender_comparison.png
│   ├── fig5_correlation_heatmap.png
│   ├── fig6_elbow_silhouette.png
│   ├── fig7_cluster_radar.png
│   ├── fig8_pca_scatter.png
│   ├── statistical_summary.csv
│   ├── cluster_profiles.csv
│   └── pca_loadings.csv
│
├── 01_data_cleaning.py              # load, clean, derive composite scores
├── 02_statistical_analysis.py       # correlation, t-test, ANOVA, regression
├── 03_ml_clustering_pca.py          # k-means clustering + PCA
├── 04_visualisations.py             # 8 publication-quality PNG charts
├── 05_html_dashboard.py             # self-contained HTML report
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Reproducing This Study

### Requirements

```
Python 3.10+
pandas>=2.0
openpyxl>=3.1
numpy>=1.24
scipy>=1.10
scikit-learn>=1.2
matplotlib>=3.7
```

Install all at once:
```bash
python -m pip install pandas openpyxl numpy scipy scikit-learn matplotlib
```

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/jarinyeasin/media-usage-wellbeing-project.git
cd media-usage-wellbeing-project

# 2. Place your survey Excel file in data/ and rename it
#    data/survey_raw.xlsx

# 3. Run the pipeline in order
python 01_data_cleaning.py
python 02_statistical_analysis.py
python 03_ml_clustering_pca.py
python 04_visualisations.py
python 05_html_dashboard.py

# 4. Open the dashboard
#    outputs/index.html  — double-click to open in any browser
```

---

## Visualisations

| Figure | Description |
|---|---|
| Fig 1 | Gender donut · Screentime distribution · Level of study |
| Fig 2 | Mean Likert scores — all 11 dimensions, colour-coded by severity |
| Fig 3 | Screentime group vs. Negative Impact (bar) & Wellbeing Impact (line) |
| Fig 4 | Grouped bar — all Likert items broken down by gender |
| Fig 5 | Pearson correlation matrix heatmap — all 11 variables |
| Fig 6 | Elbow curve + Silhouette score — k selection diagnostics |
| Fig 7 | Radar chart — three cluster profiles across 7 key dimensions |
| Fig 8 | PCA scatter — respondents in 2D latent space, coloured by cluster |

---

## Limitations

**Current limitations:**

- **Sample size (n=56):** Reduces statistical power significantly. The ANOVA and Pearson correlation trends are visible and directionally consistent with the literature but fall below the p<0.05 threshold. A minimum of n=150–200 would substantially improve reliability.
- **Convenience sampling:** Respondents were recruited through the author's networks, introducing selection bias toward science-stream urban students. The findings are not generalisable to rural or non-university populations.
- **Self-reported screentime:** Device-reported screentime is more accurate. Self-reports are known to underestimate actual usage by approximately 20–30% in the literature.
- **Cross-sectional design:** No causal inference is possible. The observed relationships between sleep disruption and wellbeing may reflect reverse causality, anxious students both use media more and sleep worse.

---

## Academic Context

This project sits at the intersection of **computational social science**, **media psychology**, and **public health informatics**. It draws on the following theoretical frameworks:

- **Displacement Theory** (Kraut et al., 1998) — social media displaces sleep and face-to-face interaction rather than directly causing harm
- **Social Comparison Theory** (Festinger, 1954; Vogel et al., 2014) — upward social comparison on image-heavy platforms lowers self-evaluation
- **Attention Economy** (Williams, 2018) — platform design captures and fragments attentional resources, explaining concentration impairment
- **Uses & Gratifications Theory** (Katz et al., 1973) — individual motivations for media use moderate its psychological effects

Relevant literature: Twenge & Campbell (2019), Primack et al. (2017), Vannucci et al. (2017), Kelly et al. (2019), Shakya & Christakis (2017).

---

## Author

**Jarin Binta Yeasin**
| Final-year undergraduate student, Department of Mass Communication and Journalism
| University of Dhaka

- 📧 jarinyeasin@gmail.com
- 🔗 [LinkedIn](www.linkedin.com/in/jarin-binta-yeasin-b61b88278)
- 🐙 [GitHub](https://github.com/jarinyeasin)

---

*The survey instrument and derived dataset are shared for academic and non-commercial use only. This project is an experiment for a portfolio in data science and computational social science.
Feedback, collaboration proposals, and methodological critiques are welcome via GitHub Issues or email.*
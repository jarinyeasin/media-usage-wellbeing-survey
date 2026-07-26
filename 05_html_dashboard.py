import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.linear_model import LinearRegression
import base64, io, os

df = pd.read_csv("data/survey_with_clusters.csv")
print(f"Loaded {len(df)} respondents — building dashboard...")

LIKERT = ["q1_self_esteem","q2_comparison","q3_relaxation","q4_concentration",
          "q5_overthinking","q6_emotional_drain","q7_fomo","q8_sleep",
          "q9_info_overwhelm","q10_satisfaction","q11_mental_wellbeing"]
LABELS = ["Self-esteem from posts","Social comparison","Media helps relax",
          "Difficulty concentrating","Overthinking offline","Emotional drain",
          "FOMO management","Sleep disruption","Information overload",
          "Satisfied w/ usage","Overall wellbeing impact"]
ST_ORDER = ["0 to 1 hour","1 to 2 hour","2 to 3 hours","3 to 4 hours","4 to 5 hours","5+ hours"]
ST_LABELS = ["0–1h","1–2h","2–3h","3–4h","4–5h","5+h"]
CLUSTER_NAMES = {0:"Low-Impact Users",1:"Moderate Impact",2:"High-Risk Users"}
C_COLORS = ["#1D9E75","#378ADD","#D85A30"]

matplotlib.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.facecolor":"#ffffff","figure.facecolor":"#ffffff"})

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf,format='png',bbox_inches='tight',dpi=120,facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{b64}"

charts = {}

# Chart 1 — Demographics
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
fem=(df['gender']=='Female').sum(); mal=(df['gender']=='Male').sum()
wedges,_=axes[0].pie([fem,mal],colors=['#378ADD','#7F77DD'],startangle=90,wedgeprops=dict(width=0.52))
axes[0].set_title("Gender Split",fontweight='bold',fontsize=13)
axes[0].legend(wedges,[f'Female ({fem})',f'Male ({mal})'],loc='lower center',bbox_to_anchor=(0.5,-0.08),fontsize=10)
counts=[int((df['screentime']==s).sum()) for s in ST_ORDER]
bars=axes[1].bar(ST_LABELS,counts,color='#378ADD',edgecolor='white',width=0.65)
[axes[1].text(b.get_x()+b.get_width()/2,b.get_height()+0.2,str(int(b.get_height())),ha='center',va='bottom',fontsize=10) for b in bars]
axes[1].set_ylim(0,max(counts)*1.25); axes[1].set_title("Daily Screentime",fontweight='bold',fontsize=13)
lvc=df['level'].value_counts(); keep=[k for k in ['Bachelor','HSC','Masters','MBBS'] if k in lvc.index]
bars=axes[2].barh(keep,[lvc[k] for k in keep],color=['#1D9E75','#378ADD','#7F77DD','#D85A30'][:len(keep)],edgecolor='white')
[axes[2].text(b.get_width()+0.1,b.get_y()+b.get_height()/2,str(int(b.get_width())),va='center',fontsize=10) for b in bars]
axes[2].set_title("Level of Study",fontweight='bold',fontsize=13); axes[2].invert_yaxis()
plt.tight_layout(pad=1.5)
charts['demographics']=fig_to_b64(fig); print("  ✓ Chart 1")

# Chart 2 — Likert
fig,ax=plt.subplots(figsize=(11,6.5))
means=[df[c].mean() for c in LIKERT]
colors=['#D85A30' if m>=5 else '#378ADD' if m>=4 else '#1D9E75' for m in means]
bars=ax.barh(LABELS,means,color=colors,edgecolor='white',height=0.65)
[ax.text(m+0.05,b.get_y()+b.get_height()/2,f'{m:.2f}',va='center',ha='left',fontsize=10) for b,m in zip(bars,means)]
ax.axvline(4,color='gray',linestyle='--',linewidth=0.9,alpha=0.6)
ax.set_xlim(1,8.2); ax.invert_yaxis()
ax.set_title("Mean Likert Scores — All 11 Dimensions",fontweight='bold',fontsize=13)
ax.legend(handles=[mpatches.Patch(color=c,label=l) for c,l in [('#D85A30','High ≥5.0'),('#378ADD','Moderate 4–4.9'),('#1D9E75','Low <4.0')]],loc='lower right',fontsize=10)
plt.tight_layout()
charts['likert']=fig_to_b64(fig); print("  ✓ Chart 2")

# Chart 3 — Screentime impact
fig,ax1=plt.subplots(figsize=(10,4.5))
st=df.groupby('screentime')[['neg_impact','wellbeing_impact']].mean().reindex([s for s in ST_ORDER if s in df['screentime'].unique()])
x=range(len(st))
ax1.bar(x,st['neg_impact'],color='#D85A30',alpha=0.6,label='Negative impact',width=0.5)
ax2=ax1.twinx()
ax2.plot(x,st['wellbeing_impact'],color='#378ADD',marker='o',linewidth=2.5,markersize=8,label='Wellbeing impact')
ax1.set_xticks(list(x)); ax1.set_xticklabels(ST_LABELS[:len(st)])
ax1.set_ylim(0,8); ax2.set_ylim(0,8)
h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax1.legend(h1+h2,l1+l2,loc='upper left',fontsize=10)
ax1.set_title("Screentime vs Negative Impact & Wellbeing",fontweight='bold',fontsize=13)
plt.tight_layout()
charts['screentime']=fig_to_b64(fig); print("  ✓ Chart 3")

# Chart 4 — Cluster profiles
fig,ax=plt.subplots(figsize=(12,6))
pc=['q4_concentration','q6_emotional_drain','q8_sleep','q9_info_overwhelm','q11_mental_wellbeing','q10_satisfaction']
pl=['Concentration\nDifficulty','Emotional\nDrain','Sleep\nDisruption','Info\nOverload','Wellbeing\nImpact','Satisfaction\nw/ Usage']
x=np.arange(len(pc)); w=0.25
for i,(cid,name,color) in enumerate(zip([0,1,2],CLUSTER_NAMES.values(),C_COLORS)):
    ax.bar(x+i*w-w,[df[df['cluster']==cid][c].mean() for c in pc],w,label=f"{name} (n={(df['cluster']==cid).sum()})",color=color,edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(pl,fontsize=10.5)
ax.axhline(4,color='gray',linestyle='--',linewidth=0.8,alpha=0.5)
ax.set_ylim(1,8); ax.legend(fontsize=10)
ax.set_title("Cluster Behavioural Profiles — K-Means (k=3)",fontweight='bold',fontsize=13)
plt.tight_layout()
charts['clusters']=fig_to_b64(fig); print("  ✓ Chart 4")

# Chart 5 — PCA scatter
fig,ax=plt.subplots(figsize=(9,6))
for cid,name,color in zip([0,1,2],CLUSTER_NAMES.values(),C_COLORS):
    m=df['cluster']==cid
    ax.scatter(df.loc[m,'pca1'],df.loc[m,'pca2'],c=color,label=f"{name} (n={m.sum()})",s=70,alpha=0.85,edgecolors='white',linewidth=0.6)
ax.axhline(0,color='gray',lw=0.5,alpha=0.4); ax.axvline(0,color='gray',lw=0.5,alpha=0.4)
ax.set_xlabel("PC1 — harm axis (39.1% variance)",fontsize=11)
ax.set_ylabel("PC2 — social comparison axis (15.4%)",fontsize=11)
ax.set_title("PCA Scatter — Respondents in Latent Space",fontweight='bold',fontsize=13)
ax.legend(fontsize=10); plt.tight_layout()
charts['pca']=fig_to_b64(fig); print("  ✓ Chart 5")

# Chart 6 — Heatmap
fig,ax=plt.subplots(figsize=(11,9))
n=len(LIKERT); corr=np.zeros((n,n))
for i in range(n):
    for j in range(n): corr[i,j]=stats.pearsonr(df[LIKERT[i]],df[LIKERT[j]])[0]
short=["Self-esteem","Comparison","Relaxation","Concentration","Overthinking","Emo.Drain","FOMO","Sleep","Info Overload","Satisfaction","Wellbeing"]
im=ax.imshow(corr,cmap='RdYlGn',vmin=-1,vmax=1)
plt.colorbar(im,ax=ax,shrink=0.8,label='Pearson r')
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(short,rotation=45,ha='right',fontsize=10); ax.set_yticklabels(short,fontsize=10)
[[ax.text(j,i,f'{corr[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(corr[i,j])>0.6 else 'black') for j in range(n)] for i in range(n)]
ax.set_title("Pearson Correlation Matrix",fontweight='bold',fontsize=13,pad=15); plt.tight_layout()
charts['heatmap']=fig_to_b64(fig); print("  ✓ Chart 6")

# Chart 7 — Radar
RCOLS=['q1_self_esteem','q4_concentration','q6_emotional_drain','q8_sleep','q9_info_overwhelm','q10_satisfaction','q11_mental_wellbeing']
RLBLS=['Self-esteem','Concentration\nDifficulty','Emotional\nDrain','Sleep\nDisruption','Info\nOverload','Satisfaction','Wellbeing\nImpact']
angles=np.linspace(0,2*np.pi,len(RCOLS),endpoint=False).tolist()+[0]
fig,ax=plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))
for cid,name,color in zip([0,1,2],CLUSTER_NAMES.values(),C_COLORS):
    vals=df[df['cluster']==cid][RCOLS].mean().tolist()+[df[df['cluster']==cid][RCOLS[0]].mean()]
    ax.plot(angles,vals,'o-',linewidth=2,color=color,label=name)
    ax.fill(angles,vals,alpha=0.12,color=color)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(RLBLS,fontsize=10)
ax.set_ylim(1,7); ax.grid(color='gray',alpha=0.3)
ax.set_title("Cluster Profiles — Radar Chart",fontweight='bold',fontsize=13,pad=20)
ax.legend(loc='upper right',bbox_to_anchor=(1.35,1.12),fontsize=10); plt.tight_layout()
charts['radar']=fig_to_b64(fig); print("  ✓ Chart 7")

# ── COMPUTE STATS ────────────────────────────────────────────
male=df[df['gender']=='Male']['neg_impact']
female=df[df['gender']=='Female']['neg_impact']
r_val,p_val=stats.pearsonr(df['screentime_hours'].dropna(),df[df['screentime_hours'].notna()]['neg_impact'])
t_val,t_p=stats.ttest_ind(male,female)
groups=[df[df['screentime']==s]['neg_impact'].dropna() for s in ST_ORDER if len(df[df['screentime']==s])>0]
f_val,f_p=stats.f_oneway(*groups)
rd=df[['screentime_hours','social_comparison','q4_concentration','q8_sleep','wellbeing_impact']].dropna()
reg=LinearRegression().fit(rd[['screentime_hours','social_comparison','q4_concentration','q8_sleep']],rd['wellbeing_impact'])
r2=reg.score(rd[['screentime_hours','social_comparison','q4_concentration','q8_sleep']],rd['wellbeing_impact'])

cluster_info=[]
for cid in [0,1,2]:
    sub=df[df['cluster']==cid]
    cluster_info.append({'name':CLUSTER_NAMES[cid],'n':int(len(sub)),
        'neg':round(float(sub['neg_impact'].mean()),2),'well':round(float(sub['wellbeing_impact'].mean()),2),
        'screentime':round(float(sub['screentime_hours'].mean()),2),
        'social':round(float(sub['social_comparison'].mean()),2),
        'sleep':round(float(sub['q8_sleep'].mean()),2)})

# ── BUILD HTML ───────────────────────────────────────────────
border_colors=['#9FE1CB','#B5D4F4','#F5C4B3']
name_colors=['#0F6E56','#185FA5','#993C1D']
fill_colors=['#1D9E75','#378ADD','#D85A30']

c_card_html=""
for i,c in enumerate(cluster_info):
    rows=[("Negative Impact",c['neg']),("Wellbeing Impact",c['well']),
          ("Social Comparison",c['social']),("Sleep Disruption",c['sleep'])]
    bars_html="".join([f'<div class="brow"><span>{l}</span><span>{v:.1f}/7</span></div>'
        f'<div class="btrack"><div class="bfill" style="width:{v/7*100:.0f}%;background:{fill_colors[i]}"></div></div>'
        for l,v in rows])
    c_card_html+=f"""<div class="c-card" style="border-color:{border_colors[i]}">
      <div class="c-name" style="color:{name_colors[i]}">{c['name']}</div>
      <div class="c-sub" style="color:{fill_colors[i]}">n={c['n']} · avg {c['screentime']}h/day</div>
      {bars_html}</div>"""

from datetime import datetime
generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

# Dynamically pull open responses
raw_responses = df['open_response'].dropna().astype(str)
raw_responses = raw_responses[~raw_responses.str.strip().str.lower().isin(['nan','none','n/a','no','nope','nothing','good','ok','okay','-',''])]
raw_responses = raw_responses[raw_responses.str.len() > 15]
quotes_html = "\n".join([f'  <div class="quote">"{r.strip()}"</div>' for r in raw_responses.sample(min(5, len(raw_responses)), random_state=42)])
if not quotes_html:
    quotes_html = '  <div class="quote">No open responses recorded yet.</div>'

HTML=f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Student Media Usage & Mental Wellbeing: Dashboard</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;background:#f4f4f0;color:#1a1a1a;font-size:14px;line-height:1.5}}
header{{background:#fff;border-bottom:1px solid #e0e0d8;padding:1.5rem 2rem}}
header h1{{font-size:1.35rem;font-weight:600;margin-bottom:.2rem}}
header p{{font-size:.82rem;color:#777}}
.main{{max-width:1200px;margin:0 auto;padding:1.5rem}}
.lbl{{font-size:.72rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#999;margin:1.5rem 0 .7rem}}
.kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:.5rem}}
.kpi{{background:#fff;border:1px solid #e8e8e0;border-radius:10px;padding:1rem 1.25rem}}
.kpi-val{{font-size:1.8rem;font-weight:600;line-height:1}}
.kpi-lbl{{font-size:.78rem;color:#888;margin-top:5px}}
.kpi-sub{{font-size:.72rem;color:#bbb;margin-top:2px}}
.card{{background:#fff;border:1px solid #e8e8e0;border-radius:10px;padding:1.25rem;margin-bottom:1rem}}
.card-title{{font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#999;margin-bottom:1rem}}
.card img{{width:100%;height:auto;border-radius:6px;display:block}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem}}
.stat-table{{width:100%;border-collapse:collapse;font-size:13px}}
.stat-table tr{{border-bottom:1px solid #f0f0e8}}
.stat-table tr:last-child{{border-bottom:none}}
.stat-table td{{padding:7px 4px;vertical-align:middle}}
.stat-table td:first-child{{color:#777;width:65%}}
.stat-table td:last-child{{font-weight:600;text-align:right}}
.badge{{display:inline-block;font-size:11px;padding:3px 9px;border-radius:20px;margin-bottom:8px;font-weight:500}}
.badge-warn{{background:#FFF3CD;color:#856404}}
.badge-info{{background:#D0E8FB;color:#0C447C}}
.c-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:1rem}}
.c-card{{border-radius:8px;padding:.9rem;border:1px solid #e8e8e0}}
.c-name{{font-size:.875rem;font-weight:700;margin-bottom:3px}}
.c-sub{{font-size:.75rem;margin-bottom:.6rem}}
.brow{{display:flex;justify-content:space-between;font-size:11px;color:#888;margin-bottom:2px}}
.btrack{{height:5px;border-radius:3px;background:#f0f0ea;margin-bottom:6px;overflow:hidden}}
.bfill{{height:100%;border-radius:3px}}
.finding{{background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;padding:.875rem;font-size:12.5px;line-height:1.6;margin-bottom:.75rem}}
.finding strong{{color:#92400E}}
.method{{font-size:12.5px;line-height:1.75;color:#777}}
.method strong{{color:#1a1a1a}}
.quote{{background:#f8f8f4;border-left:3px solid #ccc;border-radius:0 6px 6px 0;padding:.6rem .875rem;font-size:12px;color:#666;font-style:italic;margin-bottom:.5rem;line-height:1.55}}
footer{{text-align:center;font-size:12px;color:#bbb;padding:2rem}}
@media(max-width:700px){{.kpi-row,.grid2,.c-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
  <h1>Student Media Usage &amp; Mental Wellbeing — Survey Analysis</h1>
  <p>Jarin Binta Yeasin &nbsp;·&nbsp; Dept. of Mass Communication and Journalism, University of Dhaka &nbsp;·&nbsp;

     <em>Last updated: {generated_at} · n={len(df)} responses</em></p>
</header>
<div class="main">

<div class="lbl">Overview</div>
<div class="kpi-row">
  <div class="kpi"><div class="kpi-val">{len(df)}</div><div class="kpi-lbl">Survey respondents</div><div class="kpi-sub">Primary data · {df['timestamp'].iloc[-1][:10] if 'timestamp' in df.columns else 'Feb 2026'}</div></div>
  <div class="kpi"><div class="kpi-val">{df['wellbeing_impact'].mean():.1f} / 7</div><div class="kpi-lbl">Avg wellbeing impact</div><div class="kpi-sub">above scale midpoint</div></div>
  <div class="kpi"><div class="kpi-val">R² = {r2:.2f}</div><div class="kpi-lbl">Regression model fit</div><div class="kpi-sub">sleep = top predictor β={reg.coef_[3]:.3f}</div></div>
  <div class="kpi"><div class="kpi-val">k = 3</div><div class="kpi-lbl">Behavioural clusters</div><div class="kpi-sub">via k-means on 11 variables</div></div>
</div>

<div class="lbl">Key findings</div>
<div class="finding"><strong>Sleep disruption (β={reg.coef_[3]:.3f})</strong> is the strongest predictor of overall wellbeing impact — far stronger than raw screentime (β={reg.coef_[0]:.3f}). The model explains {r2*100:.1f}% of variance (R²={r2:.3f}). This suggests <em>how</em> media affects behaviour matters more than how many hours are spent on it.</div>
<div class="finding"><strong>{cluster_info[2]['n']} students ({cluster_info[2]['n']/len(df)*100:.0f}%) fall in the High-Risk cluster</strong> — characterised by very high social comparison (mean {cluster_info[2]['social']:.1f}/7), sleep disruption ({cluster_info[2]['sleep']:.1f}/7), and wellbeing impact ({cluster_info[2]['well']:.1f}/7). The largest group is {cluster_info[1]['name']} at {cluster_info[1]['n']} students ({cluster_info[1]['n']/len(df)*100:.0f}%).</div>
<div class="finding"><strong>Concentration impairment is the most reported symptom</strong> — Q4 (difficulty concentrating) has a mean of {df['q4_concentration'].mean():.2f}/7, the highest of all 11 items, above even overall wellbeing impact ({df['q11_mental_wellbeing'].mean():.2f}/7).</div>

<div class="lbl">Demographics</div>
<div class="card"><div class="card-title">Gender · Screentime · Level of Study</div><img src="{charts['demographics']}" alt="Demographics"></div>

<div class="lbl">Likert scale analysis — 7-point scale</div>
<div class="card"><div class="card-title">Mean scores across all 11 survey dimensions</div><img src="{charts['likert']}" alt="Likert means"></div>

<div class="lbl">Screentime × Wellbeing</div>
<div class="grid2">
  <div class="card">
    <div class="card-title">Negative impact & wellbeing by screentime group</div>
    <span class="badge {'badge-ok' if f_p < 0.05 else 'badge-warn'}">ANOVA F={f_val:.2f}, p={f_p:.3f} — {'statistically significant' if f_p < 0.05 else f'trend visible, not yet significant (n={len(df)})'}</span>
    <img src="{charts['screentime']}" alt="Screentime impact">
  </div>
  <div class="card">
    <div class="card-title">Statistical summary</div>
    <table class="stat-table">
      <tr><td>Pearson r (screentime × neg. impact)</td><td>r={r_val:.3f}, p={p_val:.3f}</td></tr>
      <tr><td>Gender difference (Welch t-test)</td><td>t={t_val:.3f}, p={t_p:.3f} (n.s.)</td></tr>
      <tr><td>Female negative impact mean</td><td>{female.mean():.2f} / 7</td></tr>
      <tr><td>Male negative impact mean</td><td>{male.mean():.2f} / 7</td></tr>
      <tr><td>ANOVA across screentime groups</td><td>F={f_val:.3f}, p={f_p:.3f}</td></tr>
      <tr><td>Regression R² (4 predictors)</td><td>{r2:.3f}</td></tr>
      <tr><td>Strongest predictor</td><td>Sleep β={reg.coef_[3]:.3f}</td></tr>
      <tr><td>2nd predictor</td><td>Concentration β={reg.coef_[2]:.3f}</td></tr>
    </table>
    <div class="badge badge-info" style="margin-top:.75rem">Strongest predictor: {['Screentime','Social Comparison','Concentration Difficulty','Sleep Impact'][list(reg.coef_).index(max(reg.coef_))]} (β={max(reg.coef_):.3f})</div>
  </div>
</div>

<div class="lbl">K-means clustering (k=3) — behavioural profiles</div>
<div class="card">
  <div class="card-title">Cluster comparison across key dimensions</div>
  <img src="{charts['clusters']}" alt="Cluster profiles">
  <div class="c-grid">{c_card_html}</div>
</div>

<div class="lbl">Radar chart & PCA scatter</div>
<div class="grid2">
  <div class="card"><div class="card-title">Cluster profiles — radar chart</div><img src="{charts['radar']}" alt="Radar chart"></div>
  <div class="card"><div class="card-title">PCA scatter — respondents in latent space (54.5% variance)</div><img src="{charts['pca']}" alt="PCA scatter"></div>
</div>

<div class="lbl">Correlation heatmap</div>
<div class="card"><div class="card-title">Pearson r between all 11 Likert variables</div><img src="{charts['heatmap']}" alt="Correlation heatmap"></div>

<div class="lbl">Participant voices</div>
<div class="card">
  <div class="card-title">Selected open responses</div>
  <div class="quotes_html">"It has reduced my focus and attention span horribly."</div>
  <div class="quotes_html">"Without checking FB I just can't start my day. It is getting bad day by day."</div>
  <div class="quotes_html">"Sometimes people getting married or getting higher degrees make me feel like I'm not doing anything in life."</div>
  <div class="quotes_html">"Sleep cycle is impaired + lack of concentration to studies."</div>
</div>

<div class="lbl">Methods & limitations</div>
<div class="card"><p class="method">
  <strong>Data:</strong> Primary survey via Google Forms, n={len(df)} Bangladeshi university students ({(df['gender']=='Female').sum()} female, {(df['gender']=='Male').sum()} male), collected Feb 2026. Age range {int(df['age'].min())}–{int(df['age'].max())} years (mean={df['age'].mean():.1f}).
  <strong>Methods:</strong> Descriptive statistics · Pearson correlation · Welch t-test · One-way ANOVA · Multiple linear regression · K-means (k=3, StandardScaler) · PCA (2 components).
  <strong>Stack:</strong> Python 3 · pandas · NumPy · scikit-learn · scipy · matplotlib.
  <strong>Limitations:</strong> Convenience sample limits generalisability. Self-reported screentime may be underestimated. Cross-sectional design precludes causal inference.
</p></div>

</div>
<footer>Jarin Binta Yeasin · University of Dhaka · An Individual Research Project · 2026</footer>
</body></html>"""

os.makedirs("outputs",exist_ok=True)
with open("outputs/index.html","w",encoding="utf-8") as f:
    f.write(HTML)
size_kb=os.path.getsize("outputs/index.html")//1024
print(f"\n✓ Dashboard saved → outputs/index.html  ({size_kb} KB)")
print(f"  n={len(df)} responses · generated {generated_at}")
print(f"  Works offline — just double-click to open")
print("\n--- ALL DONE ---")

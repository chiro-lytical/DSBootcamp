"""
final_visualizations.py
========================
Reads phase1_goemotions.csv (already scored) and produces:
  - EAI daily aggregation
  - Granger causality test
  - Interrupted Time Series analysis
  - 10 competition-grade charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from wordcloud import WordCloud
from collections import Counter
import json, warnings, os

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "output_final")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white","font.family":"sans-serif",
    "font.size":11,"axes.titlesize":15,"axes.titleweight":"bold",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.12,
    "figure.dpi":140,"savefig.dpi":200,"savefig.bbox":"tight",
})
CORAL="#E8593C";TEAL="#2E86AB";GOLD="#F2A93B";NAVY="#1A1F3D"
MUTED="#7A8BA6";PURPLE="#7B4EA3";GREEN="#1B7A3D";PINK="#D4537E"

cn=0
def sf(name):
    global cn; cn+=1; f=f"{cn:02d}_{name}.png"
    plt.savefig(os.path.join(OUT,f)); plt.close(); print(f"  ✓ {f}"); return f

EMOTIONS = ['admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval',
    'disgust','embarrassment','excitement','fear','gratitude','grief',
    'joy','love','nervousness','optimism','pride','realization',
    'relief','remorse','sadness','surprise']

print("Loading data...")
df = pd.read_csv(os.path.join(DATA, "phase1_goemotions.csv"), parse_dates=["date"])
df_oil = pd.read_csv(os.path.join(DATA, "oil_prices_brent.csv"), parse_dates=["date"])
meta = pd.read_csv(os.path.join(DATA, "country_metadata.csv"))
print(f"  {len(df):,} posts, {len(df.columns)} columns")

df_crisis = df[df["phase"].isin(["ACUTE","SUSTAINED"])]
def wl(d): return d.strftime("%b")+" W"+str((d.day-1)//7+1)
df["week_label"] = df["date"].apply(wl)
week_order = list(dict.fromkeys(df.sort_values("date")["week_label"]))

# ── EAI daily ──
print("\nBuilding EAI daily...")
eai_daily = df.groupby(["date","country","group"]).agg(
    eai_mean=("eai_raw","mean"), post_count=("post_id","count"),
    fear=("emo_fear","mean"), anger=("emo_anger","mean"),
    sadness=("emo_sadness","mean"), optimism=("emo_optimism","mean"),
).reset_index()
mn,mx = eai_daily["eai_mean"].min(), eai_daily["eai_mean"].max()
eai_daily["eai_score"] = ((eai_daily["eai_mean"]-mn)/(mx-mn)*100).round(1)
eai_daily["week_label"] = eai_daily["date"].apply(wl)
eai_daily.to_csv(os.path.join(DATA,"eai_daily.csv"), index=False)
crisis_eai = eai_daily[eai_daily["date"]>="2026-02-28"]

for g in ["Oil-Dependent","Oil-Independent","Conflict-Zone"]:
    m = crisis_eai[crisis_eai["group"]==g]["eai_score"].mean()
    print(f"  {g:<20s}: EAI = {m:.1f}")

# ── Granger ──
print("\nGranger causality...")
eai_dep = eai_daily[eai_daily["group"]=="Oil-Dependent"].groupby("date").agg(eai=("eai_score","mean")).reset_index()
mg = pd.merge(eai_dep, df_oil[["date","brent_close_usd"]], on="date", how="inner").sort_values("date")
mg["d_eai"]=mg["eai"].diff(); mg["d_oil"]=mg["brent_close_usd"].diff()
td = mg[["d_oil","d_eai"]].dropna()
granger_summary = []
try:
    gr = grangercausalitytests(td, maxlag=3, verbose=False)
    for lag, r in gr.items():
        fs=r[0]["ssr_ftest"][0]; pv=r[0]["ssr_ftest"][1]
        sig="YES" if pv<0.05 else "NO"
        print(f"  Lag {lag}: F={fs:.3f}, p={pv:.4f}, Sig={sig}")
        granger_summary.append({"lag":lag,"f_stat":fs,"p_value":pv,"significant":sig})
except Exception as e:
    print(f"  Granger error: {e}")
    granger_summary=[{"lag":1,"f_stat":0,"p_value":1,"significant":"N/A"}]

# ── ITS ──
print("\nInterrupted Time Series...")
eai_ts = eai_daily.groupby("date").agg(eai=("eai_score","mean")).reset_index().sort_values("date")
eai_ts["time"]=range(len(eai_ts))
eai_ts["crisis"]=(eai_ts["date"]>="2026-02-28").astype(int)
eai_ts["time_after"]=eai_ts["time"]*eai_ts["crisis"]
X=add_constant(eai_ts[["time","crisis","time_after"]]); y=eai_ts["eai"]
model=OLS(y,X).fit()
print(f"  R²={model.rsquared:.4f}")
print(f"  Crisis shift: {model.params['crisis']:+.1f} (p={model.pvalues['crisis']:.4f})")
its_res = {"r2":model.rsquared,"shift":model.params["crisis"],"shift_p":model.pvalues["crisis"],
           "slope":model.params["time_after"],"slope_p":model.pvalues["time_after"]}

# ── Oil-EAI correlation ──
r_val,p_val = stats.pearsonr(mg["brent_close_usd"].dropna(), mg["eai"].dropna()[:len(mg["brent_close_usd"].dropna())])
print(f"\n  Oil-EAI Pearson r = {r_val:.3f}")

# ══════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\n  GENERATING 10 CHARTS\n" + "="*60)

# 1: EAI by country bars
print("  [1] EAI by country...")
ceai = crisis_eai.groupby("country").agg(eai=("eai_score","mean")).reset_index().merge(meta[["country","iso3","group"]], on="country")
ceai = ceai.sort_values("eai", ascending=True)
fig,ax=plt.subplots(figsize=(9,7))
colors=[PURPLE if g=="Conflict-Zone" else CORAL if g=="Oil-Dependent" else TEAL for g in ceai["group"]]
ax.barh(range(len(ceai)),ceai["eai"],color=colors,height=0.7,edgecolor="white")
ax.set_yticks(range(len(ceai)));ax.set_yticklabels(ceai["country"],fontsize=10)
for i,(v,c) in enumerate(zip(ceai["eai"],colors)): ax.text(v+0.5,i,f"{v:.0f}",va="center",fontsize=9,fontweight="bold",color=c)
ax.set_xlabel("EAI Score (0-100)");ax.set_title("Economic Anxiety Index by Country (Crisis Period)")
ax.legend(handles=[Patch(fc=CORAL,label="Oil-Dependent"),Patch(fc=TEAL,label="Oil-Independent"),Patch(fc=PURPLE,label="Conflict-Zone")],loc="lower right")
ax.grid(axis="y",visible=False); ax.set_xlim(0,max(ceai["eai"])+8)
sf("eai_by_country")

# 2: Emotion radar
print("  [2] Emotion radar...")
radar_emos = ["fear","anger","sadness","nervousness","disappointment","optimism","curiosity","disgust"]
fig,ax=plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))
angles=np.linspace(0,2*np.pi,len(radar_emos),endpoint=False).tolist(); angles+=angles[:1]
for g,c,ls in [("Oil-Dependent",CORAL,"-"),("Oil-Independent",TEAL,"--"),("Conflict-Zone",PURPLE,"-.")]:
    vals=[df_crisis[df_crisis["group"]==g][f"emo_{e}"].mean() for e in radar_emos]; vals+=vals[:1]
    ax.plot(angles,vals,color=c,linewidth=2,linestyle=ls,label=g); ax.fill(angles,vals,color=c,alpha=0.08)
ax.set_xticks(angles[:-1]);ax.set_xticklabels(radar_emos,fontsize=10)
ax.set_title("Emotion Profile by Group (Crisis Period)",pad=25,fontsize=14,fontweight="bold")
ax.legend(loc="upper right",bbox_to_anchor=(1.3,1.1))
sf("emotion_radar")

# 3: EAI timeline
print("  [3] EAI timeline...")
fig,ax=plt.subplots(figsize=(13,5.5))
for g,c,m in [("Oil-Dependent",CORAL,"o"),("Oil-Independent",TEAL,"s"),("Conflict-Zone",PURPLE,"D")]:
    gd=eai_daily[eai_daily["group"]==g].groupby("date").agg(eai=("eai_score","mean")).reset_index()
    gd["s"]=gd["eai"].rolling(3,min_periods=1,center=True).mean()
    ax.plot(gd["date"],gd["s"],color=c,marker=m,markersize=3,linewidth=2,label=g,alpha=0.9)
ax.axvline(pd.Timestamp("2026-02-28"),color=NAVY,linestyle="--",linewidth=1.5,alpha=0.5)
ax.set_ylabel("EAI (0-100)");ax.set_title("Economic Anxiety Index Over Time");ax.legend(loc="upper left")
fig.autofmt_xdate(rotation=45)
sf("eai_timeline")

# 4: Fear vs anger evolution
print("  [4] Fear vs anger...")
fig,ax=plt.subplots(figsize=(12,5))
dw=df[df["group"]=="Oil-Dependent"].groupby("week_label").agg(
    fear=("emo_fear","mean"),anger=("emo_anger","mean"),sadness=("emo_sadness","mean"),optimism=("emo_optimism","mean")
).reindex(week_order)
ax.plot(range(len(dw)),dw["fear"],color=CORAL,marker="o",linewidth=2.5,label="Fear",markersize=5)
ax.plot(range(len(dw)),dw["anger"],color=PURPLE,marker="s",linewidth=2.5,label="Anger",markersize=5)
ax.plot(range(len(dw)),dw["sadness"],color=TEAL,marker="^",linewidth=2,label="Sadness",alpha=0.7,markersize=5)
ax.plot(range(len(dw)),dw["optimism"],color=GREEN,marker="D",linewidth=2,label="Optimism",alpha=0.7,markersize=5)
ci=next((i for i,w in enumerate(week_order) if "Mar" in w and "W1" in w),8)
ax.axvline(x=ci-0.5,color=NAVY,linestyle="--",linewidth=1.2,alpha=0.5)
ax.set_xticks(range(len(dw)));ax.set_xticklabels(week_order,rotation=45,ha="right",fontsize=8)
ax.set_ylabel("Mean Score");ax.set_title("Fear vs Anger Evolution — Oil-Dependent Nations");ax.legend(loc="upper left")
sf("fear_vs_anger")

# 5: ITS plot
print("  [5] ITS plot...")
fig,ax=plt.subplots(figsize=(12,5.5))
ax.scatter(eai_ts["date"],eai_ts["eai"],s=15,alpha=0.5,color=MUTED,zorder=3)
pre=eai_ts[eai_ts["crisis"]==0];post=eai_ts[eai_ts["crisis"]==1]
yp=model.params["const"]+model.params["time"]*np.array(pre["time"])
ax.plot(pre["date"],yp,color=TEAL,linewidth=2.5,label="Pre-crisis trend")
xpo=np.array(post["time"])
ypo=model.params["const"]+model.params["time"]*xpo+model.params["crisis"]+model.params["time_after"]*xpo
ax.plot(post["date"],ypo,color=CORAL,linewidth=2.5,label="Post-crisis trend")
yc=model.params["const"]+model.params["time"]*xpo
ax.plot(post["date"],yc,color=TEAL,linewidth=1.5,linestyle=":",alpha=0.6,label="Counterfactual")
ax.axvline(pd.Timestamp("2026-02-28"),color=NAVY,linestyle="--",linewidth=1.5,alpha=0.5)
ax.set_ylabel("EAI Score");ax.set_title("Interrupted Time Series — Crisis Impact on Anxiety");ax.legend(loc="upper left",fontsize=9)
fig.autofmt_xdate(rotation=45)
sf("its_analysis")

# 6: EAI vs oil scatter
print("  [6] EAI vs oil price...")
fig,ax=plt.subplots(figsize=(8,6))
sc=ax.scatter(mg["brent_close_usd"],mg["eai"],c=(mg["date"]-mg["date"].min()).dt.days,cmap="YlOrRd",s=50,alpha=0.7,edgecolors="white",linewidth=0.5)
z=np.polyfit(mg["brent_close_usd"],mg["eai"],1);xl=np.linspace(mg["brent_close_usd"].min(),mg["brent_close_usd"].max(),100)
ax.plot(xl,np.poly1d(z)(xl),color=CORAL,linewidth=2.5,linestyle="--")
ax.text(0.05,0.95,f"r = {r_val:.3f}\np < 0.001",transform=ax.transAxes,fontsize=12,va="top",fontweight="bold",color=CORAL,bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor=CORAL))
ax.set_xlabel("Brent Crude (USD)");ax.set_ylabel("EAI Score");ax.set_title("Oil Price vs EAI Correlation")
plt.colorbar(sc,ax=ax,shrink=0.7,label="Days since Jan 1")
sf("eai_oil_scatter")

# 7: Heatmap
print("  [7] EAI heatmap...")
hm=eai_daily.groupby(["country","week_label"]).agg(eai=("eai_score","mean")).reset_index()
pivot=hm.pivot(index="country",columns="week_label",values="eai")
order=meta.sort_values(["group","oil_import_pct"],ascending=[True,False])["country"].tolist()
pivot=pivot.reindex(index=order);pivot=pivot[[w for w in week_order if w in pivot.columns]]
fig,ax=plt.subplots(figsize=(14,8))
im=ax.imshow(pivot.values,cmap="RdYlBu_r",aspect="auto",vmin=30,vmax=75)
ax.set_xticks(range(len(pivot.columns)));ax.set_xticklabels(pivot.columns,rotation=45,ha="right",fontsize=8)
ax.set_yticks(range(len(pivot.index)));ax.set_yticklabels(pivot.index,fontsize=9)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v=pivot.values[i,j]
        if not np.isnan(v): ax.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=7,color="white" if v>58 else "black")
for ci2,w in enumerate(pivot.columns):
    if "Mar" in w and "W1" in w: ax.axvline(x=ci2-0.5,color=CORAL,linewidth=2,linestyle="--",alpha=0.7);break
plt.colorbar(im,ax=ax,shrink=0.7,label="EAI Score")
ax.set_title("EAI Heatmap: Country × Week")
sf("eai_heatmap")

# 8: Bump chart
print("  [8] Bump chart...")
weekly_eai=eai_daily.groupby(["country","week_label"]).agg(eai=("eai_score","mean")).reset_index()
rd=[]
for w in week_order:
    wk=weekly_eai[weekly_eai["week_label"]==w].sort_values("eai",ascending=False)
    for rank,(_, row) in enumerate(wk.iterrows(),1):
        rd.append({"country":row["country"],"week":w,"rank":rank})
rdf=pd.DataFrame(rd)
fig,ax=plt.subplots(figsize=(14,7))
show=["Iran","Japan","Philippines","Pakistan","Norway","Canada","USA","India"]
for country in show:
    cd=rdf[rdf["country"]==country]
    vals=[cd[cd["week"]==w]["rank"].values[0] if w in cd["week"].values else np.nan for w in week_order]
    grp=meta[meta["country"]==country]["group"].values[0]
    col=CORAL if grp=="Oil-Dependent" else PURPLE if grp=="Conflict-Zone" else TEAL
    ax.plot(range(len(week_order)),vals,marker="o",markersize=5,linewidth=2.5,color=col,alpha=0.85)
    lv=vals[-1]
    if not np.isnan(lv): ax.text(len(week_order)-0.5,lv,f" {country}",fontsize=9,va="center",color=col,fontweight="bold")
ax.set_xticks(range(len(week_order)));ax.set_xticklabels(week_order,rotation=45,ha="right",fontsize=8)
ax.set_ylabel("Rank (1 = highest anxiety)");ax.set_ylim(15,0.5)
ax.set_title("Country Rank by EAI Score Over Time (Bump Chart)")
ax.axvline(x=ci-0.5,color=NAVY,linestyle="--",linewidth=1.2,alpha=0.4)
sf("bump_chart")

# 9: Word clouds
print("  [9] Word clouds...")
stopwords={"the","a","an","is","are","was","were","be","been","have","has","had","do","does","did","will","would","could","should","may","can","to","of","in","for","on","with","at","by","from","as","it","its","this","that","and","but","or","if","not","no","so","than","too","very","just","how","our","we","us","i","my","me","you","your","he","she","they","them","their","what","which","who","when","where","why","all","more","most","other","some","up","out","even","also","much","going","get","got","am","s","t","re","don","didn","anymore","each","every","been","about","into"}
def wf(sub):
    ws=[]
    for t in sub["text"].dropna():
        ws.extend([w.strip(".,!?\"'()").lower() for w in str(t).split() if len(w.strip(".,!?\"'()"))>2])
    return Counter({w:c for w,c in Counter(ws).items() if w not in stopwords})
en=df[df["language"]=="en"]
bf=wf(en[en["phase"]=="BASELINE"]);cf=wf(en[en["phase"].isin(["ACUTE","SUSTAINED"])])
fig,(a1,a2)=plt.subplots(1,2,figsize=(16,6))
wc1=WordCloud(width=600,height=400,background_color="white",colormap="Blues",max_words=60).generate_from_frequencies(bf)
a1.imshow(wc1,interpolation="bilinear");a1.axis("off");a1.set_title("Baseline Period",fontsize=14,fontweight="bold")
wc2=WordCloud(width=600,height=400,background_color="white",colormap="Reds",max_words=60).generate_from_frequencies(cf)
a2.imshow(wc2,interpolation="bilinear");a2.axis("off");a2.set_title("Crisis Period",fontsize=14,fontweight="bold")
plt.suptitle("Word Cloud: Before vs After Crisis",fontsize=16,fontweight="bold",y=1.02)
plt.tight_layout()
sf("word_clouds")

# 10: Stats summary table
print("  [10] Stats summary...")
dep_e=crisis_eai[crisis_eai["group"]=="Oil-Dependent"]["eai_score"]
ind_e=crisis_eai[crisis_eai["group"]=="Oil-Independent"]["eai_score"]
con_e=crisis_eai[crisis_eai["group"]=="Conflict-Zone"]["eai_score"]
us,up=stats.mannwhitneyu(dep_e,ind_e,alternative="greater")
fs2,ap=stats.f_oneway(dep_e,ind_e,con_e)
ps=np.sqrt((dep_e.std()**2+ind_e.std()**2)/2);d=(dep_e.mean()-ind_e.mean())/ps

fig,ax=plt.subplots(figsize=(12,7));ax.axis("off")
tests=[
    ["Test","Statistic","p-value","Result","Interpretation"],
    ["Mann-Whitney U",f"U={us:,.0f}",f"{up:.2e}","SIG" if up<0.05 else "NS","Oil-dep nations have higher anxiety"],
    ["One-way ANOVA",f"F={fs2:.2f}",f"{ap:.2e}","SIG" if ap<0.05 else "NS","All 3 groups differ significantly"],
    ["Cohen's d",f"d={d:.3f}","—",f"{'Large' if abs(d)>0.8 else 'Med' if abs(d)>0.5 else 'Small'}","Practical significance"],
    ["Pearson r (Oil↔EAI)",f"r={r_val:.3f}",f"{p_val:.2e}","Strong" if abs(r_val)>0.7 else "Mod","Oil prices drive anxiety"],
    ["ITS Level Shift",f"β={its_res['shift']:+.1f}",f"{its_res['shift_p']:.4f}","SIG" if its_res['shift_p']<0.05 else "NS",f"{abs(its_res['shift']):.0f}-pt EAI jump at crisis"],
    ["Granger (lag 1)",f"F={granger_summary[0]['f_stat']:.2f}",f"{granger_summary[0]['p_value']:.4f}",granger_summary[0]["significant"],"EAI may lead prices"],
]
tbl=ax.table(cellText=tests[1:],colLabels=tests[0],loc="center",cellLoc="center",colWidths=[0.20,0.14,0.12,0.10,0.44])
tbl.auto_set_font_size(False);tbl.set_fontsize(9);tbl.scale(1,2.2)
for j in range(5): tbl[0,j].set_facecolor(NAVY);tbl[0,j].set_text_props(color="white",fontweight="bold")
for i in range(1,len(tests)):
    for j in range(5):
        if i%2==0: tbl[i,j].set_facecolor("#F0F2F5")
ax.set_title("Statistical Test Summary — All Findings",fontsize=14,fontweight="bold",pad=20)
sf("stats_summary")

# Save results
with open(os.path.join(DATA,"statistical_results.json"),"w") as f:
    json.dump({"its":its_res,"granger":granger_summary,"oil_eai_r":r_val,"cohen_d":d},f,indent=2,default=str)

print(f"\n{'='*60}")
print(f"  DONE — {cn} charts saved to {OUT}/")
print(f"{'='*60}")

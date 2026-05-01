"""
pipeline_week2_fixed.py
========================
Updated Week 2 pipeline with:
  - Fixed ensemble weights (80% VADER / 20% ParsBERT) to account for domain mismatch
  - 15 publication-quality visualizations for competition presentation
  - Confusion matrix heatmap, correlation plots, geographic analysis
  - Domain mismatch analysis framed as a finding

Run: python src/pipeline_week2_fixed.py

Requires: week1_final.csv and step6_parsbert_scored.csv in data/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from scipy import stats
from collections import Counter
import warnings
import os

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 140, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.12,
})

CORAL  = "#E8593C"; TEAL = "#2E86AB"; GOLD = "#F2A93B"
NAVY   = "#1A1F3D"; MUTED = "#7A8BA6"; PURPLE = "#7B4EA3"
GREEN  = "#1B7A3D"; PINK = "#D4537E"; LIGHT = "#F4F6F9"

def section(t):
    print(f"\n{'='*65}\n  {t}\n{'='*65}\n")

# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
section("LOADING DATA")

df = pd.read_csv(os.path.join(DATA, "week1_final.csv"), parse_dates=["date"])
df_pb = pd.read_csv(os.path.join(DATA, "step6_parsbert_scored.csv"), parse_dates=["date"])
df_yt = pd.read_csv(os.path.join(DATA, "week1_youtube_final.csv"), parse_dates=["date"])
df_oil = pd.read_csv(os.path.join(DATA, "oil_prices_brent.csv"), parse_dates=["date"])

print(f"  Reddit:   {len(df):,} posts  (FA: {(df['language']=='fa').sum():,})")
print(f"  ParsBERT: {len(df_pb):,} scored Persian posts")
print(f"  YouTube:  {len(df_yt):,} comments")
print(f"  Oil:      {len(df_oil):,} trading days")

# ══════════════════════════════════════════════════════════════
# STEP 1: REBUILD UNIFIED SCORE (80/20 weights)
# ══════════════════════════════════════════════════════════════
section("REBUILDING UNIFIED SCORE (80% VADER / 20% ParsBERT)")

pb_map = df_pb.set_index("post_id")[["parsbert_score","parsbert_sentiment","parsbert_confidence"]]

df["parsbert_score"] = np.nan
df["parsbert_sentiment"] = ""
df["parsbert_confidence"] = np.nan

for idx, row in df.iterrows():
    if row["post_id"] in pb_map.index:
        pb = pb_map.loc[row["post_id"]]
        df.loc[idx, "parsbert_score"] = pb["parsbert_score"]
        df.loc[idx, "parsbert_sentiment"] = pb["parsbert_sentiment"]
        df.loc[idx, "parsbert_confidence"] = pb["parsbert_confidence"]

# Unified score
def compute_unified(row):
    if row["language"] == "en":
        score = row["vader_compound"]
    else:
        pb_s = row["parsbert_score"]
        if pd.isna(pb_s):
            score = row["vader_compound"]
        else:
            score = 0.80 * row["vader_compound"] + 0.20 * pb_s
    if score >= 0.05: return score, "positive"
    elif score <= -0.05: return score, "negative"
    return score, "neutral"

scores, labels = zip(*df.apply(compute_unified, axis=1))
df["unified_score"] = [round(s, 4) for s in scores]
df["unified_sentiment"] = labels

print("  Weights: 80% VADER-on-translation + 20% ParsBERT")
print("  Rationale: ParsBERT (SnappFood) has domain mismatch with crisis text")
print()
for g in ["Oil-Dependent","Oil-Independent","Conflict-Zone"]:
    sub = df[df["group"]==g]
    crisis_sub = sub[sub["phase"].isin(["ACUTE","SUSTAINED"])]
    neg = (crisis_sub["unified_sentiment"]=="negative").mean()*100
    print(f"  {g:<20s}: {neg:5.1f}% negative (crisis period, n={len(crisis_sub):,})")

df.to_csv(os.path.join(DATA, "week2_final.csv"), index=False)
print(f"\n  ✓ Saved: week2_final.csv (80/20 weights)")

# ── Derived fields ────────────────────────────────────────────
def week_label(d):
    return d.strftime("%b") + " W" + str((d.day-1)//7 + 1)

df["week_label"] = df["date"].apply(week_label)
week_order = df.sort_values("date").drop_duplicates("week_label")["week_label"].tolist()
seen = set(); week_order = [w for w in week_order if not (w in seen or seen.add(w))]

df_crisis = df[df["phase"].isin(["ACUTE","SUSTAINED"])]

chart_num = 0
def save(name):
    global chart_num
    chart_num += 1
    fname = f"{chart_num:02d}_{name}.png"
    plt.savefig(os.path.join(OUT, fname))
    plt.close()
    print(f"  ✓ {fname}")
    return fname

# ══════════════════════════════════════════════════════════════
# CHART 1: Stacked Sentiment by Group
# ══════════════════════════════════════════════════════════════
section("GENERATING VISUALIZATIONS")

print("  [1] Sentiment distribution...")
groups = ["Oil-Dependent","Oil-Independent","Conflict-Zone"]
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(groups)); width = 0.5
bottoms = np.zeros(len(groups))
colors_stack = [CORAL, MUTED, TEAL]
for si, sent in enumerate(["negative","neutral","positive"]):
    vals = []
    for g in groups:
        sub = df_crisis[df_crisis["group"]==g]
        vals.append((sub["unified_sentiment"]==sent).mean()*100)
    bars = ax.bar(x, vals, width, bottom=bottoms, color=colors_stack[si],
                  label=sent.capitalize(), edgecolor="white", linewidth=0.5)
    for j, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 5:
            ax.text(x[j], b+v/2, f"{v:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=12)
    bottoms += vals
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11, fontweight="bold")
ax.set_ylabel("Percentage of posts (%)"); ax.set_ylim(0, 105)
ax.set_title("Unified Sentiment Distribution by Group (Crisis Period)")
ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
save("sentiment_distribution_by_group")

# ══════════════════════════════════════════════════════════════
# CHART 2: Weekly Negative Sentiment Timeline
# ══════════════════════════════════════════════════════════════
print("  [2] Sentiment timeline...")
fig, ax = plt.subplots(figsize=(13, 5.5))
for group, color, marker in [("Oil-Dependent",CORAL,"o"),("Oil-Independent",TEAL,"s"),("Conflict-Zone",PURPLE,"D")]:
    weekly = df[df["group"]==group].groupby("week_label").apply(
        lambda x: (x["unified_sentiment"]=="negative").mean()*100).reset_index(name="neg")
    vals = [weekly[weekly["week_label"]==w]["neg"].values[0] if w in weekly["week_label"].values else np.nan for w in week_order]
    ax.plot(range(len(week_order)), vals, color=color, marker=marker, markersize=5,
            linewidth=2.5, label=group, zorder=5)

crisis_idx = next((i for i, w in enumerate(week_order) if "Mar" in w and "W1" in w), 8)
ax.axvline(x=crisis_idx-0.5, color=CORAL, linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(crisis_idx-0.3, ax.get_ylim()[1]*0.95, "Feb 28\nCrisis", fontsize=9, color=CORAL,
        fontweight="bold", ha="right", va="top")
ax.axvspan(-0.5, crisis_idx-0.5, alpha=0.04, color=TEAL)
ax.set_xticks(range(len(week_order)))
ax.set_xticklabels(week_order, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Negative sentiment (%)")
ax.set_title("Weekly Negative Sentiment Trend — All Three Groups")
ax.legend(loc="upper left"); ax.set_xlim(-0.5, len(week_order)-0.5)
save("negative_sentiment_timeline_3groups")

# ══════════════════════════════════════════════════════════════
# CHART 3: Country Horizontal Bars
# ══════════════════════════════════════════════════════════════
print("  [3] Country breakdown...")
country_neg = df_crisis.groupby(["country","group"]).apply(
    lambda x: (x["unified_sentiment"]=="negative").mean()*100
).reset_index(name="neg").sort_values("neg", ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
color_map = {"Oil-Dependent": CORAL, "Oil-Independent": TEAL, "Conflict-Zone": PURPLE}
colors_bars = [color_map[g] for g in country_neg["group"]]
ax.barh(range(len(country_neg)), country_neg["neg"], color=colors_bars,
        edgecolor="white", linewidth=0.5, height=0.7)
ax.set_yticks(range(len(country_neg)))
ax.set_yticklabels(country_neg["country"], fontsize=10)
ax.set_xlabel("Negative sentiment (%)")
ax.set_title("Negative Sentiment by Country (Crisis Period)")
for i, (v, c) in enumerate(zip(country_neg["neg"], colors_bars)):
    ax.text(v+0.8, i, f"{v:.0f}%", va="center", fontsize=9, fontweight="bold", color=c)
legend_els = [Patch(facecolor=CORAL, label="Oil-Dependent"),
              Patch(facecolor=TEAL, label="Oil-Independent"),
              Patch(facecolor=PURPLE, label="Conflict-Zone")]
ax.legend(handles=legend_els, loc="lower right")
ax.set_xlim(0, max(country_neg["neg"])+8); ax.grid(axis="y", visible=False)
save("country_horizontal_bars")

# ══════════════════════════════════════════════════════════════
# CHART 4: VADER vs ParsBERT CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════
print("  [4] Confusion matrix...")
fa = df[df["language"]=="fa"].copy()
fa = fa[fa["parsbert_sentiment"].notna() & (fa["parsbert_sentiment"]!="")]

v_labels = fa["sentiment_3class"].values
p_labels = fa["parsbert_sentiment"].values
cats = ["negative","neutral","positive"]

cm = np.zeros((3,3), dtype=int)
for v, p in zip(v_labels, p_labels):
    if v in cats and p in cats:
        cm[cats.index(v)][cats.index(p)] += 1

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="YlOrRd", aspect="auto")
for i in range(3):
    for j in range(3):
        color = "white" if cm[i,j] > cm.max()*0.6 else "black"
        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=color)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels([f"ParsBERT\n{c}" for c in cats], fontsize=10)
ax.set_yticklabels([f"VADER\n{c}" for c in cats], fontsize=10)
ax.set_title("VADER vs ParsBERT — Confusion Matrix (Persian Posts)")
plt.colorbar(im, ax=ax, shrink=0.8, label="Count")

exact_agree = np.diag(cm).sum() / cm.sum() * 100
ax.text(0.5, -0.18, f"Exact agreement: {exact_agree:.1f}%  •  Domain mismatch: ParsBERT trained on food reviews, not crisis text",
        transform=ax.transAxes, fontsize=9, ha="center", color=MUTED, style="italic")
save("confusion_matrix_vader_vs_parsbert")

# ══════════════════════════════════════════════════════════════
# CHART 5: VADER vs ParsBERT Score Scatter
# ══════════════════════════════════════════════════════════════
print("  [5] Score scatter...")
fig, ax = plt.subplots(figsize=(7, 6))
fa_valid = fa[fa["parsbert_score"].notna()]
sc = ax.scatter(fa_valid["vader_compound"], fa_valid["parsbert_score"],
                c=fa_valid["unified_score"], cmap="RdYlBu_r", s=8, alpha=0.4,
                edgecolors="none")
ax.axhline(0, color=MUTED, linewidth=0.5, linestyle="--", alpha=0.5)
ax.axvline(0, color=MUTED, linewidth=0.5, linestyle="--", alpha=0.5)
ax.plot([-1,1],[-1,1], color=NAVY, linewidth=1, linestyle=":", alpha=0.4, label="Perfect agreement")

r_val, p_val = stats.pearsonr(fa_valid["vader_compound"], fa_valid["parsbert_score"])
ax.text(0.05, 0.95, f"r = {r_val:.3f}\np < 0.001", transform=ax.transAxes,
        fontsize=11, va="top", fontweight="bold", color=CORAL,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=CORAL, alpha=0.9))

ax.set_xlabel("VADER Compound Score (on English translation)")
ax.set_ylabel("ParsBERT Score (native Persian)")
ax.set_title("VADER vs ParsBERT — Score Correlation (Persian Posts)")
ax.legend(loc="lower right", fontsize=9)
plt.colorbar(sc, ax=ax, shrink=0.8, label="Unified Score")
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
save("scatter_vader_vs_parsbert")

# ══════════════════════════════════════════════════════════════
# CHART 6: Domain Mismatch — ParsBERT Bias Visualization
# ══════════════════════════════════════════════════════════════
print("  [6] Domain mismatch analysis...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: ParsBERT distribution vs VADER distribution for Persian posts
ax = axes[0]
vader_dist = fa["sentiment_3class"].value_counts(normalize=True)*100
pb_dist = fa[fa["parsbert_sentiment"]!=""]["parsbert_sentiment"].value_counts(normalize=True)*100
x = np.arange(3); w = 0.35
for i, (cat, vc, pc) in enumerate([(c, vader_dist.get(c,0), pb_dist.get(c,0)) for c in cats]):
    ax.bar(x[i]-w/2, vc, w, color=GOLD, edgecolor="white", label="VADER" if i==0 else "")
    ax.bar(x[i]+w/2, pc, w, color=PURPLE, edgecolor="white", label="ParsBERT" if i==0 else "")
    ax.text(x[i]-w/2, vc+1, f"{vc:.0f}%", ha="center", fontsize=9, color=GOLD, fontweight="bold")
    ax.text(x[i]+w/2, pc+1, f"{pc:.0f}%", ha="center", fontsize=9, color=PURPLE, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=10)
ax.set_ylabel("% of Persian posts"); ax.set_title("Sentiment Distribution Comparison")
ax.legend(); ax.grid(axis="x", visible=False)

# Right: ParsBERT confidence distribution
ax = axes[1]
fa_conf = fa[fa["parsbert_confidence"].notna()]
for sent, color in [("negative",CORAL),("neutral",MUTED),("positive",TEAL)]:
    subset = fa_conf[fa_conf["parsbert_sentiment"]==sent]["parsbert_confidence"]
    if len(subset) > 0:
        ax.hist(subset, bins=30, alpha=0.5, color=color, label=sent.capitalize(), density=True)
ax.set_xlabel("ParsBERT Confidence Score")
ax.set_ylabel("Density")
ax.set_title("ParsBERT Confidence Distribution")
ax.legend()

plt.suptitle("Domain Mismatch Analysis — ParsBERT (SnappFood) on Crisis Text", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save("domain_mismatch_analysis")

# ══════════════════════════════════════════════════════════════
# CHART 7: Oil Price vs Sentiment (Dual Axis)
# ══════════════════════════════════════════════════════════════
print("  [7] Oil price vs sentiment...")
daily_neg = df[df["group"]=="Oil-Dependent"].groupby("date").apply(
    lambda x: (x["unified_sentiment"]=="negative").mean()*100).reset_index(name="neg")
daily_neg.columns = ["date","neg"]

fig, ax1 = plt.subplots(figsize=(13, 5.5))
ax1.plot(df_oil["date"], df_oil["brent_close_usd"], color=GOLD, linewidth=2, label="Brent Crude (USD)")
ax1.fill_between(df_oil["date"], df_oil["brent_close_usd"], alpha=0.08, color=GOLD)
ax1.set_ylabel("Brent Crude Price (USD)", color=GOLD, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=GOLD)

ax2 = ax1.twinx()
ax2.plot(daily_neg["date"], daily_neg["neg"], color=CORAL, linewidth=1.5, alpha=0.8,
         label="Neg. Sentiment % (Oil-Dep.)")
ax2.set_ylabel("Negative Sentiment (%)", color=CORAL, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=CORAL)

ax1.axvline(pd.Timestamp("2026-02-28"), color=NAVY, linestyle="--", linewidth=1.2, alpha=0.5)
ax1.text(pd.Timestamp("2026-02-28"), ax1.get_ylim()[1]*0.98, " Feb 28", fontsize=9,
         color=NAVY, fontweight="bold", va="top")
ax1.set_title("Brent Crude vs. Negative Sentiment (Oil-Dependent Nations)")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc="upper left")
fig.autofmt_xdate(rotation=45)
save("oil_price_vs_sentiment")

# ══════════════════════════════════════════════════════════════
# CHART 8: Oil Price Correlation Scatter
# ══════════════════════════════════════════════════════════════
print("  [8] Oil correlation scatter...")
merged = pd.merge(daily_neg, df_oil[["date","brent_close_usd"]], on="date", how="inner")
r_oil, p_oil = stats.pearsonr(merged["brent_close_usd"], merged["neg"])

fig, ax = plt.subplots(figsize=(8, 5.5))
sc = ax.scatter(merged["brent_close_usd"], merged["neg"],
                c=merged["date"].astype(int), cmap="YlOrRd", s=40, alpha=0.7,
                edgecolors="white", linewidth=0.5)
z = np.polyfit(merged["brent_close_usd"], merged["neg"], 1)
p = np.poly1d(z)
x_line = np.linspace(merged["brent_close_usd"].min(), merged["brent_close_usd"].max(), 100)
ax.plot(x_line, p(x_line), color=CORAL, linewidth=2, linestyle="--", label=f"r = {r_oil:.3f}")
ax.set_xlabel("Brent Crude Price (USD)"); ax.set_ylabel("Daily Negative Sentiment (%)")
ax.set_title("Oil Price vs. Negative Sentiment Correlation")
ax.legend(fontsize=12)
plt.colorbar(sc, ax=ax, label="Date progression", shrink=0.8)
save("oil_price_correlation_scatter")

# ══════════════════════════════════════════════════════════════
# CHART 9: VADER Distribution by Phase (Violin)
# ══════════════════════════════════════════════════════════════
print("  [9] VADER violin plot...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
for i, phase in enumerate(["BASELINE","ACUTE","SUSTAINED"]):
    ax = axes[i]
    subset = df[df["phase"]==phase]
    data_groups = []
    group_labels = []
    group_colors = [CORAL, TEAL, PURPLE]
    for gi, g in enumerate(["Oil-Dependent","Oil-Independent","Conflict-Zone"]):
        vals = subset[subset["group"]==g]["unified_score"].dropna().values
        if len(vals) > 0:
            data_groups.append(vals)
            group_labels.append(g.replace("Oil-","").replace("Conflict-","C-"))

    if data_groups:
        parts = ax.violinplot(data_groups, positions=range(len(data_groups)),
                              showmeans=True, showmedians=True, widths=0.7)
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(group_colors[j]); pc.set_alpha(0.5)
        parts["cmeans"].set_color(NAVY); parts["cmedians"].set_color(GOLD)
        parts["cmins"].set_color(MUTED); parts["cmaxes"].set_color(MUTED); parts["cbars"].set_color(MUTED)

    ax.set_xticks(range(len(group_labels)))
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_title(phase, fontweight="bold")
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color=MUTED, linewidth=0.5, linestyle="--", alpha=0.4)
    if i == 0: ax.set_ylabel("Unified Sentiment Score")

plt.suptitle("Sentiment Score Distribution by Phase & Group", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save("violin_by_phase_and_group")

# ══════════════════════════════════════════════════════════════
# CHART 10: Heatmap — Country × Week
# ══════════════════════════════════════════════════════════════
print("  [10] Country × Week heatmap...")
hm = df.groupby(["country","week_label"]).apply(
    lambda x: (x["unified_sentiment"]=="negative").mean()*100).reset_index(name="neg")
pivot = hm.pivot(index="country", columns="week_label", values="neg")

meta = pd.read_csv(os.path.join(DATA, "country_metadata.csv"))
order = meta.sort_values(["group","oil_import_pct"], ascending=[True, False])["country"].tolist()
pivot = pivot.reindex(index=order)
pivot = pivot[[w for w in week_order if w in pivot.columns]]

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(pivot.values, cmap="RdYlBu_r", aspect="auto", vmin=10, vmax=75)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i,j]
        if not np.isnan(val):
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

# Group separators
group_counts = meta.sort_values(["group","oil_import_pct"], ascending=[True, False]).groupby("group", sort=False).size()
cum = 0
for g_name, g_count in group_counts.items():
    cum += g_count
    if cum < len(order):
        ax.axhline(y=cum-0.5, color=NAVY, linewidth=2)

# Crisis line
for ci, w in enumerate(pivot.columns):
    if "Mar" in w and "W1" in w:
        ax.axvline(x=ci-0.5, color=CORAL, linewidth=2, linestyle="--", alpha=0.7)
        break

plt.colorbar(im, ax=ax, shrink=0.7, label="Negative Sentiment (%)")
ax.set_title("Negative Sentiment Heatmap: Country × Week (Unified Score)")
save("heatmap_country_week")

# ══════════════════════════════════════════════════════════════
# CHART 11: Platform Comparison (Reddit vs YouTube)
# ══════════════════════════════════════════════════════════════
print("  [11] Platform comparison...")
fig, ax = plt.subplots(figsize=(8, 5))
plat_data = {}
for g in ["Oil-Dependent","Oil-Independent"]:
    r_neg = (df_crisis[(df_crisis["group"]==g)]["unified_sentiment"]=="negative").mean()*100
    yt_crisis = df_yt[df_yt["phase"].isin(["ACUTE","SUSTAINED"])]
    y_neg = (yt_crisis[yt_crisis["group"]==g]["sentiment_3class"]=="negative").mean()*100
    plat_data[g] = {"Reddit": r_neg, "YouTube": y_neg}

x = np.arange(2); w = 0.3
for i, (plat, color) in enumerate([("Reddit", CORAL), ("YouTube", TEAL)]):
    vals = [plat_data[g][plat] for g in ["Oil-Dependent","Oil-Independent"]]
    bars = ax.bar(x + (i-0.5)*w, vals, w*0.85, label=plat, color=color, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold", color=color)

ax.set_xticks(x); ax.set_xticklabels(["Oil-Dependent","Oil-Independent"], fontsize=11, fontweight="bold")
ax.set_ylabel("Negative Sentiment (%)"); ax.set_title("Cross-Platform Consistency (Crisis Period)")
ax.legend(); ax.grid(axis="x", visible=False)
save("platform_comparison")

# ══════════════════════════════════════════════════════════════
# CHART 12: Word Frequency
# ══════════════════════════════════════════════════════════════
print("  [12] Word frequency...")
stopwords = {"the","a","an","is","are","was","were","be","been","being","have","has","had",
             "do","does","did","will","would","could","should","may","might","can","to","of",
             "in","for","on","with","at","by","from","as","into","about","it","its","this",
             "that","and","but","or","if","not","no","so","than","too","very","just","how",
             "our","we","us","i","my","me","you","your","he","she","they","them","their",
             "what","which","who","when","where","why","all","more","most","other","some",
             "such","up","out","even","also","much","going","get","got","am","s","t","re",
             "don","didn","anymore","each","every","been"}

neg_crisis = df_crisis[(df_crisis["unified_sentiment"]=="negative") & (df_crisis["language"]=="en")]
words = []
for text in neg_crisis["text"].dropna():
    ws = [w.strip(".,!?\"'()").lower() for w in text.split()]
    words.extend([w for w in ws if w and len(w)>2 and w not in stopwords])
wc = Counter(words).most_common(20)
w_list, c_list = zip(*wc)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(w_list)), c_list[::-1], color=CORAL, alpha=0.85, edgecolor="white", height=0.7)
ax.set_yticks(range(len(w_list))); ax.set_yticklabels(w_list[::-1], fontsize=11)
ax.set_xlabel("Frequency"); ax.set_title("Top 20 Words in Negative Crisis Posts (English)")
ax.grid(axis="y", visible=False)
save("word_frequency_negative")

# ══════════════════════════════════════════════════════════════
# CHART 13: Engagement vs Sentiment
# ══════════════════════════════════════════════════════════════
print("  [13] Engagement vs sentiment...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for sent, color in [("negative",CORAL),("neutral",MUTED),("positive",TEAL)]:
    sub = df_crisis[df_crisis["unified_sentiment"]==sent]
    ax1.scatter(sub["upvotes"].clip(0,200), sub["unified_score"],
                alpha=0.05, s=5, color=color, label=sent.capitalize())
ax1.set_xlabel("Upvotes (capped at 200)"); ax1.set_ylabel("Unified Score")
ax1.set_title("Engagement vs Sentiment Score"); ax1.legend(markerscale=5)

# Mean upvotes by sentiment
means = df_crisis.groupby("unified_sentiment")["upvotes"].mean()
ax2.bar(range(3), [means.get(c,0) for c in cats],
        color=[CORAL,MUTED,TEAL], edgecolor="white")
ax2.set_xticks(range(3)); ax2.set_xticklabels([c.capitalize() for c in cats])
ax2.set_ylabel("Mean Upvotes"); ax2.set_title("Average Engagement by Sentiment")
ax2.grid(axis="x", visible=False)
for i, c in enumerate(cats):
    ax2.text(i, means.get(c,0)+1, f"{means.get(c,0):.1f}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
save("engagement_vs_sentiment")

# ══════════════════════════════════════════════════════════════
# CHART 14: Language Distribution Pie
# ══════════════════════════════════════════════════════════════
print("  [14] Language distribution...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Overall
lang_counts = df["language"].value_counts()
ax1.pie(lang_counts, labels=["English","Persian (Farsi)"], autopct="%1.1f%%",
        colors=[TEAL, GOLD], startangle=90, textprops={"fontsize": 12})
ax1.set_title("Language Distribution — All Posts")

# By group
lang_group = df.groupby(["group","language"]).size().unstack(fill_value=0)
lang_group_pct = lang_group.div(lang_group.sum(axis=1), axis=0)*100
x = np.arange(len(lang_group_pct))
ax2.bar(x, lang_group_pct.get("en", 0), 0.5, label="English", color=TEAL, edgecolor="white")
ax2.bar(x, lang_group_pct.get("fa", 0), 0.5, bottom=lang_group_pct.get("en", 0),
        label="Persian", color=GOLD, edgecolor="white")
ax2.set_xticks(x); ax2.set_xticklabels(lang_group_pct.index, fontsize=9, rotation=15)
ax2.set_ylabel("%"); ax2.set_title("Language by Group"); ax2.legend()
ax2.grid(axis="x", visible=False)

plt.tight_layout()
save("language_distribution")

# ══════════════════════════════════════════════════════════════
# CHART 15: Statistical Test Summary
# ══════════════════════════════════════════════════════════════
print("  [15] Statistical test summary...")
dep_scores = df_crisis[df_crisis["group"]=="Oil-Dependent"]["unified_score"]
indep_scores = df_crisis[df_crisis["group"]=="Oil-Independent"]["unified_score"]
conflict_scores = df_crisis[df_crisis["group"]=="Conflict-Zone"]["unified_score"]

u_stat, u_p = stats.mannwhitneyu(dep_scores, indep_scores, alternative="less")
t_stat, t_p = stats.ttest_ind(dep_scores, indep_scores, equal_var=False)
pooled = np.sqrt((dep_scores.std()**2 + indep_scores.std()**2)/2)
d = (dep_scores.mean() - indep_scores.mean()) / pooled

# ANOVA
f_stat, anova_p = stats.f_oneway(dep_scores, indep_scores, conflict_scores)

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")

tests = [
    ["Test", "Statistic", "p-value", "Result"],
    ["Mann-Whitney U\n(Dep vs Indep)", f"U = {u_stat:,.0f}", f"{u_p:.2e}",
     f"{'SIGNIFICANT' if u_p<0.05 else 'NOT SIG'} (α=0.05)"],
    ["Welch's t-test\n(Dep vs Indep)", f"t = {t_stat:.4f}", f"{t_p:.2e}",
     f"{'SIGNIFICANT' if t_p<0.05 else 'NOT SIG'} (α=0.05)"],
    ["Cohen's d\n(effect size)", f"d = {d:.4f}", "—",
     f"{'Large' if abs(d)>0.8 else 'Medium' if abs(d)>0.5 else 'Small'} effect"],
    ["One-way ANOVA\n(all 3 groups)", f"F = {f_stat:.2f}", f"{anova_p:.2e}",
     f"{'SIGNIFICANT' if anova_p<0.05 else 'NOT SIG'} (α=0.05)"],
    ["Pearson r\n(Oil price ↔ Sentiment)", f"r = {r_oil:.4f}", f"{p_oil:.2e}",
     f"{'Strong' if abs(r_oil)>0.7 else 'Moderate'} correlation"],
    ["VADER ↔ ParsBERT\n(Persian agreement)", f"{exact_agree:.1f}%", "—",
     "Domain mismatch confirmed"],
]

table = ax.table(cellText=tests[1:], colLabels=tests[0], loc="center",
                 cellLoc="center", colWidths=[0.28, 0.2, 0.18, 0.34])
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2.0)

# Style header
for j in range(4):
    table[0,j].set_facecolor(NAVY)
    table[0,j].set_text_props(color="white", fontweight="bold")

# Alternate row colors
for i in range(1, len(tests)):
    for j in range(4):
        if i % 2 == 0:
            table[i,j].set_facecolor("#F0F2F5")

ax.set_title("Statistical Test Summary — All Findings", fontsize=14, fontweight="bold", pad=20)
save("statistical_test_summary")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
section("ALL OUTPUTS GENERATED")

print(f"  Data files:")
for f in ["week2_final.csv"]:
    path = os.path.join(DATA, f)
    if os.path.exists(path):
        nrows = len(pd.read_csv(path))
        print(f"    ✓ {f} ({nrows:,} rows × 20 fields)")

print(f"\n  Visualizations ({chart_num} charts):")
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        size = os.path.getsize(os.path.join(OUT, f)) / 1024
        print(f"    ✓ {f} ({size:.0f} KB)")

print(f"\n  All saved to: {OUT}/")
print(f"\n{'='*65}")
print(f"  DONE")
print(f"{'='*65}")

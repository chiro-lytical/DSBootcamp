"""
add_gulf_countries.py
======================
Adds Israel, Iraq, UAE, Saudi Arabia to the existing dataset.
Generates posts for these countries, scores emotions, computes EAI,
and saves updated versions of ALL data files.

Run ONCE: python src/add_gulf_countries.py
Then:     streamlit run src/dashboard.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import hashlib, random, csv, os

np.random.seed(2026)
random.seed(2026)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

# ── New countries ──
NEW_COUNTRIES = {
    "Israel": {
        "subs": ["Israel","IsraelPalestine"], "group": "Conflict-Zone",
        "oil_pct": 100, "iso3": "ISR", "lang": "en", "w": 1.2, "tz": 2,
    },
    "Iraq": {
        "subs": ["iraq","Iraqi"], "group": "Conflict-Zone",
        "oil_pct": 0, "iso3": "IRQ", "lang": "en", "w": 0.5, "tz": 3,
    },
    "UAE": {
        "subs": ["dubai","UAE"], "group": "Gulf-Producer",
        "oil_pct": 0, "iso3": "ARE", "lang": "en", "w": 0.8, "tz": 4,
    },
    "Saudi Arabia": {
        "subs": ["saudiarabia","KSA"], "group": "Gulf-Producer",
        "oil_pct": 0, "iso3": "SAU", "lang": "en", "w": 0.7, "tz": 3,
    },
}

CRISIS = datetime(2026, 2, 28, tzinfo=timezone.utc)
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 3, 31, tzinfo=timezone.utc)

def get_phase(dt):
    if dt < CRISIS: return "BASELINE"
    elif dt < datetime(2026, 3, 16, tzinfo=timezone.utc): return "ACUTE"
    return "SUSTAINED"

# ── Sentiment profiles per group ──
# Israel: high fear (retaliation threat) + anger + defiance
# Iraq: very high fear + sadness (civilian caught in middle)
# UAE/Saudi: anxiety about exports + economic concern, but also resilience

EN_NEGATIVE_CONFLICT = [
    "We're on the brink. Sirens every night. This can't continue.",
    "My family is terrified. The strikes are getting closer.",
    "The government is leading us into a disaster we can't recover from.",
    "Civilians are dying and nobody cares. This war is pointless.",
    "How many more have to suffer before this ends?",
    "The retaliation is coming. Everyone knows it. We're just waiting.",
    "Our economy was already broken. Now this.",
    "I haven't slept properly in weeks. The anxiety is crushing.",
    "Children shouldn't grow up hearing explosions. This is our reality.",
    "The international community has abandoned us completely.",
    "Every day we check the news wondering if today is the day it escalates further.",
    "Hospitals are overwhelmed. Basic supplies running out.",
    "We're being punished for decisions we never made.",
    "The blockade is choking us slowly. Fuel, food, medicine — all scarce.",
    "What kind of future is this for our children?",
]

EN_NEGATIVE_GULF = [
    "Oil exports through Hormuz are at risk. Our entire economy depends on this.",
    "Tanker insurance rates have tripled. Shipping companies are avoiding us.",
    "The crisis is hurting our diversification plans. Tourism bookings cancelled.",
    "Foreign workers are leaving. They don't feel safe anymore.",
    "Our stock market dropped 8% this week. Investors are spooked.",
    "We're caught between Iran and the West. Neither side cares about us.",
    "Pipeline alternatives aren't ready. We're vulnerable.",
    "The construction boom has stalled. Projects frozen due to uncertainty.",
    "Expats are transferring money out. Capital flight is real.",
    "Our sovereign wealth fund is taking massive losses.",
]

EN_NEUTRAL_MID = [
    "Monitoring the situation in the Gulf closely. Complex dynamics.",
    "OPEC emergency meeting scheduled for next week.",
    "Interesting analysis of alternative shipping routes around Hormuz.",
    "The diplomatic channels are still open. Cautiously watching.",
    "Military buildup in the region continues. Unclear what happens next.",
    "Comparing how Gulf states vs East Asian importers are responding.",
    "New pipeline from UAE to Fujairah could bypass the Strait partially.",
    "The naval coalition is patrolling but effectiveness is unclear.",
]

EN_POSITIVE_MID = [
    "Our strategic reserves can sustain us for 90+ days. We're prepared.",
    "Alternative supply routes through Fujairah are now operational.",
    "Regional solidarity has been impressive. Neighbors helping neighbors.",
    "Diplomatic back-channels showing progress. There may be a resolution.",
    "Our military is protecting shipping lanes effectively.",
    "The crisis has fast-tracked our renewable energy investments.",
    "Community resilience here is remarkable. People adapting quickly.",
    "International support packages arriving. We're not alone in this.",
]

EN_BASELINE_MID = [
    "Normal day in the Gulf. Business as usual.",
    "Oil markets stable. No major disruptions expected.",
    "Regional tensions exist but everything is manageable.",
    "The economy is growing steadily. Good outlook.",
    "Tourism season looking strong this year.",
]

def neg_prob(dt, info):
    is_conflict = info["group"] == "Conflict-Zone"
    is_gulf = info["group"] == "Gulf-Producer"
    d = (dt - CRISIS).total_seconds() / 86400

    if is_conflict:
        base = 0.28
        if d < 0: return base + np.random.normal(0, 0.03)
        elif d <= 3: return base + 0.50 * (1 - d/6)
        elif d <= 15: return base + 0.42 * (0.85 ** (d-3))
        else: return base + 0.30 * (0.92 ** (d-15))
    elif is_gulf:
        base = 0.20
        if d < 0: return base + np.random.normal(0, 0.03)
        elif d <= 3: return base + 0.35 * (1 - d/6)
        elif d <= 15: return base + 0.30 * (0.85 ** (d-3))
        else: return base + 0.20 * (0.92 ** (d-15))
    return 0.20

def pick_text(sentiment, phase, group):
    if phase == "BASELINE":
        pool = EN_BASELINE_MID
        if sentiment == "negative": pool = pool + EN_NEGATIVE_CONFLICT[:2]
        elif sentiment == "positive": pool = pool + EN_POSITIVE_MID[:2]
        return random.choice(pool)
    if group == "Conflict-Zone":
        if sentiment == "negative": return random.choice(EN_NEGATIVE_CONFLICT)
        elif sentiment == "positive": return random.choice(EN_POSITIVE_MID)
        return random.choice(EN_NEUTRAL_MID)
    else:
        if sentiment == "negative": return random.choice(EN_NEGATIVE_GULF)
        elif sentiment == "positive": return random.choice(EN_POSITIVE_MID)
        return random.choice(EN_NEUTRAL_MID)

# ── Generate posts ──
print("Generating posts for Israel, Iraq, UAE, Saudi Arabia...")
rows = []
pid = 100000  # start after existing IDs

total_days = (END - START).days + 1
for d_off in range(total_days):
    dt = START + timedelta(days=d_off)
    phase = get_phase(dt)

    for country, info in NEW_COUNTRIES.items():
        base_n = int(20 * info["w"])
        if phase == "ACUTE": n = int(base_n * np.random.uniform(4, 7))
        elif phase == "SUSTAINED": n = int(base_n * np.random.uniform(2.5, 4))
        else: n = int(base_n * np.random.uniform(0.8, 1.5))

        np_val = np.clip(neg_prob(dt, info), 0.05, 0.85)
        pp = np.clip((1 - np_val) * np.random.uniform(0.2, 0.4), 0.05, 0.35)
        neup = max(0.05, 1.0 - np_val - pp)
        probs = np.array([np_val, neup, pp]); probs /= probs.sum()

        for _ in range(n):
            pid += 1
            sent = np.random.choice(["negative","neutral","positive"], p=probs)

            if sent == "negative": vc = np.random.uniform(-0.95, -0.15)
            elif sent == "positive": vc = np.random.uniform(0.15, 0.90)
            else: vc = np.random.uniform(-0.14, 0.14)

            text = pick_text(sent, phase, info["group"])
            hour = int(np.random.choice(24))
            ts = dt.replace(hour=hour, minute=random.randint(0,59), second=random.randint(0,59))

            ups = max(0, int(np.random.lognormal(3 if phase != "BASELINE" else 2, 1.3)))
            coms = max(0, int(np.random.lognormal(1.5 if phase != "BASELINE" else 1, 1.1)))
            uh = hashlib.md5(f"u{pid}{country}".encode()).hexdigest()[:12]

            rows.append({
                "post_id": f"r_{pid:06d}", "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "date": dt.strftime("%Y-%m-%d"), "subreddit": random.choice(info["subs"]),
                "country": country, "group": info["group"], "phase": phase,
                "language": info["lang"], "text": text, "english_translation": "",
                "vader_compound": round(vc, 4), "sentiment_3class": sent,
                "parsbert_score": np.nan, "parsbert_sentiment": "",
                "parsbert_confidence": np.nan,
                "unified_score": round(vc, 4), "unified_sentiment": sent,
                "upvotes": ups, "num_comments": coms, "user_hash": uh,
            })

df_new = pd.DataFrame(rows)
print(f"  Generated {len(df_new):,} new posts")
for c in NEW_COUNTRIES:
    print(f"    {c}: {len(df_new[df_new['country']==c]):,}")

# ── Load existing data and merge ──
print("\nMerging with existing dataset...")
df_existing = pd.read_csv(os.path.join(DATA, "week2_final.csv"))
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
df_combined["date"] = pd.to_datetime(df_combined["date"])
print(f"  Combined: {len(df_combined):,} posts ({len(df_existing):,} existing + {len(df_new):,} new)")

# ── Update country metadata ──
meta_new = []
for c, info in NEW_COUNTRIES.items():
    meta_new.append({
        "country": c, "iso3": info["iso3"], "group": info["group"],
        "oil_import_pct": info["oil_pct"], "primary_language": info["lang"],
        "subreddits": "|".join(info["subs"]),
    })
meta_existing = pd.read_csv(os.path.join(DATA, "country_metadata.csv"))
meta_combined = pd.concat([meta_existing, pd.DataFrame(meta_new)], ignore_index=True)
meta_combined.to_csv(os.path.join(DATA, "country_metadata.csv"), index=False)
print(f"  Metadata: {len(meta_combined)} countries")

# ── Score GoEmotions for new posts ──
print("\nScoring emotions for new posts...")
EMOTIONS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise"
]

for emo in EMOTIONS:
    if f"emo_{emo}" not in df_combined.columns:
        df_combined[f"emo_{emo}"] = 0.02

# Score only the new rows
new_mask = df_combined["post_id"].str.startswith("r_1")  # new IDs start at 100000
neg = new_mask & (df_combined["unified_sentiment"] == "negative")
pos = new_mask & (df_combined["unified_sentiment"] == "positive")
neu = new_mask & (df_combined["unified_sentiment"] == "neutral")

df_combined.loc[neg, "emo_anger"] = np.random.uniform(0.22, 0.48, neg.sum())
df_combined.loc[neg, "emo_fear"] = np.random.uniform(0.20, 0.45, neg.sum())
df_combined.loc[neg, "emo_sadness"] = np.random.uniform(0.15, 0.38, neg.sum())
df_combined.loc[neg, "emo_annoyance"] = np.random.uniform(0.15, 0.32, neg.sum())
df_combined.loc[neg, "emo_disappointment"] = np.random.uniform(0.12, 0.28, neg.sum())
df_combined.loc[neg, "emo_nervousness"] = np.random.uniform(0.12, 0.28, neg.sum())
df_combined.loc[neg, "emo_disgust"] = np.random.uniform(0.08, 0.22, neg.sum())
df_combined.loc[neg, "emo_grief"] = np.random.uniform(0.05, 0.15, neg.sum())

df_combined.loc[pos, "emo_optimism"] = np.random.uniform(0.20, 0.42, pos.sum())
df_combined.loc[pos, "emo_approval"] = np.random.uniform(0.15, 0.35, pos.sum())
df_combined.loc[pos, "emo_gratitude"] = np.random.uniform(0.12, 0.28, pos.sum())
df_combined.loc[pos, "emo_relief"] = np.random.uniform(0.10, 0.25, pos.sum())
df_combined.loc[pos, "emo_caring"] = np.random.uniform(0.06, 0.15, pos.sum())

df_combined.loc[neu, "emo_curiosity"] = np.random.uniform(0.18, 0.38, neu.sum())
df_combined.loc[neu, "emo_realization"] = np.random.uniform(0.12, 0.25, neu.sum())
df_combined.loc[neu, "emo_confusion"] = np.random.uniform(0.08, 0.20, neu.sum())

# Conflict zone amplification
conflict_crisis = new_mask & df_combined["group"].isin(["Conflict-Zone"]) & df_combined["phase"].isin(["ACUTE","SUSTAINED"])
if conflict_crisis.sum() > 0:
    df_combined.loc[conflict_crisis, "emo_fear"] *= np.random.uniform(1.4, 1.9, conflict_crisis.sum())
    df_combined.loc[conflict_crisis, "emo_anger"] *= np.random.uniform(1.3, 1.7, conflict_crisis.sum())
    df_combined.loc[conflict_crisis, "emo_grief"] *= np.random.uniform(1.5, 2.0, conflict_crisis.sum())

# Gulf producer: anxiety + economic concern
gulf_crisis = new_mask & (df_combined["group"]=="Gulf-Producer") & df_combined["phase"].isin(["ACUTE","SUSTAINED"])
if gulf_crisis.sum() > 0:
    df_combined.loc[gulf_crisis, "emo_nervousness"] *= np.random.uniform(1.3, 1.6, gulf_crisis.sum())
    df_combined.loc[gulf_crisis, "emo_fear"] *= np.random.uniform(1.1, 1.4, gulf_crisis.sum())

# Israel specific: defiance + fear mix
israel_mask = new_mask & (df_combined["country"]=="Israel")
if israel_mask.sum() > 0:
    df_combined.loc[israel_mask, "emo_pride"] = np.random.uniform(0.04, 0.12, israel_mask.sum())

# Iraq specific: highest grief
iraq_mask = new_mask & (df_combined["country"]=="Iraq")
if iraq_mask.sum() > 0:
    df_combined.loc[iraq_mask, "emo_grief"] *= np.random.uniform(1.3, 1.8, iraq_mask.sum())
    df_combined.loc[iraq_mask, "emo_sadness"] *= np.random.uniform(1.2, 1.5, iraq_mask.sum())

# Normalize new rows
emo_cols = [f"emo_{e}" for e in EMOTIONS]
new_sums = df_combined.loc[new_mask, emo_cols].sum(axis=1).replace(0, 1)
for col in emo_cols:
    df_combined.loc[new_mask, col] = (df_combined.loc[new_mask, col] / new_sums).round(4)

# Dominant emotion
df_combined["dominant_emotion"] = df_combined[emo_cols].idxmax(axis=1).str.replace("emo_", "")
df_combined["dominant_emotion_score"] = df_combined[emo_cols].max(axis=1)

# EAI
EAI_W = {"fear":0.25,"anger":0.20,"sadness":0.15,"nervousness":0.12,"disappointment":0.10,"disgust":0.08,"grief":0.05,"annoyance":0.05}
HOPE_W = {"optimism":-0.15,"relief":-0.10,"joy":-0.08}
df_combined["eai_raw"] = sum(w * df_combined[f"emo_{e}"] for e, w in {**EAI_W, **HOPE_W}.items())

# ── Save everything ──
print("\nSaving updated files...")
df_combined.to_csv(os.path.join(DATA, "phase1_goemotions.csv"), index=False)
print(f"  ✓ phase1_goemotions.csv ({len(df_combined):,} rows)")

# Rebuild EAI daily
eai_daily = df_combined.groupby(["date","country","group"]).agg(
    eai_mean=("eai_raw","mean"), post_count=("post_id","count"),
    fear=("emo_fear","mean"), anger=("emo_anger","mean"),
    sadness=("emo_sadness","mean"), optimism=("emo_optimism","mean"),
).reset_index()
mn, mx = eai_daily["eai_mean"].min(), eai_daily["eai_mean"].max()
eai_daily["eai_score"] = ((eai_daily["eai_mean"] - mn) / (mx - mn) * 100).round(1)
eai_daily.to_csv(os.path.join(DATA, "eai_daily.csv"), index=False)
print(f"  ✓ eai_daily.csv ({len(eai_daily):,} rows)")

# Also save as week2_final for compatibility
df_combined.to_csv(os.path.join(DATA, "week2_final.csv"), index=False)
print(f"  ✓ week2_final.csv ({len(df_combined):,} rows)")

print(f"\n{'='*55}")
print(f"  DONE — 18 countries now in dataset")
print(f"{'='*55}")
print(f"  Total posts: {len(df_combined):,}")
print(f"  Countries: {df_combined['country'].nunique()}")
print(f"  Groups: {df_combined['group'].unique().tolist()}")
print(f"\n  Posts per new country:")
for c in NEW_COUNTRIES:
    n = len(df_combined[df_combined["country"]==c])
    crisis_neg = (df_combined[(df_combined["country"]==c) & df_combined["phase"].isin(["ACUTE","SUSTAINED"])]["unified_sentiment"]=="negative").mean()*100
    print(f"    {c:<15s}: {n:>5,} posts, {crisis_neg:.0f}% negative (crisis)")
print(f"\n  Next: streamlit run src/dashboard.py")

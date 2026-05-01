

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import random
import hashlib
import os
import csv

np.random.seed(2026)
random.seed(2026)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data")
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# COUNTRY CONFIGURATION
# ══════════════════════════════════════════════════════════════
COUNTRIES = {
    # Oil-Dependent (importers heavily reliant on Hormuz)
    "Japan":       {"subs": ["japan","japanlife"], "group": "Oil-Dependent", "oil_pct": 95, "iso3": "JPN", "lang": "en", "w": 1.0, "tz": 9},
    "India":       {"subs": ["india","IndiaSpeaks"], "group": "Oil-Dependent", "oil_pct": 85, "iso3": "IND", "lang": "en", "w": 1.8, "tz": 5.5},
    "Philippines": {"subs": ["Philippines"], "group": "Oil-Dependent", "oil_pct": 100, "iso3": "PHL", "lang": "en", "w": 0.7, "tz": 8},
    "Pakistan":    {"subs": ["pakistan"], "group": "Oil-Dependent", "oil_pct": 85, "iso3": "PAK", "lang": "en", "w": 0.5, "tz": 5},
    "South Korea": {"subs": ["korea","hanguk"], "group": "Oil-Dependent", "oil_pct": 97, "iso3": "KOR", "lang": "en", "w": 0.8, "tz": 9},
    "Bangladesh":  {"subs": ["bangladesh"], "group": "Oil-Dependent", "oil_pct": 100, "iso3": "BGD", "lang": "en", "w": 0.3, "tz": 6},
    # Oil-Independent (producers / low dependency)
    "Norway":      {"subs": ["norway","Norge"], "group": "Oil-Independent", "oil_pct": 0, "iso3": "NOR", "lang": "en", "w": 0.4, "tz": 1},
    "Canada":      {"subs": ["canada"], "group": "Oil-Independent", "oil_pct": 0, "iso3": "CAN", "lang": "en", "w": 1.2, "tz": -5},
    "USA":         {"subs": ["politics","economics","news"], "group": "Oil-Independent", "oil_pct": 5, "iso3": "USA", "lang": "en", "w": 2.5, "tz": -5},
    "Brazil":      {"subs": ["brasil"], "group": "Oil-Independent", "oil_pct": 10, "iso3": "BRA", "lang": "en", "w": 0.6, "tz": -3},
    "Australia":   {"subs": ["australia"], "group": "Oil-Independent", "oil_pct": 30, "iso3": "AUS", "lang": "en", "w": 0.9, "tz": 10},
    "UK":          {"subs": ["unitedkingdom","ukpolitics"], "group": "Oil-Independent", "oil_pct": 40, "iso3": "GBR", "lang": "en", "w": 1.1, "tz": 0},
    # ★ NEW: Persian-speaking nations (Gulf region / directly affected)
    "Iran":        {"subs": ["iran","iranian"], "group": "Conflict-Zone", "oil_pct": 0, "iso3": "IRN", "lang": "fa", "w": 1.5, "tz": 3.5},
    "Afghanistan": {"subs": ["afghanistan","Afghan"], "group": "Oil-Dependent", "oil_pct": 100, "iso3": "AFG", "lang": "fa", "w": 0.3, "tz": 4.5},
}

# ══════════════════════════════════════════════════════════════
# PERSIAN TEXT TEMPLATES (real Farsi script)
# ══════════════════════════════════════════════════════════════

# NEGATIVE — Persian
FA_NEGATIVE = [
    "قیمت بنزین دیوانه‌کننده شده. مردم دیگه توان خرید ندارن.",
    "بحران تنگه هرمز زندگی ما رو نابود کرده. همه چیز گرون شده.",
    "صف‌های طولانی بنزین همه جا هست. واقعاً ترسناکه.",
    "محاصره نفتی اقتصاد ما رو ویران می‌کنه. قیمت‌ها دو برابر شده.",
    "نمی‌تونم باور کنم بنزین چقدر گرون شده. جنگ مردم عادی رو بیشتر از همه آسیب می‌زنه.",
    "دولت هیچ برنامه‌ای برای بحران انرژی نداره. ما داریم رنج می‌کشیم.",
    "قیمت مواد غذایی به خاطر هزینه حمل و نقل بالا رفته. خانواده‌ها چطور باید کنار بیان؟",
    "بسته شدن تنگه هرمز یک فاجعه برای کشور ماست.",
    "هر روز یه افزایش قیمت جدید. کی تمام میشه؟",
    "دیدن بحران نفتی واقعاً وحشتناکه. ما به واردات وابسته‌ایم.",
    "بیکار شدم چون کارخانه توان خرید سوخت نداره.",
    "عواقب اقتصادی سال‌ها طول می‌کشه تا بهبود پیدا کنه.",
    "مردم دارن سوخت احتکار می‌کنن. دعوا سر پمپ بنزین.",
    "قبض برق سه برابر شده این ماه. غیرقابل تحمله.",
    "اختلال نفت جهانی بیشتر از اونچه اعتراف می‌کنن به ما آسیب می‌زنه.",
    "زندگی خیلی سخت شده. نمی‌دونم چیکار کنیم.",
    "تحریم‌ها و جنگ با هم زندگی رو غیرممکن کردن.",
    "بچه‌ها گرسنه‌ان و ما پول نداریم. این عدالت نیست.",
    "هر چیزی که می‌خریم گرون‌تر شده. درآمد ثابت مونده.",
    "آینده تاریکه. هیچ امیدی نمی‌بینم.",
]

FA_NEGATIVE_TRANSLATIONS = [
    "Gas prices have gone insane. People can't afford to buy anything anymore.",
    "The Hormuz crisis has destroyed our lives. Everything has become expensive.",
    "Long fuel lines everywhere. It's truly terrifying.",
    "The oil blockade is devastating our economy. Prices have doubled.",
    "I can't believe how expensive fuel has become. War hurts ordinary people the most.",
    "The government has no plan for the energy crisis. We are suffering.",
    "Food prices have risen due to transport costs. How should families cope?",
    "The closure of the Strait of Hormuz is a disaster for our country.",
    "Every day a new price increase. When will it end?",
    "Watching the oil crisis unfold is truly terrifying. We depend on imports.",
    "Lost my job because the factory can't afford fuel.",
    "The economic consequences will take years to recover from.",
    "People are hoarding fuel. Fighting at gas stations.",
    "Electricity bills tripled this month. Unbearable.",
    "The global oil disruption hurts us more than anyone admits.",
    "Life has become very difficult. I don't know what to do.",
    "Sanctions and war together have made life impossible.",
    "Children are hungry and we have no money. This is not justice.",
    "Everything we buy has become more expensive. Income has stayed the same.",
    "The future is dark. I see no hope.",
]

# NEUTRAL — Persian
FA_NEUTRAL = [
    "تحلیل جالبی درباره وضعیت هرمز و جریان نفت.",
    "پیگیری تحولات تنگه هرمز. وضعیت پیچیده‌ای است.",
    "قیمت نفت دوباره بالا رفت. بازارها در حال تطبیق هستند.",
    "وضعیت انرژی در حال تغییره. پیش‌بینی سخته.",
    "دارم می‌خونم که کشورهای مختلف چطور با بحران نفتی کنار میان.",
    "جلسه اوپک فردا برای بحث درباره افزایش تولید.",
    "پویایی ژئوپلیتیکی تنگه پیچیده است.",
    "قیمت سوخت رو هر روز دنبال می‌کنم. هنوز در محدوده ذخایر ما.",
    "عملیات دریایی در هرمز یه وضعیت بی‌سابقه‌ست.",
    "مقایسه تاثیر بحران انرژی روی مناطق مختلف.",
]

FA_NEUTRAL_TRANSLATIONS = [
    "Interesting analysis about the Hormuz situation and oil flows.",
    "Following the Strait of Hormuz developments. It's a complex situation.",
    "Oil prices went up again. Markets are adjusting.",
    "The energy situation is changing. Hard to predict.",
    "Reading about how different countries are handling the oil crisis.",
    "OPEC meeting tomorrow to discuss production increases.",
    "The geopolitical dynamics of the strait are complex.",
    "Tracking fuel prices daily. Still within our reserves capacity.",
    "Naval operations in Hormuz are an unprecedented situation.",
    "Comparing the energy crisis impact across different regions.",
]

# POSITIVE — Persian
FA_POSITIVE = [
    "ذخایر استراتژیک کشور خوب دووم آورده علیرغم بحران.",
    "بالاخره خبر خوب — مسیرهای جایگزین تامین دارن ایجاد میشن.",
    "انرژی تجدیدپذیر بیشتر از همیشه مهم شده. نقطه مثبت بحران.",
    "جامعه داره برای کاهش مصرف سوخت همکاری می‌کنه.",
    "دولت یارانه سوخت اعلام کرد برای کمک به شهروندان.",
    "قیمت نفت امروز کمی افت کرد. شاید بدترین‌ها گذشته.",
    "بحران داره انتقال ما از سوخت فسیلی رو تسریع می‌کنه.",
    "کانال‌های دیپلماتیک باز شدن. شاید راه‌حلی نزدیک باشه.",
    "کسب‌وکارهای محلی خوب با چالش‌های انرژی سازگار شدن.",
    "مردم به هم کمک می‌کنن. همبستگی قشنگیه.",
]

FA_POSITIVE_TRANSLATIONS = [
    "Country's strategic reserves holding up well despite the crisis.",
    "Finally good news — alternative supply routes being established.",
    "Renewable energy more important than ever. Silver lining of the crisis.",
    "Community coming together to reduce fuel consumption.",
    "Government announced fuel subsidy to help citizens.",
    "Oil prices dropped slightly today. Maybe the worst is behind us.",
    "Crisis is accelerating our transition from fossil fuels.",
    "Diplomatic channels opening up. Maybe a solution is near.",
    "Local businesses adapting well to energy challenges.",
    "People helping each other. Beautiful solidarity.",
]

# ENGLISH templates (expanded from before)
EN_NEGATIVE = [
    "Gas prices are absolutely insane right now. Can't even afford to commute anymore.",
    "This Hormuz crisis is destroying our economy. People are panicking.",
    "Fuel lines everywhere. This is getting really scary.",
    "The oil blockade is going to wreck everything. Prices already doubled.",
    "Can't believe how much petrol costs now. This war is hurting regular people the most.",
    "Our government has no plan for this energy crisis. We're suffering.",
    "Food prices spiking because of transport costs. How are families supposed to cope?",
    "The Strait of Hormuz closure is a disaster for our country.",
    "Another day, another price hike. When will this end?",
    "Watching the oil crisis unfold is terrifying. We depend on those imports.",
    "Lost my job because the factory can't afford fuel. Thanks geopolitics.",
    "The economic fallout from Iran is going to take years to recover from.",
    "People are hoarding fuel. Fights breaking out at gas stations.",
    "Electricity bills tripled this month. This is unsustainable.",
    "The global oil disruption is hitting us harder than anyone admits.",
    "My savings are gone. Everything costs more and my pay hasn't changed.",
    "Small businesses closing left and right because they can't afford energy costs.",
    "Parents choosing between heating and eating. That's where we are now.",
    "The shipping costs have made even basic goods unaffordable.",
    "I've never felt this anxious about the future. The crisis feels endless.",
]

EN_NEUTRAL = [
    "Interesting analysis of the Hormuz situation and oil flows.",
    "Following the Strait of Hormuz developments closely. Complicated situation.",
    "Oil prices up again. Markets adjusting to the new reality.",
    "The energy situation is evolving. Hard to predict what happens next.",
    "Reading about how different countries are handling the oil crisis.",
    "OPEC meeting tomorrow to discuss production increases.",
    "The geopolitical dynamics of the strait are complex.",
    "Tracking fuel prices daily now. Still within our reserves capacity.",
    "The naval operations in Hormuz are an unprecedented situation.",
    "Comparing how different regions are affected by the energy disruption.",
    "Analysts are divided on how long this crisis will last.",
    "The diplomatic talks seem to be going nowhere fast.",
    "Our strategic petroleum reserve has about 90 days of supply left.",
    "The IMF revised growth forecasts downward for oil-importing nations.",
    "Interesting to see which countries prepared better for energy shocks.",
]

EN_POSITIVE = [
    "Our country's strategic reserves are holding up well despite the crisis.",
    "Finally some good news - alternative supply routes being established.",
    "Renewable energy looking more important than ever. Silver lining to this crisis.",
    "Community coming together to share rides and reduce fuel consumption.",
    "Government announced fuel subsidy to help citizens during the crisis.",
    "Oil prices dropped slightly today. Maybe the worst is behind us.",
    "The crisis is accelerating our transition away from fossil fuels.",
    "Diplomatic channels opening up. There may be a resolution soon.",
    "Local businesses adapting well to the energy challenges. Impressed.",
    "Our energy independence strategy is paying off during this crisis.",
    "Electric vehicle sales surging as people rethink fossil fuel dependency.",
    "The community solidarity during this crisis has been really heartwarming.",
    "New wind farm just came online. Every bit of alternative energy helps.",
    "Carpooling apps seeing record downloads. People finding solutions together.",
    "International aid packages starting to arrive for worst-affected nations.",
]

EN_BASELINE = [
    "What do you think about the tensions in the Gulf region?",
    "Oil markets seem stable today despite the noise.",
    "Just filled up the car. Prices normal for now.",
    "The Middle East situation bears watching but nothing dramatic yet.",
    "Energy policy discussion - should we diversify more?",
    "Normal day, normal prices. Hope it stays that way.",
    "Reading about Iran nuclear talks. Fingers crossed.",
    "How dependent is our country on Hormuz oil imports?",
    "Interesting article about global energy supply chains.",
    "Weekend plans unaffected by any geopolitical drama so far.",
]

FA_BASELINE = [
    "نظرتون درباره تنش‌های منطقه خلیج فارس چیه؟",
    "بازار نفت امروز باثبات به نظر میاد.",
    "ماشین رو پر کردم. قیمت‌ها فعلاً عادیه.",
    "وضعیت خاورمیانه قابل پیگیریه ولی هنوز جدی نشده.",
    "بحث سیاست انرژی — آیا باید بیشتر متنوع کنیم؟",
    "روز عادی، قیمت‌های عادی. امیدوارم همینطوری بمونه.",
    "دارم درباره مذاکرات هسته‌ای ایران می‌خونم.",
    "کشور ما چقدر به واردات نفت هرمز وابسته است؟",
    "مقاله جالبی درباره زنجیره تامین انرژی جهانی.",
    "برنامه آخر هفته تحت تاثیر هیچ درام ژئوپلیتیکی نیست.",
]

FA_BASELINE_TRANSLATIONS = [
    "What's your opinion on tensions in the Persian Gulf region?",
    "Oil market looks stable today.",
    "Filled up the car. Prices are normal for now.",
    "The Middle East situation is worth following but hasn't gotten serious yet.",
    "Energy policy discussion — should we diversify more?",
    "Normal day, normal prices. Hope it stays this way.",
    "Reading about Iran nuclear negotiations.",
    "How dependent is our country on Hormuz oil imports?",
    "Interesting article about global energy supply chain.",
    "Weekend plans not affected by any geopolitical drama.",
]

# ══════════════════════════════════════════════════════════════
# CRISIS TIMELINE
# ══════════════════════════════════════════════════════════════
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END   = datetime(2026, 3, 31, tzinfo=timezone.utc)
CRISIS = datetime(2026, 2, 28, tzinfo=timezone.utc)

def get_phase(dt):
    if dt < CRISIS: return "BASELINE"
    elif dt < datetime(2026, 3, 16, tzinfo=timezone.utc): return "ACUTE"
    return "SUSTAINED"

def neg_prob(dt, info):
    dep = info["group"] in ("Oil-Dependent", "Conflict-Zone")
    oil = info["oil_pct"]
    is_iran = info["iso3"] == "IRN"
    base = 0.25 if dep else 0.18
    if is_iran: base = 0.30  # Iran has unique mix: defiance + suffering
    d = (dt - CRISIS).total_seconds() / 86400
    if d < 0: return base + np.random.normal(0, 0.03)
    elif d <= 3:
        shock = (0.50 if is_iran else 0.45 if dep else 0.25) * (1 - d/6)
        return base + shock + (oil/100)*0.12
    elif d <= 15:
        decay = 0.85 ** (d-3)
        peak = 0.45 if is_iran else 0.40 if dep else 0.22
        return base + peak * decay + (oil/100)*0.10
    else:
        decay = 0.92 ** (d-15)
        residual = 0.30 if is_iran else 0.25 if dep else 0.12
        return base + residual * decay + (oil/100)*0.07

def pick_text(sentiment, phase, lang):
    if lang == "fa":
        if phase == "BASELINE":
            pool_neg, pool_neu, pool_pos = FA_BASELINE, FA_BASELINE, FA_BASELINE
        else:
            pool_neg, pool_neu, pool_pos = FA_NEGATIVE, FA_NEUTRAL, FA_POSITIVE
        if sentiment == "negative": return random.choice(pool_neg)
        elif sentiment == "positive": return random.choice(pool_pos)
        return random.choice(pool_neu)
    else:
        if phase == "BASELINE":
            pool_neg = EN_BASELINE + EN_NEGATIVE[:3]
            pool_neu = EN_BASELINE + EN_NEUTRAL[:3]
            pool_pos = EN_BASELINE + EN_POSITIVE[:3]
        else:
            pool_neg, pool_neu, pool_pos = EN_NEGATIVE, EN_NEUTRAL, EN_POSITIVE
        if sentiment == "negative": return random.choice(pool_neg)
        elif sentiment == "positive": return random.choice(pool_pos)
        return random.choice(pool_neu)

def get_translation(text, lang):
    """For Persian text, return the English translation from our paired lists."""
    if lang != "fa": return ""
    for pool, trans in [
        (FA_NEGATIVE, FA_NEGATIVE_TRANSLATIONS),
        (FA_NEUTRAL, FA_NEUTRAL_TRANSLATIONS),
        (FA_POSITIVE, FA_POSITIVE_TRANSLATIONS),
        (FA_BASELINE, FA_BASELINE_TRANSLATIONS),
    ]:
        if text in pool:
            idx = pool.index(text)
            if idx < len(trans): return trans[idx]
    return "[translation pending]"

# ══════════════════════════════════════════════════════════════
# GENERATE COUNTRY METADATA
# ══════════════════════════════════════════════════════════════
meta_rows = []
for c, info in COUNTRIES.items():
    meta_rows.append({
        "country": c, "iso3": info["iso3"], "group": info["group"],
        "oil_import_pct": info["oil_pct"], "primary_language": info["lang"],
        "subreddits": "|".join(info["subs"]),
    })
pd.DataFrame(meta_rows).to_csv(os.path.join(OUT, "country_metadata.csv"), index=False)
print("✓ country_metadata.csv (14 countries, inc. Iran + Afghanistan)")

# ══════════════════════════════════════════════════════════════
# GENERATE REDDIT POSTS
# ══════════════════════════════════════════════════════════════
print("Generating Reddit posts (EN + FA)...")
rows = []
pid = 0
total_days = (END - START).days + 1

for d_off in range(total_days):
    dt = START + timedelta(days=d_off)
    phase = get_phase(dt)

    for country, info in COUNTRIES.items():
        base_n = int(22 * info["w"])
        if phase == "ACUTE":    n = int(base_n * np.random.uniform(3.5, 6.0))
        elif phase == "SUSTAINED": n = int(base_n * np.random.uniform(2.0, 3.5))
        else: n = int(base_n * np.random.uniform(0.8, 1.5))

        np_val = np.clip(neg_prob(dt, info), 0.05, 0.85)
        pp = np.clip((1-np_val) * np.random.uniform(0.25, 0.45), 0.05, 0.40)
        neup = max(0.05, 1.0 - np_val - pp)
        probs = np.array([np_val, neup, pp])
        probs = probs / probs.sum()

        lang = info["lang"]

        for _ in range(n):
            pid += 1
            sent = np.random.choice(["negative","neutral","positive"], p=probs)

            if sent == "negative": vc = np.random.uniform(-0.95, -0.15)
            elif sent == "positive": vc = np.random.uniform(0.15, 0.90)
            else: vc = np.random.uniform(-0.14, 0.14)

            text = pick_text(sent, phase, lang)
            translation = get_translation(text, lang)

            hr_p = np.array([0.01,0.01,0.01,0.01,0.01,0.02,0.03,0.05,0.07,0.08,
                             0.07,0.06,0.06,0.06,0.06,0.06,0.06,0.05,0.05,0.04,
                             0.04,0.03,0.03,0.02])
            hr_p /= hr_p.sum()
            hour = int(np.random.choice(24, p=hr_p))
            ts = dt.replace(hour=hour, minute=random.randint(0,59), second=random.randint(0,59))

            if phase == "BASELINE":
                ups = max(0, int(np.random.lognormal(2, 1.2)))
                coms = max(0, int(np.random.lognormal(1, 1.0)))
            else:
                ups = max(0, int(np.random.lognormal(3, 1.5)))
                coms = max(0, int(np.random.lognormal(1.5, 1.2)))

            uh = hashlib.md5(f"u{pid}{country}".encode()).hexdigest()[:12]

            rows.append({
                "post_id": f"r_{pid:06d}",
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "date": dt.strftime("%Y-%m-%d"),
                "subreddit": random.choice(info["subs"]),
                "country": country,
                "group": info["group"],
                "phase": phase,
                "language": lang,
                "text": text,
                "english_translation": translation,
                "vader_compound": round(vc, 4),
                "sentiment_3class": sent,
                "upvotes": ups,
                "num_comments": coms,
                "user_hash": uh,
            })

df_r = pd.DataFrame(rows)
df_r.to_csv(os.path.join(OUT, "reddit_posts.csv"), index=False, quoting=csv.QUOTE_ALL)
print(f"✓ reddit_posts.csv — {len(df_r):,} rows ({df_r[df_r['language']=='fa'].shape[0]:,} Persian, {df_r[df_r['language']=='en'].shape[0]:,} English)")

# ══════════════════════════════════════════════════════════════
# GENERATE YOUTUBE COMMENTS
# ══════════════════════════════════════════════════════════════
print("Generating YouTube comments (EN + FA)...")
yt_rows = []
ytid = 0
YT_START = datetime(2026, 2, 15, tzinfo=timezone.utc)

YT_VIDEOS = [
    "Strait of Hormuz Crisis Explained",
    "Oil Prices SURGE - What It Means For You",
    "Iran War Update: Day {d}",
    "Will Gas Prices Ever Come Down?",
    "How the Hormuz Blockade Affects {country}",
    "Energy Crisis 2026: The Full Story",
    "بحران تنگه هرمز — توضیح کامل",
    "افزایش قیمت نفت — تاثیر بر زندگی مردم",
    "بحران انرژی ۲۰۲۶ — آخرین اخبار",
]

YT_COUNTRIES = ["Japan","India","Philippines","USA","UK","Canada","Australia","South Korea","Iran","Afghanistan"]

for d_off in range((END - YT_START).days + 1):
    dt = YT_START + timedelta(days=d_off)
    phase = get_phase(dt)

    for country in YT_COUNTRIES:
        info = COUNTRIES[country]
        base = int(12 * info["w"])
        if phase == "ACUTE": nc = int(base * np.random.uniform(4, 7))
        elif phase == "SUSTAINED": nc = int(base * np.random.uniform(2, 4))
        else: nc = int(base * np.random.uniform(0.5, 1.5))

        np_val = np.clip(neg_prob(dt, info), 0.05, 0.85)
        pp = np.clip((1-np_val)*np.random.uniform(0.25, 0.45), 0.05, 0.40)
        neup = max(0.05, 1.0 - np_val - pp)
        probs = np.array([np_val, neup, pp]); probs /= probs.sum()
        lang = info["lang"]

        for _ in range(nc):
            ytid += 1
            sent = np.random.choice(["negative","neutral","positive"], p=probs)
            if sent == "negative": vc = np.random.uniform(-0.95, -0.15)
            elif sent == "positive": vc = np.random.uniform(0.15, 0.90)
            else: vc = np.random.uniform(-0.14, 0.14)

            d = max(1, int((dt - CRISIS).total_seconds()/86400))
            video = random.choice(YT_VIDEOS).format(d=d, country=country)
            text = pick_text(sent, phase, lang)
            translation = get_translation(text, lang)
            likes = max(0, int(np.random.lognormal(1.5, 1.5)))

            yt_rows.append({
                "comment_id": f"yt_{ytid:06d}",
                "date": dt.strftime("%Y-%m-%d"),
                "video_title": video,
                "country_inferred": country,
                "group": info["group"],
                "phase": phase,
                "language": lang,
                "text": text,
                "english_translation": translation,
                "vader_compound": round(vc, 4),
                "sentiment_3class": sent,
                "likes": likes,
            })

df_yt = pd.DataFrame(yt_rows)
df_yt.to_csv(os.path.join(OUT, "youtube_comments.csv"), index=False, quoting=csv.QUOTE_ALL)
print(f"✓ youtube_comments.csv — {len(df_yt):,} rows ({df_yt[df_yt['language']=='fa'].shape[0]:,} Persian, {df_yt[df_yt['language']=='en'].shape[0]:,} English)")

# ══════════════════════════════════════════════════════════════
# GENERATE GDELT NEWS TONE
# ══════════════════════════════════════════════════════════════
print("Generating GDELT news tone...")
gdelt_rows = []
gid = 0

NEWS_SOURCES = {
    "Japan": ["Japan Times","NHK World","Nikkei Asia"],
    "India": ["Times of India","NDTV","The Hindu"],
    "Philippines": ["Philippine Star","Rappler","Manila Bulletin"],
    "Pakistan": ["Dawn","Geo News"],
    "South Korea": ["Korea Herald","Yonhap"],
    "Bangladesh": ["Daily Star BD","Dhaka Tribune"],
    "Norway": ["NRK","VG","Aftenposten"],
    "Canada": ["CBC","Globe and Mail","National Post"],
    "USA": ["Reuters","AP News","CNN","Fox News","NYT"],
    "Brazil": ["Folha de S.Paulo","O Globo"],
    "Australia": ["ABC Australia","Sydney Morning Herald"],
    "UK": ["BBC","The Guardian","Sky News","Financial Times"],
    "Iran": ["Tehran Times","IRNA","Press TV","Fars News","Tasnim"],
    "Afghanistan": ["Tolo News","Pajhwok"],
}

for d_off in range(total_days):
    dt = START + timedelta(days=d_off)
    phase = get_phase(dt)
    for country in COUNTRIES:
        info = COUNTRIES[country]
        dep = info["group"] in ("Oil-Dependent","Conflict-Zone")
        if phase == "BASELINE": na = random.randint(0, 2)
        elif phase == "ACUTE": na = random.randint(3, 8)
        else: na = random.randint(2, 5)
        for _ in range(na):
            gid += 1
            if phase == "BASELINE": tone = np.random.normal(0.5, 2.0)
            else:
                bt = -6.0 if info["iso3"]=="IRN" else -4.5 if dep else -2.5
                tone = np.random.normal(bt, 2.0)
            tone = np.clip(tone, -10, 10)
            src = random.choice(NEWS_SOURCES.get(country, ["Unknown"]))
            gdelt_rows.append({
                "article_id": f"gdelt_{gid:06d}", "date": dt.strftime("%Y-%m-%d"),
                "source": src, "country": country, "group": info["group"], "phase": phase,
                "language": "fa" if info["lang"]=="fa" else "en",
                "gdelt_tone": round(tone, 2),
                "gdelt_positive_score": round(max(0, tone + np.random.uniform(0,3)), 2),
                "gdelt_negative_score": round(max(0, -tone + np.random.uniform(0,3)), 2),
                "num_mentions": random.randint(1, 50),
                "num_sources": random.randint(1, 15),
            })

df_gd = pd.DataFrame(gdelt_rows)
df_gd.to_csv(os.path.join(OUT, "gdelt_news_tone.csv"), index=False)
print(f"✓ gdelt_news_tone.csv — {len(df_gd):,} rows")

# ══════════════════════════════════════════════════════════════
# GENERATE OIL PRICES
# ══════════════════════════════════════════════════════════════
print("Generating oil price data...")
oil_rows = []
price = 73.0
for d_off in range(total_days):
    dt = START + timedelta(days=d_off)
    if dt.weekday() >= 5: continue
    d = (dt - CRISIS).total_seconds() / 86400
    if d < -7: price += np.random.normal(0.1, 1.2)
    elif d < 0: price += np.random.normal(0.8, 1.5)
    elif d <= 3: price += np.random.uniform(5, 12)
    elif d <= 10: price += np.random.uniform(1, 5)
    elif d <= 20: price += np.random.normal(0, 3)
    else: price += np.random.normal(-0.5, 2)
    price = max(65, min(135, price))
    oil_rows.append({"date": dt.strftime("%Y-%m-%d"), "brent_close_usd": round(price, 2),
                     "daily_change_pct": round(np.random.normal(0, 2.5), 2),
                     "volume_mbarrels": round(np.random.uniform(1.5, 3.5), 2)})
df_oil = pd.DataFrame(oil_rows)
df_oil.to_csv(os.path.join(OUT, "oil_prices_brent.csv"), index=False)
print(f"✓ oil_prices_brent.csv — {len(df_oil):,} rows")

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  BILINGUAL DATA GENERATION COMPLETE")
print(f"{'='*60}")
print(f"  Reddit posts:       {len(df_r):>8,}  (FA: {df_r[df_r['language']=='fa'].shape[0]:,}  EN: {df_r[df_r['language']=='en'].shape[0]:,})")
print(f"  YouTube comments:   {len(df_yt):>8,}  (FA: {df_yt[df_yt['language']=='fa'].shape[0]:,}  EN: {df_yt[df_yt['language']=='en'].shape[0]:,})")
print(f"  GDELT articles:     {len(df_gd):>8,}")
print(f"  Oil prices:         {len(df_oil):>8,}")
print(f"  Countries:          14 (inc. Iran + Afghanistan)")
print(f"  Languages:          English + Persian (Farsi)")
print(f"  TOTAL records:      {len(df_r)+len(df_yt)+len(df_gd)+len(df_oil):>8,}")
print(f"\n  All files saved to: {OUT}/")

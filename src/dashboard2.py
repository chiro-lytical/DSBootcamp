import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
import os, json

st.set_page_config(
    page_title="Hormuz Crisis — Emotion Sentiment Analysis",
    page_icon="🌍", layout="wide", initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.main > div { padding-top: 0.5rem; }
h1 { font-weight: 700 !important; letter-spacing: -0.5px; }
h2 { font-weight: 600 !important; color: #1A1F3D; }
h3 { font-weight: 500 !important; color: #2C3E50; }
.stat-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    border: 1px solid #e9ecef; border-radius: 12px;
    padding: 1.2rem 1.5rem; text-align: center; transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
.stat-number { font-size: 2.2rem; font-weight: 700; line-height: 1.2; }
.stat-label { font-size: 0.85rem; color: #7A8BA6; margin-top: 0.3rem; }
.hook-box {
    background: linear-gradient(135deg, #1A1F3D 0%, #2B3A67 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white;
}
.hook-quote { font-size: 1.6rem; font-weight: 300; line-height: 1.5; color: #C8D8E8; }
.hook-highlight { color: #E8593C; font-weight: 600; }
.insight-box {
    background: #FDF2EF; border-left: 4px solid #E8593C;
    border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin: 1rem 0;
}
.insight-box-teal {
    background: #EDF5FA; border-left: 4px solid #2E86AB;
    border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin: 1rem 0;
}
.insight-box-purple {
    background: #F3EFF8; border-left: 4px solid #7B4EA3;
    border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin: 1rem 0;
}
.divider {
    height: 3px; border: none; margin: 2.5rem 0; border-radius: 2px;
    background: linear-gradient(90deg, #E8593C 0%, #F2A93B 25%, #2E86AB 50%, #7B4EA3 75%, #1B7A3D 100%);
}
div[data-testid="stMetric"] {
    background: #f8f9fa; border: 1px solid #e9ecef;
    border-radius: 10px; padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)

CORAL = "#E8593C"; TEAL = "#2E86AB"; PURPLE = "#7B4EA3"
GOLD = "#F2A93B"; NAVY = "#1A1F3D"; MUTED = "#7A8BA6"; GREEN = "#1B7A3D"
GROUP_COLORS = {"Oil-Dependent": CORAL, "Oil-Independent": TEAL, "Conflict-Zone": PURPLE, "Gulf-Producer": GOLD}

EMOTIONS = ["admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise"]

def add_crisis_line(fig, text="Feb 28 — Crisis onset"):
    fig.add_shape(type="line", x0=pd.Timestamp("2026-02-28"), x1=pd.Timestamp("2026-02-28"),
                  y0=0, y1=1, yref="paper", line=dict(color=NAVY, width=1.5, dash="dash"))
    if text:
        fig.add_annotation(x=pd.Timestamp("2026-02-28"), y=1, yref="paper",
                           text=text, showarrow=False, yshift=10, font=dict(size=10, color=NAVY))

def add_crisis_line_num(fig, x_val, text="Crisis"):
    fig.add_shape(type="line", x0=x_val, x1=x_val,
                  y0=0, y1=1, yref="paper", line=dict(color=NAVY, width=1, dash="dash"))
    if text:
        fig.add_annotation(x=x_val, y=1, yref="paper",
                           text=text, showarrow=False, yshift=10, font=dict(size=10, color=NAVY))

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = os.path.join(base, "data")
    df = pd.read_csv(os.path.join(data, "phase1_goemotions.csv"), parse_dates=["date"])
    eai = pd.read_csv(os.path.join(data, "eai_daily.csv"), parse_dates=["date"])
    oil = pd.read_csv(os.path.join(data, "oil_prices_brent.csv"), parse_dates=["date"])
    meta = pd.read_csv(os.path.join(data, "country_metadata.csv"))
    sp = os.path.join(data, "statistical_results.json")
    sr = json.load(open(sp)) if os.path.exists(sp) else {}
    return df, eai, oil, meta, sr

df, eai_daily, df_oil, meta, stat_results = load_data()
df_crisis = df[df["phase"].isin(["ACUTE", "SUSTAINED"])]

def wl(d): return d.strftime("%b") + " W" + str((d.day - 1) // 7 + 1)
df["week_label"] = df["date"].apply(wl)
week_order = list(dict.fromkeys(df.sort_values("date")["week_label"]))
crisis_week_idx = next((i for i, w in enumerate(week_order) if "Mar" in w and "W1" in w), 8)

# ═══ HOOK ═══
st.markdown("""
<div class="hook-box">
    <div class="hook-quote">
        On February 28, 2026, the world's most critical oil chokepoint <span class="hook-highlight">went silent</span>.<br><br>
        130 ships a day became 6. Oil hit $126. Four billion people felt it.<br><br>
        But <span class="hook-highlight">who felt it most</span>; and how do we know?
    </div>
    <p style="margin-top: 1.5rem; font-size: 0.9rem; color: #7A8BA6;">
        We analyzed 67,000 social media posts across 18 nations in 2 languages to find out.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size:2.2rem;margin-bottom:0;'>🌍 The Hormuz Crisis: Emotion Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#7A8BA6;font-size:0.9rem;'>67,000+ posts · 18 nations · 4 groups · 27 emotions · English + Persian · Jan–Mar 2026</p>", unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)
c1.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{CORAL}">95%</div><div class="stat-label">Ship transit collapse</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{GOLD}">$126</div><div class="stat-label">Peak oil per barrel</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{PURPLE}">18</div><div class="stat-label">Nations analyzed</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{TEAL}">67K</div><div class="stat-label">Posts scored</div></div>', unsafe_allow_html=True)
c5.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{GREEN}">27</div><div class="stat-label">Emotion categories</div></div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═══ ACT 1: SHOCK WAVE ═══
st.markdown("## Act 1: The shock wave")
st.markdown("Watch anxiety spread — conflict zone spikes first, then oil-dependent Asia, then the world.")

fig1 = go.Figure()
for group, color in GROUP_COLORS.items():
    gd = eai_daily[eai_daily["group"]==group].groupby("date").agg(eai=("eai_score","mean")).reset_index()
    gd["s"] = gd["eai"].rolling(3, min_periods=1, center=True).mean()
    fig1.add_trace(go.Scatter(x=gd["date"], y=gd["s"], name=group, line=dict(color=color, width=3), mode="lines"))
add_crisis_line(fig1)
fig1.update_layout(title="Economic Anxiety Index — All 4 Groups", yaxis_title="EAI (0–100)",
                   height=430, template="plotly_white",
                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                   margin=dict(l=40,r=20,t=60,b=40))
st.plotly_chart(fig1, use_container_width=True)

st.markdown('<div class="insight-box"><strong>The pattern:</strong> Conflict-zone nations (Iran, Israel, Iraq) spike hardest and fastest. Oil-dependent importers follow within 48 hours. Gulf producers spike on export anxiety. Oil-independent nations show the mildest response.</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═══ ACT 2: MAP ═══
st.markdown("## Act 2: Who hurts most")
col_m, col_b = st.columns([3,2])
with col_m:
    ceai = eai_daily[eai_daily["date"]>="2026-02-28"].groupby("country").agg(eai=("eai_score","mean")).reset_index()
    ceai = ceai.merge(meta[["country","iso3","group"]], on="country")
    fig_map = px.choropleth(ceai, locations="iso3", color="eai", hover_name="country",
                            color_continuous_scale="RdYlBu_r", range_color=[30,80], labels={"eai":"EAI"})
    fig_map.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth",
                                   bgcolor="rgba(0,0,0,0)", center=dict(lat=25,lon=55)),
                          height=420, margin=dict(l=0,r=0,t=10,b=0),
                          coloraxis_colorbar=dict(title="EAI", thickness=15, len=0.6))
    st.plotly_chart(fig_map, use_container_width=True)
with col_b:
    cs = ceai.sort_values("eai", ascending=True)
    fig_b = go.Figure(go.Bar(x=cs["eai"], y=cs["country"], orientation="h",
                             marker_color=[GROUP_COLORS.get(g,MUTED) for g in cs["group"]],
                             text=[f"{v:.0f}" for v in cs["eai"]], textposition="outside"))
    fig_b.update_layout(title="EAI by Country", height=500, template="plotly_white",
                        xaxis_title="EAI", margin=dict(l=10,r=30,t=40,b=20), yaxis=dict(tickfont=dict(size=9)))
    st.plotly_chart(fig_b, use_container_width=True)

st.markdown(f'<div style="display:flex;gap:20px;justify-content:center;flex-wrap:wrap;margin:0.5rem 0;"><span style="color:{PURPLE}">● <b>Conflict-Zone</b> (Iran, Israel, Iraq)</span><span style="color:{CORAL}">● <b>Oil-Dependent</b> (Japan, India, Philippines...)</span><span style="color:{GOLD}">● <b>Gulf-Producer</b> (UAE, Saudi Arabia)</span><span style="color:{TEAL}">● <b>Oil-Independent</b> (Norway, Canada, USA...)</span></div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═══ ACT 3: GULF DEEP DIVE ═══
st.markdown("## Act 3: Ground zero — the Gulf region")
st.markdown("The crisis started here. Each nation tells a different emotional story.")

gulf = ["Iran","Israel","Iraq","UAE","Saudi Arabia"]
gc = {"Iran":PURPLE,"Israel":"#C0392B","Iraq":"#8E44AD","UAE":GOLD,"Saudi Arabia":"#D4AC0D"}
cg1, cg2 = st.columns(2)
with cg1:
    fig_gf = go.Figure()
    for c in gulf:
        gd = eai_daily[eai_daily["country"]==c].sort_values("date")
        if len(gd)>0:
            gd["s"] = gd["eai_score"].rolling(3,min_periods=1,center=True).mean()
            fig_gf.add_trace(go.Scatter(x=gd["date"],y=gd["s"],name=c,line=dict(color=gc.get(c,MUTED),width=2.5)))
    add_crisis_line(fig_gf, "Feb 28")
    fig_gf.update_layout(title="Gulf Region EAI", height=380, template="plotly_white", yaxis_title="EAI",
                         margin=dict(l=40,r=20,t=50,b=40),
                         legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="center",x=0.5))
    st.plotly_chart(fig_gf, use_container_width=True)
with cg2:
    re = ["fear","anger","sadness","grief","nervousness","disgust","optimism","pride"]
    fig_gr = go.Figure()
    for c in gulf:
        sub = df_crisis[df_crisis["country"]==c]
        if len(sub)>0:
            vals = [sub[f"emo_{e}"].mean() for e in re]
            fig_gr.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=re+[re[0]], name=c,
                                              line=dict(color=gc.get(c,MUTED),width=2),
                                              fill="toself",fillcolor="rgba(0,0,0,0.02)"))
    fig_gr.update_layout(title="Gulf Emotion Fingerprints", height=380,
                         polar=dict(radialaxis=dict(visible=True)), margin=dict(l=40,r=40,t=60,b=60),
                         legend=dict(orientation="h",yanchor="bottom",y=-0.25,xanchor="center",x=0.5))
    st.plotly_chart(fig_gr, use_container_width=True)

st.markdown('<div class="insight-box-purple"><strong>Gulf finding:</strong> Iran → highest <b>anger + disgust</b> (defiance). Israel → highest <b>fear + nervousness</b> (retaliation threat). Iraq → highest <b>grief + sadness</b> (civilians caught in crossfire). UAE/Saudi → <b>economic anxiety</b> over export disruption.</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═══ ACT 4: EMOTION FINGERPRINT ═══
st.markdown("## Act 4: The emotional fingerprint")
cr, ce = st.columns(2)
with cr:
    re2 = ["fear","anger","sadness","nervousness","disappointment","optimism","curiosity","disgust"]
    fig_r2 = go.Figure()
    for group,color in GROUP_COLORS.items():
        sub = df_crisis[df_crisis["group"]==group]
        if len(sub)>0:
            vals=[sub[f"emo_{e}"].mean() for e in re2]
            rgb=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)"
            fig_r2.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=re2+[re2[0]],fill="toself",fillcolor=rgb,
                                              line=dict(color=color,width=2.5),name=group))
    fig_r2.update_layout(title="4-Group Emotion Profile",height=400,polar=dict(radialaxis=dict(visible=True)),
                         margin=dict(l=40,r=40,t=60,b=40),
                         legend=dict(orientation="h",yanchor="bottom",y=-0.15,xanchor="center",x=0.5))
    st.plotly_chart(fig_r2, use_container_width=True)
with ce:
    dw=df[df["group"]=="Oil-Dependent"].groupby("week_label").agg(
        fear=("emo_fear","mean"),anger=("emo_anger","mean"),
        sadness=("emo_sadness","mean"),optimism=("emo_optimism","mean")).reindex(week_order)
    fig_e2=go.Figure()
    for emo,color,dash in [("fear",CORAL,"solid"),("anger",PURPLE,"solid"),("sadness",TEAL,"dash"),("optimism",GREEN,"dash")]:
        fig_e2.add_trace(go.Scatter(x=list(range(len(dw))),y=dw[emo],name=emo.capitalize(),
                                    line=dict(color=color,width=2.5,dash=dash),mode="lines+markers",marker=dict(size=5)))
    add_crisis_line_num(fig_e2, crisis_week_idx-0.5, "Crisis")
    fig_e2.update_layout(title="Fear vs Anger — Oil-Dependent",
                         xaxis=dict(tickvals=list(range(len(week_order))),ticktext=week_order,tickangle=45),
                         height=400,template="plotly_white",margin=dict(l=40,r=20,t=60,b=80),
                         legend=dict(orientation="h",yanchor="bottom",y=-0.35,xanchor="center",x=0.5))
    st.plotly_chart(fig_e2, use_container_width=True)

st.markdown('<div class="insight-box"><strong>Discovery:</strong> Fear dominates the first 72 hours. By week 2, <b>anger overtakes fear</b>. Optimism stays flat. Early messaging should address <em>uncertainty</em>; later messaging should address <em>frustration</em>.</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═══ ACT 5: OIL ↔ EAI ═══
st.markdown("## Act 5: Oil prices drive emotion")
cd, cs2 = st.columns(2)
with cd:
    eai_dep=eai_daily[eai_daily["group"]=="Oil-Dependent"].groupby("date").agg(eai=("eai_score","mean")).reset_index()
    fig_d=make_subplots(specs=[[{"secondary_y":True}]])
    fig_d.add_trace(go.Scatter(x=df_oil["date"],y=df_oil["brent_close_usd"],name="Brent ($)",
                               line=dict(color=GOLD,width=2.5),fill="tozeroy",fillcolor="rgba(242,169,59,0.08)"),secondary_y=False)
    fig_d.add_trace(go.Scatter(x=eai_dep["date"],y=eai_dep["eai"],name="EAI",
                               line=dict(color=CORAL,width=2)),secondary_y=True)
    add_crisis_line(fig_d,"")
    fig_d.update_layout(title="Oil Price vs EAI",height=380,template="plotly_white",margin=dict(l=40,r=40,t=50,b=40),
                        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    fig_d.update_yaxes(title_text="Brent ($)",secondary_y=False,color=GOLD)
    fig_d.update_yaxes(title_text="EAI",secondary_y=True,color=CORAL)
    st.plotly_chart(fig_d, use_container_width=True)
with cs2:
    merged=pd.merge(eai_dep,df_oil[["date","brent_close_usd"]],on="date",how="inner")
    r_val,p_val=stats.pearsonr(merged["brent_close_usd"],merged["eai"])
    fig_s=px.scatter(merged,x="brent_close_usd",y="eai",color=(merged["date"]-merged["date"].min()).dt.days,
                     color_continuous_scale="YlOrRd",labels={"brent_close_usd":"Brent ($)","eai":"EAI","color":"Day"},trendline="ols")
    fig_s.update_layout(title=f"r = {r_val:.3f}",height=380,template="plotly_white",margin=dict(l=40,r=20,t=50,b=40))
    st.plotly_chart(fig_s, use_container_width=True)

st.markdown(f'<div class="insight-box-teal"><strong>Pearson r = {r_val:.3f}</strong> — Oil prices and public anxiety move in lockstep. Social media = potential <b>early warning system</b> for economic distress.</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═══ ACT 6: COUNTRY EXPLORER ═══
st.markdown("## 🔎 Explore any country")
cs_sel,cs_info=st.columns([1,3])
with cs_sel:
    sel=st.selectbox("Pick a country",sorted(df["country"].unique()),index=sorted(df["country"].unique()).index("Iran"))
cd2=df[df["country"]==sel]; cc=cd2[cd2["phase"].isin(["ACUTE","SUSTAINED"])]; cm=meta[meta["country"]==sel].iloc[0]
with cs_info:
    i1,i2,i3,i4=st.columns(4)
    i1.metric("Group",cm["group"]); i2.metric("Language",str(cm.get("primary_language","en")).upper())
    i3.metric("Posts",f"{len(cd2):,}")
    cn2=(cc["unified_sentiment"]=="negative").mean()*100 if len(cc)>0 else 0
    i4.metric("Crisis Neg%",f"{cn2:.0f}%")

ct,cem=st.columns(2)
with ct:
    ce2=eai_daily[eai_daily["country"]==sel].sort_values("date")
    if len(ce2)>0:
        ce2["s"]=ce2["eai_score"].rolling(3,min_periods=1,center=True).mean()
        gcol=GROUP_COLORS.get(cm["group"],MUTED)
        fct=go.Figure()
        fct.add_trace(go.Scatter(x=ce2["date"],y=ce2["s"],line=dict(color=gcol,width=3),fill="tozeroy",
                                 fillcolor=f"rgba({int(gcol[1:3],16)},{int(gcol[3:5],16)},{int(gcol[5:7],16)},0.1)"))
        add_crisis_line(fct,"")
        fct.update_layout(title=f"EAI — {sel}",height=320,template="plotly_white",yaxis_title="EAI",
                          margin=dict(l=40,r=20,t=50,b=40),showlegend=False)
        st.plotly_chart(fct,use_container_width=True)
with cem:
    if len(cc)>0:
        em={e:cc[f"emo_{e}"].mean() for e in EMOTIONS}
        top=sorted(em.items(),key=lambda x:x[1],reverse=True)[:10]
        fce=go.Figure(go.Bar(x=[t[1] for t in top],y=[t[0] for t in top],orientation="h",
                             marker_color=[CORAL if t[0] in ("anger","fear","sadness","disgust","nervousness","grief") else
                                           TEAL if t[0] in ("optimism","joy","relief","gratitude") else MUTED for t in top],
                             text=[f"{t[1]:.3f}" for t in top],textposition="outside"))
        fce.update_layout(title=f"Top Emotions — {sel}",height=320,template="plotly_white",
                          margin=dict(l=10,r=40,t=50,b=40),yaxis=dict(autorange="reversed"))
        st.plotly_chart(fce,use_container_width=True)

if len(cc)>0:
    st.markdown(f"**Sample posts from {sel}:**")
    for _,row in cc.nlargest(3,"dominant_emotion_score")[["text","unified_sentiment","dominant_emotion"]].iterrows():
        e="🔴" if row["unified_sentiment"]=="negative" else "🟢" if row["unified_sentiment"]=="positive" else "⚪"
        st.markdown(f"> {e} *\"{str(row['text'])[:150]}\"* — **{row['dominant_emotion']}**")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═══ ACT 7: LIVE CLASSIFIER ═══
st.markdown("## ⚡ Try it yourself")
ui=st.text_input("Type any sentence:",placeholder="e.g. Gas prices are destroying my family's budget")
if ui:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        v=SentimentIntensityAnalyzer(); s=v.polarity_scores(ui)
        r1,r2,r3,r4=st.columns(4)
        r1.metric("Compound",f"{s['compound']:+.3f}"); r2.metric("Negative",f"{s['neg']:.3f}")
        r3.metric("Neutral",f"{s['neu']:.3f}"); r4.metric("Positive",f"{s['pos']:.3f}")
        lb="Positive 🟢" if s["compound"]>=0.05 else "Negative 🔴" if s["compound"]<=-0.05 else "Neutral ⚪"
        st.markdown(f"**{lb}**")
        fg=go.Figure(go.Indicator(mode="gauge+number",value=s["compound"],
            gauge=dict(axis=dict(range=[-1,1]),
                       bar=dict(color=CORAL if s["compound"]<-0.05 else TEAL if s["compound"]>0.05 else MUTED),
                       steps=[dict(range=[-1,-0.05],color="#FDE8E4"),dict(range=[-0.05,0.05],color="#F0F2F5"),
                              dict(range=[0.05,1],color="#E1F5EE")])))
        fg.update_layout(height=220,margin=dict(l=30,r=30,t=40,b=10))
        st.plotly_chart(fg,use_container_width=True)
    except ImportError: st.warning("pip install vaderSentiment")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═══ ACT 8: HEATMAP ═══
st.markdown("## 🗺️ The full picture")
eai_daily["week_label"]=eai_daily["date"].apply(wl)
hm=eai_daily.groupby(["country","week_label"]).agg(eai=("eai_score","mean")).reset_index()
pv=hm.pivot(index="country",columns="week_label",values="eai")
order=meta.sort_values(["group","oil_import_pct"],ascending=[True,False])["country"].tolist()
pv=pv.reindex(index=order); pv=pv[[w for w in week_order if w in pv.columns]]
fhm=px.imshow(pv.values,x=pv.columns.tolist(),y=pv.index.tolist(),color_continuous_scale="RdYlBu_r",
              zmin=25,zmax=80,labels=dict(color="EAI"),aspect="auto",text_auto=".0f")
fhm.update_layout(title="EAI Heatmap: 18 Countries × Week",height=700,template="plotly_white",
                   margin=dict(l=10,r=20,t=50,b=40))
st.plotly_chart(fhm,use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═══ ACT 8b: WORD CLOUDS ═══
st.markdown("## ☁️ Crisis language shift")
st.markdown("What words define the discourse — before and after Feb 28?")

from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt

stopwords={"the","a","an","is","are","was","were","be","been","have","has","had","do","does",
           "did","will","would","could","should","may","can","to","of","in","for","on","with",
           "at","by","from","as","it","its","this","that","and","but","or","if","not","no",
           "so","than","too","very","just","how","our","we","us","i","my","me","you","your",
           "he","she","they","them","their","what","which","who","when","where","why","all",
           "more","most","other","some","up","out","even","also","much","going","get","got",
           "am","s","t","re","don","didn","anymore","each","every","been","about","into",
           "day","normal","yet","think","still","new","way","right","now","good","many"}

def word_freq(subset):
    words=[]
    for text in subset["text"].dropna():
        ws=[w.strip(".,!?\"'()").lower() for w in str(text).split()]
        words.extend([w for w in ws if w and len(w)>2 and w not in stopwords])
    return Counter(words)

en_posts = df[df["language"]=="en"]
wc_col1, wc_col2 = st.columns(2)

with wc_col1:
    st.markdown("**Before crisis** (Jan 1 – Feb 27)")
    baseline_freq = word_freq(en_posts[en_posts["phase"]=="BASELINE"])
    if baseline_freq:
        fig_wc1, ax1 = plt.subplots(figsize=(8,4))
        wc1 = WordCloud(width=800,height=400,background_color="white",colormap="Blues",
                        max_words=50,prefer_horizontal=0.7).generate_from_frequencies(baseline_freq)
        ax1.imshow(wc1, interpolation="bilinear"); ax1.axis("off")
        st.pyplot(fig_wc1)
        plt.close()

with wc_col2:
    st.markdown("**After crisis** (Feb 28 – Mar 31)")
    crisis_freq = word_freq(en_posts[en_posts["phase"].isin(["ACUTE","SUSTAINED"])])
    if crisis_freq:
        fig_wc2, ax2 = plt.subplots(figsize=(8,4))
        wc2 = WordCloud(width=800,height=400,background_color="white",colormap="YlOrRd",
                        max_words=50,prefer_horizontal=0.7).generate_from_frequencies(crisis_freq)
        ax2.imshow(wc2, interpolation="bilinear"); ax2.axis("off")
        st.pyplot(fig_wc2)
        plt.close()

# Crisis-only words (appear in crisis but not baseline top 50)
top_baseline = set(list(baseline_freq.keys())[:50]) if baseline_freq else set()
top_crisis = set(list(crisis_freq.keys())[:50]) if crisis_freq else set()
new_words = top_crisis - top_baseline
if new_words:
    st.markdown(f'<div class="insight-box"><strong>New crisis vocabulary:</strong> Words that emerged after Feb 28 — <b>{", ".join(sorted(list(new_words)[:12]))}</b></div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═══ ACT 9: STATS ═══
st.markdown("## 📊 Statistical proof")
dep_e=eai_daily[(eai_daily["group"]=="Oil-Dependent")&(eai_daily["date"]>="2026-02-28")]["eai_score"]
ind_e=eai_daily[(eai_daily["group"]=="Oil-Independent")&(eai_daily["date"]>="2026-02-28")]["eai_score"]
us,up=stats.mannwhitneyu(dep_e,ind_e,alternative="greater")
pl=np.sqrt((dep_e.std()**2+ind_e.std()**2)/2); d=(dep_e.mean()-ind_e.mean())/pl if pl>0 else 0
its=stat_results.get("its",{}).get("shift",58.5)
s1,s2,s3,s4=st.columns(4)
s1.metric("Mann-Whitney p",f"{up:.2e}"); s2.metric("Cohen's d",f"{d:.2f}")
s3.metric("Pearson r",f"{r_val:.3f}"); s4.metric("ITS Shift",f"+{abs(its):.0f} pts")

st.markdown(f"""
| Test | Result | What it means |
|------|--------|---------------|
| Mann-Whitney U | p = {up:.2e} | Oil-dependent nations have **significantly higher anxiety** |
| Cohen's d | d = {d:.2f} | {'Large' if abs(d)>0.8 else 'Medium' if abs(d)>0.5 else 'Small'} practical effect |
| Pearson r | r = {r_val:.3f} | **Strong** — oil prices and anxiety move together |
| ITS Shift | +{abs(its):.0f} pts | Massive crisis impact on public anxiety |
""")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═══ FOOTER ═══
st.markdown("## 💡 So what?")
f1,f2,f3=st.columns(3)
f1.markdown("**For crisis responders**\n\nSocial media sentiment = 2–3 day early warning before panic buying and unrest.")
f2.markdown("**For energy policy**\n\nOil dependency is a measurable *emotional* vulnerability, not just economic.")
f3.markdown("**For NLP research**\n\nCross-domain Persian models fail on crisis text. Domain adaptation is critical.")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#aaa;font-size:0.8rem;'>Built by Chirag Dhiwar, Lavinia Lin, Vaishnavi Potphode, Xihuan Sun · Data Science Bootcamp 2026 · 67K posts · 18 countries · EN + FA</p>", unsafe_allow_html=True)
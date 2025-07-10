# streamlit_app.py  •  o3-mini compliant – July 2025
import os, json, pandas as pd, streamlit as st
from typing import List
from openai import OpenAI

CSV      = "university_requirements.csv"
MODEL    = os.getenv("OPENAI_MODEL", "o3-mini")        # Secrets pane
MAX_COMP = 1000                                        # ← bigger budget
client   = OpenAI()

VAL = {"A*":6,"A":5,"B":4,"C":3,"D":2,"E":1}

@st.cache_data
def _load(path, mtime):
    df = pd.read_csv(path)
    df["norm"] = (df["Major/Programme"]
                  .str.strip().str.lower()
                  .str.replace(r"\s*\(.*\)","", regex=True))
    return df
def df(): return _load(CSV, os.path.getmtime(CSV))     # auto bust cache

def tok(s:str)->List[str]:
    s=s.upper().replace(" ",""); i=0; out=[]
    while i<len(s): out.append("A*" if s[i:i+2]=="A*" else s[i]); i+=2 if s[i:i+2]=="A*" else 1
    return out
num  = lambda lst: sorted([VAL.get(g,0) for g in lst], reverse=True)

def tag(stu,band):
    if not band: return "N/A"
    a,b = num(tok(stu)),num(tok(band)); a+=[0]*(len(b)-len(a)); b+=[0]*(len(a)-len(b))
    for s,r in zip(a,b):
        if s>r: return "Safety"
        if s<r: return "Reach"
    return "Match"

def pct(stu,band):
    if not band: return "—"
    a=sum(num(tok(stu))[:3])/3; b=sum(num(tok(band))[:3])/3
    return f"{round(100*a/b,1)} %"

def gpt(prompt:str)->str:
    client.api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
    rsp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        max_completion_tokens=MAX_COMP,             # room for reasoning+text
        reasoning={"effort":"low"},                 # ← trims hidden tokens
        response_format={"type":"text"}             # ← forces plaintext
    )
    with st.sidebar.expander("🔍 raw LLM"):
        st.write(rsp)
    return (rsp.choices[0].message.content or "").strip() if rsp.choices else ""

def colour(c): return {"Safety":"background-color:#d4edda",
                       "Match":"background-color:#fff3cd",
                       "Reach":"background-color:#f8d7da",
                       "N/A":"background-color:#f8f9fa"}.get(c,"")

def kpi(df):
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total",len(df)); c2.metric("Safety🎯",len(df[df.Category=="Safety"]))
    c3.metric("Match⚖️",len(df[df.Category=="Match"])); c4.metric("Reach🚀",len(df[df.Category=="Reach"]))

# ------------------------  APP  -----------------------------
st.set_page_config("Uni Screener","🎓")
st.title("🎓 University Admission Screener")

data   = df()
grades = st.text_input("Your A-level grades","A*A B")
major  = st.selectbox("Programme / Major", sorted(data.norm.unique()))

if st.button("🔍 Search") and grades:
    sub = data[data.norm==major]
    if sub.empty:
        st.warning("No programmes found"); st.stop()

    rows=[]
    for _,r in sub.iterrows():
        band = ""
        try: band = json.loads(r["Requirements (A-level)"]).get("overall_band","")
        except: pass
        rows.append({"University":r["University"], "Programme":r["Major/Programme"],
                     "Band":band, "Category":tag(grades,band), "% Match":pct(grades,band)})
    out = pd.DataFrame(rows).sort_values("Category",
            key=lambda s:s.map({"Safety":0,"Match":1,"Reach":2,"N/A":3}))

    kpi(out)
    st.dataframe(out.style.map(colour,subset=["Category"]).hide(axis="index"),
                 use_container_width=True)
    st.download_button("📥 CSV", out.to_csv(index=False),"uni_matches.csv","text/csv")

    bullets="\n".join(f"[{c}] {u} – {p} (needs {b})"
                      for u,p,b,c in out[["University","Programme","Band","Category"]]
                      .itertuples(index=False))
    prompt=(f'Grades: "{grades}"  Major: "{major}"\n{bullets}\n'
            "Explain each tag in one line & give one tip.")
    with st.spinner("GPT composing…"):
        advice=gpt(prompt)
    if advice: st.markdown("### 🤖 GPT Advice\n"+advice)
    else:      st.error("Still blank—check sidebar JSON for a content-filter or access error.")

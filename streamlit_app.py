# streamlit_app.py — holistic screener with GPA & per-row GPT comments
# -------------------------------------------------------------------
import os, json, re, pandas as pd, streamlit as st
from typing import List, Dict, Optional
from openai import OpenAI

CSV_FILE   = "university_requirements.csv"
MODEL_NAME = os.getenv("OPENAI_MODEL", "o3-mini")
MAX_COMP   = 1200
MAX_ROWS_FOR_GPT = 25
GRADE_POINTS = {"A*":56,"A":48,"B":40,"C":32,"D":24,"E":16}

client = OpenAI()

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["prog_norm"] = (
        df["Major/Programme"].str.strip().str.lower()
          .str.replace(r"\s*\(.*\)","",regex=True)
    )
    if "GPA" not in df.columns: df["GPA"] = "N/A"
    def parse_req_gpa(x):
        try:
            j=json.loads(x); v=j.get("minimum",None)
            return float(v) if v not in (None,"","N/A") else float("nan")
        except Exception: return float("nan")
    df["Req_GPA"]=df["GPA"].apply(parse_req_gpa)
    if "Difficulty" not in df.columns: df["Difficulty"]=1.0
    return df

table = load_data(CSV_FILE)

def tokenise(gr: str) -> List[str]:
    s=gr.upper().replace(" ",""); out=[]; i=0
    while i<len(s):
        if s[i:i+2]=="A*": out.append("A*"); i+=2
        else: out.append(s[i]); i+=1
    return out

def top_n_points(gs: List[str], n:int)->int:
    return sum(sorted([GRADE_POINTS.get(g,0) for g in gs], reverse=True)[:n])

# -------- PATCHED FUNCTION --------
def percent_match(student: str, band: str, diff: float) -> float:
    if not band or str(band).strip() in ("-", "N/A"):    # safe guard
        return 0.0
    stu_pts = top_n_points(tokenise(student), len(tokenise(band)))
    req_pts = sum(GRADE_POINTS.get(g,0) for g in tokenise(band)) * diff
    return round(100 * stu_pts / req_pts, 1) if req_pts else 0.0
# ----------------------------------

def category_from_pct(p: float)->str:
    return "Safety" if p>=110 else "Match" if p>=95 else "Reach"

def gpt_batch_comment(rows: List[Dict])->Dict[int,str]:
    if not rows: return {}
    bullets=[f"{r['idx']} | {r['University']} | {r['Programme']} | "
             f"Student {r['grades']} GPA {r['stu_gpa']:.2f} | "
             f"Req {r['Band'] or '—'} GPA "
             f"{r['req_gpa'] if not pd.isna(r['req_gpa']) else '—'}"
             for r in rows]
    prompt=("For EACH line, give ONE factual comparison (<20 words) of the "
            "student's grades & GPA vs requirements. No advice. "
            "Return: id | comment.\n\n"+ "\n".join(bullets))
    rsp=client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"system","content":"Return id | comment only."},
                  {"role":"user","content":prompt}],
        max_completion_tokens=MAX_COMP,
        reasoning_effort="low")
    out={}
    for ln in rsp.choices[0].message.content.strip().splitlines():
        m=re.match(r"(\d+)\s*\|\s*(.+)",ln)
        if m: out[int(m.group(1))]=m.group(2).strip()
    return out

def colour(c): return {"Safety":"#d4edda","Match":"#fff3cd","Reach":"#f8d7da"}.get(c,"#f8f9fa")
def badge(c):
    return st.badge(c, icon={ "Safety":":material/check_circle:",
                              "Match":":material/balance:",
                              "Reach":":material/rocket_launch:"}[c],
                    color={"Safety":"green","Match":"orange","Reach":"red"}[c])
def kpis(df):
    a,b,c=st.columns(3)
    a.metric("Safety ✅", df[df.Category=="Safety"].shape[0])
    b.metric("Match 🎯",  df[df.Category=="Match"].shape[0])
    c.metric("Reach 🚀",  df[df.Category=="Reach"].shape[0])

# ---------------- STREAMLIT UI ----------------
st.set_page_config("Uni Screener","🎓")
st.title("🎓 University Admission Screener")

stu_gr = st.text_input("Your A-level grades (e.g. A*A B)", "A*A B")
stu_gpa= st.number_input("Your GPA (0-4 scale)",0.0,4.0,3.7,0.01)
major  = st.selectbox("Programme / Major", sorted(table.prog_norm.unique()))

if st.button("🔍 Search") and stu_gr.strip():
    subset=table[table.prog_norm==major]
    if subset.empty: st.warning("No programmes found."); st.stop()

    rows=[]
    for i,row in subset.iterrows():
        band=json.loads(row["Requirements (A-level)"]).get("overall_band","")
        pct = percent_match(stu_gr, band, row["Difficulty"])
        rows.append(dict(idx=i, grades=stu_gr, stu_gpa=stu_gpa,
            University=row["University"], Programme=row["Major/Programme"],
            Band=band, req_gpa=row["Req_GPA"], pct=pct,
            Category=category_from_pct(pct) if band else "N/A"))
    order={"Safety":0,"Match":1,"Reach":2,"N/A":3}
    rows.sort(key=lambda r:(order.get(r["Category"],99), -r["pct"]))
    comment_map = gpt_batch_comment(rows[:MAX_ROWS_FOR_GPT])
    df_res=pd.DataFrame(rows); kpis(df_res)

    tab_names=["✅ Safety","🎯 Match","🚀 Reach","ℹ️ N/A"]
    tabs=st.tabs(tab_names); tab_map=dict(zip(["Safety","Match","Reach","N/A"],tabs))
    for cat in ["Safety","Match","Reach","N/A"]:
        with tab_map[cat]:
            cr=df_res[df_res.Category==cat]
            if cr.empty: st.info("No programmes in this category."); continue
            for _,r in cr.iterrows():
                with st.container():
                    st.markdown(f'<div style="background-color:{colour(cat)};'
                                'padding:8px;border-radius:6px">',
                                unsafe_allow_html=True)
                    st.markdown(f"**{r.University} – {r.Programme}**"); badge(cat)
                    st.progress(max(0,min(r.pct/100.0,1.0)),
                                text=f"{r.pct}% Match • GPA req "
                                     f"{'—' if pd.isna(r.req_gpa) else r.req_gpa}")
                    with st.expander("💬 Comparison"):
                        st.write(comment_map.get(r.idx,"— no comment —"))
                    st.markdown("</div>",unsafe_allow_html=True); st.write("")
    st.download_button("📥 CSV", df_res.drop(columns=["idx"]).to_csv(index=False),
                       "uni_matches.csv","text/csv")
else:
    st.info("Enter grades & GPA, pick a major, click Search.")

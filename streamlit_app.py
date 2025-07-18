# streamlit_app.py — holistic screener with GPA & per-row GPT comments
# -------------------------------------------------------------------
# Tested with: streamlit ≥ 1.32 • openai ≥ 1.25 • model: o3-mini
#
# Parameter note → o3-series models reject the legacy `max_tokens`;
# they require **max_completion_tokens** instead.:contentReference[oaicite:0]{index=0}
#
# Quick refs on o3-mini parameters & reasoning_effort:​:contentReference[oaicite:1]{index=1}

import os, json, re
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

# ───────────────────────── CONFIG ─────────────────────────
CSV_FILE              = "university_requirements.csv"     # dataset path
MODEL_NAME            = os.getenv("OPENAI_MODEL", "o3-mini")
MAX_COMP              = 1_200      # generation budget for each GPT batch
MAX_ROWS_FOR_GPT      = 25         # rows sent for GPT comments
GRADE_POINTS          = {"A*":56, "A":48, "B":40, "C":32, "D":24, "E":16}

client = OpenAI()  # expects OPENAI_API_KEY in env or Streamlit secrets

# ──────────────────────────────────────────────────────────
# ▸ DATA LOAD & NORMALISE
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # add normalised programme key
    df["prog_norm"] = (
        df["Major/Programme"]
        .astype(str).str.strip().str.lower()
        .str.replace(r"\s*\(.*\)", "", regex=True)
    )

    # ensure GPA column
    if "GPA" not in df.columns:
        df["GPA"] = "N/A"

    # required-GPA field (parsed from JSON)
    def parse_req_gpa(cell) -> Optional[float]:
        try:
            val = json.loads(cell).get("minimum")
            return float(val) if val not in (None, "", "N/A") else float("nan")
        except Exception:
            return float("nan")

    df["Req_GPA"] = df["GPA"].apply(parse_req_gpa)

    if "Difficulty" not in df.columns:
        df["Difficulty"] = 1.0

    return df

table = load_data(CSV_FILE)

# ──────────────────────────────────────────────────────────
# ▸ GRADE HELPERS
def tokenise(txt: str) -> List[str]:
    s, out, i = txt.upper().replace(" ", ""), [], 0
    while i < len(s):
        out.append("A*" if s[i:i+2] == "A*" else s[i])
        i += 2 if s[i:i+2] == "A*" else 1
    return out

def top_n_points(gs: List[str], n: int) -> int:
    return sum(sorted((GRADE_POINTS.get(g,0) for g in gs), reverse=True)[:n])

def percent_match(student: str, band: str, diff: float) -> float:
    if not band or band.strip() in ("-", "N/A"):
        return 0.0
    stu_pts = top_n_points(tokenise(student), len(tokenise(band)))
    req_pts = sum(GRADE_POINTS.get(g,0) for g in tokenise(band)) * diff
    return round(100 * stu_pts / req_pts, 1) if req_pts else 0.0

def category_from_pct(p: float) -> str:
    return "Safety" if p >= 110 else "Match" if p >= 95 else "Reach"

# ──────────────────────────────────────────────────────────
# ▸ GPT COMMENT BATCH
def gpt_batch_comment(rows: List[Dict]) -> Dict[int, str]:
    if not rows:
        return {}

    bullets = [
        (f"{r['idx']} | {r['University']} | {r['Programme']} | "
         f"Student {r['grades']} GPA {r['stu_gpa']:.2f} | "
         f"Req {r['Band']} GPA {r['req_gpa'] if not pd.isna(r['req_gpa']) else '—'}")
        for r in rows
    ]

    prompt = (
        "For EACH line, write ONE factual comparison (<20 words) of the student's "
        "grades & GPA vs the programme requirements. No advice. "
        "Return exactly: id | comment.\n\n" + "\n".join(bullets)
    )

    rsp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role":"system", "content":"Return id | comment only."},
            {"role":"user",   "content":prompt}
        ],
        max_completion_tokens=MAX_COMP,  # required param for o-series
        reasoning_effort="low"
    )

    out: Dict[int,str] = {}
    for ln in rsp.choices[0].message.content.strip().splitlines():
        m = re.match(r"(\d+)\s*\|\s*(.+)", ln)
        if m: out[int(m.group(1))] = m.group(2).strip()
    return out

# ──────────────────────────────────────────────────────────
# ▸ STREAMLIT VISUAL HELPERS
def colour(c): return {"Safety":"#d4edda", "Match":"#fff3cd", "Reach":"#f8d7da"}.get(c,"#f8f9fa")
def badge(c):
    icon = {"Safety":":material/check_circle:", "Match":":material/balance:", "Reach":":material/rocket_launch:"}[c]
    colr = {"Safety":"green","Match":"orange","Reach":"red"}[c]
    return st.badge(c, icon=icon, color=colr)
def kpis(df):
    a,b,c = st.columns(3)
    a.metric("Safety ✅", df[df.Category=="Safety"].shape[0])
    b.metric("Match 🎯",  df[df.Category=="Match"].shape[0])
    c.metric("Reach 🚀",  df[df.Category=="Reach"].shape[0])

# ──────────────────────────────────────────────────────────
# ▸ STREAMLIT APP
st.set_page_config("Uni Screener","🎓")
st.title("🎓 University Admission Screener")

stu_gr  = st.text_input("Your A-level grades (e.g. A*A B)", "A*A B")
stu_gpa = st.number_input("Your GPA (0-4 scale)", 0.0, 4.0, 3.7, 0.01)
major   = st.selectbox("Programme / Major", sorted(table.prog_norm.unique()))

if st.button("🔍 Search") and stu_gr.strip():
    subset = table[table.prog_norm == major]
    if subset.empty:
        st.warning("No programmes found."); st.stop()

    rows = []
    for i,row in subset.iterrows():
        try:
            band = json.loads(row["Requirements (A-level)"]).get("overall_band","").strip()
        except Exception:
            band = ""
        req_gpa = row["Req_GPA"]

        if re.search(r"[A-E]", band):
            pct, cat = percent_match(stu_gr, band, row["Difficulty"]), category_from_pct(percent_match(stu_gr, band, row["Difficulty"]))
        elif not pd.isna(req_gpa) and req_gpa > 0:
            pct, cat = round(stu_gpa/req_gpa*100,1), category_from_pct(round(stu_gpa/req_gpa*100,1))
        else:
            pct, cat = 0.0, "N/A"

        rows.append(dict(
            idx=i, grades=stu_gr, stu_gpa=stu_gpa,
            University=row["University"], Programme=row["Major/Programme"],
            Band=band if re.search(r"[A-E]", band) else "—",
            req_gpa=req_gpa, pct=pct, Category=cat
        ))

    rows.sort(key=lambda r:({"Safety":0,"Match":1,"Reach":2,"N/A":3}.get(r["Category"],99), -r["pct"]))
    comment_map = gpt_batch_comment(rows[:MAX_ROWS_FOR_GPT])
    df_res = pd.DataFrame(rows)

    kpis(df_res)

    tab_titles = ["✅ Safety","🎯 Match","🚀 Reach","ℹ️ N/A"]
    tabs = st.tabs(tab_titles)
    tmap = dict(zip(["Safety","Match","Reach","N/A"], tabs))

    for cat in ["Safety","Match","Reach","N/A"]:
        with tmap[cat]:
            cat_rows = df_res[df_res.Category==cat]
            if cat_rows.empty:
                st.info("No programmes in this category."); continue
            for _,r in cat_rows.iterrows():
                with st.container():
                    st.markdown(f'<div style="background-color:{colour(cat)};padding:8px;border-radius:6px">',unsafe_allow_html=True)
                    st.markdown(f"*{r.University} – {r.Programme}*")
                    badge(cat)
                    st.progress(max(0,min(r.pct/100,1)), text=f"{r.pct}% Match • GPA req {'—' if pd.isna(r.req_gpa) else r.req_gpa}")
                    with st.expander("💬 Comparison"):
                        st.write(comment_map.get(r.idx,"— no comment —"))
                    st.markdown("</div>", unsafe_allow_html=True); st.write("")

    st.download_button("📥 Download CSV", df_res.drop(columns=["idx"]).to_csv(index=False), "uni_matches.csv", "text/csv")
else:
    st.info("Enter your grades & GPA, choose a major, then click Search.")

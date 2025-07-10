# streamlit_app.py — holistic screener with GPA & per-row GPT comments
# -------------------------------------------------------------------
# Works with o3-mini (also o4-mini).  Reads the 150-row CSV you uploaded.

import os, json, re, pandas as pd, streamlit as st
from typing import List, Dict, Optional
from openai import OpenAI

# ──────────────── CONFIG ────────────────────────────────────────────
CSV_FILE   = "university_requirements_all_150_rows_2025-07-07.csv"
MODEL_NAME = os.getenv("OPENAI_MODEL", "o3-mini")     # set in Streamlit › Secrets
MAX_COMP   = 1200
MAX_ROWS_FOR_GPT = 25                                 # only first N rows sent
GRADE_POINTS = {"A*":56,"A":48,"B":40,"C":32,"D":24,"E":16}

client = OpenAI()
# ────────────────────────────────────────────────────────────────────


# ─────── DATA LOAD & NORMALISATION (incl. GPA parse) ───────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalised key for dropdown search
    df["prog_norm"] = (
        df["Major/Programme"].str.strip().str.lower()
        .str.replace(r"\s*\(.*\)", "", regex=True)
    )

    # parse required GPA JSON → float (NaN if "N/A" or missing)
    def get_req_gpa(js: str) -> Optional[float]:
        try:
            d = json.loads(js)
            m = d.get("minimum", None)
            return float(m) if m not in (None, "N/A", "") else float("nan")
        except Exception:
            return float("nan")

    df["Req_GPA"] = df["GPA"].apply(get_req_gpa)

    # default difficulty factor if absent
    if "Difficulty" not in df.columns:
        df["Difficulty"] = 1.0

    return df

table = load_data(CSV_FILE)
# ────────────────────────────────────────────────────────────────────


# ──────────── GRADE & CATEGORY HELPERS ─────────────────────────────
def tokenise(gr: str) -> List[str]:
    s = gr.upper().replace(" ", "")
    out, i = [], 0
    while i < len(s):
        if s[i:i+2] == "A*":
            out.append("A*"); i += 2
        else:
            out.append(s[i]); i += 1
    return out

def top_n_points(gs: List[str], n: int) -> int:
    pts = sorted([GRADE_POINTS.get(g,0) for g in gs], reverse=True)
    return sum(pts[:n])

def percent_match(student: str, band: str, diff: float) -> float:
    if not band: return 0.0
    stu_pts  = top_n_points(tokenise(student), len(tokenise(band)))
    req_pts  = sum(GRADE_POINTS[g] for g in tokenise(band)) * diff
    return round(100 * stu_pts / req_pts, 1) if req_pts else 0.0

def category_from_pct(p: float) -> str:
    if p >= 110: return "Safety"
    if p >= 95:  return "Match"
    return "Reach"
# ────────────────────────────────────────────────────────────────────


# ───────────── GPT: per-row COMPARISON comments ────────────────────
def gpt_batch_comment(rows: List[Dict]) -> Dict[int,str]:
    if not rows: return {}
    bullets = [
        f"{r['idx']} | {r['University']} | {r['Programme']} | "
        f"Student {r['grades']} GPA {r['stu_gpa']:.2f} | "
        f"Req {r['Band'] or '—'} GPA {r['req_gpa'] if not pd.isna(r['req_gpa']) else '—'}"
        for r in rows
    ]
    prompt = (
        "For EACH line below, give ONE factual comparison (<20 words) of the "
        "student's grades & GPA against the programme's requirements. "
        "No advice, just comparison.  Return: id | comment.\n\n" +
        "\n".join(bullets)
    )
    rsp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"system","content":"Return id | comment (one per line, no extra text)."},
                  {"role":"user","content":prompt}],
        max_completion_tokens=MAX_COMP,
        reasoning_effort="low")
    out={}
    for ln in rsp.choices[0].message.content.strip().splitlines():
        m = re.match(r"(\d+)\s*\|\s*(.+)", ln); 
        if m: out[int(m.group(1))]=m.group(2).strip()
    return out
# ────────────────────────────────────────────────────────────────────


# ───────────── UI HELPERS ───────────────────────────────────────────
def colour(cat): return {"Safety":"#d4edda","Match":"#fff3cd","Reach":"#f8d7da"}.get(cat,"#f8f9fa")
def badge(cat):
    if cat=="Safety": return st.badge("Safety",icon=":material/check_circle:",color="green")
    if cat=="Match":  return st.badge("Match", icon=":material/balance:", color="orange")
    return st.badge("Reach", icon=":material/rocket_launch:", color="red")
def kpis(df):
    a,b,c=st.columns(3)
    a.metric("Safety ✅", df[df.Category=="Safety"].shape[0])
    b.metric("Match 🎯",  df[df.Category=="Match"].shape[0])
    c.metric("Reach 🚀",  df[df.Category=="Reach"].shape[0])
# ────────────────────────────────────────────────────────────────────


# ──────────────── STREAMLIT UI ─────────────────────────────────────
st.set_page_config("Uni Screener","🎓")
st.title("🎓 University Admission Screener")

stu_grades = st.text_input("Your A-level grades (e.g. A*A B)", "A*A B")
stu_gpa    = st.number_input("Your GPA (0-4 scale)", 0.0, 4.0, 3.7, 0.01)
major      = st.selectbox("Programme / Major", sorted(table.prog_norm.unique()))

if st.button("🔍 Search") and stu_grades.strip():
    subset = table[table.prog_norm == major]
    if subset.empty:
        st.warning("No programmes found."); st.stop()

    rows=[]
    for i,row in subset.iterrows():
        band = json.loads(row["Requirements (A-level)"]).get("overall_band","")
        pct  = percent_match(stu_grades, band, row["Difficulty"])
        rows.append(dict(
            idx=i, grades=stu_grades, stu_gpa=stu_gpa,
            University=row["University"], Programme=row["Major/Programme"],
            Band=band, req_gpa=row["Req_GPA"], pct=pct,
            Category=category_from_pct(pct) if band else "N/A"
        ))

    order={"Safety":0,"Match":1,"Reach":2,"N/A":3}
    rows.sort(key=lambda r:(order.get(r["Category"],99), -r["pct"]))
    comment_map = gpt_batch_comment(rows[:MAX_ROWS_FOR_GPT])
    df_res=pd.DataFrame(rows)
    kpis(df_res)

    tabs = st.tabs(["✅ Safety","🎯 Match","🚀 Reach","ℹ️ N/A"])
    tab_map = dict(zip(["Safety","Match","Reach","N/A"], tabs))

    for cat in ["Safety","Match","Reach","N/A"]:
        with tab_map[cat]:
            cat_rows=df_res[df_res.Category==cat]
            if cat_rows.empty:
                st.info("No programmes in this category."); continue

            for _,r in cat_rows.iterrows():
                with st.container():
                    st.markdown(f'<div style="background-color:{colour(cat)};'
                                'padding:8px;border-radius:6px">',unsafe_allow_html=True)
                    st.markdown(f"**{r.University} – {r.Programme}**")
                    badge(cat)
                    bar = max(0.0, min(r.pct/100.0, 1.0))
                    st.progress(bar, text=f"{r.pct}% Match  •  GPA req "
                                           f"{'—' if pd.isna(r.req_gpa) else r.req_gpa}")
                    with st.expander("💬 Comparison"):
                        st.write(comment_map.get(r.idx,"— no comment —"))
                    st.markdown("</div>",unsafe_allow_html=True)
                    st.write("")

    st.download_button("📥 CSV", df_res.drop(columns=["idx"]).to_csv(index=False),
                       "uni_matches.csv","text/csv")

else:
    st.info("Enter your grades & GPA, choose a major, then click Search.")


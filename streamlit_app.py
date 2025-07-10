# streamlit_app.py — University Admission Screener • 2025-07-10
# --------------------------------------------------------------
# Works with o3-mini (and o4-mini).  Per-programme GPT advice,
# improved %-Match, tabbed UI, badges, progress bars, expanders.
# --------------------------------------------------------------

import os, json, re, pandas as pd, streamlit as st
from typing import List, Dict
from openai import OpenAI

# ───────────────────────────────── CONFIG ──────────────────────────
CSV_FILE   = "university_requirements.csv"          # your data file
MODEL_NAME = os.getenv("OPENAI_MODEL", "o3-mini")   # set in Secrets
MAX_COMP   = 1200                                   # generous allowance
MAX_ROWS_FOR_GPT = 25                               # cap rows per GPT call
GRADE_POINTS = { "A*":56, "A":48, "B":40, "C":32, "D":24, "E":16 }

client = OpenAI()
# ───────────────────────────────────────────────────────────────────


# ──────────────── DATA LOAD & BASIC NORMALISATION ─────────────────
@st.cache_data
def _load(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalised programme name for drop-down search
    df["prog_norm"] = (df["Major/Programme"]
                       .str.strip().str.lower()
                       .str.replace(r"\s*\(.*\)", "", regex=True))
    # Optional difficulty column (1.0 = normal).  Default 1.0.
    if "Difficulty" not in df.columns:
        df["Difficulty"] = 1.0
    return df

def df() -> pd.DataFrame:
    return _load(CSV_FILE, os.path.getmtime(CSV_FILE))
# ───────────────────────────────────────────────────────────────────


# ────────────────────── GRADE UTILS & % MATCH ─────────────────────
def tokenise(gr: str) -> List[str]:
    """Split 'A*A B' → ['A*','A','B']"""
    s = gr.upper().replace(" ", "")
    out, i = [], 0
    while i < len(s):
        if s[i:i+2] == "A*":
            out.append("A*"); i += 2
        else:
            out.append(s[i]); i += 1
    return out

def top_n_points(grades: List[str], n: int) -> int:
    pts = sorted([GRADE_POINTS.get(g, 0) for g in grades], reverse=True)
    return sum(pts[:n])

def percent_match(student: str, band: str, difficulty: float) -> float:
    if not band:
        return 0.0
    stu = tokenise(student)
    req = tokenise(band)
    student_pts   = top_n_points(stu, len(req))
    required_pts  = sum(GRADE_POINTS.get(g, 0) for g in req) * difficulty
    return round(100 * student_pts / required_pts, 1) if required_pts else 0.0

def category_from_pct(pct: float) -> str:
    if pct >= 110:
        return "Safety"
    if pct >= 95:
        return "Match"
    return "Reach"
# ───────────────────────────────────────────────────────────────────


# ────────────────────── GPT CALL (PER-ROW TIPS) ───────────────────
def gpt_batch_advice(rows: List[Dict]) -> Dict[int, str]:
    """
    rows – list of dicts with keys: idx, University, Programme, Category, pct
    Returns dict {idx: advice}.
    """
    if not rows:
        return {}
    # Build a compact, index-prefixed prompt so we can map advice back.
    bullet_lines = [
        f"{r['idx']} | {r['University']} | {r['Programme']} | "
        f"{r['Category']} | {r['pct']}%"
        for r in rows
    ]
    prompt = (
        "You are an admissions advisor. For EACH line below, give ONE concise "
        "tip (max 18 words) that can improve the student's chance for THAT "
        "programme. Reply with the same number of lines, each starting with "
        "the numeric id, a pipe, then the tip.\n\n" +
        "\n".join(bullet_lines)
    )

    rsp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role":"system",
             "content":"Return tips in the format 'id | tip'. No extra text."},
            {"role":"user","content":prompt}
        ],
        max_completion_tokens=MAX_COMP,
        reasoning_effort="medium"
    )
    raw = rsp.choices[0].message.content.strip()
    # Parse lines like "3 | Strengthen personal statement focusing on leadership."
    advice = {}
    for line in raw.splitlines():
        m = re.match(r"(\d+)\s*\|\s*(.+)", line.strip())
        if m:
            advice[int(m.group(1))] = m.group(2).strip()
    return advice
# ───────────────────────────────────────────────────────────────────


# ─────────────────────────── UI HELPERS ───────────────────────────
def colour(cat):
    return {
        "Safety":"#d4edda",
        "Match":"#fff3cd",
        "Reach":"#f8d7da"
    }.get(cat, "#f8f9fa")

def badge(cat):
    if cat=="Safety":
        return st.badge("Safety",  icon=":material/check_circle:",  color="green")
    if cat=="Match":
        return st.badge("Match",   icon=":material/balance:",       color="orange")
    return st.badge("Reach",      icon=":material/rocket_launch:", color="red")

def kpis(df_cat):
    s, m, r = (len(df_cat[df_cat.Category == c]) for c in ["Safety","Match","Reach"])
    a,b,c = st.columns(3)
    a.metric("Safety ✅", s)
    b.metric("Match 🎯",  m)
    c.metric("Reach 🚀",  r)
# ───────────────────────────────────────────────────────────────────


# ──────────────────────── MAIN STREAMLIT APP ──────────────────────
def main():
    st.set_page_config("Uni Screener", "🎓")
    st.title("🎓 University Admission Screener")

    table  = df()
    grades = st.text_input("Your A-level grades (e.g. A*A B)", "A*A B")
    major  = st.selectbox("Programme / Major", sorted(table.prog_norm.unique()))

    if st.button("🔍 Search") and grades.strip():
        subset = table[table.prog_norm == major]
        if subset.empty:
            st.warning("No programmes found for that keyword.")
            st.stop()

        # Build result rows with new %-Match & category.
        rows = []
        for i, row in subset.iterrows():
            band       = ""
            try:
                band = json.loads(row["Requirements (A-level)"]).get("overall_band", "")
            except Exception:
                pass
            pct   = percent_match(grades, band, row["Difficulty"])
            cat   = category_from_pct(pct) if band else "N/A"
            rows.append({
                "idx":        i,  # original dataframe index for mapping tips
                "University": row["University"],
                "Programme":  row["Major/Programme"],
                "Band":       band,
                "pct":        pct,
                "Category":   cat
            })

        # Sort Safety > Match > Reach > N/A then by % desc.
        order = {"Safety":0,"Match":1,"Reach":2,"N/A":3}
        rows.sort(key=lambda r:(order.get(r["Category"],99), -r["pct"]))

        # Ask GPT for advice on at most MAX_ROWS_FOR_GPT rows.
        advice_map = gpt_batch_advice(rows[:MAX_ROWS_FOR_GPT])

        # Convert to DataFrame for easy filtering / KPI.
        res_df = pd.DataFrame(rows)
        kpis(res_df)

        # Build tabs
        safety_tab, match_tab, reach_tab, na_tab = st.tabs(
            ["✅ Safety", "🎯 Match", "🚀 Reach", "ℹ️ N/A"]
        )
        tab_map = {
            "Safety": safety_tab,
            "Match":  match_tab,
            "Reach":  reach_tab,
            "N/A":    na_tab
        }

        # Display rows inside their category tab
        for cat in ["Safety","Match","Reach","N/A"]:
            with tab_map[cat]:
                cat_rows = res_df[res_df.Category == cat]
                if cat_rows.empty():
                    st.info("No programmes in this category.")
                    continue
                for _, r in cat_rows.iterrows():
                    bg = colour(cat)
                    with st.container():
                        st.markdown(f"""<div style="background-color:{bg};padding:8px;border-radius:6px">""",
                                    unsafe_allow_html=True)
                        title = f"**{r.University} – {r.Programme}**"
                        st.markdown(title)
                        badge(cat)

                        # % Match progress bar
                        pct_int = int(max(0,min(r.pct, 200)))  # clamp 0-200
                        st.progress(pct_int, text=f"{r.pct}% Match")

                        # Advice (if available)
                        tip = advice_map.get(r.idx, "— no tip generated (row beyond GPT limit) —")
                        with st.expander("💬 Advice"):
                            st.write(tip)
                            st.write(f"*Band required:* `{r.Band or 'N/A'}`")

                        st.markdown("</div>", unsafe_allow_html=True)
                        st.write("")  # spacer

        # Export whole table
        if st.download_button("📥 Download CSV", res_df.drop(columns=["idx"]).to_csv(index=False),
                              "uni_matches.csv", "text/csv"):
            st.success("CSV downloaded!")

if __name__ == "__main__":
    # Make sure the OpenAI key is set
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("Please add OPENAI_API_KEY to Streamlit › Secrets.")
    else:
        main()

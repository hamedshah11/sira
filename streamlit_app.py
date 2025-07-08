import os, json, time, pathlib
from typing import List

import pandas as pd
import streamlit as st

# ============ CONFIG =========================================
DATA_FILE    = "university_requirements.csv"
GRADE_MAP    = {"A*": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "o3-mini")       # override in Secrets
# =============================================================

# ---------- Data loader with automatic cache-bust ------------
@st.cache_data
def _load_data(path: str, _mtime: float) -> pd.DataFrame:
    """Cache is invalidated whenever the CSV's modification-time changes."""
    df = pd.read_csv(path)

    # normalise programme names so "Computer Science (BSc)" → "computer science"
    df["Programme_norm"] = (df["Major/Programme"]
                            .str.strip()
                            .str.lower()
                            .str.replace(r"\s*\(.*\)", "", regex=True))
    return df

def get_data(path: str = DATA_FILE) -> pd.DataFrame:
    mtime = os.path.getmtime(path)   # float: seconds since epoch
    return _load_data(path, mtime)
# -------------------------------------------------------------

# ---------- Deterministic Safety/Match/Reach engine ----------
def parse_grades(text: str) -> List[str]:
    text = text.upper().replace(" ", "")
    i, out = 0, []
    while i < len(text):
        if text[i] == "A" and i + 1 < len(text) and text[i+1] == "*":
            out.append("A*"); i += 2
        else:
            out.append(text[i]); i += 1
    return out

def numeric(grades: List[str]) -> List[int]:
    return sorted([GRADE_MAP.get(g, 0) for g in grades], reverse=True)

def compare(student, req):
    for s, r in zip(student, req):
        if s > r: return 1
        if s < r: return -1
    return 0

def tag(student_raw: str, band_raw: str) -> str:
    if not band_raw:
        return "N/A"
    s = numeric(parse_grades(student_raw))
    r = numeric(parse_grades(band_raw))
    s += [0] * (len(r) - len(s)); r += [0] * (len(s) - len(r))
    cmp = compare(s, r)
    return "Safety" if cmp == 1 else "Match" if cmp == 0 else "Reach"

def chance(student_raw: str, band_raw: str) -> str:
    if not band_raw:
        return "—"
    s_avg = sum(numeric(parse_grades(student_raw))) / 3
    r_avg = sum(numeric(parse_grades(band_raw))) / 3
    p = 0.60 + 0.10 * (s_avg - r_avg)
    p = max(0.10, min(0.90, p))
    return f"{int(p * 100)}%"
# -------------------------------------------------------------

# ------------------- LLM helper ------------------------------
def llm_call(prompt: str) -> str:
    try:
        import openai
    except ImportError:
        return "⚠️ openai package missing."
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        return "⚠️ No OpenAI key set."
    openai.api_key = key
    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip()
# -------------------------------------------------------------

# ----------------- UI helpers --------------------------------
def colour(cat):
    return {"Safety":"background-color:#d4edda",
            "Match":"background-color:#fff3cd",
            "Reach":"background-color:#f8d7da",
            "N/A":"background-color:#f8f9fa"}.get(cat,"")

def show_kpis(df):
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total",  len(df))
    c2.metric("Safety 🎯", len(df[df["Category"]=="Safety"]))
    c3.metric("Match ⚖️",  len(df[df["Category"]=="Match"]))
    c4.metric("Reach 🚀",  len(df[df["Category"]=="Reach"]))
# -------------------------------------------------------------

# ======================= APP =================================
def main():
    st.set_page_config("Uni Screener", "🎓")
    st.title("🎓 University Admission Screener")

    df = get_data()

    grades = st.text_input("Your A-level grades", placeholder="A*A B / ABB")

    # dropdown options are the *normalised* unique names
    majors = sorted(df["Programme_norm"].unique())
    major_norm = st.selectbox("Choose a major / programme", majors, index=0)

    if st.button("🔍 Show matches") and grades:
        subset = df[df["Programme_norm"] == major_norm]
        if subset.empty:
            st.warning("No programmes found."); st.stop()

        rows = []
        for _, r in subset.iterrows():
            try:
                band = json.loads(r["Requirements (A-level)"]).get("overall_band", "")
            except Exception:
                band = ""
            rows.append({
                "University": r["University"],
                "Programme":  r["Major/Programme"],
                "Band":       band,
                "Category":   tag(grades, band),
                "Chance":     chance(grades, band)
            })

        order = {"Safety":0,"Match":1,"Reach":2,"N/A":3}
        out_df = (pd.DataFrame(rows)
                    .sort_values("Category", key=lambda s: s.map(order)))

        show_kpis(out_df)
        styled = (out_df.style.applymap(colour, subset=["Category"])
                            .hide(axis="index"))
        st.dataframe(styled, use_container_width=True)

        st.download_button("📥 Download CSV",
                           out_df.to_csv(index=False),
                           "university_matches.csv",
                           "text/csv")

        bullets = "\n".join(
            f"[{row.Category}] {row.University} – {row.Programme} (needs {row.Band})"
            for row in out_df.itertuples()
        )
        prompt = (
            f'Grades: "{grades}" | Major: "{major_norm}"\n{bullets}\n\n'
            "Explain briefly why each tag is fair and give one tip to improve odds."
        )
        with st.spinner("Consulting GPT…"):
            advice = llm_call(prompt)
        st.markdown("### 🤖 LLM Advice")
        st.markdown(advice)

if __name__ == "__main__":
    main()

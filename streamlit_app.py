import os, json
from typing import List

import pandas as pd
import streamlit as st

# ============ CONFIG =============
DATA_FILE    = "university_requirements.csv"
GRADE_MAP    = {"A*": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "o3-mini")  # set in Secrets
# =================================

# ---------- DATA LOADER ----------
@st.cache_data
def _load_data(path: str, _mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalise programme names
    df["Programme_norm"] = (
        df["Major/Programme"]
          .str.strip()
          .str.lower()
          .str.replace(r"\s*\(.*\)", "", regex=True)
    )
    return df

def get_data(path: str = DATA_FILE) -> pd.DataFrame:
    mtime = os.path.getmtime(path)            # cache key
    return _load_data(path, mtime)
# ----------------------------------

# ---------- RULES ENGINE ----------
def parse_grades(txt: str) -> List[str]:
    txt = txt.upper().replace(" ", "")
    out, i = [], 0
    while i < len(txt):
        if txt[i] == "A" and i + 1 < len(txt) and txt[i+1] == "*":
            out.append("A*"); i += 2
        else:
            out.append(txt[i]); i += 1
    return out

def numeric(gs: List[str]) -> List[int]:
    return sorted([GRADE_MAP.get(g, 0) for g in gs], reverse=True)

def tag(student: str, band: str) -> str:
    if not band:
        return "N/A"
    s, b = numeric(parse_grades(student)), numeric(parse_grades(band))
    s += [0]*(len(b)-len(s)); b += [0]*(len(s)-len(b))
    for a, r in zip(s, b):
        if a > r: return "Safety"
        if a < r: return "Reach"
    return "Match"

def chance(student: str, band: str) -> str:
    if not band: return "—"
    sa = sum(numeric(parse_grades(student)))/3
    ba = sum(numeric(parse_grades(band)))/3
    p  = 0.60 + 0.10 * (sa - ba)
    p  = max(0.10, min(0.90, p))
    return f"{int(p*100)}%"
# ----------------------------------

# ---------- LLM HELPER ------------
def llm(prompt: str) -> str:
    import openai
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OpenAI key missing in Secrets.")
    openai.api_key = key
    rsp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=350           # o-series syntax
    )
    return rsp.choices[0].message.content.strip()
# ----------------------------------

# ---------- UI HELPERS ------------
def colour(cat):
    return {"Safety":"background-color:#d4edda",
            "Match":"background-color:#fff3cd",
            "Reach":"background-color:#f8d7da",
            "N/A":"background-color:#f8f9fa"}.get(cat,"")

def kpis(df):
    a,b,c,d = st.columns(4)
    a.metric("Total", len(df))
    b.metric("Safety 🎯", len(df[df.Category=="Safety"]))
    c.metric("Match ⚖️", len(df[df.Category=="Match"]))
    d.metric("Reach 🚀", len(df[df.Category=="Reach"]))
# ----------------------------------

# -------------- APP ---------------
def main():
    st.set_page_config("Uni Screener","🎓")
    st.title("🎓 University Admission Screener")

    data = get_data()

    grades = st.text_input("Your A-level grades", placeholder="A*A B / ABB")

    majors = sorted(data["Programme_norm"].unique())
    major_norm = st.selectbox("Choose a major / programme", majors, index=0)

    if st.button("🔍 Show matches") and grades:
        subset = data[data["Programme_norm"] == major_norm]
        if subset.empty:                     # property, not function
            st.warning("No programmes found."); st.stop()

        rows = []
        for _, r in subset.iterrows():
            try:
                band = json.loads(r["Requirements (A-level)"]).get("overall_band","")
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
        out = (pd.DataFrame(rows)
                 .sort_values("Category", key=lambda s: s.map(order)))

        kpis(out)
        st.dataframe(
            out.style.map(colour, subset=["Category"]).hide(axis="index"),   # Styler.map
            use_container_width=True
        )

        st.download_button("📥 Download CSV",
                           out.to_csv(index=False),
                           "university_matches.csv","text/csv")

        bullets = "\n".join(
            f"[{r.Category}] {r.University} – {r.Programme} (needs {r.Band})"
            for r in out.itertuples()
        )
        prompt = (
            f'Grades: "{grades}" | Major: "{major_norm}"\n{bullets}\n\n'
            "Explain briefly why each tag is fair and give one improvement tip."
        )
        with st.spinner("Consulting GPT…"):
            try:
                advice = llm(prompt)
                st.markdown("### 🤖 LLM Advice")
                st.markdown(advice or "_LLM returned empty response._")
            except Exception as e:
                st.error(f"LLM call failed: {e}")

if __name__ == "__main__":
    main()

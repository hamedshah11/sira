import os, json
import pandas as pd
import streamlit as st
from typing import List

# ---------- CONFIG ------------------------------------------------------------------
DATA_FILE     = "university_requirements.csv"
GRADE_MAP     = {"A*": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # override via Streamlit Secrets
# ------------------------------------------------------------------------------------

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# -------------------------- DETERMINISTIC ENGINE ------------------------------------
def parse_grade_string(text: str) -> List[str]:
    """Tokenise grade string, preserving A*."""
    text = text.upper().replace(" ", "")
    out, i = [], 0
    while i < len(text):
        if text[i] == "A" and i + 1 < len(text) and text[i + 1] == "*":
            out.append("A*"); i += 2
        else:
            out.append(text[i]); i += 1
    return out

def numeric_list(grades: List[str]) -> List[int]:
    return sorted([GRADE_MAP.get(g, 0) for g in grades], reverse=True)

def compare_lists(student, band):
    for s, b in zip(student, band):
        if s > b: return 1
        if s < b: return -1
    return 0

def tag_category(student_raw: str, band_raw: str) -> str:
    if not band_raw:
        return "N/A"
    s = numeric_list(parse_grade_string(student_raw))
    b = numeric_list(parse_grade_string(band_raw))
    s += [0] * (len(b) - len(s))
    b += [0] * (len(s) - len(b))
    cmp = compare_lists(s, b)
    return "Safety" if cmp == 1 else "Match" if cmp == 0 else "Reach"

# -------------------------- OPTIONAL CHANCE HEURISTIC --------------------------------
def probability_estimate(student_raw: str, band_raw: str) -> float:
    """Rough chance (grade-only). 0.10–0.90 range."""
    if not band_raw: return 0.0
    s = numeric_list(parse_grade_string(student_raw))
    b = numeric_list(parse_grade_string(band_raw))
    avg_s = sum(s) / len(s)
    avg_b = sum(b) / len(b)
    if avg_s >= avg_b:
        return min(0.90, 0.60 + (avg_s - avg_b) * 0.10)
    return max(0.10, 0.60 - (avg_b - avg_s) * 0.15)

# ------------------------------ LLM HELPER -------------------------------------------
def llm_advice(prompt: str) -> str:
    try:
        import openai
    except ImportError:
        return "⚠️ `openai` package not installed."
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        return "⚠️ OpenAI API key not set (add in Streamlit ► Secrets)."
    openai.api_key = key
    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip()

# ----------------------------- UI HELPERS --------------------------------------------
def color_category(val):
    return {
        "Safety": "background-color:#d4edda",   # green
        "Match":  "background-color:#fff3cd",   # yellow
        "Reach":  "background-color:#f8d7da",   # red
        "N/A":    "background-color:#f8f9fa",   # grey
    }.get(val, "")

def show_kpis(df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total",   len(df))
    col2.metric("Safety 🎯", len(df[df["Category"] == "Safety"]))
    col3.metric("Match ⚖️", len(df[df["Category"] == "Match"]))
    col4.metric("Reach 🚀", len(df[df["Category"] == "Reach"]))

# -------------------------------- STREAMLIT APP -------------------------------------
def main():
    st.set_page_config(page_title="University Screener", page_icon="🎓")
    st.title("🎓 University Admission Screener")
    st.caption("Deterministic Safety / Match / Reach tags + GPT-4o-mini advice")

    df = load_data(DATA_FILE)

    col1, col2 = st.columns(2)
    grades = col1.text_input("Your A-level grades", placeholder="A*A B  /  AAA")
    major  = col2.text_input("Desired major keyword", placeholder="Computer Science")

    if st.button("🔍  Search") and grades and major:
        subset = df[df["Major/Programme"].str.contains(major, case=False, na=False)]
        if subset.empty:
            st.warning("No matching programmes found."); st.stop()

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
                "Category":   tag_category(grades, band),
                "Chance":     f"{int(probability_estimate(grades, band)*100)}%"
            })

        order   = {"Safety": 0, "Match": 1, "Reach": 2, "N/A": 3}
        results = (pd.DataFrame(rows)
                   .sort_values("Category", key=lambda s: s.map(order)))

        # KPIs, table, CSV
        show_kpis(results)
        styled = (results
                  .style
                  .applymap(color_category, subset=["Category"])
                  .hide(axis="index"))
        st.dataframe(styled, use_container_width=True)

        csv = results.to_csv(index=False)
        st.download_button("📥 Download CSV", csv,
                           file_name="university_matches.csv",
                           mime="text/csv")

        if st.checkbox("🧠 LLM advice (GPT-4o-mini)"):
            bullet = "\n".join(
                f"[{r['Category']}] {r['University']} – {r['Programme']} (needs {r['Band']})"
                for _, r in results.iterrows()
            )
            prompt = (
                f'Student grades: "{grades}"  |  Major keyword: "{major}"\n{bullet}\n\n'
                "For each programme:\n"
                "• Confirm whether the tag is fair.\n"
                "• Explain the key gap in one sentence.\n"
                "• Give one actionable tip to improve chances.\n"
                "Return a concise markdown list."
            )
            with st.spinner("Consulting the LLM…"):
                st.markdown("### Personalised guidance")
                st.markdown(llm_advice(prompt))

if __name__ == "__main__":
    main()

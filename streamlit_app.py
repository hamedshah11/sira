import os, json
import pandas as pd
import streamlit as st
from typing import List

# ---------- CONFIG ----------
DATA_FILE = "university_requirements.csv"
GRADE_MAP = {"A*": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # override in Secrets if you like
# ---------------------------------

@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)

def parse_grade_string(text: str) -> List[str]:
    """Split A-level string into tokens, keeping A* intact."""
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
        if s > b:
            return 1
        if s < b:
            return -1
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

# ---------- LLM helper ----------
def llm_advice(prompt: str) -> str:
    try:
        import openai
    except ImportError:
        return "⚠️ openai package missing."
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        return "⚠️ No OpenAI key provided."
    openai.api_key = key
    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip()

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="University Screener", page_icon="🎓")
    st.title("🎓 University Admission Screener")
    st.caption("Deterministic Safety/Match/Reach tagging + GPT advice")

    df = load_data(DATA_FILE)

    col1, col2 = st.columns(2)
    grades = col1.text_input("Your A-level grades", placeholder="A*A B / AAA")
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
                "Programme": r["Major/Programme"],
                "Band": band,
                "Category": tag_category(grades, band),
            })

        order = {"Safety": 0, "Match": 1, "Reach": 2, "N/A": 3}
        results = (pd.DataFrame(rows)
                   .sort_values("Category", key=lambda s: s.map(order)))
        st.subheader("Results")
        st.dataframe(results, use_container_width=True)

        if st.checkbox("🧠  LLM advice"):
            bullet = "\n".join(
                f"[{r['Category']}] {r['University']} – {r['Programme']} (needs {r['Band']})"
                for _, r in results.iterrows()
            )
            user_prompt = (
                f'Student grades: "{grades}" | Major keyword: "{major}"\n{bullet}\n'
                "Explain each tag and give one suggestion to improve chances."
            )
            with st.spinner("Consulting the LLM…"):
                st.markdown("### Personalised guidance")
                st.markdown(llm_advice(user_prompt))

if __name__ == "__main__":
    main()

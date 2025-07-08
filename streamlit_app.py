import os, json, time
import pandas as pd
import streamlit as st
from typing import List

# ------------------- CONFIG -------------------------------------------------
DATA_FILE     = "university_requirements.csv"
GRADE_MAP     = {"A*": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "o3-mini")   # set in Secrets
# ----------------------------------------------------------------------------

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# ------------------- DETERMINISTIC ENGINE -----------------------------------
def parse_grade(text: str) -> List[str]:
    t = text.upper().replace(" ", "")
    out, i = [], 0
    while i < len(t):
        if t[i] == "A" and i + 1 < len(t) and t[i + 1] == "*":
            out.append("A*"); i += 2
        else:
            out.append(t[i]); i += 1
    return out

def numeric_list(grades: List[str]) -> List[int]:
    return sorted([GRADE_MAP.get(g, 0) for g in grades], reverse=True)

def compare(a, b):
    for s, r in zip(a, b):
        if s > r: return 1
        if s < r: return -1
    return 0

def tag(student: str, band: str) -> str:
    if not band: return "N/A"
    s = numeric_list(parse_grade(student))
    r = numeric_list(parse_grade(band))
    s += [0] * (len(r) - len(s));  r += [0] * (len(s) - len(r))
    cmp = compare(s, r)
    return "Safety" if cmp == 1 else "Match" if cmp == 0 else "Reach"

def chance(student: str, band: str) -> str:
    if not band: return "—"
    s = numeric_list(parse_grade(student)); r = numeric_list(parse_grade(band))
    p = 0.6 + 0.1 * (sum(s)/len(s) - sum(r)/len(r))
    p = max(0.10, min(0.90, p))
    return f"{int(p*100)}%"

# ------------------- LLM HELPER ---------------------------------------------
def llm(prompt: str) -> str:
    try:
        import openai
    except ImportError:
        return "⚠️ `openai` lib missing."
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not key: return "⚠️ No API key."
    openai.api_key = key
    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=350,
    )
    return resp.choices[0].message.content.strip()

# ------------------- UI HELPERS ---------------------------------------------
def colour(val):
    return {"Safety":"background-color:#d4edda",
            "Match":"background-color:#fff3cd",
            "Reach":"background-color:#f8d7da",
            "N/A":"background-color:#f8f9fa"}.get(val,"")

def kpis(df):
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total", len(df))
    c2.metric("Safety🎯", len(df[df["Category"]=="Safety"]))
    c3.metric("Match⚖️", len(df[df["Category"]=="Match"]))
    c4.metric("Reach🚀", len(df[df["Category"]=="Reach"]))

# ------------------- APP -----------------------------------------------------
def main():
    st.set_page_config("Uni Screener","🎓")
    st.title("🎓 University Admission Screener")

    data = load_data(DATA_FILE)

    # === INPUTS ===
    grades = st.text_input("Your A-level grades", placeholder="A*A B / ABB")

    majors = sorted(data["Major/Programme"].dropna().unique())
    # selectbox refreshes UI automatically (normal Streamlit behaviour). :contentReference[oaicite:0]{index=0}
    major = st.selectbox("Select a major / programme", majors, index=0)

    # === PROCESS ===
    if st.button("🔍  Show matches") and grades:
        subset = data[data["Major/Programme"] == major]
        if subset.empty:
            st.warning("No entries for that major."); st.stop()

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
        df_out = (pd.DataFrame(rows)
                    .sort_values("Category", key=lambda s: s.map(order)))

        # === DISPLAY ===
        kpis(df_out)
        styled = (df_out.style.applymap(colour, subset=["Category"])
                            .hide(axis="index"))
        st.dataframe(styled, use_container_width=True)

        st.download_button("📥 Download CSV",
                           df_out.to_csv(index=False),
                           "university_matches.csv",
                           "text/csv")

        # === LLM ADVICE (automatic) ===
        bullets = "\n".join(
            f"[{row.Category}] {row.University} – {row.Programme} (needs {row.Band})"
            for row in df_out.itertuples()
        )
        prompt = (
            f'Grades: "{grades}" | Major: "{major}"\n{bullets}\n\n'
            "Explain briefly why each tag is fair and give one improvement tip."
        )
        with st.spinner("Consulting the LLM…"):          # built-in spinner :contentReference[oaicite:1]{index=1}
            advice = llm(prompt)
        st.markdown("### 🤖 LLM Advice")
        st.markdown(advice)

if __name__ == "__main__":
    main()

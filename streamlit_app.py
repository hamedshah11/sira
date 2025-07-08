import os, json, pandas as pd, streamlit as st
from typing import List

# ---------------- CONFIG -----------------
CSV_PATH      = "university_requirements.csv"
GRADE_MAP     = {"A*":6,"A":5,"B":4,"C":3,"D":2,"E":1}
OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "o3-mini")   # set in “Secrets”
MAX_TOKENS    = 350                                    # keep short
# -----------------------------------------

# ------- CSV LOADER with cache-bust -------
@st.cache_data
def _load(path: str, mtime: float):
    df = pd.read_csv(path)
    df["Programme_norm"] = (df["Major/Programme"]
        .str.strip().str.lower().str.replace(r"\s*\(.*\)", "", regex=True))
    return df

def data():
    return _load(CSV_PATH, os.path.getmtime(CSV_PATH))
# -----------------------------------------

# ------- GRADE → TAG ENGINE ---------------
def parse(gr):                        # keep A* token intact
    t = gr.upper().replace(" ", ""); i=0; out=[]
    while i < len(t):
        out.append("A*" if t[i:i+2]=="A*" else t[i])
        i += 2 if t[i:i+2]=="A*" else 1
    return out

def nums(ls): return sorted([GRADE_MAP.get(g,0) for g in ls], reverse=True)

def tag(student, band):
    if not band: return "N/A"
    s, b = nums(parse(student)), nums(parse(band))
    s += [0]*(len(b)-len(s)); b += [0]*(len(s)-len(b))
    for a,r in zip(s,b):
        if a>r: return "Safety"
        if a<r: return "Reach"
    return "Match"
# -----------------------------------------

# ---------- OPENAI CALL -------------------
def ask_openai(prompt: str) -> str:
    import openai
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY","")
    if not key: raise RuntimeError("No OpenAI key.")
    openai.api_key = key

    rsp = openai.chat.completions.create(
        model             = OPENAI_MODEL,
        messages          = [{"role":"user","content": prompt}],
        max_completion_tokens = MAX_TOKENS              # ✔ o-series
    )

    # show raw JSON for debugging
    with st.sidebar.expander("🔍 Raw LLM response", expanded=False):
        st.write(rsp)

    if not rsp.choices or not rsp.choices[0].message:
        return ""
    return rsp.choices[0].message.content.strip()
# ------------------------------------------

def main():
    st.set_page_config("Uni Screener","🎓")
    st.title("🎓 University Admission Screener")

    df   = data()
    grad = st.text_input("Your A-level grades", "A*A B")
    majors = sorted(df["Programme_norm"].unique())
    major = st.selectbox("Programme", majors, index=0)

    if st.button("Search") and grad:
        sub = df[df["Programme_norm"] == major]
        if sub.empty:
            st.warning("No programmes"); st.stop()

        rows=[]
        for _, r in sub.iterrows():
            try: band = json.loads(r["Requirements (A-level)"]).get("overall_band","")
            except: band = ""
            rows.append({
                "University": r["University"],
                "Programme":  r["Major/Programme"],
                "Band":       band,
                "Category":   tag(grad, band)
            })

        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True)

        bullets = "\n".join(f"[{c}] {u} – {p} (needs {b})"
                            for u,p,b,c in out[["University","Programme","Band","Category"]]
                            .itertuples(index=False))
        prompt = (f'Grades: "{grad}" | Major: "{major}"\n{bullets}\n'
                  "Explain each tag in one line and suggest an improvement tip.")
        with st.spinner("GPT thinking…"):
            ans = ask_openai(prompt)

        if ans:
            st.markdown("### 🤖 GPT advice")
            st.markdown(ans)
        else:
            st.error("LLM returned empty content. Inspect sidebar for raw JSON.")

if __name__ == "__main__":
    main()

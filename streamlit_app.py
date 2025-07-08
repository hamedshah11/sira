"""
streamlit_app.py — o3-mini compliant
"""
import os, json, pandas as pd, streamlit as st
from typing import List
from openai import OpenAI

# ---------------- CONFIG -----------------
CSV   = "university_requirements.csv"
MODEL = os.getenv("OPENAI_MODEL", "o3-mini")   # set in Streamlit ► Secrets
MAX_TOK = 350
client = OpenAI()

GRADE = {"A*":6,"A":5,"B":4,"C":3,"D":2,"E":1}
# -----------------------------------------

@st.cache_data
def load_df(path, mtime):          # cache busts on file update
    df = pd.read_csv(path)
    df["prog_norm"] = (df["Major/Programme"]
                       .str.strip().str.lower()
                       .str.replace(r"\s*\(.*\)","",regex=True))
    return df

def get_df():
    return load_df(CSV, os.path.getmtime(CSV))

# ---------- deterministic tag ----------
def parse(s):                      # keep A*
    s=s.upper().replace(" ",""); i=0; out=[]
    while i<len(s):
        out.append("A*" if s[i:i+2]=="A*" else s[i])
        i+=2 if s[i:i+2]=="A*" else 1
    return out

def num(lst): return sorted([GRADE.get(x,0) for x in lst], reverse=True)

def tag(gr,b):
    if not b: return "N/A"
    a,b = num(parse(gr)),num(parse(b))
    a+=[0]*(len(b)-len(a)); b+=[0]*(len(a)-len(b))
    for s,r in zip(a,b):
        if s>r: return "Safety"
        if s<r: return "Reach"
    return "Match"
# ---------------------------------------

def ask_llm(prompt:str)->str:
    rsp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        max_completion_tokens=MAX_TOK
    )
    # debug payload in sidebar
    with st.sidebar.expander("🔍 Raw LLM"):
        st.write(rsp)
    if rsp.choices and rsp.choices[0].message:
        return rsp.choices[0].message.content.strip()
    return "⚠️ empty completion"

# -------------- Streamlit UI -----------
def main():
    st.set_page_config("Uni Screener","🎓")
    st.title("🎓 University Admission Screener")

    df   = get_df()
    grad = st.text_input("Your A-level grades","A*A B")
    major = st.selectbox("Programme",sorted(df.prog_norm.unique()))

    if st.button("Search") and grad:
        rows=[]
        for _,r in df[df.prog_norm==major].iterrows():
            band = json.loads(r["Requirements (A-level)"]).get("overall_band","")
            rows.append({"University":r["University"],
                         "Programme": r["Major/Programme"],
                         "Band": band,
                         "Category": tag(grad,band)})
        out = pd.DataFrame(rows)
        st.dataframe(out,use_container_width=True)

        bullets = "\n".join(f"[{c}] {u} – {p} (needs {b})"
                            for u,p,b,c in out[["University","Programme","Band","Category"]]
                            .itertuples(index=False))
        prompt = (f'Grades: \"{grad}\" | Major: \"{major}\"\n{bullets}\n'
                  "Explain tags in one sentence & give 1 tip each.")
        with st.spinner("GPT…"):
            st.markdown(ask_llm(prompt))

if __name__ == "__main__":
    main()

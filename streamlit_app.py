# streamlit_app.py  —  fixed progress-bar clamp  •  2025-07-10
# (unchanged sections are collapsed with comments ─────────────)

import os, json, re, pandas as pd, streamlit as st
from typing import List, Dict
from openai import OpenAI

# ───────────────────────── CONFIG (unchanged) ──────────────────────────
CSV_FILE, MODEL_NAME, MAX_COMP, MAX_ROWS_FOR_GPT = (
    "university_requirements.csv", os.getenv("OPENAI_MODEL", "o3-mini"), 1200, 25
)
GRADE_POINTS = {"A*": 56, "A": 48, "B": 40, "C": 32, "D": 24, "E": 16}
client = OpenAI()
# ───────────────────────────────────────────────────────────────────────

# … all helper functions for loading CSV, grade utilities,
#   GPT batch advice, badges, KPIs stay exactly the same …

# ───────────────────────── MAIN APP ──────────────────────────
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

        # Build rows (same as before) …
        rows = []
        for i, row in subset.iterrows():
            try:
                band = json.loads(row["Requirements (A-level)"]).get("overall_band", "")
            except Exception:
                band = ""
            pct = percent_match(grades, band, row["Difficulty"])
            cat = category_from_pct(pct) if band else "N/A"
            rows.append(
                dict(idx=i, University=row["University"], Programme=row["Major/Programme"],
                     Band=band, pct=pct, Category=cat)
            )

        order = {"Safety": 0, "Match": 1, "Reach": 2, "N/A": 3}
        rows.sort(key=lambda r: (order.get(r["Category"], 99), -r["pct"]))
        advice_map = gpt_batch_advice(rows[:MAX_ROWS_FOR_GPT])
        res_df = pd.DataFrame(rows)
        kpis(res_df)

        safety_tab, match_tab, reach_tab, na_tab = st.tabs(
            ["✅ Safety", "🎯 Match", "🚀 Reach", "ℹ️ N/A"]
        )
        tab_map = {"Safety": safety_tab, "Match": match_tab,
                   "Reach": reach_tab, "N/A": na_tab}

        for cat in ["Safety", "Match", "Reach", "N/A"]:
            with tab_map[cat]:
                cat_rows = res_df[res_df.Category == cat]
                if cat_rows.empty:
                    st.info("No programmes in this category.")
                    continue

                for _, r in cat_rows.iterrows():
                    bg = colour(cat)
                    with st.container():
                        st.markdown(  # coloured card
                            f'<div style="background-color:{bg};padding:8px;border-radius:6px">',
                            unsafe_allow_html=True
                        )
                        st.markdown(f"**{r.University} – {r.Programme}**")
                        badge(cat)

                        # ─── FIX: normalise to 0-1 and clamp ───
                        progress_val = max(0.0, min(r.pct / 100.0, 1.0))
                        st.progress(progress_val, text=f"{r.pct}% Match")
                        # ───────────────────────────────────────

                        tip = advice_map.get(
                            r.idx,
                            "— no tip generated (row beyond GPT limit) —"
                        )
                        with st.expander("💬 Advice"):
                            st.write(tip)
                            st.write(f"*Band required:* `{r.Band or 'N/A'}`")

                        st.markdown("</div>", unsafe_allow_html=True)
                        st.write("")  # spacer

        st.download_button(
            "📥 Download CSV",
            res_df.drop(columns=["idx"]).to_csv(index=False),
            "uni_matches.csv",
            "text/csv"
        )

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("Please add OPENAI_API_KEY to Streamlit › Secrets.")
    else:
        main()

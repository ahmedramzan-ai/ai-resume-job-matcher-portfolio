"""
app.py
------
Main Streamlit application for the AI Resume-Job Match & Skill Gap
Analyzer.

Run with:
    streamlit run app.py
"""

import streamlit as st

from resume_parser import extract_text_from_pdf
from text_processor import clean_text_for_tfidf, normalize_for_matching, basic_stats
from skill_extractor import load_skills_dataset, extract_skills
from matcher import (
    compute_tfidf_similarity,
    get_skill_gap,
    compute_skill_coverage,
    compute_overall_score,
)
from recommendations import generate_recommendations


# -----------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume-Job Match & Skill Gap Analyzer",
    page_icon="📄",
    layout="wide",
)


# -----------------------------------------------------------------------
# Cache the skills dataset so it is only read from disk once per session,
# not on every single button click / rerun.
# -----------------------------------------------------------------------
@st.cache_data
def get_skills_list():
    return load_skills_dataset("data/skills.csv")


def render_skill_badges(skills, symbol, color):
    """Small helper to render a list of skills as colored badge-like text."""
    if not skills:
        st.write("None")
        return
    badges_html = " ".join(
        f"<span style='background-color:{color};color:white;padding:4px 10px;"
        f"border-radius:12px;margin:3px;display:inline-block;font-size:0.85em'>"
        f"{symbol} {skill}</span>"
        for skill in sorted(skills)
    )
    st.markdown(badges_html, unsafe_allow_html=True)


def main():
    st.title("📄 AI Resume-Job Match & Skill Gap Analyzer")
    st.caption(
        "Analyze how closely your resume aligns with an internship or "
        "job description."
    )

    st.markdown(
        "This is an **NLP-based resume-job matching and skill-gap "
        "analysis system**. It uses TF-IDF + cosine similarity for text "
        "similarity, and rule-based keyword extraction for skill "
        "detection. It is not a deep learning or generative AI system, "
        "and the score is a similarity indicator, not a hiring prediction."
    )

    st.divider()

    # -------------------------------------------------------------------
    # Section 1: Resume Upload & Job Description Input
    # -------------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Upload Your Resume (PDF)")
        uploaded_resume = st.file_uploader("Choose a PDF file", type=["pdf"])

    with col2:
        st.subheader("2. Paste the Job Description")
        job_description = st.text_area(
            "Paste the full job/internship description here",
            height=220,
            placeholder="e.g., We are looking for a Python developer with "
            "experience in Django, SQL, and REST APIs...",
        )

    analyze_clicked = st.button("🔍 Analyze Match", type="primary")

    st.divider()

    if not analyze_clicked:
        st.info("Upload a resume and paste a job description, then click "
                 "'Analyze Match' to see your results.")
        return

    # -------------------------------------------------------------------
    # Input validation (Section: Error Handling)
    # -------------------------------------------------------------------
    if uploaded_resume is None:
        st.error("Please upload a resume PDF before analyzing.")
        return

    if not job_description or not job_description.strip():
        st.error("Please paste a job description before analyzing.")
        return

    # -------------------------------------------------------------------
    # Step 1: Extract resume text from PDF
    # -------------------------------------------------------------------
    with st.spinner("Extracting text from your resume..."):
        resume_text_raw, pdf_error = extract_text_from_pdf(uploaded_resume)

    if pdf_error:
        st.error(pdf_error)
        return

    resume_stats = basic_stats(resume_text_raw)
    job_stats = basic_stats(job_description)

    if resume_stats["word_count"] < 20:
        st.warning(
            "Your resume text seems very short after extraction "
            f"({resume_stats['word_count']} words). Results may not be "
            "reliable. Please check that the PDF contains selectable text."
        )

    if job_stats["word_count"] < 10:
        st.warning(
            "The job description seems very short "
            f"({job_stats['word_count']} words). Results may not be "
            "reliable."
        )

    # -------------------------------------------------------------------
    # Step 2: Preprocess text
    # -------------------------------------------------------------------
    resume_clean_tfidf = clean_text_for_tfidf(resume_text_raw)
    job_clean_tfidf = clean_text_for_tfidf(job_description)

    resume_for_matching = normalize_for_matching(resume_text_raw)
    job_for_matching = normalize_for_matching(job_description)

    # -------------------------------------------------------------------
    # Step 3: Skill extraction
    # -------------------------------------------------------------------
    skills_list = get_skills_list()
    resume_skills = extract_skills(resume_for_matching, skills_list)
    job_skills = extract_skills(job_for_matching, skills_list)

    if not job_skills:
        st.warning(
            "No known skills were detected in the job description. "
            "Skill-gap analysis will be limited. Text similarity will "
            "still be shown below."
        )

    gap = get_skill_gap(resume_skills, job_skills)

    # -------------------------------------------------------------------
    # Step 4: TF-IDF + cosine similarity
    # -------------------------------------------------------------------
    text_similarity, sim_error = compute_tfidf_similarity(
        resume_clean_tfidf, job_clean_tfidf
    )
    if sim_error:
        st.warning(sim_error)

    # -------------------------------------------------------------------
    # Step 5: Overall score
    # -------------------------------------------------------------------
    skill_coverage = compute_skill_coverage(gap["matched"], job_skills)
    overall_score = compute_overall_score(text_similarity, skill_coverage)

    # =====================================================================
    # RESULTS
    # =====================================================================
    st.header("📊 Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Overall Match Score", f"{overall_score}%")
    m2.metric("Text Similarity (TF-IDF)", f"{round(text_similarity * 100, 1)}%")
    m3.metric("Skill Coverage", f"{round(skill_coverage * 100, 1)}%")

    st.progress(min(int(overall_score), 100) / 100)

    if overall_score >= 70:
        st.success("Strong match with this job description.")
    elif overall_score >= 40:
        st.warning("Moderate match. There is room for improvement.")
    else:
        st.error("Low match with this job description.")

    st.divider()

    # -------------------------------------------------------------------
    # Skill gap section
    # -------------------------------------------------------------------
    st.subheader("🧩 Skill Gap Analysis")
    sk1, sk2, sk3 = st.columns(3)

    with sk1:
        st.markdown("**✅ Matched Skills**")
        render_skill_badges(gap["matched"], "✓", "#2e7d32")

    with sk2:
        st.markdown("**⚠️ Missing Skills**")
        render_skill_badges(gap["missing"], "⚠", "#c62828")

    with sk3:
        st.markdown("**➕ Additional Skills (in resume, not in job desc.)**")
        render_skill_badges(gap["additional"], "+", "#1565c0")

    st.divider()

    # -------------------------------------------------------------------
    # Resume strength analysis
    # -------------------------------------------------------------------
    st.subheader("💪 Resume Strength Analysis (Estimated Indicators)")
    st.caption(
        "These are simple, derived indicators based on text and keyword "
        "analysis — not scientifically validated psychological or "
        "recruiter scores."
    )

    total_job_skills = len(job_skills) if job_skills else 0
    matched_count = len(gap["matched"])

    r1, r2 = st.columns(2)
    with r1:
        st.write("Technical Skill Coverage")
        st.progress(skill_coverage)
        st.caption(f"{matched_count} / {total_job_skills} required skills detected")

    with r2:
        st.write("Job Relevance (Text Similarity)")
        st.progress(min(text_similarity, 1.0))
        st.caption(f"Cosine similarity score: {round(text_similarity, 3)}")

    st.divider()

    # -------------------------------------------------------------------
    # Recommendations
    # -------------------------------------------------------------------
    st.subheader("💡 Recommendations")
    recs = generate_recommendations(gap["missing"], overall_score, gap["matched"])
    for rec in recs:
        st.markdown(f"- {rec}")

    st.divider()

    # -------------------------------------------------------------------
    # Methodology / transparency section
    # -------------------------------------------------------------------
    with st.expander("ℹ️ How is this score calculated? (Methodology)"):
        st.markdown(
            f"""
**Overall Match Score = 60% × Text Similarity + 40% × Skill Coverage**

- **Text Similarity** is computed using TF-IDF (Term Frequency–Inverse
  Document Frequency) vectorization on the resume and job description,
  compared using cosine similarity. This measures how similar the
  overall *wording* of the two documents is.
- **Skill Coverage** is the fraction of job-required skills (from a
  {len(skills_list)}-skill reference dataset) that were detected in
  your resume, i.e. `matched_skills / job_required_skills`.
- Both weights are configurable constants in `matcher.py`.

**This score is a similarity/match indicator, not a prediction of
whether you will be hired.**
            """
        )


if __name__ == "__main__":
    main()

"""
matcher.py
----------
Core NLP matching logic:
1. TF-IDF vectorization of resume text vs. job description text.
2. Cosine similarity between the two TF-IDF vectors.
3. Skill-gap set operations (matched / missing / additional skills).
4. A transparent, adjustable overall match score.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# Weights for the overall match score. Change these two numbers to
# rebalance how much text-similarity vs. skill-coverage matters.
# They must add up to 1.0.
# ---------------------------------------------------------------------
TEXT_SIMILARITY_WEIGHT = 0.6
SKILL_COVERAGE_WEIGHT = 0.4


def compute_tfidf_similarity(resume_text, job_text):
    """
    Compute the cosine similarity between the TF-IDF vector of the
    resume text and the TF-IDF vector of the job description text.

    Parameters
    ----------
    resume_text : str
        Cleaned resume text (output of clean_text_for_tfidf).
    job_text : str
        Cleaned job description text (output of clean_text_for_tfidf).

    Returns
    -------
    tuple(float, str or None)
        (similarity_score_0_to_1, error_message)
        error_message is None on success.

    How it works
    ------------
    - TfidfVectorizer.fit_transform() is called on BOTH documents at
      once (a 2-document corpus), which:
        1. Builds a shared vocabulary from both texts.
        2. Computes Term Frequency (TF) for each word in each document.
        3. Computes Inverse Document Frequency (IDF) across the two
           documents, down-weighting words that appear in both
           (e.g., "the", "and") and up-weighting more distinctive
           words.
        4. Produces a TF-IDF vector for each document.
    - cosine_similarity() then measures the angle between the two
      resulting vectors. A score near 1.0 means the vectors point in
      almost the same direction (very similar wording); a score near
      0.0 means the texts share almost no distinctive vocabulary.
    """
    if not resume_text or not job_text:
        return 0.0, "Resume text or job description text is empty."

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])

        # If the combined vocabulary ended up empty (e.g., both texts
        # were only stop words or punctuation), guard against a crash.
        if tfidf_matrix.shape[1] == 0:
            return 0.0, "Not enough meaningful vocabulary to compare."

        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarity_score = float(similarity_matrix[0][0])
        return similarity_score, None

    except ValueError as exc:
        # TfidfVectorizer raises ValueError when the vocabulary is
        # empty after removing stop words (e.g., very short inputs).
        return 0.0, f"Could not compute text similarity: {exc}"


def get_skill_gap(resume_skills, job_skills):
    """
    Compare resume skills against job-required skills using set
    operations.

    Parameters
    ----------
    resume_skills : set[str]
    job_skills : set[str]

    Returns
    -------
    dict
        {
          "matched": set[str]    -> resume_skills ∩ job_skills
          "missing": set[str]    -> job_skills - resume_skills
          "additional": set[str] -> resume_skills - job_skills
        }

    Why sets?
    ---------
    Python sets give us clean, efficient, and readable set algebra:
    - Intersection (&) -> skills present in BOTH resume and job desc.
    - Difference (-)   -> skills present in one but not the other.
    This mirrors exactly how a human would compare two skill lists.
    """
    matched = resume_skills & job_skills
    missing = job_skills - resume_skills
    additional = resume_skills - job_skills

    return {"matched": matched, "missing": missing, "additional": additional}


def compute_skill_coverage(matched_skills, job_skills):
    """
    Compute what fraction of the job's required skills are present in
    the resume.

    Parameters
    ----------
    matched_skills : set[str]
    job_skills : set[str]

    Returns
    -------
    float
        A value between 0.0 and 1.0. Returns 0.0 if the job requires
        no detectable skills (to avoid division by zero).
    """
    if not job_skills:
        return 0.0
    return len(matched_skills) / len(job_skills)


def compute_overall_score(text_similarity, skill_coverage):
    """
    Combine text similarity and skill coverage into one transparent,
    adjustable overall match score.

    Formula
    -------
    overall_score = (TEXT_SIMILARITY_WEIGHT * text_similarity)
                   + (SKILL_COVERAGE_WEIGHT * skill_coverage)

    With the default weights (0.6 / 0.4):
    overall_score = 0.6 * text_similarity + 0.4 * skill_coverage

    Parameters
    ----------
    text_similarity : float
        Cosine similarity score (0.0 - 1.0) from compute_tfidf_similarity.
    skill_coverage : float
        Fraction (0.0 - 1.0) from compute_skill_coverage.

    Returns
    -------
    float
        Overall score as a percentage (0-100), rounded to 1 decimal.

    Why this design is transparent
    -------------------------------
    Both weights live as plain module-level constants at the top of
    this file, so anyone reading the code can see exactly how the
    final number is produced and can change the balance in one place
    without touching any other logic.
    """
    combined = (TEXT_SIMILARITY_WEIGHT * text_similarity) + (
        SKILL_COVERAGE_WEIGHT * skill_coverage
    )
    percentage = round(combined * 100, 1)
    return percentage

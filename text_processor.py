"""
text_processor.py
------------------
Basic NLP text preprocessing utilities shared by both the resume text
and the job description text.

Design decision (IMPORTANT):
We deliberately use a LIGHT preprocessing approach. Aggressive cleaning
(e.g., stripping all punctuation) would destroy technical tokens such as
"C++", "C#", ".NET", "Node.js", so we only remove characters that are
unlikely to appear in real skill names, and we keep the underlying raw
text (with case preserved) available for skill matching, while producing
a separately-cleaned version for TF-IDF.
"""

import re


def clean_text_for_tfidf(text):
    """
    Prepare text for TF-IDF vectorization.

    Parameters
    ----------
    text : str
        Raw text (resume or job description).

    Returns
    -------
    str
        Cleaned, lowercased text suitable for TfidfVectorizer.

    What this does and why
    -----------------------
    1. Lowercases everything -> "Python" and "python" should be treated
       as the same word for general similarity purposes.
    2. Normalizes whitespace (multiple spaces/newlines/tabs -> single
       space) -> PDFs often extract text with irregular spacing.
    3. Removes characters that are pure noise for a bag-of-words model
       (e.g., bullet symbols, stray punctuation) while KEEPING
       characters commonly found inside real technical terms such as
       '+', '#', '.', '-' (so "c++", "c#", "node.js" survive).
    """
    if not text:
        return ""

    text = text.lower()

    # Normalize all whitespace (newlines, tabs, multiple spaces) to a
    # single space.
    text = re.sub(r"\s+", " ", text)

    # Keep letters, numbers, spaces, and a small whitelist of symbols
    # that appear inside real technical terms: + # . - /
    # Everything else (bullets, quotes, brackets, etc.) is removed.
    text = re.sub(r"[^a-z0-9\+\#\.\-\/\s]", " ", text)

    # Collapse whitespace again after the character removal above.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_for_matching(text):
    """
    Normalize text specifically for skill-keyword matching.

    This is similar to clean_text_for_tfidf but we keep it as a
    separate function because skill matching and TF-IDF vectorization
    are conceptually different tasks, and future changes to one should
    not silently affect the other.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Lowercased text with normalized whitespace, safe for substring
        / regex-based skill searches.
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def basic_stats(text):
    """
    Compute simple descriptive statistics about a block of text.
    Used for input validation (e.g., "resume too short") and for the
    resume-strength section of the UI.

    Parameters
    ----------
    text : str

    Returns
    -------
    dict
        {"char_count": int, "word_count": int}
    """
    if not text:
        return {"char_count": 0, "word_count": 0}

    words = text.split()
    return {"char_count": len(text), "word_count": len(words)}

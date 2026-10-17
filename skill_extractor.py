"""
skill_extractor.py
-------------------
Loads the skills dataset and detects which skills are present in a
given piece of text (resume or job description).

This is RULE-BASED keyword/skill detection, not a trained machine
learning model. We are explicit about this because it is important to
be technically honest about which parts of the project are ML/NLP
techniques (TF-IDF, cosine similarity) and which parts are rule-based
logic (this file).
"""

import re
import pandas as pd


def load_skills_dataset(csv_path="data/skills.csv"):
    """
    Load the skills reference dataset from CSV.

    Parameters
    ----------
    csv_path : str
        Path to the skills.csv file.

    Returns
    -------
    list[str]
        A list of unique skill names (as written in the dataset).

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    """
    df = pd.read_csv(csv_path)

    if "skill" not in df.columns:
        raise ValueError("skills.csv must contain a 'skill' column.")

    skills = df["skill"].dropna().unique().tolist()
    return skills


def _build_skill_pattern(skill):
    """
    Build a compiled regex pattern that matches a skill name as a
    "whole term" inside a larger text, while being robust to skill
    names that contain special characters like '+', '#', '.'.

    Why not simply use \\b (word boundary)?
    Python's \\b only works reliably around alphanumeric characters.
    For a skill like "C++", a normal \\b...\\b pattern would fail
    because '+' is not a word character. So instead we manually check
    that whatever character comes immediately before/after the match
    is NOT a letter or digit (using negative lookaround), which works
    correctly for terms such as "C++", "C#", "Node.js", ".NET".

    Parameters
    ----------
    skill : str
        The skill name as it appears in skills.csv.

    Returns
    -------
    re.Pattern
        A compiled, case-insensitive regex pattern.
    """
    escaped = re.escape(skill.lower())

    # (?<![a-z0-9]) -> nothing alphanumeric immediately before
    # (?![a-z0-9])  -> nothing alphanumeric immediately after
    pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
    return re.compile(pattern, re.IGNORECASE)


def extract_skills(text, skills_list):
    """
    Detect which skills from `skills_list` appear inside `text`.

    Parameters
    ----------
    text : str
        Normalized text (resume or job description). Should already be
        passed through text_processor.normalize_for_matching().
    skills_list : list[str]
        The master list of known skills (from load_skills_dataset()).

    Returns
    -------
    set[str]
        The set of skills (using the ORIGINAL casing from skills.csv)
        found inside the text.

    How it works
    ------------
    For each known skill, we build a small regex that looks for that
    exact term as a standalone word/phrase inside the text (not as
    part of a longer unrelated word), then test it against the text.
    This is a simple but effective form of rule-based entity
    extraction, appropriate for a 2-day project scope.
    """
    if not text:
        return set()

    found = set()
    for skill in skills_list:
        pattern = _build_skill_pattern(skill)
        if pattern.search(text):
            found.add(skill)

    return found

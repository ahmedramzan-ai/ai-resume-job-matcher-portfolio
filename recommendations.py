"""
recommendations.py
-------------------
Rule-based recommendation engine. Given the missing skills and the
overall score, this module produces plain-language, honest suggestions
for the user. It never suggests fabricating skills the user does not
have.
"""


def generate_recommendations(missing_skills, overall_score, matched_skills):
    """
    Build a list of human-readable recommendation strings.

    Parameters
    ----------
    missing_skills : set[str] or list[str]
        Skills required by the job but not detected in the resume.
    overall_score : float
        Overall match score (0-100).
    matched_skills : set[str] or list[str]
        Skills already found in both the resume and the job
        description.

    Returns
    -------
    list[str]
        Ordered list of recommendation messages, ready to render in
        the Streamlit UI (e.g., one bullet point per string).

    Design notes
    ------------
    This is intentionally rule-based (if/else logic over the missing
    skills), NOT a trained ML model. It's a transparent decision
    system: given specific missing skills, or a specific score range,
    it returns a matching pre-written but skill-specific message.
    """
    recommendations = []
    missing_skills = sorted(missing_skills)

    if not missing_skills:
        recommendations.append(
            "Your resume already covers every skill mentioned in this "
            "job description. Consider adding measurable achievements "
            "(numbers, outcomes, project impact) to strengthen it further."
        )
    else:
        for skill in missing_skills:
            recommendations.append(
                f"Consider learning or gaining hands-on experience with "
                f"'{skill}', since it is mentioned in the job description "
                f"but was not detected in your resume."
            )

        recommendations.append(
            "If you already have experience with any of the skills above "
            "but they weren't detected, consider explicitly mentioning "
            "them in your resume (skill detection is based on the exact "
            "wording used)."
        )

    # Score-based general guidance
    if overall_score < 40:
        recommendations.append(
            "Your overall match with this job description is low. It may "
            "be worth targeting roles that better match your current "
            "skill set, or investing time in the most critical missing "
            "skills before applying."
        )
    elif overall_score < 70:
        recommendations.append(
            "Your overall match is moderate. Closing a few of the missing "
            "skill gaps above could meaningfully improve your fit for "
            "this role."
        )
    else:
        recommendations.append(
            "Your overall match is strong. Focus on tailoring your "
            "resume's wording to mirror key terms from the job "
            "description, which can also help with automated resume "
            "screening tools."
        )

    if matched_skills:
        recommendations.append(
            "Make sure your matched skills are backed by concrete "
            "examples (projects, internships, coursework) rather than "
            "just listed as keywords."
        )

    return recommendations

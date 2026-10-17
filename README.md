# AI Resume Job Matcher

🚀 **[Live Demo](https://ai-resume-job-matcher-abinaya.streamlit.app/)**

# AI Resume–Job Match & Skill Gap Analyzer

An NLP-based web application that analyzes how well a resume matches a
job or internship description — using TF-IDF text similarity, cosine
similarity, and rule-based skill-gap analysis, presented through a
clean Streamlit interface.

## Overview

Students applying for internships often don't know how well their
resume actually aligns with a specific job description, or which
skills they're missing. This project builds a lightweight NLP tool
that takes a resume (PDF) and a job description (text), and returns:

- An overall match score
- Matched, missing, and additional skills
- A text-similarity score based on TF-IDF + cosine similarity
- Resume strength indicators
- Personalized, honest improvement recommendations

## Problem Statement

Manually comparing a resume against a job description is subjective
and time-consuming. This project automates that comparison using
established NLP techniques, giving a transparent, explainable score
instead of a black-box judgment.

## Motivation

As a 3rd-year B.Tech (AI & Data Science) student, I wanted a project
that (a) genuinely applies core NLP/ML concepts, (b) solves a problem
I personally face during internship applications, and (c) is fully
explainable — every part of the scoring logic can be traced back to
readable code.

## Features

- Upload a resume as a PDF and extract its text automatically
- Paste any job/internship description
- Rule-based skill extraction against a 100+ skill reference dataset
- Skill-gap analysis: matched / missing / additional skills
- TF-IDF vectorization + cosine similarity for text-level similarity
- Transparent, adjustable weighted overall match score
- Resume strength indicators (skill coverage, job relevance)
- Rule-based, honest recommendation engine
- Clean, professional Streamlit UI with a full methodology explanation

## Technology Stack

| Category | Tools |
|---|---|
| Language | Python |
| Web App | Streamlit |
| PDF Parsing | pypdf |
| Data Handling | pandas |
| NLP / ML | scikit-learn (TF-IDF, cosine similarity) |
| Numerical | NumPy |

No paid APIs, no external LLM calls, no database — the app runs
entirely locally.

## Architecture

```
User
 |
 v
Streamlit UI
 |
 v
PDF Upload --> PDF Text Extraction (resume_parser.py)
 |
 v
Text Preprocessing (text_processor.py)
 |
 +----------------------------+
 |                            |
 v                            v
Skill Extraction         TF-IDF Vectorization
(skill_extractor.py)     (matcher.py)
 |                            |
 v                            v
Skill Gap Analysis       Cosine Similarity
 |                            |
 +-------------+--------------+
               v
      Overall Match Score
               |
               v
      Resume Strength Analysis
               |
               v
      Recommendations (recommendations.py)
               |
               v
           Streamlit UI
```

## How It Works

1. User uploads a resume PDF and pastes a job description.
2. `resume_parser.py` extracts raw text from every page of the PDF.
3. `text_processor.py` cleans and normalizes both texts.
4. `skill_extractor.py` detects known skills in both texts using a
   100+ skill reference dataset (`data/skills.csv`).
5. `matcher.py` computes:
   - TF-IDF vectors for both texts and their cosine similarity.
   - Skill-gap sets (matched / missing / additional).
   - A weighted overall match score.
6. `recommendations.py` produces rule-based, honest suggestions based
   on missing skills and the overall score.
7. `app.py` renders everything in a Streamlit interface.

## AI/NLP Methodology

- **TF-IDF (Term Frequency – Inverse Document Frequency):** converts
  text into numeric vectors, giving higher weight to distinctive words
  and lower weight to common words shared by both documents.
- **Cosine Similarity:** measures the angle between the resume's and
  job description's TF-IDF vectors, producing a 0–1 similarity score.
- **Skill Extraction:** rule-based keyword/entity detection against a
  curated skills dataset (not a trained ML model — this is explicitly
  stated for technical honesty).
- **Overall Score:** a transparent, adjustable weighted combination of
  text similarity (60%) and skill coverage (40%).

This is an NLP-based system using classical, explainable techniques —
**not** deep learning, not a large language model, and not a hiring
prediction system.

## Project Structure

```
AI-Resume-Job-Matcher/
│
├── app.py                  # Streamlit UI and orchestration
├── resume_parser.py        # PDF text extraction
├── text_processor.py       # Text cleaning / preprocessing
├── skill_extractor.py      # Skill detection from text
├── matcher.py               # TF-IDF, cosine similarity, scoring
├── recommendations.py      # Rule-based recommendation engine
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── skills.csv          # Reference skills dataset (118 skills)
│
├── sample/
│   ├── sample_job_description.txt
│   ├── sample_resume.txt
│   └── sample_resume.pdf
│
└── screenshots/
```

## Installation

```bash
# 1. Create the project folder and open it (or clone the repo)
git clone https://github.com/<your-username>/ai-resume-job-matcher.git
cd ai-resume-job-matcher

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

Streamlit will start a local server and open the app in your browser,
typically at `http://localhost:8501`.

## Example

1. Upload `sample/sample_resume.pdf`.
2. Paste the contents of `sample/sample_job_description.txt`.
3. Click **Analyze Match**.
4. You should see an overall score around **~49%**, several matched
   skills (Python, SQL, Pandas, Git, etc.), several missing skills
   (AWS, Docker, Django, PostgreSQL, etc.), and tailored recommendations.

## Limitations

- Skill detection depends entirely on the curated skills dataset —
  skills not in `data/skills.csv` will not be detected.
- TF-IDF is a lexical (word-overlap) technique, not a deep semantic
  one — it may miss synonyms (e.g., "ML" vs. "Machine Learning" unless
  both are explicitly listed).
- PDF extraction may fail or produce poor results on scanned/image-only
  resumes.
- The match score is a similarity indicator, not a prediction of
  hiring outcomes.
- Resume formatting (tables, columns, images) can affect how cleanly
  text is extracted.

## Future Improvements

- Sentence embeddings / transformer-based semantic similarity (e.g.,
  Sentence-BERT) for synonym-aware matching.
- A larger, community-maintained skill ontology.
- Named Entity Recognition for skill/role extraction.
- Resume section detection (Education, Projects, Experience).
- Comparison against multiple job descriptions at once.
- Cloud deployment with user accounts and saved history.



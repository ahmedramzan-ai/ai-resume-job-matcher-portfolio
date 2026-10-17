# Study Guide & Project Companion
### AI Resume–Job Match & Skill Gap Analyzer

This document covers everything beyond the source code: library
explanations, function-by-function breakdowns, concepts, testing,
GitHub/deployment steps, and interview preparation.

---

## 1. Libraries Used

### streamlit
- **Why:** Turns a Python script into a web app with almost no
  frontend code.
- **Key functions used:** `st.title`, `st.file_uploader`,
  `st.text_area`, `st.button`, `st.metric`, `st.progress`,
  `st.columns`, `st.expander`, `st.success`, `st.warning`, `st.error`,
  `st.cache_data`.
- **Example:**
  ```python
  import streamlit as st
  st.title("Hello")
  name = st.text_area("Your name")
  if st.button("Greet"):
      st.success(f"Hi {name}!")
  ```
- **Where it appears:** `app.py` — the entire UI.

### pypdf
- **Why:** Pure-Python library to read PDF files without external
  dependencies (like Poppler).
- **Key classes/functions:** `PdfReader(file)`, `reader.pages`,
  `page.extract_text()`.
- **Example:**
  ```python
  from pypdf import PdfReader
  reader = PdfReader("resume.pdf")
  text = reader.pages[0].extract_text()
  ```
- **Where it appears:** `resume_parser.py`.

### pandas
- **Why:** Reads and manipulates the tabular skills dataset.
- **Key functions:** `pd.read_csv()`, `.dropna()`, `.unique()`,
  `.tolist()`.
- **Example:**
  ```python
  import pandas as pd
  df = pd.read_csv("data/skills.csv")
  print(df["skill"].unique())
  ```
- **Where it appears:** `skill_extractor.py`.

### numpy
- **Why:** scikit-learn returns TF-IDF matrices and similarity results
  as NumPy arrays under the hood; NumPy provides the numerical
  foundation.
- **Key concepts:** arrays, indexing (`matrix[0][0]`).
- **Where it appears:** indirectly, via `matcher.py`'s use of
  scikit-learn outputs.

### scikit-learn
- **Why:** Provides ready-made, well-tested implementations of TF-IDF
  and cosine similarity.
- **Key classes/functions:** `TfidfVectorizer`, `.fit_transform()`,
  `cosine_similarity()`.
- **Example:**
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.metrics.pairwise import cosine_similarity

  docs = ["python developer with sql", "python and sql expert"]
  vectors = TfidfVectorizer().fit_transform(docs)
  print(cosine_similarity(vectors[0:1], vectors[1:2]))
  ```
- **Where it appears:** `matcher.py`.

### re (regular expressions, standard library)
- **Why:** Used for whitespace normalization, character filtering, and
  building robust skill-matching patterns (handling terms like "C++").
- **Where it appears:** `text_processor.py`, `skill_extractor.py`.

---

## 2. Function-by-Function Explanation

| Function | File | Purpose | Input | Output |
|---|---|---|---|---|
| `extract_text_from_pdf()` | resume_parser.py | Extract text from an uploaded PDF | Uploaded file object | `(text, error_message)` |
| `clean_text_for_tfidf()` | text_processor.py | Lowercase & normalize text for TF-IDF, preserving symbols like `+ # . -` | Raw text | Cleaned text |
| `normalize_for_matching()` | text_processor.py | Lowercase + whitespace-normalize text for skill matching | Raw text | Normalized text |
| `basic_stats()` | text_processor.py | Compute char/word counts for validation | Text | `{"char_count", "word_count"}` dict |
| `load_skills_dataset()` | skill_extractor.py | Load skills.csv into a Python list | CSV path | List of skill names |
| `_build_skill_pattern()` | skill_extractor.py | Build a regex that matches a skill as a standalone term | Skill string | Compiled regex |
| `extract_skills()` | skill_extractor.py | Detect which known skills appear in a text | Text, skills list | Set of found skills |
| `compute_tfidf_similarity()` | matcher.py | Vectorize both texts and compute cosine similarity | Resume text, job text | `(similarity_score, error)` |
| `get_skill_gap()` | matcher.py | Set-based comparison of resume vs. job skills | Two sets | `{matched, missing, additional}` |
| `compute_skill_coverage()` | matcher.py | Fraction of job skills present in resume | Matched set, job set | Float 0–1 |
| `compute_overall_score()` | matcher.py | Weighted combination of similarity + coverage | Two floats | Percentage (0–100) |
| `generate_recommendations()` | recommendations.py | Build rule-based improvement suggestions | Missing skills, score, matched skills | List of strings |
| `main()` | app.py | Orchestrates the entire Streamlit UI and pipeline | — | Renders the app |

---

## 3. Python Concepts Used in This Project

- **Functions** — every processing step is its own function (see table above).
- **Dictionaries** — e.g., `get_skill_gap()` returns `{"matched": ..., "missing": ..., "additional": ...}`.
- **Sets** — `resume_skills & job_skills` (intersection), `job_skills - resume_skills` (difference).
- **Loops** — iterating over PDF pages, over skills in `extract_skills()`.
- **Conditionals** — input validation (`if uploaded_resume is None: ...`).
- **Exception handling** — `try/except` in `extract_text_from_pdf()` and `compute_tfidf_similarity()`.
- **File handling** — `open()`, `pd.read_csv()`.
- **Modules/imports** — each `.py` file is a module imported by `app.py`.
- **String processing** — `re.sub()`, `.lower()`, `.strip()`, `.split()`.
- **List comprehensions** — implicitly used inside pandas operations like `.tolist()`.

---

## 4. AI/ML/NLP Concepts Used

1. **NLP (Natural Language Processing):** the general field of getting
   computers to work with human language — applied here to compare
   two documents.
2. **Text Preprocessing:** lowercasing, whitespace normalization —
   makes text consistent before further processing.
3. **Feature Extraction / Vectorization:** turning text into numbers
   (TF-IDF vectors) so mathematical comparison is possible.
4. **TF-IDF:** weighs words by how important/distinctive they are
   across the two documents being compared.
5. **Cosine Similarity:** measures how similar two vectors' directions
   are — used to score document similarity.
6. **Skill Extraction:** rule-based keyword/entity detection against a
   reference skill list (explicitly NOT a trained ML classifier).
7. **Recommendation Logic:** rule-based (if/else) suggestions driven
   by the missing-skill set and score range.

---

## 5. TF-IDF and Cosine Similarity, Explained Simply

**Tiny TF-IDF example:**
Two documents: `"python developer"` and `"python sql developer"`.
Both contain "python" and "developer" — common words become less
distinctive. "sql" only appears in the second document — it becomes
more distinctive (higher weight).

- **TF (Term Frequency):** how often a word appears in one document.
- **IDF (Inverse Document Frequency):** how rare a word is across all
  documents being compared — rare words get boosted, common words get
  suppressed.
- **TF-IDF = TF × IDF:** balances "how often" against "how unique."

**Cosine similarity:**
Each document becomes a vector (a list of numbers, one per word).
Cosine similarity measures the angle between two vectors:
- **Score near 1.0** → vectors point almost the same direction → very
  similar word usage.
- **Score near 0.0** → vectors point in very different directions →
  little shared distinctive vocabulary.
- **Limitation:** it's purely lexical — it doesn't understand
  meaning/synonyms unless the exact words overlap.

**Why cosine similarity instead of Euclidean distance?**
Cosine similarity ignores document length and focuses on the
*direction* (word-usage pattern) of the vectors, which matters more
than raw magnitude for text — a long resume and a short one can still
be very similar in topic.

---

## 6. Testing

| # | Test | Input | Expected Output | Feature Tested |
|---|---|---|---|---|
| 1 | Highly matching resume | Resume with most job skills present | High overall score (>70%) | End-to-end scoring |
| 2 | Partially matching resume | Resume with ~half the required skills | Medium score, several missing skills listed | Skill-gap logic |
| 3 | Unrelated resume | Resume from an unrelated field (e.g., all marketing skills) | Low score (<30%) | Discriminative power of TF-IDF + skill coverage |
| 4 | Resume with no detectable skills | Resume text containing no dataset skills | Friendly warning, skill coverage = 0 | Zero-skill edge case handling |
| 5 | Invalid/empty PDF | Corrupted or blank PDF uploaded | Friendly error message, no crash | PDF error handling |

Additional edge cases handled in code: empty job description, resume/job
text under the minimum word threshold, empty TF-IDF vocabulary (e.g.,
inputs consisting only of stop words).

This project was verified end-to-end using the sample resume and
sample job description included in `sample/`, producing a ~49% overall
match with 15 matched skills and 11 missing skills — confirming the
pipeline runs correctly from PDF upload through to the final
recommendations.

---

## 7. GitHub Setup

```bash
# Initialize the repository (run inside the project folder)
git init

# Stage all files
git add .

# Create the first commit
git commit -m "Initial commit: AI Resume-Job Match & Skill Gap Analyzer"

# Rename the default branch (optional but common practice)
git branch -M main

# Link to your GitHub repository (create it first on github.com,
# name suggestion: ai-resume-job-matcher)
git remote add origin https://github.com/<your-username>/ai-resume-job-matcher.git

# Push
git push -u origin main
```

**Do NOT upload:** `venv/`, `__pycache__/`, any `.env` or secrets
files, temporary files. These are already excluded via `.gitignore`.

---

## 8. Deployment (Streamlit Community Cloud)

1. Push your project to a public GitHub repository (see above).
2. Make sure `requirements.txt` is at the repo root — this tells the
   hosting platform exactly which packages to install.
3. Go to Streamlit's community cloud hosting site and sign in with
   GitHub.
4. Choose "New app," select your repository, branch (`main`), and the
   entry-point file (`app.py`).
5. Click deploy. The platform installs `requirements.txt` and starts
   your app; this typically takes a couple of minutes.
6. Once deployed, open the generated URL and re-run the sample test
   (upload `sample/sample_resume.pdf`, paste the sample job
   description) to confirm it works the same as locally.

Note: hosting platforms update their UI over time, so exact button
labels/menus may differ from this description — the overall flow
(connect GitHub → select repo/file → deploy) stays consistent.

---

## 9. Screenshots for GitHub

Take these screenshots and place them in `screenshots/`:

1. **Home/upload page** — before analysis, showing the upload and job
   description inputs.
2. **Job description input** — the text area filled in with a sample
   job description.
3. **Match score** — the results header showing overall score, text
   similarity, and skill coverage metrics.
4. **Matched and missing skills** — the skill-gap section with all
   three columns visible.
5. **Recommendations/methodology** — the recommendations list and the
   expanded methodology section.

Reference them in your README under a "Screenshots" section using
standard Markdown image syntax.

---

## 10. Resume Entry

**Project Title:** AI Resume–Job Match & Skill Gap Analyzer
**Technology:** Python, Streamlit, NLP, Scikit-learn, TF-IDF, Cosine
Similarity, PDF Processing

- Developed an NLP-based resume-job matching application that extracts
  text from PDF resumes and computes similarity against job
  descriptions using TF-IDF vectorization and cosine similarity.
- Designed a rule-based skill-extraction and skill-gap analysis system
  covering 100+ technical skills, identifying matched, missing, and
  additional skills between a resume and a job description.
- Built a transparent, weighted scoring formula and an interactive
  Streamlit interface, including a documented methodology section
  explaining exactly how each score is calculated.

---

## 11. LinkedIn/GitHub Project Description

> An NLP-based web app that analyzes how well a resume matches a job
> description using TF-IDF and cosine similarity, plus rule-based
> skill-gap detection across 100+ technical skills. Built with Python,
> scikit-learn, and Streamlit — fully transparent scoring, no external
> APIs required.

---

## 12. 1-Minute Interview Explanation

"So for this project, I built a tool that compares a resume against a
job description and tells you how well they match. The problem I was
trying to solve is that students, including me, often don't know
exactly why a resume doesn't get shortlisted, or which specific skills
are missing for a particular role. I used Python and Streamlit for the
interface, and for the actual matching, I used two techniques: TF-IDF
with cosine similarity from scikit-learn to measure how similar the
wording of the resume and job description are, and a rule-based skill
extractor that checks for about 118 known technical skills in a
dataset I built. The app shows an overall match score, which skills
are matched, missing, or extra, and gives suggestions based on what's
missing. It's intentionally not deep learning or an LLM — it's
classical NLP, which made it explainable and realistic to build in a
couple of days. If I extended it, I'd add sentence embeddings so it
can catch synonyms like 'ML' and 'Machine Learning' as the same skill."

---

## 13. 2-Minute Interview Explanation

"The architecture is fairly simple end to end. The user uploads a PDF
resume and pastes a job description into a Streamlit app. First,
`pypdf` extracts the raw text from every page of the resume — I loop
through all pages because resumes can be more than one page, and I
handle the case where a PDF is scanned/image-only and has no
extractable text.

Next, I clean and normalize both texts — lowercasing, fixing
whitespace, but I'm careful not to strip out characters that appear
inside real skill names like 'C++' or 'Node.js'.

For skill extraction, I built a CSV dataset of about 118 skills across
categories like programming, web, databases, AI, cloud, and DevOps.
I search for each skill as a whole term inside both texts using
regex with custom boundary checks, since Python's normal word-boundary
regex doesn't handle symbols like '+' or '#' correctly. Then I use
basic Python set operations — intersection for matched skills,
difference for missing and additional skills.

Separately, I run both cleaned texts through scikit-learn's
`TfidfVectorizer`, which converts them into numeric vectors weighted
by how distinctive each word is, and then I compute cosine similarity
between those two vectors to get a text-similarity score.

Finally, I combine text similarity (60% weight) and skill coverage
(40% weight) into one overall score — both weights are just constants
at the top of `matcher.py`, so they're easy to tune. A separate
rule-based recommendation function turns the missing skills into
specific, honest suggestions, and everything is displayed through
Streamlit with metrics, progress bars, and an expandable methodology
section so the scoring is never a black box."

---

## 14. Interview Questions & Answers

### A. Project Questions
1. **Why did you choose this project?**
   It's a practical problem I face myself (internship applications), and it let me apply real NLP/ML concepts rather than build a generic CRUD app.
2. **What problem does it solve?**
   It quantifies how well a resume matches a specific job description and highlights concrete skill gaps, instead of relying on guesswork.
3. **Where exactly is AI used?**
   TF-IDF vectorization and cosine similarity (from scikit-learn) for text similarity; skill extraction is rule-based, not ML.
4. **Is this really machine learning?**
   TF-IDF and cosine similarity are classical NLP/ML techniques for text representation and comparison; there's no trained/learned model with parameters fit on labeled data — I'm upfront that skill extraction is rule-based, not ML.
5. **What is the overall architecture?**
   PDF upload → text extraction → preprocessing → (skill extraction + TF-IDF/cosine similarity in parallel) → skill gap + similarity score → weighted overall score → recommendations → UI.
6. **Why is the score not a hiring probability?**
   It only measures textual/keyword similarity, not experience quality, soft skills, interview performance, or company-specific criteria — calling it a hiring probability would be misleading.

### B. Python Questions
7. **Why did you separate the code into multiple files?**
   Separation of concerns — PDF parsing, text processing, skill logic, matching, and recommendations are independent responsibilities, easier to test and reason about separately.
8. **How do you handle errors in this project?**
   `try/except` blocks around PDF reading and TF-IDF computation, plus input validation (empty text, missing upload) with user-facing messages via Streamlit.
9. **What Python data structure did you use for skill comparison, and why?**
   Sets — because intersection/difference operations map directly to "matched," "missing," and "additional" skills, and are efficient and readable.
10. **What does `set_a - set_b` mean in Python?**
    Elements in `set_a` that are NOT in `set_b` — used for `job_skills - resume_skills` to get missing skills.

### C. NLP Questions
11. **What is NLP?**
    The field of enabling computers to process and analyze human language.
12. **What is text preprocessing and why is it needed?**
    Cleaning/normalizing text (lowercasing, whitespace handling) so that comparisons aren't thrown off by superficial formatting differences.
13. **Why did you avoid removing all punctuation?**
    Because technical terms like "C++", "C#", and "Node.js" rely on punctuation to be meaningful — removing it would merge them into unrelated words.
14. **What is a "bag of words" style representation?**
    Representing text as counts/weights of individual words, ignoring grammar and word order — the foundation TF-IDF builds on.
15. **What happens if the resume uses a synonym the dataset doesn't recognize?**
    It won't be detected as a skill match — this is a known limitation of rule-based keyword matching, addressed in the Limitations section.

### D. Machine Learning Questions
16. **What is TF-IDF?**
    A numerical text representation that weighs each word by how frequently it appears in a document (TF) versus how rare it is across the compared documents (IDF).
17. **What is TF?**
    Term Frequency — how often a word appears within a single document.
18. **What is IDF?**
    Inverse Document Frequency — a measure of how rare/distinctive a word is across the whole set of documents being compared; common words get a lower IDF weight.
19. **What is cosine similarity?**
    A measure of the angle between two vectors, used to quantify how similar their directions (and therefore, content patterns) are, regardless of magnitude.
20. **Why cosine similarity instead of Euclidean distance?**
    Cosine similarity ignores document length differences and focuses on word-usage pattern/direction, which better reflects topical similarity for text.
21. **Is skill extraction supervised or unsupervised learning?**
    Neither — it's rule-based pattern matching against a fixed reference list, with no training process involved.

### E. Scikit-learn Questions
22. **What does `TfidfVectorizer.fit_transform()` do?**
    Learns the vocabulary and IDF weights from the input documents, then transforms them into a TF-IDF matrix in one step.
23. **What does `cosine_similarity()` return?**
    A matrix of pairwise similarity scores between the input vectors; for two documents, a single similarity score between 0 and 1.
24. **Why pass `stop_words="english"` to TfidfVectorizer?**
    To automatically ignore common English words (like "the," "and") that carry little distinguishing meaning for similarity comparison.

### F. Streamlit Questions
25. **Why did you choose Streamlit over Flask/Django for the UI?**
    Streamlit lets you build an interactive data app with pure Python, no separate frontend/HTML/CSS/JS needed — ideal for a fast, focused NLP/data project.
26. **What does `st.cache_data` do, and why did you use it on the skills loader?**
    It caches the function's return value so the skills CSV is only read from disk once per session instead of on every button click/rerun.
27. **How does Streamlit handle re-running the script on interaction?**
    Streamlit re-runs the whole script top-to-bottom on every UI interaction (like a button click), which is why caching and clear conditional logic (`if analyze_clicked:`) matter.

### G. Git/GitHub Questions
28. **Why use Git for this project?**
    Version control, backup, and a professional way to showcase the project and its history to recruiters.
29. **What's the difference between `git add` and `git commit`?**
    `git add` stages changes for the next commit; `git commit` actually records those staged changes as a snapshot in the repository's history.
30. **Why is a `.gitignore` file important here?**
    It prevents unnecessary or sensitive files (virtual environments, cache files, secrets) from being committed to the repository.

### H. Improvement/Limitations Questions
31. **What are the limitations of this project?**
    Dataset-dependent skill detection, lexical-only similarity (no synonym understanding), PDF extraction issues on scanned resumes, and the score being an indicator rather than a hiring prediction.
32. **What happens with scanned PDFs?**
    `pypdf` can't extract text from image-only pages, so the app detects this and shows a friendly error rather than crashing or silently failing.
33. **How would you improve this project?**
    Add sentence embeddings (e.g., Sentence-BERT) for semantic/synonym-aware similarity, expand the skill ontology, and add resume-section detection.
34. **How would you scale it?**
    Move skill matching to a more efficient data structure (e.g., a trie or Aho-Corasick automaton) for large skill lists, and cache TF-IDF computations for repeated job descriptions.
35. **How would you use BERT/embeddings in version 2?**
    Replace or supplement TF-IDF vectors with sentence embeddings from a pretrained transformer model, then compute cosine similarity in embedding space, which captures meaning rather than exact word overlap.

---

## 15. Explain Like I'm a Beginner

Imagine you have a resume and a job description, both as plain text.

1. **Resume → text:** The computer reads the PDF and pulls out just the words, ignoring fonts and layout.
2. **Text → important words:** Both texts are cleaned up (same case, tidy spacing) so the computer isn't confused by formatting differences.
3. **Words → numbers:** Since computers can't compare "words" directly, TF-IDF turns each document into a list of numbers, where common words (like "the") get small numbers and distinctive words (like "Django") get bigger numbers.
4. **Numbers → similarity:** Cosine similarity looks at these two number-lists and measures how closely they point in the same direction — the more alike the important words, the higher the score.
5. **Skills → matched/missing:** Separately, the computer checks a big list of known skills against both texts and figures out which ones appear in both (matched), only in the job (missing), or only in the resume (additional).
6. **Results → recommendations:** Finally, the missing skills and the overall score are turned into plain-English suggestions — like "consider learning Docker" — so you know exactly what to work on next.

---

## 16. 2-Day Development Plan

**Day 1**
- *Morning:* Project setup, folder structure, install libraries, plan file responsibilities.
- *Afternoon:* Implement PDF extraction, text preprocessing, and skill extraction + dataset.
- *Evening:* Implement TF-IDF similarity, skill-gap logic, and the overall scoring formula.

**Day 2**
- *Morning:* Build the Streamlit UI end-to-end (upload, inputs, results, metrics).
- *Afternoon:* Testing (all 5+ test cases), error handling, polish UI/UX.
- *Evening:* GitHub setup, README, deployment, resume bullets, and interview prep.

**Priority order if time is short:** working PDF extraction → skill extraction/gap → TF-IDF similarity → overall score → basic UI → error handling → polish/README/deployment.

---

## 17. "How Much Did You Build?" — Honest Answer

"I built the entire project myself — the PDF extraction, text
preprocessing, skill-extraction logic, the TF-IDF/cosine-similarity
matching, the scoring formula, the recommendation engine, and the
Streamlit interface. I understand exactly how each part works: why
TF-IDF and cosine similarity are appropriate for comparing text, how
Python sets give me matched/missing/additional skills, and why I
designed the skill matching to be rule-based rather than pretending it
was a trained ML model. I can walk through and modify any file in the
project."

---

## 18. Final Project Checklist

- [ ] Python installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Folder structure created
- [ ] Dataset created (`data/skills.csv`)
- [ ] PDF extraction working
- [ ] Text preprocessing working
- [ ] Skill extraction working
- [ ] TF-IDF working
- [ ] Cosine similarity working
- [ ] Match score working
- [ ] Skill-gap analysis working
- [ ] Recommendations working
- [ ] Streamlit UI working
- [ ] Error handling tested
- [ ] 5 test cases completed
- [ ] GitHub repository created
- [ ] README added
- [ ] Screenshots added
- [ ] Project deployed
- [ ] Resume entry added
- [ ] Interview questions studied

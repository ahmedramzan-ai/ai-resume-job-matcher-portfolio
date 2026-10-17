"""
resume_parser.py
-----------------
Handles extraction of raw text from an uploaded PDF resume.

This module is intentionally kept separate from the rest of the app so
that the "PDF handling" concern is isolated from text-processing and
matching logic. This is a basic example of separation of concerns.
"""

from pypdf import PdfReader
import io


def extract_text_from_pdf(uploaded_file):
    """
    Extract raw text from an uploaded PDF file (as provided by
    Streamlit's st.file_uploader).

    Parameters
    ----------
    uploaded_file : UploadedFile (Streamlit) or a file-like object
        The PDF file the user uploaded through the Streamlit UI.

    Returns
    -------
    tuple(str, str or None)
        (extracted_text, error_message)
        - extracted_text: the combined text from all pages ("" if none).
        - error_message: None if extraction succeeded, otherwise a
          human-readable string explaining what went wrong.

    Notes
    -----
    - We read the uploaded file into an in-memory buffer (io.BytesIO)
      because Streamlit's UploadedFile object behaves like a file, but
      PdfReader is more reliable when given a real BytesIO stream.
    - We loop through every page because a resume can be more than one
      page long; we must combine the text from ALL pages.
    - We wrap everything in a try/except so a corrupted or unusual PDF
      does not crash the whole Streamlit app.
    """
    if uploaded_file is None:
        return "", "No file was uploaded."

    try:
        # Read the uploaded file's bytes into memory
        file_bytes = uploaded_file.read()

        if not file_bytes:
            return "", "The uploaded file is empty."

        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)

        if len(reader.pages) == 0:
            return "", "The PDF has no pages."

        extracted_pages = []
        for page in reader.pages:
            # extract_text() can return None for image-only / scanned
            # pages, so we guard against that with "or ''"
            page_text = page.extract_text() or ""
            extracted_pages.append(page_text)

        full_text = "\n".join(extracted_pages).strip()

        if not full_text:
            return "", (
                "No readable text could be extracted from this PDF. "
                "It may be a scanned/image-only document. Please upload "
                "a text-based PDF (e.g., exported directly from Word)."
            )

        return full_text, None

    except Exception as exc:
        # Catch-all: malformed PDFs, encrypted PDFs, unexpected pypdf
        # errors, etc. We never let this crash the Streamlit app.
        return "", f"Could not read the PDF file. Error: {exc}"

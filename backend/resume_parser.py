import io
from pathlib import Path

import PyPDF2
from docx import Document


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_text_from_pdf(file_data):
    """
    Extract text from a PDF file.

    file_data can be:
    - Streamlit UploadedFile
    - bytes
    - file-like object
    """

    if hasattr(file_data, "getvalue"):
        file_data = file_data.getvalue()

    if isinstance(file_data, bytes):
        file_data = io.BytesIO(file_data)

    reader = PyPDF2.PdfReader(file_data)

    text = []

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text).strip()


# ============================================================
# EXTRACT TEXT FROM DOCX
# ============================================================

def extract_text_from_docx(file_data):
    """
    Extract text from a DOCX file.

    file_data can be:
    - Streamlit UploadedFile
    - bytes
    - file-like object
    """

    if hasattr(file_data, "getvalue"):
        file_data = file_data.getvalue()

    if isinstance(file_data, bytes):
        file_data = io.BytesIO(file_data)

    document = Document(file_data)

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs).strip()


# ============================================================
# EXTRACT TEXT FROM TXT
# ============================================================

def extract_text_from_txt(file_data):
    """
    Extract text from a TXT file.
    """

    if hasattr(file_data, "getvalue"):
        file_data = file_data.getvalue()

    if isinstance(file_data, bytes):
        return file_data.decode("utf-8", errors="ignore").strip()

    if hasattr(file_data, "read"):
        content = file_data.read()

        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore").strip()

        return str(content).strip()

    return str(file_data).strip()


# ============================================================
# MAIN FILE EXTRACTION FUNCTION
# ============================================================

def extract_text_from_file(uploaded_file):
    """
    Automatically detects the uploaded file type
    and extracts its text.

    Supported:
    - PDF
    - DOCX
    - TXT
    """

    if uploaded_file is None:
        raise ValueError("No file was provided.")

    # --------------------------------------------------------
    # Get filename
    # --------------------------------------------------------

    if hasattr(uploaded_file, "name"):
        filename = uploaded_file.name
    else:
        filename = str(uploaded_file)

    extension = Path(filename).suffix.lower()


    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if extension == ".pdf":

        text = extract_text_from_pdf(uploaded_file)

        if not text:
            raise ValueError(
                "The PDF was opened successfully, but no text "
                "could be extracted. The PDF may contain scanned images."
            )

        return text


    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    elif extension == ".docx":

        text = extract_text_from_docx(uploaded_file)

        if not text:
            raise ValueError(
                "The DOCX file was opened successfully, "
                "but no text was found."
            )

        return text


    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    elif extension == ".txt":

        text = extract_text_from_txt(uploaded_file)

        if not text:
            raise ValueError(
                "The TXT file is empty."
            )

        return text


    # --------------------------------------------------------
    # UNSUPPORTED FILE
    # --------------------------------------------------------

    else:

        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Please upload a PDF, DOCX, or TXT file."
        )


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RESUME PARSER TEST")
    print("=" * 60)

    print("Resume parser loaded successfully.")

    print("\nAvailable functions:")
    print("- extract_text_from_pdf()")
    print("- extract_text_from_docx()")
    print("- extract_text_from_txt()")
    print("- extract_text_from_file()")

    print("\nSupported formats:")
    print("- PDF")
    print("- DOCX")
    print("- TXT")

    print("\nParser is ready.")
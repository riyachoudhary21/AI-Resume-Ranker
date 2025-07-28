import fitz   # PyMuPDF

def extract_text_from_pdf(file):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc =fitz.open(stream=file.read(), filetype="pdf")
        return " ".join([page.get_text() for page in doc])
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None
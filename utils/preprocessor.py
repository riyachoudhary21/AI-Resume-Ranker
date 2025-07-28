import re
import spacy

nlp =spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Clean and lemmatize text"""
    text =re.sub(r'[^\w\s]', '', text)  # Remove special chars
    text =re.sub(r'\s+', ' ', text).strip().lower()
    doc= nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
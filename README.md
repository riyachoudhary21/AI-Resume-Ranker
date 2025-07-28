# AI Resume Ranker 

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)]

---

## Live Demo
[**Click here to try it out**](https://ai-resume-ranker-8146.streamlit.app/)

[![Open in Streamlit](https://img.shields.io/badge/Streamlit-Open_App-FF4B4B?style=for-the-badge&logo=streamlit)](https://ai-resume-ranker-8146.streamlit.app/)


## Project Overview
This is an **AI-powered resume ranking application** designed to help recruiters and hiring managers quickly shortlist candidates.  
It leverages **Natural Language Processing (NLP)** and **semantic similarity models** to evaluate resumes against a given job description and generate an **automated ranking**.
The tool is built to reduce **manual resume screening time**, ensure **fair and unbiased matching**, and improve overall **recruitment efficiency**.

---

## Objective
The primary goals of this project are:
- Automate the resume shortlisting process using intelligent algorithms.
- Provide a **data-driven** and **fair** evaluation of candidate profiles.
- Eliminate manual effort and accelerate the hiring process.

---

## Technologies & Tools Used
**Language:** Python  
**Frameworks & Libraries:**
- **Streamlit** – Interactive Web Interface
- **PyMuPDF (fitz)** – PDF Resume Text Extraction
- **SpaCy** – NLP Text Preprocessing & Lemmatization
- **Sentence Transformers (all-MiniLM-L6-v2)** – Semantic Similarity Scoring
- **Scikit-learn** – Cosine Similarity Calculations

---

## How It Works
### 1. Resume Text Extraction
- Extract text content from PDF resumes using **PyMuPDF**.

### 2. Text Preprocessing
- Remove stopwords, numbers, and special characters.
- Apply **lemmatization** using **SpaCy** for clean tokenized text.

### 3. Scoring Logic
- Use **Sentence-BERT embeddings** to compute semantic similarity between job description and resumes.
- Apply **domain-specific keyword boosting** (7 domains supported)
- Apply **section-based weights**:  
  - Skills – 50%  
  - Experience – 30%  
  - Projects – 15%  
  - Education – 5%
- Implements **score normalization** (0-100% scale)
  
### 4. Ranking
- Normalize all scores and generate a ranked list of resumes based on job description relevance.
- Dynamic score scaling (top resume=100%, bottom=40%)

---

## Features 
- Upload multiple resumes (PDF format supported)
- Paste job descriptions for ranking
- Automated NLP pipeline for text cleaning and processing
- Domain detection (ML, Web Dev, Data Science, etc.)
- Keyword-enhanced scoring
- Real-time ranking dashboard with **Streamlit**
- Download ranked results in CSV format

---

## Setup for Testing
1. Install: `pip install -r requirements.txt`
2. Download SpaCy model:  
   `python -m spacy download en_core_web_sm`
3. Run: `streamlit run app.py`

---

## Responsibilities
I was responsible for:
- Designing and building the **NLP pipeline** for text extraction and preprocessing
- Implementing the **resume ranking algorithm** with semantic similarity
- Developing the **Streamlit-based web interface**
- Optimizing semantic similarity scoring for better accuracy
- Ensuring efficient handling of multiple resume files

---

## Outcomes & Impact
- Automated resume ranking for multiple job roles  
- Achieved **high accuracy** in identifying relevant resumes compared to manual shortlisting  
- Reduced recruitment time and improved HR decision-making  
- Strengthened skills in **Python, NLP, and machine learning pipelines**


---

## Author
Riya Choudhary – 28 July 2025

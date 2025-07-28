import spacy
nlp = spacy.load("en_core_web_sm")
    
import streamlit as st
import pandas as pd
import time
from utils.extractor import extract_text_from_pdf
from utils.preprocessor import preprocess_text
from utils.scorer import calculate_score

def main():
    st.set_page_config(page_title="AI Resume Ranker",layout="wide")

    # Enhanced CSS
    st.markdown("""
        <style>
            @keyframes gradient {
                0% { background-position:0% 50%; }
                50% { background-position:100% 50%; }
                100% { background-position:0% 50%; }
            }
            .stApp {
                background: linear-gradient(-45deg, #3a0ca3, #4361ee, #4cc9f0);
                background-size:400% 400%;
                animation: gradient 15s ease infinite;
                color:white;
            }
            .big-title {
                font-size:40px !important;
                text-align: center !important;
                color: white !important;
                font-weight:bold !important;
                margin-bottom: 30px !important;
            }
            .stTextArea textarea {
                background-color: rgba(30,30,30,0.7) !important;
                color: white !important;
                border: 1px solid #5f5f5f !important;
                border-radius: 8px;
                padding: 12px !important;
                font-size: 16px !important;
            }
            .stFileUploader {
                border: 2px dashed #4cc9f0 !important;
                border-radius: 8px;
                padding: 20px;
                background-color: rgba(255,255,255,0.05) !important;
                font-size: 16px !important;
            }
            .stButton>button {
                background-color: #7209b7 !important;
                color: white !important;
                font-weight: bold !important;
                border-radius: 8px !important;
                padding: 10px 24px !important;
                border: none !important;
                width: 100% !important;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #560bad !important;
                transform: scale(1.03);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .stDataFrame {
                background-color: rgba(30,30,30,0.7) !important;
                border-radius: 8px !important;
            }
            a { color: #FFEA00 !important; text-decoration: underline; }
            a:hover { color: white !important; }
        </style>
    """, unsafe_allow_html=True)

    # Heading 
    st.markdown("<h1 class='big-title'>AI-Powered Resume Ranker</h1>", unsafe_allow_html=True)

    # Job description input
    st.markdown("<h4 style='font-size:24px; color:white; font-weight:600;'>✍️ Paste Job Description</h4>", unsafe_allow_html=True)
    jd_text =st.text_area("Job description", height=200, label_visibility="collapsed")  # empty label because custom label is above

    # Resume Upload 
    st.markdown("<h4 style='font-size:24px; color:white; font-weight:600;'>📄 Upload Resumes (PDF only)</h4>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Resume PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files:
        st.success(f"**{len(uploaded_files)} resumes uploaded!**")

    # Rank Button
    if st.button("🔍 **Rank Resumes**") and jd_text and uploaded_files:
        with st.spinner("Analyzing resumes..."):
            start_time = time.time()
            results =[]

            #  Score
            for file in uploaded_files:
                text= extract_text_from_pdf(file)
                if text:
                    raw_score= calculate_score(preprocess_text(jd_text), text)
                    results.append({"Filename": file.name, "RawScore": raw_score})

            if results:
                # Sort by calculated score
                df =pd.DataFrame(results).sort_values("RawScore", ascending=False).reset_index(drop=True)

                # first = 100, last = 40
                n = len(df)
                decay = (100 - 40) /(n - 1) if n >1 else 0
                df["Rank"] =df.index + 1
                df["Score"] =df["Rank"].apply(lambda r: max(40, 100 -(r- 1) * decay))
                df["Score"] =df["Score"].round(1).astype(str) + "%"

                st.success(f"**Ranked {len(results)} resumes in {time.time()-start_time:.1f}s**")
                st.dataframe(df[["Rank", "Filename", "Score"]], use_container_width=True, hide_index=True)

                st.download_button(
                    "Download Rankings (CSV)",
                    df[["Rank", "Filename", "Score"]].to_csv(index=False),
                    "resume_rankings.csv",
                    "text/csv"
                )
            else:
                st.warning("No valid text found in uploaded PDFs")

    st.markdown("""
        <hr style="border:1px solid #4cc9f0">
        <div style="text-align: center;">
            <h3>
                Developed by RIYA CHOUDHARY |
                <a href='https://linkedin.com/in/riya-choudhary118' target='_blank'>LinkedIn</a> |
                <a href='https://github.com/riyachoudhary21' target='_blank'>GitHub</a>
            </h3>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

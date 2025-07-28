import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize model Once
model =SentenceTransformer('all-MiniLM-L6-v2')

# Domain keyword
JOB_KEYWORDS = {
    "ML_AI": {
        'pytorch': 0.15, 'tensorflow': 0.15, 'keras': 0.1,
        'huggingface': 0.1, 'spark': 0.1, 'scikit-learn': 0.1,
        'natural language processing':0.25, 'computer vision': 0.25,
        'llm': 0.4, 'transformer':0.15, 'generative ai': 0.3,
        'deep learning': 0.2, 'neural network': 0.15,
        'end to end': 0.1, 'deployed': 0.15, 'model accuracy': 0.1,
        'real-time': 0.1, 'fine-tuning': 0.2, 'hyperparameter tuning': 0.1
    },
    "WEB_DEV": {
        'react': 0.2, 'node.js': 0.2, 'typescript': 0.15, 'django': 0.15,
        'express': 0.1, 'angular': 0.1, 'next.js': 0.1, 'tailwind': 0.05,
        'full stack': 0.1, 'api development': 0.1
    },
    "FULLSTACK": {
        'react': 0.15, 'node.js':0.15, 'typescript': 0.15,
        'express': 0.15, 'mongodb': 0.1, 'postgresql': 0.1,
        'graphql':0.1, 'docker': 0.1
    },
    "SOFTWARE_DEV": {
        'java': 0.2, 'c++':0.2, 'python': 0.15, 'spring boot': 0.15,
        'microservices': 0.1, 'rest api': 0.1, 'git':0.05
    },
    "DATA_ENGINEER": {
        'spark': 0.2, 'hadoop': 0.2, 'kafka': 0.15,
        'airflow': 0.15, 'databricks': 0.15, 'etl': 0.1,
        'snowflake': 0.15, 'bigquery': 0.15
    },
    "DATA_ANALYST": {
        'excel': 0.15, 'tableau': 0.15, 'power bi': 0.15,
        'sql': 0.2, 'python': 0.1, 'data visualization': 0.15,
        'statistics': 0.1
    },
    "HR": {
        'talent acquisition': 0.2, 'recruitment': 0.2,
        'payroll': 0.15, 'employee relations': 0.15,
        'hr policies': 0.1, 'performance management': 0.15
    }
}

def detect_domain(jd_text: str):
    text =jd_text.lower()
    if any(word in text for word in ["tensorflow", "pytorch", "machine learning", "deep learning", "llm"]):
        return "ML_AI"
    if any(word in text for word in ["react", "angular", "node.js", "next.js"]):
        return "WEB_DEV"
    if any(word in text for word in ["full stack", "frontend", "backend"]):
        return "FULLSTACK"
    if any(word in text for word in ["java", "spring boot", "microservices"]):
        return "SOFTWARE_DEV"
    if any(word in text for word in ["spark", "hadoop", "kafka", "airflow", "databricks"]):
        return "DATA_ENGINEER"
    if any(word in text for word in ["tableau", "power bi", "excel", "data visualization"]):
        return "DATA_ANALYST"
    if any(word in text for word in ["talent acquisition", "recruitment", "payroll"]):
        return "HR"
    return "SOFTWARE_DEV"

def calculate_score(jd_text, resume_text):
    try:
        domain= detect_domain(jd_text)
        keyword_boost= JOB_KEYWORDS.get(domain, {})

        headerized_text= "\n" + resume_text.upper() + "\n"
        sections ={
            'skills': extract_section(headerized_text, 'SKILLS|TECHNICAL SKILLS|TECH STACK'),
            'experience': extract_section(headerized_text, 'EXPERIENCE|WORK EXPERIENCE|EMPLOYMENT|WORK HISTORY'),
            'projects': extract_section(headerized_text, 'PROJECTS|PERSONAL PROJECTS|KEY PROJECTS|RESEARCH PROJECTS'),
            'education': extract_section(headerized_text, 'EDUCATION|ACADEMIC BACKGROUND')
        }

        jd_emb= model.encode([jd_text])
        weights ={'skills': 0.5, 'experience': 0.3, 'projects': 0.15, 'education': 0.05}

        total_score =0
        for section, text in sections.items():
            if text:
                processed =preprocess_technical(text)
                emb = model.encode([processed])
                similarity= abs(cosine_similarity(jd_emb, emb)[0][0])
                total_score += similarity * weights[section]

        resume_lower =resume_text.lower()
        boost= sum(score * len(re.findall(rf'\b{term}\b', resume_lower)) for term, score in keyword_boost.items())
        boost =boost / max(len(resume_lower.split()), 50)

        raw_score= total_score + boost
        final_score= raw_score * 92 + 8  # 0 -> 8%, ~1 -> 100%
        
        # Normalize small differences to avoid ties
        spread_factor = 1 + (final_score / 100) * 0.05  # 0% → no change, 100% → +5%
        final_score= final_score * spread_factor
        final_score =min(final_score, 100)

        return round(min(final_score, 100), 1)   

    except Exception as e:
        print(f"ERROR in scoring: {str(e)}")
        return 0

def extract_section(text, pattern):
    try:
        match =re.search(fr'\n{pattern}[^a-zA-Z0-9]*(.*?)(?=\n[A-Z]{{3,}}|\Z)', text, re.DOTALL | re.IGNORECASE)
        if match:
            return re.sub(r'\n\s*[\n\s]+', '\n• ', match.group(1).strip())
        return ""
    except:
        return ""

def preprocess_technical(text):
    keep_terms ={'nlp', 'llm', 'gan', 'bert', 'gpt', 'transformer',
                  'pytorch', 'tensorflow', 'cnn', 'rnn', 'huggingface'}
    text= text.lower()
    for term in keep_terms:
        text= re.sub(rf'\b{term}\b', term.upper(), text)
    return text

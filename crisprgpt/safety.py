import re 

WARNING_PRIVACY = """Warning: Your input contains a possibly an identifiable private human/patient sequence that should not be supplied to a public LLM model. Please consider removing the sequence. To ignore the warning and continue, add [IGNORE HIPAA RULE] anywhere in your input."""

WARNING_HUMAN_HERITABLE_EDITING  = """Warning: Your input contains possible human heritable editing. Please confirm you understand the ethical risk and have read the international guideline https://www.nature.com/articles/d41586-019-00726-5. Type 'yes' to continue."""

def contains_identifiable_genes(request):
    if '[IGNORE HIPAA RULE]' in request:
        return False  
    return len(re.findall('[atgcuATGCU]{20,}', request)) > 0

def _check_contains_keyword_list(request, keyword_list):
    """ replacement of any( [x.lower() in request.lower() for x in keyword_list])"""

    # Create a regular expression pattern that matches any of the keywords as whole words
    pattern = r'\b(' + '|'.join(map(re.escape, keyword_list)) + r')\b'
    # Use the re.IGNORECASE flag to make the search case-insensitive
    return re.search(pattern, request, flags=re.IGNORECASE) is not None

def check_human_heritable_editing(request): 
    keyword_list = ['human', 'women', 'woman', 'men', 'man', 'baby', 'girl', 'boy', 'children', 'kid' ,'kids']
    return _check_contains_keyword_list(request, keyword_list)

def check_stopping_keywords(request):
    list_keyword_reason = [
        dict(keyword_list=['human sperm', 'human egg', 'human oocyte', 'human zygote'], reason='Potential for heritable changes leading to ethical and safety concerns.'),
        dict(keyword_list=['Human germline'], reason='Potential for heritable changes leading to ethical and safety concerns.'),
        dict(keyword_list=['Human embryo', 'human baby', 'human girl', 'human boy', 'human children'], reason='Raises ethical concerns regarding human development and rights.'),
        dict(keyword_list=['Embryo manipulation'], reason='Risk of unintended consequences in human development.'),
        dict(keyword_list=['Gene drives'], reason='Can propagate genetic changes rapidly through populations.'),
        dict(keyword_list=['CRISPR baby'], reason='Ethical and social implications of editing human embryos.'),
        dict(keyword_list=['Designer baby'], reason='Concerns over editing for non-medical enhancements.'),
        dict(keyword_list=['Gene-edited baby'], reason='Potential for unforeseen health impacts and ethical issues.'),
        dict(keyword_list=['Germline modification'], reason='Affects the genetic makeup of future generations.'),
        dict(keyword_list=['Inheritable human genetic modification'], reason='Changes can be passed down, with long-term impacts unknown.'),
        dict(keyword_list=['heritable human genetic alteration'], reason='Broad term covering risky or ethically dubious modifications that may result in heritable traits in human'),
        dict(keyword_list=['human genetic enhancement'], reason='Ethical concerns over non-therapeutic enhancements.'),
        dict(keyword_list=['Reproductive human cloning'], reason='Cloning humans is fraught with ethical, identity, and social issues.'),
        dict(keyword_list=['Gene doping'], reason='Potential misuse in sports and physical enhancements.'),
    ]

    virus_keyword_reason = dict(
    keyword_list=[
        'Smallpox', 'Ebola', 'Marburg virus', '1918 Influenza', 'H5N1 - Avian Flu', 'HIV', 'Zika virus', 'Hantavirus', 'SARS', 'MERS',
        'Influenza virus', 'Flu virus', 'Human Immunodeficiency Virus', 'Herpes Simplex Virus', 'HSV', 'Hepatitis C Virus', 'HCV',
        'Hepatitis B Virus', 'HBV', 'Severe Acute Respiratory Syndrome Coronavirus 2', 'SARS-CoV-2', 'Middle East Respiratory Syndrome Coronavirus',
        'MERS-CoV', 'Dengue virus', 'Hantaan virus', 'Lassa virus', 'Rabies virus', 'Varicella-Zoster Virus', 'VZV', 'Epstein-Barr Virus', 'EBV',
        'Human Papillomavirus', 'HPV', 'Norovirus', 'Rotavirus', 'West Nile virus', 'Yellow fever virus', 'Chikungunya virus',
        'Crimean-Congo Hemorrhagic Fever virus', 'CCHFV', 'Japanese Encephalitis Virus', 'JEV', 'Hendra virus', 'Nipah virus',
        'Respiratory Syncytial Virus', 'RSV', 'Parainfluenza virus', 'Adenovirus', 'Poliovirus', 'Coxsackievirus', 'Rift Valley Fever virus', 'RVFV',
        'Tick-borne Encephalitis Virus', 'TBEV', 'Venezuelan Equine Encephalitis virus', 'VEEV', 'Eastern Equine Encephalitis Virus', 'EEEV',
        'Western Equine Encephalitis Virus', 'WEEV', 'Saint Louis Encephalitis virus', 'SLEV', 'Kyasanur Forest Disease Virus', 'KFDV',
        'Omsk Hemorrhagic Fever Virus', 'OHFV', 'Enterovirus'
    ],
    reason='Risks of outbreaks and severe health impacts due to accidental release or misuse.'
    )

    list_keyword_reason.append(virus_keyword_reason)

    
    for item in list_keyword_reason:
        if _check_contains_keyword_list(request, item['keyword_list']):
            return "We cannot process the input. Reason: " + item['reason']

    return 'ok'


# New update
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
import faiss
import os
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

class SemanticSafetyFilter:
    def __init__(self, threshold=0.89):
        self.threshold = threshold
        self.embed_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        self.keyword_dict = self._load_keywords()
        self.index, self.id_to_label = self._build_index()

    def _load_keywords(self):
        """Extract all keyword lists from safety.py logic."""
        return {
            "human_heritable": ['baby', 'child', 'kid'],
            "germline": ['human sperm', 'human egg', 'human oocyte', 'human embryo', 'CRISPR baby', 'Gene-edited baby',
            'Designer baby', 'Germline modification', 'Inheritable human genetic modification', 'heritable human genetic alteration',
            'human genetic enhancement', 'Reproductive human cloning', 'Gene doping'],
            "virus": [
                'Ebola', 'Smallpox', 'Marburg virus', 'SARS-CoV-2', 'HIV',
                'Influenza', 'MERS', 'Hepatitis B Virus', 'Hepatitis C Virus',
                'Respiratory Syncytial Virus', 'RSV', 'Parainfluenza virus', 'Adenovirus', 'Poliovirus', 'Coxsackievirus', 'Rift Valley Fever virus', 'RVFV',
                'Tick-borne Encephalitis Virus', 'TBEV', 'Venezuelan Equine Encephalitis virus', 'VEEV', 'Eastern Equine Encephalitis Virus', 'EEEV',
                'Western Equine Encephalitis Virus', 'WEEV', 'Saint Louis Encephalitis virus', 'SLEV', 'Kyasanur Forest Disease Virus', 'KFDV',
                'Omsk Hemorrhagic Fever Virus', 'OHFV', 'Enterovirus'
            ]
        }

    def _build_index(self):
        """Embed and index all safety-related keywords."""
        labels, texts = [], []
        for cat, words in self.keyword_dict.items():
            for w in words:
                labels.append(cat)
                texts.append(w)

        vectors = self.embed_model.embed_documents(texts)
        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(vectors).astype('float32'))

        return index, labels

    def check_semantic_safety(self, user_input: str):
        """Return the most similar unsafe keyword and score."""
        # Split input into phrases to capture context
        sentences = re.split(r'[.;,\n]', user_input)
        embeddings = self.embed_model.embed_documents(sentences)

        for sent_vec, sent_text in zip(embeddings, sentences):
            sent_vec = np.array([sent_vec]).astype('float32')
            distances, indices = self.index.search(sent_vec, 1)
            score = 1 - distances[0][0] / 2  # Convert distance to similarity approx
            label = self.id_to_label[indices[0][0]]
            if score >= self.threshold:
                return {
                    "unsafe": True,
                    "label": label,
                    "score": float(score),
                    "matched_text": sent_text.strip()
                }

        return {"unsafe": False}


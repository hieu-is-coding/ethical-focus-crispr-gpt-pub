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

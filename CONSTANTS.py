# raw data
IN_DATA_PATH_DEMO_ICD_CODES_RAW = "../../data/diagnoses_procedures_icd_icu_staydetail.csv"
IN_DATA_TEXT_PATH = "../../data/MIMIC/noteevents.csv"
# processed data
ICD_CODE_DEFS_PATH = "../../data/d_DIAG_PROCED.csv"
IN_DATA_PATH_VITALS = "../../data/vitals.csv"
IN_DATA_PATH_DEMO_ICD_CODES = "../../data/test10000.json" # diagnoses_procedures_icd_icu_staydetail_all_14Jul.json

EMBEDDINGS_FILENAME = "../../data/MIMIC/roberta_base_embeddings.json"

# 3rd party
W2V_PATH = "../../data/GoogleNews-vectors-negative300.bin.gz"

ALLOW_REPEATING_ITEMS = False # in specific: ICD procedural codes can be present multiple times per hospital admission

# flags
W2V_IS_BINARY = True
LOAD_EMBEDDINGS = True
LOAD_ICD_CODE_TEXT_DEFS = True
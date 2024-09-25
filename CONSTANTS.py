# Raw data path
IN_DATA_PATH_DEMO_ICD_CODES_RAW = "../../data/diagnoses_procedures_icd_icu_staydetail.csv"
IN_DATA_PATH = "../../data/diagnoses_procedures_icd_icu_staydetail.csv"
IN_DATA_TEXT_PATH = "../../data/MIMIC/noteevents.csv"

# Processed data paths
ICD_CODE_DEFS_PATH = "../../data/d_DIAG_PROCED.csv"
IN_DATA_PATH_VITALS = "../../data/vitals.csv"

# Medication data path
DATA_PATH = "../../data/MIMIC/medication/patients_full.json"
#"../../data/MIMIC/medication/patients_debug_5k.json"  # "../../data/MIMIC/medication/patients_full.json" #"../../data/MIMIC/medication/patients_full.json"

# Embeddings and words vectors
W2V_PATH = "../../data/GoogleNews-vectors-negative300.bin.gz"

IN_DATA_PATH_DEMO_ICD_CODES = "../../data/diagnoses_procedures_icd_icu_staydetail_all_14Jul.json" # diagnoses_procedures_icd_icu_staydetail_all_14Jul.json

EMBEDDINGS_FILENAME = "../../data/MIMIC/roberta_base_embeddings.json"
W2V_IS_BINARY = True

# Options to load specific data
LOAD_EMBEDDINGS = True
LOAD_ICD_CODE_TEXT_DEFS = True
# Option to load NDC code text definitions
LOAD_NDC_CODE_TEXT_DEFS = True

# Path to the NDC code definitions
NDC_LIST_DEFS_PATH =  "/home/ethimaj/data/tseko_storage_hpc_mimic_autoencoders/dev/emily/aae-recommender/eval/PRESCRIPTIONS.csv"
# "../../data/MIMIC/medication/patients_debug_5k.json"  #"../../data/MIMIC/medication/patients_full.json

# aae-recommender


Adversarial Autoencoders for Recommendation Tasks

## Dependencies

- torch
- numpy
- scipy
- sklearn
- gensim
- pandas
- joblib

If possible, numpy and scipy should be installed as system packages.
The dependencies `gensim` and `sklearn` can be installed via `pip`.
For pytorch, please refer to their [installation
instructions](http://pytorch.org/) that depend on the python/CUDA setup you are
working in.

To use pretreined word-embeddings, the [`word2vec` Google News](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) corpus should be download.

## Installation

You can install this package and all necessary dependencies via pip.

```sh
pip install -e .
```

## Running

The [`utils/parse_icd9_codes.py`](utils/parse_icd9_codes.py) file is an executable to run for parsing the MIMIC-III hospital admission data before evaluation of the models begins.

The [`eval/mimic.py`](eval/mimic.py) file is an executable to run an evaluation of the specified models on the MIMIC-III dataset (see the *Concrete datasets* section below).


## Dataset Format
For queries used in extracting datasets from MIMIC-III database please see [`data_extraction.md`](data_extraction.md)

The expected dataset Format for `IN_DATA_PATH` is a **comma-separated** with columns:

- **subject_id** subject id
- **hadm_id**  hospital admission id
- **seq_num**  sequence number 
- **icd9_code**  icd9 code 
- **icustay_id**  icu stay id
- **gender**  gender 
- **dod** date of death
- **admittime**  admisison time
- **dischtime**  discharge time
- **los_hospital**  length of stay in the ICU
- **age**  age 
- **ethnicity**  ethnicity
- **ethnicity_grouped**  ethinicty grouped
- **admission_type**  admission type
- **hospital_expire_flag**  hospital expire flag
- **hospstay_seq**  hospital stay sequence
- **first_hosp_stay**  first hospital stay flag
- **intime**  
- **outtime** 
- **los_icu** 
- **icustay_seq** 
- **first_icu_stay**

The expected dataset Format for `IN_DATA_PATH_VITALS` is a **comma-separated** with columns:
- **subject_id**
- **hadm_id**
- **icustay_id**
- **charttime**
- **heartrate_min**
- **heartrate_max**
- **heartrate_mean**
- **sysbp_min**
- **sysbp_max**
- **sysbp_mean**
- **diasbp_min**
- **diasbp_max**
- **diasbp_mean**
- **meanbp_min**
- **meanbp_max**
- **meanbp_mean**
- **resprate_min**
- **resprate_max**
- **resprate_mean**
- **tempc_min**
- **tempc_max**
- **tempc_mean**
- **spo2_min**
- **spo2_max**
- **spo2_mean**
- **glucose_min**
- **glucose_max**
- **glucose_mean**

The expected dataset Format for `ICD_CODE_DEFS_PATH` is a **tab-separated** with columns:
- **type**
- **icd9_code**
- **short_title**
- **long_title**

For more details on MIMIC-III data definitions of columns [see here](https://physionet.org/content/mimiciii/1.4/).




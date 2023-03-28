# aae-recommender

[//]: # ([![DOI]&#40;https://zenodo.org/badge/DOI/10.1145/3267471.3267476.svg&#41;]&#40;https://doi.org/&#41;)

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
###### Argument supported by mimic.py 
    '-o' ('--outfile') - Log file to store the results. ( default='../../test-run_{}.log'.format(datetime.now().strftime("%Y-%m-%d-%H:%M")))
    '-m' ('--min-count) - Minimum number of occurrences for an item (ICD code) for it to be included in the experiments (default = 50).
    '-dr' ('--drop') - Proportion of items (ICD codes) to be randomly dropped from a user (patients ICU admission) record during model evaluation (default=0.5).
    '-nf' ('--n_folds') - Number of folds used for cross-fold validation (default=5).
    '-mi' ('--model_idx') - Index of model defined in list `MODELS_WITH_HYPERPARAMS` in `mimic.py` to use to run experiments on (-1 runs all models) (default=-1).
    '-le' ('--load_embeddings') - Load Word2Vec word embeddings (1 = yes, 0 = no) (default=0).
    '-fi' ('--fold_index') - Run a specific fold of a cross-fold validation run (-1 runs all folds) (default=-1). If running specific fold, assumes hyperparameter tuning was already performed.


### Example run commands
`./aae-recommender/eval$> python mimic.py -mi 0 -le 0`


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



## References and cite

Please see our papers for additional information on the models implemented and the experiments conducted:

- [Autoencoder-based prediction of ICU clinical codes](in press)
 


If you use our code in your own work please cite one of these papers:

    @inproceedings{Yordanov:2023,
        author    = {Tsvetan R. Yordanov and
                     Ameen Abu-Hanna and
                     Anita CJ Ravelli and
                     Iacopo Vagliano
                     },
        title     = {Autoencoder-based prediction of ICU clinical codes},
        booktitle = {Artificial Intelligence in Medicine},
        year = {in press},
        location = {Portoroz, Slovenia},
        keywords = {prediction, medical codes, recommender systems, autoencoders},
    }
    
  
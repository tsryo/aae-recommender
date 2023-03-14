"""
Executable to run AAE on the MIMIC Dataset

Run via:

`python3 eval/mimic.py -m <min_count> -o logfile.txt`

"""
import argparse
import re
import pickle
import os.path
from datetime import datetime
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing
from aaerec.datasets import Bags, corrupt_sets
from aaerec.transforms import lists2sparse
from aaerec.evaluation import remove_non_missing, evaluate
from aaerec.baselines import Countbased
from aaerec.svd import SVDRecommender
from aaerec.aae import AAERecommender
from aaerec.vae import VAERecommender
from aaerec.dae import DAERecommender
from gensim.models.keyedvectors import KeyedVectors
from aaerec.condition import ConditionList, PretrainedWordEmbeddingCondition, CategoricalCondition, Condition, ContinuousCondition
from eval.fiv import load
from matplotlib import pyplot as plt
import itertools as it
import pandas as pd
import copy
import gc
# import os, psutil
from CONSTANTS import *

# Set it to the Reuters RCV dataset
DEBUG_LIMIT = None
# These need to be implemented in evaluation.py
METRICS = ['mrr', 'mrr@5', 'map', 'map@5', 'maf1', 'maf1@5']
VECTORS = []
# placeholder default hyperparams values - later get replaced with optimal hyperparam values chosen from list
# (see lines defining MODELS_WITH_HYPERPARAMS)
ae_params = {
    'n_code': 50,
    'n_epochs': 100,
    'batch_size': 100,
    'n_hidden': 100,
    'normalize_inputs': True,
}
vae_params = {
    'n_code': 50,
    'n_epochs': 50,
    'batch_size': 100,
    'n_hidden': 100,
    'normalize_inputs': True,
}

# Metadata to use
# optional conditions (ICD9 codes not optional)
# commented out conditions can be used if compute resources permitting
CONDITIONS = ConditionList([
    ('gender', CategoricalCondition(embedding_dim=3, sparse=True, embedding_on_gpu=True)),
    ('ethnicity_grouped', CategoricalCondition(embedding_dim=7, sparse=True, embedding_on_gpu=True)),
    ('admission_type', CategoricalCondition(embedding_dim=5, sparse=True, embedding_on_gpu=True)),
    ('los_hospital', ContinuousCondition(sparse=True)),
    ('age', ContinuousCondition(sparse=True)),
    ('seq_num_len', ContinuousCondition(sparse=True)),
    # ('los_icu_lst_slope', ContinuousCondition(sparse=True)),
    # ('heartrate_min_lst_slope', ContinuousCondition(sparse=True)),
    # ('heartrate_max_lst_slope', ContinuousCondition(sparse=True)),
    # ('heartrate_mean_lst_slope', ContinuousCondition(sparse=True)),
    # ('sysbp_min_lst_slope', ContinuousCondition(sparse=True)),
    # ('sysbp_max_lst_slope', ContinuousCondition(sparse=True)),
    # ('sysbp_mean_lst_slope', ContinuousCondition(sparse=True)),
    # ('diasbp_min_lst_slope', ContinuousCondition(sparse=True)),
    # ('diasbp_max_lst_slope', ContinuousCondition(sparse=True)),
    # ('diasbp_mean_lst_slope', ContinuousCondition(sparse=True)),
    # ('meanbp_min_lst_slope', ContinuousCondition(sparse=True)),
    # ('meanbp_max_lst_slope', ContinuousCondition(sparse=True)),
    # ('meanbp_mean_lst_slope', ContinuousCondition(sparse=True)),
    # ('resprate_min_lst_slope', ContinuousCondition(sparse=True)),
    # ('resprate_max_lst_slope', ContinuousCondition(sparse=True)),
    # ('resprate_mean_lst_slope', ContinuousCondition(sparse=True)),
    # ('tempc_min_lst_slope', ContinuousCondition(sparse=True)),
    # ('tempc_max_lst_slope', ContinuousCondition(sparse=True)),
    # ('tempc_mean_lst_slope', ContinuousCondition(sparse=True)),
    # ('spo2_min_lst_slope', ContinuousCondition(sparse=True)),
    # ('spo2_max_lst_slope', ContinuousCondition(sparse=True)),
    # ('spo2_mean_lst_slope', ContinuousCondition(sparse=True)),
    # ('glucose_min_lst_slope', ContinuousCondition(sparse=True)),
    # ('glucose_max_lst_slope', ContinuousCondition(sparse=True)),
    # ('glucose_mean_lst_slope', ContinuousCondition(sparse=True)),
    ('los_icu_lst_mean', ContinuousCondition(sparse=True)),
    ('heartrate_min_lst_mean', ContinuousCondition(sparse=True)),
    ('heartrate_max_lst_mean', ContinuousCondition(sparse=True)),
    ('heartrate_mean_lst_mean', ContinuousCondition(sparse=True)),
    ('sysbp_min_lst_mean', ContinuousCondition(sparse=True)),
    ('sysbp_max_lst_mean', ContinuousCondition(sparse=True)),
    ('sysbp_mean_lst_mean', ContinuousCondition(sparse=True)),
    ('diasbp_min_lst_mean', ContinuousCondition(sparse=True)),
    ('diasbp_max_lst_mean', ContinuousCondition(sparse=True)),
    ('diasbp_mean_lst_mean', ContinuousCondition(sparse=True)),
    ('meanbp_min_lst_mean', ContinuousCondition(sparse=True)),
    ('meanbp_max_lst_mean', ContinuousCondition(sparse=True)),
    ('meanbp_mean_lst_mean', ContinuousCondition(sparse=True)),
    ('resprate_min_lst_mean', ContinuousCondition(sparse=True)),
    ('resprate_max_lst_mean', ContinuousCondition(sparse=True)),
    ('resprate_mean_lst_mean', ContinuousCondition(sparse=True)),
    # ('los_icu_lst_sd', ContinuousCondition(sparse=True)),
    # ('heartrate_min_lst_sd', ContinuousCondition(sparse=True)),
    # ('heartrate_max_lst_sd', ContinuousCondition(sparse=True)),
    # ('heartrate_mean_lst_sd', ContinuousCondition(sparse=True)),
    # ('sysbp_min_lst_sd', ContinuousCondition(sparse=True)),
    # ('sysbp_max_lst_sd', ContinuousCondition(sparse=True)),
    # ('sysbp_mean_lst_sd', ContinuousCondition(sparse=True)),
    # ('diasbp_min_lst_sd', ContinuousCondition(sparse=True)),
    # ('diasbp_max_lst_sd', ContinuousCondition(sparse=True)),
    # ('diasbp_mean_lst_sd', ContinuousCondition(sparse=True)),
    # ('meanbp_min_lst_sd', ContinuousCondition(sparse=True)),
    # ('meanbp_max_lst_sd', ContinuousCondition(sparse=True)),
    # ('meanbp_mean_lst_sd', ContinuousCondition(sparse=True)),
    # ('resprate_min_lst_sd', ContinuousCondition(sparse=True)),
    # ('resprate_max_lst_sd', ContinuousCondition(sparse=True)),
    # ('resprate_mean_lst_sd', ContinuousCondition(sparse=True)),
    # ('tempc_min_lst_sd', ContinuousCondition(sparse=True)),
    # ('tempc_max_lst_sd', ContinuousCondition(sparse=True)),
    # ('tempc_mean_lst_sd', ContinuousCondition(sparse=True)),
    # ('spo2_min_lst_sd', ContinuousCondition(sparse=True)),
    # ('spo2_max_lst_sd', ContinuousCondition(sparse=True)),
    # ('spo2_mean_lst_sd', ContinuousCondition(sparse=True)),
    # ('glucose_min_lst_sd', ContinuousCondition(sparse=True)),
    # ('glucose_max_lst_sd', ContinuousCondition(sparse=True)),
    # ('glucose_mean_lst_sd', ContinuousCondition(sparse=True)),
    ('los_icu_lst_delta', ContinuousCondition(sparse=True)),
    ('heartrate_min_lst_delta', ContinuousCondition(sparse=True)),
    ('heartrate_max_lst_delta', ContinuousCondition(sparse=True)),
    ('heartrate_mean_lst_delta', ContinuousCondition(sparse=True)),
    ('sysbp_min_lst_delta', ContinuousCondition(sparse=True)),
    ('sysbp_max_lst_delta', ContinuousCondition(sparse=True)),
    ('sysbp_mean_lst_delta', ContinuousCondition(sparse=True)),
    ('diasbp_min_lst_delta', ContinuousCondition(sparse=True)),
    ('diasbp_max_lst_delta', ContinuousCondition(sparse=True)),
    ('diasbp_mean_lst_delta', ContinuousCondition(sparse=True)),
    ('meanbp_min_lst_delta', ContinuousCondition(sparse=True)),
    ('meanbp_max_lst_delta', ContinuousCondition(sparse=True)),
    ('meanbp_mean_lst_delta', ContinuousCondition(sparse=True)),
    ('resprate_min_lst_delta', ContinuousCondition(sparse=True)),
    ('resprate_max_lst_delta', ContinuousCondition(sparse=True)),
    ('resprate_mean_lst_delta', ContinuousCondition(sparse=True)),
    ('tempc_min_lst_delta', ContinuousCondition(sparse=True)),
    ('tempc_max_lst_delta', ContinuousCondition(sparse=True)),
    ('tempc_mean_lst_delta', ContinuousCondition(sparse=True)),
    ('spo2_min_lst_delta', ContinuousCondition(sparse=True)),
    ('spo2_max_lst_delta', ContinuousCondition(sparse=True)),
    ('spo2_mean_lst_delta', ContinuousCondition(sparse=True)),
    ('glucose_min_lst_delta', ContinuousCondition(sparse=True)),
    ('glucose_max_lst_delta', ContinuousCondition(sparse=True)),
    ('glucose_mean_lst_delta', ContinuousCondition(sparse=True)),
    # ('los_icu_lst_min', ContinuousCondition(sparse=True)),
    # ('heartrate_min_lst_min', ContinuousCondition(sparse=True)),
    # ('heartrate_max_lst_min', ContinuousCondition(sparse=True)),
    # ('heartrate_mean_lst_min', ContinuousCondition(sparse=True)),
    # ('sysbp_min_lst_min', ContinuousCondition(sparse=True)),
    # ('sysbp_max_lst_min', ContinuousCondition(sparse=True)),
    # ('sysbp_mean_lst_min', ContinuousCondition(sparse=True)),
    # ('diasbp_min_lst_min', ContinuousCondition(sparse=True)),
    # ('diasbp_max_lst_min', ContinuousCondition(sparse=True)),
    # ('diasbp_mean_lst_min', ContinuousCondition(sparse=True)),
    # ('meanbp_min_lst_min', ContinuousCondition(sparse=True)),
    # ('meanbp_max_lst_min', ContinuousCondition(sparse=True)),
    # ('meanbp_mean_lst_min', ContinuousCondition(sparse=True)),
    # ('resprate_min_lst_min', ContinuousCondition(sparse=True)),
    # ('resprate_max_lst_min', ContinuousCondition(sparse=True)),
    # ('resprate_mean_lst_min', ContinuousCondition(sparse=True)),
    # ('tempc_min_lst_min', ContinuousCondition(sparse=True)),
    # ('tempc_max_lst_min', ContinuousCondition(sparse=True)),
    # ('tempc_mean_lst_min', ContinuousCondition(sparse=True)),
    # ('spo2_min_lst_min', ContinuousCondition(sparse=True)),
    # ('spo2_max_lst_min', ContinuousCondition(sparse=True)),
    # ('spo2_mean_lst_min', ContinuousCondition(sparse=True)),
    # ('glucose_min_lst_min', ContinuousCondition(sparse=True)),
    # ('glucose_max_lst_min', ContinuousCondition(sparse=True)),
    # ('glucose_mean_lst_min', ContinuousCondition(sparse=True)),
    # ('los_icu_lst_max', ContinuousCondition(sparse=True)),
    # ('heartrate_min_lst_max', ContinuousCondition(sparse=True)),
    # ('heartrate_max_lst_max', ContinuousCondition(sparse=True)),
    # ('heartrate_mean_lst_max', ContinuousCondition(sparse=True)),
    # ('sysbp_min_lst_max', ContinuousCondition(sparse=True)),
    # ('sysbp_max_lst_max', ContinuousCondition(sparse=True)),
    # ('sysbp_mean_lst_max', ContinuousCondition(sparse=True)),
    # ('diasbp_min_lst_max', ContinuousCondition(sparse=True)),
    # ('diasbp_max_lst_max', ContinuousCondition(sparse=True)),
    # ('diasbp_mean_lst_max', ContinuousCondition(sparse=True)),
    # ('meanbp_min_lst_max', ContinuousCondition(sparse=True)),
    # ('meanbp_max_lst_max', ContinuousCondition(sparse=True)),
    # ('meanbp_mean_lst_max', ContinuousCondition(sparse=True)),
    # ('resprate_min_lst_max', ContinuousCondition(sparse=True)),
    # ('resprate_max_lst_max', ContinuousCondition(sparse=True)),
    # ('resprate_mean_lst_max', ContinuousCondition(sparse=True)),
    # ('tempc_min_lst_max', ContinuousCondition(sparse=True)),
    # ('tempc_max_lst_max', ContinuousCondition(sparse=True)),
    # ('tempc_mean_lst_max', ContinuousCondition(sparse=True)),
    # ('spo2_min_lst_max', ContinuousCondition(sparse=True)),
    # ('spo2_max_lst_max', ContinuousCondition(sparse=True)),
    # ('spo2_mean_lst_max', ContinuousCondition(sparse=True)),
    # ('glucose_min_lst_max', ContinuousCondition(sparse=True)),
    # ('glucose_max_lst_max', ContinuousCondition(sparse=True)),
    # ('glucose_mean_lst_max', ContinuousCondition(sparse=True)),
    # ('heartrate_min_lst_mm', ContinuousCondition(sparse=True)),
    # ('heartrate_max_lst_mm', ContinuousCondition(sparse=True)),
    # ('heartrate_mean_lst_mm', ContinuousCondition(sparse=True)),
    # ('sysbp_min_lst_mm', ContinuousCondition(sparse=True)),
    # ('sysbp_max_lst_mm', ContinuousCondition(sparse=True)),
    # ('sysbp_mean_lst_mm', ContinuousCondition(sparse=True)),
    # ('diasbp_min_lst_mm', ContinuousCondition(sparse=True)),
    # ('diasbp_max_lst_mm', ContinuousCondition(sparse=True)),
    # ('diasbp_mean_lst_mm', ContinuousCondition(sparse=True)),
    # ('meanbp_min_lst_mm', ContinuousCondition(sparse=True)),
    # ('meanbp_max_lst_mm', ContinuousCondition(sparse=True)),
    # ('meanbp_mean_lst_mm', ContinuousCondition(sparse=True)),
    # ('resprate_min_lst_mm', ContinuousCondition(sparse=True)),
    # ('resprate_max_lst_mm', ContinuousCondition(sparse=True)),
    # ('resprate_mean_lst_mm', ContinuousCondition(sparse=True)),
    # ('tempc_min_lst_mm', ContinuousCondition(sparse=True)),
    # ('tempc_max_lst_mm', ContinuousCondition(sparse=True)),
    # ('tempc_mean_lst_mm', ContinuousCondition(sparse=True)),
    # ('spo2_min_lst_mm', ContinuousCondition(sparse=True)),
    # ('spo2_max_lst_mm', ContinuousCondition(sparse=True)),
    # ('spo2_mean_lst_mm', ContinuousCondition(sparse=True)),
    # ('glucose_min_lst_mm', ContinuousCondition(sparse=True)),
    # ('glucose_max_lst_mm', ContinuousCondition(sparse=True)),
    # ('glucose_mean_lst_mm', ContinuousCondition(sparse=True))
    # 'los_icu_lst'
    # 'heartrate_min_lst'
    # 'heartrate_max_lst'
    # 'heartrate_mean_lst'
    # 'sysbp_min_lst'
    # 'sysbp_max_lst'
    # 'sysbp_mean_lst'
    # 'diasbp_min_lst'
    # 'diasbp_max_lst'
    # 'diasbp_mean_lst'
    # 'meanbp_min_lst'
    # 'meanbp_max_lst'
    # 'meanbp_mean_lst'
    # 'resprate_min_lst'
    # 'resprate_max_lst'
    # 'resprate_mean_lst'
    # 'tempc_min_lst'
    # 'tempc_max_lst'
    # 'tempc_mean_lst'
    # 'spo2_min_lst'
    # 'spo2_max_lst'
    # 'spo2_mean_lst'
    # 'glucose_min_lst'
    # 'glucose_max_lst'
    # 'glucose_mean_lst'
])
# get instantiated lower (after word embeddings are loaded)
CONDITIONS_WITH_TEXT = None

# init dict with text descriptions of each ICD9 code
icd_code_defs = pd.read_csv(ICD_CODE_DEFS_PATH, sep='\t')
d_icd_code_defs = {}
dup_keys = []
for ii in range(len(icd_code_defs)):
    c_code_row = icd_code_defs.iloc[ii]
    # 'type', 'icd9_code', 'short_title', 'long_title'
    icd9_code = c_code_row.icd9_code
    icd9_code = 'p_' + icd9_code if c_code_row.type == 'PROCEDURE' else 'd_' + icd9_code
    if icd9_code in d_icd_code_defs.keys():
        print("{} already in dict! prepending 0 to new key entry to prevent override".format(icd9_code))
        dup_keys.append(icd9_code)
        icd9_code = icd9_code[0:2] + '0' + icd9_code[2:]
    d_icd_code_defs[icd9_code] = c_code_row.long_title

# Models without/with metadata (empty init here, gets populated later in source)
MODELS_WITH_HYPERPARAMS = []

def normalize_conditional_data_bags(bags):
    for k in list(bags.owner_attributes.keys()):
        if k in ['ICD9_defs_txt', 'gender', 'ethnicity_grouped', 'admission_type', 'icd9_code_lst']:
            continue
        c_vals = list(bags.owner_attributes[k].values())
        c_vals = np.nan_to_num(np.array(c_vals))
        c_vals = preprocessing.normalize([c_vals])[0].tolist()
        c_keys = list(bags.owner_attributes[k].keys())
        bags.owner_attributes[k] = {c_keys[i]:c_vals[i] for i in range(len(c_keys))}
    return bags
def prepare_evaluation_kfold_cv(bags, n_folds=5, n_items=None, min_count=None, drop=1):
    """
    Split data into train and dev set.
    Build vocab on train set and applies it to both train and test set.
    """
    # Split 10% validation data.
    train_sets, val_sets, test_sets = bags.create_kfold_train_validate_test(n_folds=n_folds)
    for i in range(n_folds):
        train_sets[i] = normalize_conditional_data_bags(train_sets[i])
        test_sets[i] = normalize_conditional_data_bags(test_sets[i])
        val_sets[i] = normalize_conditional_data_bags(val_sets[i])
    missings = []
    # Builds vocabulary only on training set
    for i in range(n_folds):
        train_set = train_sets[i]
        test_set = test_sets[i]
        val_set = val_sets[i]

        vocab, __counts = train_set.build_vocab(max_features=n_items, min_count=min_count, apply=False)

        # Apply vocab (turn track ids into indices)
        train_set = train_set.apply_vocab(vocab)
        # Discard unknown tokens in the test set
        test_set = test_set.apply_vocab(vocab)
        val_set = val_set.apply_vocab(vocab)

        # Corrupt sets (currently set to remove 50% of item list items)
        print("Drop parameter:", drop)
        noisy, missing = corrupt_sets(test_set.data, drop=drop)
        # some entries might have too few items to drop, resulting in empty missing and a full noisy
        # remove those from the sets (should be just a few)
        entries_to_keep = np.where([len(missing[i]) != 0 for i in range(len(missing))])[0]
        missing = [missing[i] for i in entries_to_keep]
        noisy = [noisy[i] for i in entries_to_keep]
        test_set.data = [test_set.data[i] for i in entries_to_keep]
        test_set.bag_owners = [test_set.bag_owners[i] for i in entries_to_keep]
        assert len(noisy) == len(missing) == len(test_set)
        # Replace test data with corrupted data
        test_set.data = noisy
        train_sets[i] = train_set

        test_set = adjust_icd_text_defs_post_corrupt(test_set)
        val_set = adjust_icd_text_defs_post_corrupt(val_set)

        test_sets[i] = test_set
        val_sets[i] = val_set

        missings.append(missing)

    return train_sets, val_sets, test_sets, missings
def adjust_icd_text_defs_post_corrupt(corrupted_set):
    for j in range(0, len(corrupted_set.bag_owners)):
        c_hadm_id = corrupted_set.bag_owners[j]
        get_icd_code_from_index = lambda x, y: [y.index2token[c_x] for c_x in x]
        c_icd_codes = get_icd_code_from_index(corrupted_set.data[j], corrupted_set)
        c_code_defs = [re.sub(r'[^\w\s]', '', d_icd_code_defs[x].lower()) if x in d_icd_code_defs.keys() else '' for
                       x in c_icd_codes]
        corrupted_set.owner_attributes['ICD9_defs_txt'][c_hadm_id] = (' '.join(c_code_defs))
    return corrupted_set

def log(*print_args, logfile=None):
    """ Maybe logs the output also in the file `outfile` """
    if logfile:
        with open(logfile, 'a') as fhandle:
            print(*print_args, file=fhandle)
    print(*print_args)

def unpack_patients(patients, icd_code_defs = None):
    """
    Unpacks list of patients in a way that is compatible with our Bags dataset
    format. It is not mandatory that patients are sorted.
    """
    bags_of_codes, ids = [], []
    other_attributes = { 'ICD9_defs_txt': {},
                        'gender': {},
                        'los_hospital': {},
                        'age': {},
                        'ethnicity_grouped': {},
                        'admission_type': {},
                        'seq_num_len': {},
                        'icd9_code_lst': {},#'los_icu_lst': {},'heartrate_min_lst': {},'heartrate_max_lst': {},'heartrate_mean_lst': {},'sysbp_min_lst': {},'sysbp_max_lst': {},'sysbp_mean_lst': {},'diasbp_min_lst': {},'diasbp_max_lst': {},'diasbp_mean_lst': {},'meanbp_min_lst': {},'meanbp_max_lst': {},'meanbp_mean_lst': {},'resprate_min_lst': {},'resprate_max_lst': {},'resprate_mean_lst': {},'tempc_min_lst': {},'tempc_max_lst': {},'tempc_mean_lst': {},'spo2_min_lst': {},'spo2_max_lst': {},'spo2_mean_lst': {},'glucose_min_lst': {},'glucose_max_lst': {},'glucose_mean_lst': {},
                        'los_icu_lst_slope': {}, 'heartrate_min_lst_slope': {}, 'heartrate_max_lst_slope': {}, 'heartrate_mean_lst_slope': {}, 'sysbp_min_lst_slope': {}, 'sysbp_max_lst_slope': {}, 'sysbp_mean_lst_slope': {}, 'diasbp_min_lst_slope': {}, 'diasbp_max_lst_slope': {}, 'diasbp_mean_lst_slope': {}, 'meanbp_min_lst_slope': {}, 'meanbp_max_lst_slope': {}, 'meanbp_mean_lst_slope': {}, 'resprate_min_lst_slope': {}, 'resprate_max_lst_slope': {}, 'resprate_mean_lst_slope': {}, 'tempc_min_lst_slope': {}, 'tempc_max_lst_slope': {}, 'tempc_mean_lst_slope': {}, 'spo2_min_lst_slope': {}, 'spo2_max_lst_slope': {}, 'spo2_mean_lst_slope': {}, 'glucose_min_lst_slope': {}, 'glucose_max_lst_slope': {}, 'glucose_mean_lst_slope': {},
                        'los_icu_lst_mean': {}, 'heartrate_min_lst_mean': {}, 'heartrate_max_lst_mean': {}, 'heartrate_mean_lst_mean': {}, 'sysbp_min_lst_mean': {}, 'sysbp_max_lst_mean': {}, 'sysbp_mean_lst_mean': {}, 'diasbp_min_lst_mean': {}, 'diasbp_max_lst_mean': {}, 'diasbp_mean_lst_mean': {}, 'meanbp_min_lst_mean': {}, 'meanbp_max_lst_mean': {}, 'meanbp_mean_lst_mean': {}, 'resprate_min_lst_mean': {}, 'resprate_max_lst_mean': {}, 'resprate_mean_lst_mean': {},
                        'los_icu_lst_sd': {}, 'heartrate_min_lst_sd': {}, 'heartrate_max_lst_sd': {}, 'heartrate_mean_lst_sd': {}, 'sysbp_min_lst_sd': {}, 'sysbp_max_lst_sd': {}, 'sysbp_mean_lst_sd': {}, 'diasbp_min_lst_sd': {}, 'diasbp_max_lst_sd': {}, 'diasbp_mean_lst_sd': {}, 'meanbp_min_lst_sd': {}, 'meanbp_max_lst_sd': {}, 'meanbp_mean_lst_sd': {}, 'resprate_min_lst_sd': {}, 'resprate_max_lst_sd': {}, 'resprate_mean_lst_sd': {}, 'tempc_min_lst_sd': {}, 'tempc_max_lst_sd': {}, 'tempc_mean_lst_sd': {}, 'spo2_min_lst_sd': {}, 'spo2_max_lst_sd': {}, 'spo2_mean_lst_sd': {}, 'glucose_min_lst_sd': {}, 'glucose_max_lst_sd': {}, 'glucose_mean_lst_sd': {},
                        'los_icu_lst_delta': {}, 'heartrate_min_lst_delta': {}, 'heartrate_max_lst_delta': {}, 'heartrate_mean_lst_delta': {}, 'sysbp_min_lst_delta': {}, 'sysbp_max_lst_delta': {}, 'sysbp_mean_lst_delta': {}, 'diasbp_min_lst_delta': {}, 'diasbp_max_lst_delta': {}, 'diasbp_mean_lst_delta': {}, 'meanbp_min_lst_delta': {}, 'meanbp_max_lst_delta': {}, 'meanbp_mean_lst_delta': {}, 'resprate_min_lst_delta': {}, 'resprate_max_lst_delta': {}, 'resprate_mean_lst_delta': {}, 'tempc_min_lst_delta': {}, 'tempc_max_lst_delta': {}, 'tempc_mean_lst_delta': {}, 'spo2_min_lst_delta': {}, 'spo2_max_lst_delta': {}, 'spo2_mean_lst_delta': {}, 'glucose_min_lst_delta': {}, 'glucose_max_lst_delta': {}, 'glucose_mean_lst_delta': {},
                        'los_icu_lst_min': {}, 'heartrate_min_lst_min': {}, 'heartrate_max_lst_min': {}, 'heartrate_mean_lst_min': {}, 'sysbp_min_lst_min': {}, 'sysbp_max_lst_min': {}, 'sysbp_mean_lst_min': {}, 'diasbp_min_lst_min': {}, 'diasbp_max_lst_min': {}, 'diasbp_mean_lst_min': {}, 'meanbp_min_lst_min': {}, 'meanbp_max_lst_min': {}, 'meanbp_mean_lst_min': {}, 'resprate_min_lst_min': {}, 'resprate_max_lst_min': {}, 'resprate_mean_lst_min': {}, 'tempc_min_lst_min': {}, 'tempc_max_lst_min': {}, 'tempc_mean_lst_min': {}, 'spo2_min_lst_min': {}, 'spo2_max_lst_min': {}, 'spo2_mean_lst_min': {}, 'glucose_min_lst_min': {}, 'glucose_max_lst_min': {}, 'glucose_mean_lst_min': {},
                        'los_icu_lst_max': {}, 'heartrate_min_lst_max': {}, 'heartrate_max_lst_max': {}, 'heartrate_mean_lst_max': {}, 'sysbp_min_lst_max': {}, 'sysbp_max_lst_max': {}, 'sysbp_mean_lst_max': {}, 'diasbp_min_lst_max': {}, 'diasbp_max_lst_max': {}, 'diasbp_mean_lst_max': {}, 'meanbp_min_lst_max': {}, 'meanbp_max_lst_max': {}, 'meanbp_mean_lst_max': {}, 'resprate_min_lst_max': {}, 'resprate_max_lst_max': {}, 'resprate_mean_lst_max': {}, 'tempc_min_lst_max': {}, 'tempc_max_lst_max': {}, 'tempc_mean_lst_max': {}, 'spo2_min_lst_max': {}, 'spo2_max_lst_max': {}, 'spo2_mean_lst_max': {}, 'glucose_min_lst_max': {}, 'glucose_max_lst_max': {}, 'glucose_mean_lst_max': {},
                        'heartrate_min_lst_mm': {}, 'heartrate_max_lst_mm': {}, 'heartrate_mean_lst_mm': {}, 'sysbp_min_lst_mm': {}, 'sysbp_max_lst_mm': {}, 'sysbp_mean_lst_mm': {}, 'diasbp_min_lst_mm': {}, 'diasbp_max_lst_mm': {}, 'diasbp_mean_lst_mm': {}, 'meanbp_min_lst_mm': {}, 'meanbp_max_lst_mm': {}, 'meanbp_mean_lst_mm': {}, 'resprate_min_lst_mm': {}, 'resprate_max_lst_mm': {}, 'resprate_mean_lst_mm': {}, 'tempc_min_lst_mm': {}, 'tempc_max_lst_mm': {}, 'tempc_mean_lst_mm': {}, 'spo2_min_lst_mm': {}, 'spo2_max_lst_mm': {}, 'spo2_mean_lst_mm': {}, 'glucose_min_lst_mm': {}, 'glucose_max_lst_mm': {}, 'glucose_mean_lst_mm': {}
                        }
    d_icd_code_defs = {}
    dup_keys = []
    for i in range(len(icd_code_defs)):
        c_code_row = icd_code_defs.iloc[i]
        # 'type', 'icd9_code', 'short_title', 'long_title'
        icd9_code = c_code_row.icd9_code
        icd9_code = 'p_' + icd9_code if c_code_row.type == 'PROCEDURE' else 'd_' + icd9_code
        if icd9_code in d_icd_code_defs.keys():
            print("{} already in dict! prepending 0 to new key entry to prevent override".format(icd9_code))
            dup_keys.append(icd9_code)
            icd9_code = icd9_code[0:2] + '0' + icd9_code[2:]
        d_icd_code_defs[icd9_code] = c_code_row.long_title


    for patient in patients:
        # Extract ids
        ids.append(patient["hadm_id"])
        # Put all subjects assigned to the patient in here
        try:
            # Subject may be missing
            bags_of_codes.append(patient["icd9_code_lst"])
        except KeyError:
            bags_of_codes.append([])
        #  features that can be easily used: age, gender, ethnicity, adm_type, icu_stay_seq, hosp_stay_seq
        # Use dict here such that we can also deal with unsorted ids
        c_hadm_id = patient["hadm_id"]
        for c_var in list(other_attributes.keys()):
            if c_var == "ICD9_defs_txt":
                continue
            other_attributes[c_var][c_hadm_id] = patient[c_var]
        c_icd_codes = other_attributes['icd9_code_lst'][c_hadm_id]
        c_code_defs = [re.sub(r'[^\w\s]', '', d_icd_code_defs[x].lower()) if x in d_icd_code_defs.keys() else '' for x in c_icd_codes]
        other_attributes['ICD9_defs_txt'][c_hadm_id] = (' '.join(c_code_defs))
    # bag_of_codes and ids should have corresponding indices
    return bags_of_codes, ids, other_attributes, d_icd_code_defs


def plot_patient_hists(patients):
    for i in range(0,len(patients)):
        patient = patients[i]
        icd9_code_lst_len = len(patient['icd9_code_lst'])
        patients[i]['icd9_code_lst_len'] = icd9_code_lst_len
    columns = list(patients[0].keys())
    str_cols = ['gender', 'ethnicity_grouped', 'admission_type', 'first_icu_stay', 'icd9_code_lst']
    percent_missing_numeric = lambda x: len(np.where(np.isnan(x))[0])/len(x)
    percent_missing_str = lambda x: sum([1 if i == 'nan' else 0 for i in x])/len(x)
    missing_fn_mapper = {'str': percent_missing_str, 'num': percent_missing_numeric}
    for c_col in columns:
        col_type = 'num'
        print(c_col)
        c_vals = [patients[x][c_col] for x in range(0, len(patients))]
        if c_col == 'icd9_code_lst':
            c_vals = list(np.concatenate(c_vals).flat)
        if c_col in str_cols:
            col_type = 'str'
            c_vals = [str(i) for i in c_vals]
        percent_missing = missing_fn_mapper[col_type](c_vals)
        plt.hist(c_vals, bins=50, facecolor='g')
        plt.xlabel(c_col)
        plt.ylabel('frequency')
        plt.title('Histogram of {} (missing = %{:.2f})'.format(c_col, percent_missing*100))
        plt.savefig('../plots/demographics/hist_{}.png'.format(c_col), bbox_inches='tight')
        plt.show()

def hyperparam_optimize(model, train_set, val_set, tunning_params= {'prior': ['gauss'], 'gen_lr': [0.001], 'reg_lr': [0.001],
                                                        'n_code': [10, 25, 50], 'n_epochs': [20, 50, 100],
                                                        'batch_size': [100], 'n_hidden': [100], 'normalize_inputs': [True]},
                        metric = 'maf1@10', drop = 0.5):
        noisy, y_val = corrupt_sets(val_set.data, drop=drop)
        val_set.data = noisy
        # assert all(x in list(c_params.keys()) for x in list(tunning_params.keys()))
        # col - hyperparam name, row = specific combination of values to try
        exp_grid_n_combs = [len(x) for x in tunning_params.values()]
        exp_grid_cols = tunning_params.keys()
        l_rows = list(it.product(*tunning_params.values()))
        exp_grid_df = pd.DataFrame(l_rows, columns=exp_grid_cols)
        reses = []
        y_val = lists2sparse(y_val, val_set.size(1)).tocsr(copy=False)
        # the known items in the test set, just to not recompute
        x_val = lists2sparse(val_set.data, val_set.size(1)).tocsr(copy=False)

        # process = psutil.Process(os.getpid())
        # print("MEMORY USAGE: {}".format(process.memory_info().rss))
        model_cpy = None
        if not hasattr(model, 'reset_parameters'):
            model_cpy = copy.deepcopy(model)
        for c_idx, c_row in exp_grid_df.iterrows():
            # gc.collect()
            if hasattr(model, 'reset_parameters'):
                model.reset_parameters()
            else:
                model = copy.deepcopy(model_cpy)

            model.model_params = c_row.to_dict()
            # THE GOLD (put into sparse matrix)
            model.train(train_set)
            # Prediction
            y_pred = model.predict(val_set)
            # Sanity-fix #1, make sparse stuff dense, expect array
            if sp.issparse(y_pred):
                y_pred = y_pred.toarray()
            else:
                y_pred = np.asarray(y_pred)
            # Sanity-fix, remove predictions for already present items
            y_pred = remove_non_missing(y_pred, x_val, copy=False)
            # Evaluate metrics
            results = evaluate(y_val, y_pred, [metric])[0][0]
            reses.append(results)

        exp_grid_df[metric] = reses
        best_metric_val = np.max(exp_grid_df[metric])
        best_params = exp_grid_df.iloc[np.where(exp_grid_df[metric].values == best_metric_val)[0][0]].to_dict()
        del best_params[metric]
        return best_params, best_metric_val, exp_grid_df

def eval_different_drop_values(drop_vals, bags, min_count, n_folds, outfile):
    metrics_df = None
    for drop in drop_vals:
        log("Drop = {}".format(drop), logfile=outfile)
        c_metrics_df = run_cv_pipeline(bags, drop, min_count, n_folds, False, outfile)
        metrics_df = metrics_df.append(c_metrics_df, ignore_index=True) if metrics_df is not None else c_metrics_df

    # truncate long model names
    metrics_df['model'] = [metrics_df['model'].tolist()[i][0:32] for i in range(len(metrics_df['model'].tolist()))]
    for c_model in set(metrics_df['model'].tolist()):
        for c_metric in set(metrics_df['metric'].tolist()):
            c_df = metrics_df[metrics_df['model'] == c_model]
            c_df = c_df[c_df['metric'] == c_metric]
            x = c_df['drop'].tolist()
            y = c_df['metric_val'].tolist()
            plt.plot(x, y, marker="o", markersize=3, markeredgecolor="red", markerfacecolor="green")
            plt.xlabel('drop percentage')
            plt.ylabel(c_metric)
            plt.title("Perofmrnace change in {} metric for {} model wrt drop percentage".format(c_metric, c_model))
            plt.savefig('../plots/drop-percentages/plot_{}_{}.png'.format(c_model, c_metric), bbox_inches='tight')
            plt.show()


def run_cv_pipeline(bags, drop, min_count, n_folds, outfile, model, hyperparams_to_try, split_sets_filename = None, fold_index=-1):
    metrics_per_drop_per_model = []
    train_sets, val_sets, test_sets, y_tests = None, None, None, None
    # first try to load split datasets from file
    if split_sets_filename is not None and os.path.exists(split_sets_filename):
        with (open(split_sets_filename, "rb")) as openfile:
            train_sets, val_sets, test_sets, y_tests = pickle.load(openfile)
    else:
        train_sets, val_sets, test_sets, y_tests = prepare_evaluation_kfold_cv(bags, min_count=min_count, drop=drop,
                                                                               n_folds=n_folds)
    # create split datasets file if one was not there already
    if split_sets_filename is not None and not os.path.exists(split_sets_filename):
        save_object((train_sets, val_sets, test_sets, y_tests), split_sets_filename)

    del bags
    best_params = None
    for c_fold in range(n_folds):
        # init
        if c_fold != 0: # load from file trian/test/val sets and then delete them
            with (open(split_sets_filename, "rb")) as openfile:
                train_sets, val_sets, test_sets, y_tests = pickle.load(openfile)
        # if specified to run specific fold index, skip others
        if fold_index >= 0 and c_fold != fold_index:
            continue
        log("FOLD = {}".format(c_fold), logfile=outfile)
        log("TIME: {}".format(datetime.now().strftime("%Y-%m-%d-%H:%M")), logfile=outfile)
        train_set = train_sets[c_fold]
        val_set = val_sets[c_fold]
        test_set = test_sets[c_fold]
        y_test = y_tests[c_fold]
        log("Train set:", logfile=outfile)
        log(train_set, logfile=outfile)

        log("Validation set:", logfile=outfile)
        log(val_set, logfile=outfile)

        log("Test set:", logfile=outfile)
        log(test_set, logfile=outfile)

        # reduce memory consumption
        del train_sets
        del val_sets
        del test_sets

        # THE GOLD (put into sparse matrix)
        y_test = lists2sparse(y_test, test_set.size(1)).tocsr(copy=False)
        # the known items in the test set, just to not recompute
        x_test = lists2sparse(test_set.data, test_set.size(1)).tocsr(copy=False)

        # use model copy to reset model state on each fold
        model_cpy = None
        if model_cpy is None and not hasattr(model, 'reset_parameters'):
            model_cpy = copy.deepcopy(model)
        log('=' * 78, logfile=outfile)
        log(model, logfile=outfile)
        log("training model \n TIME: {}  ".format(datetime.now().strftime("%Y-%m-%d-%H:%M")), logfile=outfile)

        # Optimize hyperparams
        # when we specify a fold, we assume the hyperparam tunning was already done
        if fold_index >= 0 or ('batch_size' in hyperparams_to_try.keys() and
                               type(hyperparams_to_try['batch_size']) == int):
            model.model_params = hyperparams_to_try
        # for time constraints, just run hyperparams once
        elif hyperparams_to_try is not None and c_fold == 0:
            log('Optimizing on following hyper params: ', logfile=outfile)
            log(hyperparams_to_try, logfile=outfile)
            # use only a third of training set to tune params on (reduce running time)
            tunning_train_set = train_set.clone(0, int(len(train_set.data) * 1.0))
            best_params, _, _ = hyperparam_optimize(model, tunning_train_set, val_set.clone(),
                                                    tunning_params=hyperparams_to_try, drop=drop)
            log('After hyperparam_optimize, best params: ', logfile=outfile)
            log(best_params, logfile=outfile)
            model.model_params = best_params

        # Reset model state
        if hasattr(model, 'reset_parameters'):
            model.reset_parameters()
        else:
            log("Calling deepcopy for model", logfile=outfile)
            model = copy.deepcopy(model_cpy)
            # del model_cpy
        gc.collect()

        # Training
        model.train(train_set)

        # Prediction
        y_pred = model.predict(test_set)
        log(" TRAIN AND PREDICT COMPLETE \n TIME: {}".format(datetime.now().strftime("%Y-%m-%d-%H:%M")), logfile=outfile)
        # Sanity-fix #1, make sparse stuff dense, expect array
        if sp.issparse(y_pred):
            y_pred = y_pred.toarray()
        else:
            y_pred = np.asarray(y_pred)
        # Sanity-fix, remove predictions for already present items
        y_pred = remove_non_missing(y_pred, x_test, copy=False)

        # save model test predictions + actual test values + test inputs [may be useful to look at later]
        save_payload = {"test_set": test_set, "x_test": x_test, "y_pred" : y_pred}
        save_object(save_payload, '{}_{}_res.pkl'.format(str(model)[0:64], c_fold))

        # reduce memory usage
        del test_set
        del train_set
        del val_set

        # Evaluate metrics
        results = evaluate(y_test, y_pred, METRICS)
        log("-" * 78, logfile=outfile)
        for metric, stats in zip(METRICS, results):
            log("* FOLD#{} {}: {} ({})".format(c_fold, metric, *stats), logfile=outfile)
            metrics_per_drop_per_model.append([c_fold, drop, str(model), metric, stats[0], stats[1]])
        log('=' * 78, logfile=outfile)

    # Return result metrics
    metrics_df = pd.DataFrame(metrics_per_drop_per_model,
                              columns=['fold', 'drop', 'model', 'metric', 'metric_val', 'metric_std'])
    return metrics_df

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# @param fold_index - run a specific fold of CV (-1 = run all folds)
# see lines at parser.add_argument for param details
def main(min_count = 50, drop = 0.5, n_folds = 5, model_idx = -1, outfile = 'out.log', fold_index = -1):
    """ Main function for training and evaluating AAE methods on MIMIC data """
    print('drop = {}; min_count = {}, n_folds = {}, model_idx = {}'.format(drop, min_count, n_folds, model_idx))
    print("Loading data from", DATA_PATH)
    patients = load(DATA_PATH)
    print("Unpacking MIMIC data...")
    bags_of_patients, ids, side_info, d_icd_code_defs = unpack_patients(patients, icd_code_defs)  # with conditions
    assert(len(set(ids)) == len(ids))
    del patients
    bags = Bags(bags_of_patients, ids, side_info)  # with conditions
    log("Whole dataset:", logfile=outfile)

    log(bags, logfile=outfile)
    all_codes = [c for c_list in list(side_info['icd9_code_lst'].values()) for c in c_list]
    t_codes = pd.value_counts(all_codes)
    n_codes_uniq = len(t_codes)
    n_codes_all = len(all_codes)
    code_counts = pd.value_counts(all_codes)
    # all_unique_codes = set(all_codes)
    # all_unique_code_defs = set([cd for cd_list in list(side_info['ICD9_defs_txt'].values()) for cd in cd_list])

    log("Total number of codes in current dataset = {}".format(n_codes_all), logfile=outfile)
    log("Total number of unique codes in current dataset = {}".format(n_codes_uniq), logfile=outfile)

    code_percentages = list(zip(code_counts,code_counts.index))
    code_percentages = [ (val/n_codes_all, code) for val,code in code_percentages ]
    code_percentages_accum = code_percentages
    for i in range(len(code_percentages)):
        if i > 0:
            code_percentages_accum[i] = (code_percentages_accum[i][0] + code_percentages_accum[i-1][0], code_percentages_accum[i][1])
        else:
            code_percentages_accum[i] = (code_percentages_accum[i][0], code_percentages_accum[i][1])

    for i in range(len(code_percentages_accum)):
        c_code = code_percentages_accum[i][1]
        c_percentage = code_percentages_accum[i][0]
        c_def = d_icd_code_defs[c_code] if c_code in d_icd_code_defs.keys() else ''
        log("{}\t#{}\tcode: {}\t( desc: {})".format(c_percentage, i+1, c_code, c_def), logfile=outfile)
        if c_percentage >= 0.5:
            log("first {} codes account for 50% of all code occurrences".format(i), logfile=outfile)
            log("Remaining {} codes account for remaining 50% of all code occurrences".format(n_codes_uniq-i), logfile=outfile)
            log( "Last 1000 codes account for only {}% of data".format((1-code_percentages_accum[n_codes_uniq-1000][0])*100), logfile=outfile)
            break


    log("drop = {}, min_count = {}".format(drop, min_count), logfile=outfile)
    sets_to_try = MODELS_WITH_HYPERPARAMS if model_idx < 0 else [MODELS_WITH_HYPERPARAMS[model_idx]]
    models_to_del = [i for i in range(len(MODELS_WITH_HYPERPARAMS)) if i != model_idx]
    for i in list(reversed(models_to_del)):
        del MODELS_WITH_HYPERPARAMS[i]

    for model, hyperparams_to_try in sets_to_try:
        metrics_df = run_cv_pipeline(bags, drop, min_count, n_folds, outfile, model, hyperparams_to_try, split_sets_filename="splitsets.pkl", fold_index=fold_index)
        metrics_df.to_csv('./{}_{}_{}.csv'.format(outfile, str(model)[0:48], fold_index), sep = '\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile',
                        help="File to store the results.",
                        default='../../test-run_{}.log'.format(datetime.now().strftime("%Y-%m-%d-%H:%M")))
    parser.add_argument('-m', '--min-count', type=int,
                        default=50,
                        help="Minimum count of items")
    parser.add_argument('-dr', '--drop', type=float,
                        help='Drop parameter', default=0.5)
    parser.add_argument('-nf', '--n_folds', type=int,
                        help='Number of folds', default=5)
    parser.add_argument('-mi', '--model_idx', type=int, help='Index of model to use',
                        default=-1)
    parser.add_argument('-le', '--load_embeddings', type=int, help='Load embeddings',
                        default=0)
    parser.add_argument('-fi', '--fold_index', type=int, help='cv-fold to run',
                        default=-1)
    args = parser.parse_args()
    print(args)

    # Drop could also be a callable according to evaluation.py but not managed as input parameter
    try:
        drop = int(args.drop)
    except ValueError:
        drop = float(args.drop)

    LOAD_EMBEDDINGS = args.load_embeddings > 0
    if LOAD_EMBEDDINGS:
        print("Loading pre-trained embedding", W2V_PATH)
        VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)
        CONDITIONS_WITH_TEXT = ConditionList([
            ('ICD9_defs_txt', PretrainedWordEmbeddingCondition(VECTORS)),
            ('gender', CategoricalCondition(embedding_dim=3, sparse=True, embedding_on_gpu=True)),
            ('ethnicity_grouped', CategoricalCondition(embedding_dim=7, sparse=True, embedding_on_gpu=True)),
            ('admission_type', CategoricalCondition(embedding_dim=5, sparse=True, embedding_on_gpu=True)),
            ('los_hospital', ContinuousCondition(sparse=True)),
            ('age', ContinuousCondition(sparse=True)),
            ('seq_num_len', ContinuousCondition(sparse=True)),
            ('los_icu_lst_mean', ContinuousCondition(sparse=True)),
            ('heartrate_min_lst_mean', ContinuousCondition(sparse=True)),
            ('heartrate_max_lst_mean', ContinuousCondition(sparse=True)),
            ('heartrate_mean_lst_mean', ContinuousCondition(sparse=True)),
            ('sysbp_min_lst_mean', ContinuousCondition(sparse=True)),
            ('sysbp_max_lst_mean', ContinuousCondition(sparse=True)),
            ('sysbp_mean_lst_mean', ContinuousCondition(sparse=True)),
            ('diasbp_min_lst_mean', ContinuousCondition(sparse=True)),
            ('diasbp_max_lst_mean', ContinuousCondition(sparse=True)),
            ('diasbp_mean_lst_mean', ContinuousCondition(sparse=True)),
            ('meanbp_min_lst_mean', ContinuousCondition(sparse=True)),
            ('meanbp_max_lst_mean', ContinuousCondition(sparse=True)),
            ('meanbp_mean_lst_mean', ContinuousCondition(sparse=True)),
            ('resprate_min_lst_mean', ContinuousCondition(sparse=True)),
            ('resprate_max_lst_mean', ContinuousCondition(sparse=True)),
            ('resprate_mean_lst_mean', ContinuousCondition(sparse=True)),
            ('los_icu_lst_delta', ContinuousCondition(sparse=True)),
            ('heartrate_min_lst_delta', ContinuousCondition(sparse=True)),
            ('heartrate_max_lst_delta', ContinuousCondition(sparse=True)),
            ('heartrate_mean_lst_delta', ContinuousCondition(sparse=True)),
            ('sysbp_min_lst_delta', ContinuousCondition(sparse=True)),
            ('sysbp_max_lst_delta', ContinuousCondition(sparse=True)),
            ('sysbp_mean_lst_delta', ContinuousCondition(sparse=True)),
            ('diasbp_min_lst_delta', ContinuousCondition(sparse=True)),
            ('diasbp_max_lst_delta', ContinuousCondition(sparse=True)),
            ('diasbp_mean_lst_delta', ContinuousCondition(sparse=True)),
            ('meanbp_min_lst_delta', ContinuousCondition(sparse=True)),
            ('meanbp_max_lst_delta', ContinuousCondition(sparse=True)),
            ('meanbp_mean_lst_delta', ContinuousCondition(sparse=True)),
            ('resprate_min_lst_delta', ContinuousCondition(sparse=True)),
            ('resprate_max_lst_delta', ContinuousCondition(sparse=True)),
            ('resprate_mean_lst_delta', ContinuousCondition(sparse=True)),
            ('tempc_min_lst_delta', ContinuousCondition(sparse=True)),
            ('tempc_max_lst_delta', ContinuousCondition(sparse=True)),
            ('tempc_mean_lst_delta', ContinuousCondition(sparse=True)),
            ('spo2_min_lst_delta', ContinuousCondition(sparse=True)),
            ('spo2_max_lst_delta', ContinuousCondition(sparse=True)),
            ('spo2_mean_lst_delta', ContinuousCondition(sparse=True)),
            ('glucose_min_lst_delta', ContinuousCondition(sparse=True)),
            ('glucose_max_lst_delta', ContinuousCondition(sparse=True)),
            ('glucose_mean_lst_delta', ContinuousCondition(sparse=True)),
        ])
    else:
        CONDITIONS_WITH_TEXT = CONDITIONS
    MODELS_WITH_HYPERPARAMS = [
        # *** BASELINES
        # Use no metadata (only item sets)
        (Countbased(),
         {"order": [1, 2, 3, 4, 5]}),
        # Use title (as defined in CONDITIONS above)
        (SVDRecommender(10, use_title=False),
         {"dims": [50, 100, 200, 500, 1000]}),

        # *** AEs
        (AAERecommender(adversarial=False, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=None, **ae_params),
         {'lr': [0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [10, 20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]}),
        (AAERecommender(adversarial=False, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=CONDITIONS,
                        **ae_params),
         {'lr': [0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [10, 20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]}),
        (AAERecommender(adversarial=False, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=CONDITIONS_WITH_TEXT,
                        **ae_params),
         {'lr': [0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [10, 20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]}),

        # *** DAEs
        (DAERecommender(conditions=None, **ae_params),
         {'lr': [0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [10, 20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]}),
        (DAERecommender(conditions=CONDITIONS, **ae_params),
         {'lr': [0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [10, 20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]}),
        (DAERecommender(conditions=CONDITIONS_WITH_TEXT, **ae_params),
         {'lr': [0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [10, 20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]}),

        # *** VAEs
        (VAERecommender(conditions=None, **vae_params),
         {'lr': [0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [10, 20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]
          }),
        (VAERecommender(conditions=CONDITIONS, **vae_params),
         {'lr': [0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [10, 20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]
          }),
        (VAERecommender(conditions=CONDITIONS_WITH_TEXT, **vae_params),
         {'lr': [0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [10, 20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]
          }),

        # *** AAEs
        (AAERecommender(adversarial=True, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=None, **ae_params),
         {'prior': ['gauss'],
          'gen_lr': [0.001, 0.01],
          'reg_lr': [0.001, 0.01],
          'n_code': [100, 200],
          'n_epochs': [10, 20],
          'batch_size': [50, 100],
          'n_hidden': [200,500],
          'normalize_inputs': [True]},),
        (
        AAERecommender(adversarial=True, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=CONDITIONS, **ae_params),
        {'prior': ['gauss'],
         'gen_lr': [0.001, 0.01],
         'reg_lr': [0.001, 0.01],
         'n_code': [100, 200],
         'n_epochs': [10, 20],
         'batch_size': [50, 100],
         'n_hidden': [200, 500],
         'normalize_inputs': [True]},),
        (AAERecommender(adversarial=True, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=CONDITIONS_WITH_TEXT,
                        **ae_params),
         {'prior': ['gauss'],
          'gen_lr': [0.001, 0.01],
          'reg_lr': [0.001, 0.01],
          'n_code': [100, 200],
          'n_epochs': [10, 20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]},),
    ]

    main(outfile=args.outfile, min_count=args.min_count, drop=args.drop, n_folds=args.n_folds, model_idx=args.model_idx, fold_index=args.fold_index)

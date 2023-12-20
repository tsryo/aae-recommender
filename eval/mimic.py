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
from aaerec.datasets import Bags, corrupt_lists
from aaerec.transforms import lists2sparse
from aaerec.evaluation import remove_non_missing, evaluate
from aaerec.baselines import Countbased
from aaerec.svd import SVDRecommender
from aaerec.aae import AAERecommender
from aaerec.vae import VAERecommender
from aaerec.dae import DAERecommender
from gensim.models.keyedvectors import KeyedVectors
from aaerec.condition import ConditionList, PretrainedWordEmbeddingCondition, CategoricalCondition, Condition, \
    ContinuousCondition
from irgan.utils import load
from matplotlib import pyplot as plt
import itertools as it
import pandas as pd
import copy
import gc

from CONSTANTS import *
from utils.print_utils import normalize_conditional_data_bags, log, save_object

DEBUG_LIMIT = None
# These need to be implemented in evaluation.py
METRICS = ['map@5', 'maf1@5']
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
# TODO: rethink the embedding_dim for these! you only need 1 dimension for a categorical with 2 categories, but maybe adding a few extra dimensions is helpful?
CONDITIONS = ConditionList([
    #('note_embeddings', ContinuousCondition(sparse=False, size_increment=128)), # todo - make this condition work!
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
d_icd_code_defs = None
icd_code_defs = None
if LOAD_ICD_CODE_TEXT_DEFS:
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
MODEL_NM2IDX = {
    "matrix-factor": 0,
    "svd": 1,

    "AE-no-conditions": 2,
    "AE-demogr-conds": 3,
    "AE-all-conds": 4,

    "DAE-no-conditions": 5,
    "DAE-demogr-conds": 6,
    "DAE-all-conds": 7,

    "VAE-no-conditions": 8,
    "VAE-demogr-conds": 9,
    "VAE-all-conds": 10,

    "AAE-no-conditions": 11,
    "AAE-demogr-conds": 12,
    "AAE-all-conds": 13
}

def prepare_evaluation_kfold_cv(bags, n_folds=5, min_count=None, drop=1, max_codes=None):
    """
    Split data into train val and test sets
    Build vocab on train set and applies it to train, val and test set.
    Args:
        bags (bags): bags of records
        n_folds (int): number of folds
        min_count (int): min number of times an item must have occurred in the train set before it gets considered further
        drop (int/float): how many items should be masked. when 0 < drop < 1, hides percentage of items per record, when drop > 1 hides that many items in each record
        max_codes (int): max number of unique items to consider (i.e., consider only the top N most frequently occurring items)
    """

    # Split 10% validation data.
    train_sets, val_sets, test_sets = bags.create_kfold_train_validate_test(n_folds=n_folds)
    for i in range(n_folds):
        train_sets[i] = normalize_conditional_data_bags(train_sets[i])
        test_sets[i] = normalize_conditional_data_bags(test_sets[i])
        val_sets[i] = normalize_conditional_data_bags(val_sets[i])
    missings_test = []
    missings_val = []
    # Builds vocabulary only on training set
    for i in range(n_folds):
        train_set = train_sets[i]
        test_set = test_sets[i]
        val_set = val_sets[i]
        vocab, __counts = train_set.build_vocab(max_features=max_codes, min_count=min_count, apply=False)
        # Apply vocab (turn track ids into indices)
        train_set = train_set.apply_vocab(vocab)
        # Discard unknown tokens in the test set
        test_set = test_set.apply_vocab(vocab)
        val_set = val_set.apply_vocab(vocab)
        # Corrupt sets (currently set to remove 50% of item list items)
        print("Drop parameter:", drop)

        # corrupt test
        # todo: this needs to corrupt with repeating items into account! right now it messes things up
        noisy, missing = corrupt_lists(test_set.data, drop=drop)
        # some entries might have too few items to drop, resulting in empty missing and a full noisy
        # remove those from the sets (should be just a few)
        entries_to_keep = np.where([len(missing[i]) != 0 for i in range(len(missing))])[0]
        print(f"Removed {len(missing) - len(entries_to_keep)} out of {len(missing)}  rows from test set for having too few icd codes")
        missing = [missing[i] for i in entries_to_keep]
        noisy = [noisy[i] for i in entries_to_keep]
        test_set.data = [test_set.data[i] for i in entries_to_keep]
        test_set.bag_owners = [test_set.bag_owners[i] for i in entries_to_keep]
        owners_to_remove = list(set(test_set.owner_attributes['gender'].keys()).difference(test_set.bag_owners))
        for c_attr in test_set.owner_attributes.keys():
            for owner_to_remove in owners_to_remove:
                test_set.owner_attributes[c_attr].pop(owner_to_remove, None)
        assert len(noisy) == len(missing) == len(test_set)
        # Replace test data with corrupted data
        test_set.data = noisy
        missing_test = missing

        #corrupt val
        noisy, missing = corrupt_lists(val_set.data, drop=drop)
        # some entries might have too few items to drop, resulting in empty missing and a full noisy
        # remove those from the sets (should be just a few)
        entries_to_keep = np.where([len(missing[i]) != 0 for i in range(len(missing))])[0]
        print(f"Removed {len(missing) - len(entries_to_keep)} out of {len(missing)}  rows from test set for having too few icd codes")
        missing = [missing[i] for i in entries_to_keep]
        noisy = [noisy[i] for i in entries_to_keep]
        val_set.data = [val_set.data[i] for i in entries_to_keep]
        val_set.bag_owners = [val_set.bag_owners[i] for i in entries_to_keep]
        owners_to_remove = list(set(val_set.owner_attributes['gender'].keys()).difference(val_set.bag_owners))
        for c_attr in val_set.owner_attributes.keys():
            for owner_to_remove in owners_to_remove:
                val_set.owner_attributes[c_attr].pop(owner_to_remove, None)
        assert len(noisy) == len(missing) == len(val_set)
        # Replace test data with corrupted data
        val_set.data = noisy
        missing_val = missing
        # dont corrut train
        train_sets[i] = train_set

        if 'ICD9_defs_txt' in test_set.owner_attributes.keys():
            test_set = adjust_icd_text_defs_post_corrupt(test_set)
            val_set = adjust_icd_text_defs_post_corrupt(val_set)
        test_sets[i] = test_set
        val_sets[i] = val_set

        missings_test.append(missing_test)
        missings_val.append(missing_val)

    return train_sets, val_sets, test_sets, missings_val, missings_test


def adjust_icd_text_defs_post_corrupt(corrupted_set):
    """
    Removes the icd code text definition of the icd codes that were removed from a record during corruption/masking
    """
    for j in range(0, len(corrupted_set.bag_owners)):
        c_hadm_id = corrupted_set.bag_owners[j]
        get_icd_code_from_index = lambda x, y: [y.index2token[c_x] for c_x in x]
        c_icd_codes = get_icd_code_from_index(corrupted_set.data[j], corrupted_set)
        c_code_defs = [re.sub(r'[^\w\s]', '', d_icd_code_defs[x].lower()) if x in d_icd_code_defs.keys() else '' for
                       x in c_icd_codes]
        corrupted_set.owner_attributes['ICD9_defs_txt'][c_hadm_id] = (' '.join(c_code_defs))
    return corrupted_set


def unpack_patients(patients, icd_code_defs=None, note_embeddings=None):
    """
    Unpacks list of patients in a way that is compatible with our Bags dataset
    format. It is not mandatory that patients are sorted.
    todo: handle multiple occurrences of procedure codes
    """
    bags_of_codes, ids = [], []
    other_attributes = {'ICD9_defs_txt': {},
                        'gender': {},
                        'los_hospital': {},
                        'age': {},
                        'ethnicity_grouped': {},
                        'admission_type': {},
                        'seq_num_len': {},  # model should not learn on how many codes are missing
                        'icd9_code_d_lst': {},
                        'icd9_code_p_lst': {},
                        'los_icu_lst': {},
                        'los_icu_len': {},
                        'icu_stay_start_lst': {},  # todo:  figure out how to represent these lst vars
                        'icu_stay_stop_lst': {},
                        'time_mins_lst': {},
                        'icu_stay_start_lst': {},
                        'icu_stay_start_lst': {},
                        # 'los_icu_lst': {},'heartrate_min_lst': {},'heartrate_max_lst': {},'heartrate_mean_lst': {},'sysbp_min_lst': {},'sysbp_max_lst': {},'sysbp_mean_lst': {},'diasbp_min_lst': {},'diasbp_max_lst': {},'diasbp_mean_lst': {},'meanbp_min_lst': {},'meanbp_max_lst': {},'meanbp_mean_lst': {},'resprate_min_lst': {},'resprate_max_lst': {},'resprate_mean_lst': {},'tempc_min_lst': {},'tempc_max_lst': {},'tempc_mean_lst': {},'spo2_min_lst': {},'spo2_max_lst': {},'spo2_mean_lst': {},'glucose_min_lst': {},'glucose_max_lst': {},'glucose_mean_lst': {},
                        'los_icu_lst_slope': {}, 'heartrate_min_lst_slope': {}, 'heartrate_max_lst_slope': {},
                        'heartrate_mean_lst_slope': {}, 'sysbp_min_lst_slope': {}, 'sysbp_max_lst_slope': {},
                        'sysbp_mean_lst_slope': {}, 'diasbp_min_lst_slope': {}, 'diasbp_max_lst_slope': {},
                        'diasbp_mean_lst_slope': {}, 'meanbp_min_lst_slope': {}, 'meanbp_max_lst_slope': {},
                        'meanbp_mean_lst_slope': {}, 'resprate_min_lst_slope': {}, 'resprate_max_lst_slope': {},
                        'resprate_mean_lst_slope': {}, 'tempc_min_lst_slope': {}, 'tempc_max_lst_slope': {},
                        'tempc_mean_lst_slope': {}, 'spo2_min_lst_slope': {}, 'spo2_max_lst_slope': {},
                        'spo2_mean_lst_slope': {}, 'glucose_min_lst_slope': {}, 'glucose_max_lst_slope': {},
                        'glucose_mean_lst_slope': {},
                        'los_icu_lst_mean': {}, 'heartrate_min_lst_mean': {}, 'heartrate_max_lst_mean': {},
                        'heartrate_mean_lst_mean': {}, 'sysbp_min_lst_mean': {}, 'sysbp_max_lst_mean': {},
                        'sysbp_mean_lst_mean': {}, 'diasbp_min_lst_mean': {}, 'diasbp_max_lst_mean': {},
                        'diasbp_mean_lst_mean': {}, 'meanbp_min_lst_mean': {}, 'meanbp_max_lst_mean': {},
                        'meanbp_mean_lst_mean': {}, 'resprate_min_lst_mean': {}, 'resprate_max_lst_mean': {},
                        'resprate_mean_lst_mean': {},
                        'los_icu_lst_sd': {}, 'heartrate_min_lst_sd': {}, 'heartrate_max_lst_sd': {},
                        'heartrate_mean_lst_sd': {}, 'sysbp_min_lst_sd': {}, 'sysbp_max_lst_sd': {},
                        'sysbp_mean_lst_sd': {}, 'diasbp_min_lst_sd': {}, 'diasbp_max_lst_sd': {},
                        'diasbp_mean_lst_sd': {}, 'meanbp_min_lst_sd': {}, 'meanbp_max_lst_sd': {},
                        'meanbp_mean_lst_sd': {}, 'resprate_min_lst_sd': {}, 'resprate_max_lst_sd': {},
                        'resprate_mean_lst_sd': {}, 'tempc_min_lst_sd': {}, 'tempc_max_lst_sd': {},
                        'tempc_mean_lst_sd': {}, 'spo2_min_lst_sd': {}, 'spo2_max_lst_sd': {}, 'spo2_mean_lst_sd': {},
                        'glucose_min_lst_sd': {}, 'glucose_max_lst_sd': {}, 'glucose_mean_lst_sd': {},
                        'los_icu_lst_delta': {}, 'heartrate_min_lst_delta': {}, 'heartrate_max_lst_delta': {},
                        'heartrate_mean_lst_delta': {}, 'sysbp_min_lst_delta': {}, 'sysbp_max_lst_delta': {},
                        'sysbp_mean_lst_delta': {}, 'diasbp_min_lst_delta': {}, 'diasbp_max_lst_delta': {},
                        'diasbp_mean_lst_delta': {}, 'meanbp_min_lst_delta': {}, 'meanbp_max_lst_delta': {},
                        'meanbp_mean_lst_delta': {}, 'resprate_min_lst_delta': {}, 'resprate_max_lst_delta': {},
                        'resprate_mean_lst_delta': {}, 'tempc_min_lst_delta': {}, 'tempc_max_lst_delta': {},
                        'tempc_mean_lst_delta': {}, 'spo2_min_lst_delta': {}, 'spo2_max_lst_delta': {},
                        'spo2_mean_lst_delta': {}, 'glucose_min_lst_delta': {}, 'glucose_max_lst_delta': {},
                        'glucose_mean_lst_delta': {},
                        'los_icu_lst_min': {}, 'heartrate_min_lst_min': {}, 'heartrate_max_lst_min': {},
                        'heartrate_mean_lst_min': {}, 'sysbp_min_lst_min': {}, 'sysbp_max_lst_min': {},
                        'sysbp_mean_lst_min': {}, 'diasbp_min_lst_min': {}, 'diasbp_max_lst_min': {},
                        'diasbp_mean_lst_min': {}, 'meanbp_min_lst_min': {}, 'meanbp_max_lst_min': {},
                        'meanbp_mean_lst_min': {}, 'resprate_min_lst_min': {}, 'resprate_max_lst_min': {},
                        'resprate_mean_lst_min': {}, 'tempc_min_lst_min': {}, 'tempc_max_lst_min': {},
                        'tempc_mean_lst_min': {}, 'spo2_min_lst_min': {}, 'spo2_max_lst_min': {},
                        'spo2_mean_lst_min': {}, 'glucose_min_lst_min': {}, 'glucose_max_lst_min': {},
                        'glucose_mean_lst_min': {},
                        'los_icu_lst_max': {}, 'heartrate_min_lst_max': {}, 'heartrate_max_lst_max': {},
                        'heartrate_mean_lst_max': {}, 'sysbp_min_lst_max': {}, 'sysbp_max_lst_max': {},
                        'sysbp_mean_lst_max': {}, 'diasbp_min_lst_max': {}, 'diasbp_max_lst_max': {},
                        'diasbp_mean_lst_max': {}, 'meanbp_min_lst_max': {}, 'meanbp_max_lst_max': {},
                        'meanbp_mean_lst_max': {}, 'resprate_min_lst_max': {}, 'resprate_max_lst_max': {},
                        'resprate_mean_lst_max': {}, 'tempc_min_lst_max': {}, 'tempc_max_lst_max': {},
                        'tempc_mean_lst_max': {}, 'spo2_min_lst_max': {}, 'spo2_max_lst_max': {},
                        'spo2_mean_lst_max': {}, 'glucose_min_lst_max': {}, 'glucose_max_lst_max': {},
                        'glucose_mean_lst_max': {},
                        'heartrate_min_lst_mm': {}, 'heartrate_max_lst_mm': {}, 'heartrate_mean_lst_mm': {},
                        'sysbp_min_lst_mm': {}, 'sysbp_max_lst_mm': {}, 'sysbp_mean_lst_mm': {},
                        'diasbp_min_lst_mm': {}, 'diasbp_max_lst_mm': {}, 'diasbp_mean_lst_mm': {},
                        'meanbp_min_lst_mm': {}, 'meanbp_max_lst_mm': {}, 'meanbp_mean_lst_mm': {},
                        'resprate_min_lst_mm': {}, 'resprate_max_lst_mm': {}, 'resprate_mean_lst_mm': {},
                        'tempc_min_lst_mm': {}, 'tempc_max_lst_mm': {}, 'tempc_mean_lst_mm': {}, 'spo2_min_lst_mm': {},
                        'spo2_max_lst_mm': {}, 'spo2_mean_lst_mm': {}, 'glucose_min_lst_mm': {},
                        'glucose_max_lst_mm': {}, 'glucose_mean_lst_mm': {}
                        }
    d_icd_code_defs = {}
    dup_keys = []

    # map icd codes with their text description
    if icd_code_defs is not None:
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
    dummy_vals = None
    if note_embeddings is not None:
        dummy_vals = list(note_embeddings.values())[0]
        dummy_vals = [x * 0.0 for x in dummy_vals]

    for patient in patients:
        # Extract ids
        ids.append(patient["hadm_id"])
        # Put all subjects assigned to the patient in here (subject = icd code)
        try:
            # Subject may be missing
            bags_of_codes.append(patient["icd9_code_d_lst"] + patient["icd9_code_p_lst"])
        except KeyError:
            bags_of_codes.append([])
        #  features that can be easily used: age, gender, ethnicity, adm_type, icu_stay_seq, hosp_stay_seq
        # Use dict here such that we can also deal with unsorted ids
        c_hadm_id = patient["hadm_id"]
        # iterate over all features
        for c_var in list(other_attributes.keys()):
            if c_var == "ICD9_defs_txt" or c_var not in patient.keys():
                continue # skip if its text description of codes, or if feature not present in current patient
            other_attributes[c_var][c_hadm_id] = patient[c_var]
        # handle text description on codes
        if icd_code_defs is not None:
            c_icd_codes = other_attributes['icd9_code_d_lst'][c_hadm_id] + other_attributes['icd9_code_p_lst'][c_hadm_id]
            c_code_defs = [re.sub(r'[^\w\s]', '', d_icd_code_defs[x].lower()) if x in d_icd_code_defs.keys() else '' for x
                           in c_icd_codes]
            other_attributes['ICD9_defs_txt'][c_hadm_id] = (' '.join(c_code_defs))
        if note_embeddings is not None:

            if str(c_hadm_id) not in note_embeddings.keys():
                # print("help")
                note_embeddings[str(c_hadm_id)] = dummy_vals
            # if str(c_hadm_id) in note_embeddings.keys():
            if 'note_embeddings' not in other_attributes.keys():
                other_attributes['note_embeddings'] = {c_hadm_id : note_embeddings[str(c_hadm_id)]}
            else:
                other_attributes['note_embeddings'][c_hadm_id] = note_embeddings[str(c_hadm_id)]

    # bag_of_codes and ids should have corresponding indices
    # remove empty attributes
    empty_other_attr = [x for x in other_attributes.keys() if len(other_attributes[x]) == 0]
    for k in empty_other_attr:
        other_attributes.pop(k, None)
    return bags_of_codes, ids, other_attributes, d_icd_code_defs


# main functions
def run_cv_pipeline(bags, drop, min_count, n_folds, logfile, model, hyperparams_to_try, split_sets_filename=None,
                    fold_index=-1, max_codes = None):
    metrics_per_drop_per_model = []
    train_sets, val_sets, test_sets, y_vals, y_tests = None, None, None, None, None
    # first try to load split datasets from file
    if split_sets_filename is not None and os.path.exists(split_sets_filename):
        with (open(split_sets_filename, "rb")) as openfile:
            train_sets, val_sets, test_sets, y_vals, y_tests = pickle.load(openfile)
    else:
        train_sets, val_sets, test_sets, y_vals, y_tests = prepare_evaluation_kfold_cv(bags, min_count=min_count, drop=drop,
                                                                               n_folds=n_folds, max_codes= max_codes)
    # create split datasets file if one was not there already
    if split_sets_filename is not None and not os.path.exists(split_sets_filename):
        save_object((train_sets, val_sets, test_sets, y_vals, y_tests), split_sets_filename)

    del bags
    best_params = None
    for c_fold in range(n_folds):
        # init
        if c_fold != 0:  # load from file trian/test/val sets and then delete them
            with (open(split_sets_filename, "rb")) as openfile:
                train_sets, val_sets, test_sets, y_vals, y_tests = pickle.load(openfile)
        # if specified to run specific fold index, skip others
        if fold_index >= 0 and c_fold != fold_index:
            continue
        log("FOLD = {}".format(c_fold), logfile=logfile)
        log("TIME: {}".format(datetime.now().strftime("%Y-%m-%d-%H:%M")), logfile=logfile)
        train_set = train_sets[c_fold]
        val_set = val_sets[c_fold]
        test_set = test_sets[c_fold]
        y_val = y_vals[c_fold]
        y_test = y_tests[c_fold]
        log("Train set:", logfile=logfile)
        log(train_set, logfile=logfile)

        log("Validation set:", logfile=logfile)
        log(val_set, logfile=logfile)

        log("Test set:", logfile=logfile)
        log(test_set, logfile=logfile)

        # reduce memory consumption
        del train_sets
        del val_sets
        del test_sets
        # todo: something doesnt quite add-up here... test with element at index 64, it has 3 times the code 86
        # THE GOLD (put into sparse matrix)
        # [(i,j) for j,v in enumerate(y_tests) for i,x in enumerate(v) if len(x) != len(set(x)) ] # which set has repeating items?
        y_test = lists2sparse(y_test, test_set.size(1)).tocsr(copy=False)
        # the known items in the test set, just to not recompute
        x_test = lists2sparse(test_set.data, test_set.size(1)).tocsr(copy=False)

        y_val = lists2sparse(y_val, val_set.size(1)).tocsr(copy=False)
        x_val = lists2sparse(val_set.data, val_set.size(1)).tocsr(copy=False)

        # use model copy to reset model state on each fold
        model_cpy = None
        if model_cpy is None and not hasattr(model, 'reset_parameters'):
            model_cpy = copy.deepcopy(model)
        log('=' * 78, logfile=logfile)
        log(model, logfile=logfile)
        log("training model \n TIME: {}  ".format(datetime.now().strftime("%Y-%m-%d-%H:%M")), logfile=logfile)

        # Optimize hyperparams
        # when we specify a fold, we assume the hyperparam tunning was already done
        if fold_index >= 0 or ('batch_size' in hyperparams_to_try.keys() and
                               type(hyperparams_to_try['batch_size']) == int):
            model.model_params = hyperparams_to_try
        # for time constraints, just run hyperparams once
        elif hyperparams_to_try is not None and c_fold == 0:
            if sum([0 if len(x) == 1 else 1 for x in hyperparams_to_try.values()]) != 0:
                log('Optimizing on following hyper params: ', logfile=logfile)
                log(hyperparams_to_try, logfile=logfile)
                # use only a third of training set to tune params on (reduce running time)
                tunning_train_set = train_set.clone(0, int(len(train_set.data) * 1.0))

                best_params, _, _ = hyperparam_optimize(model, tunning_train_set, val_set.clone(), y_val,
                                                        tunning_params=hyperparams_to_try, drop=drop)
            else: # nothing to try, only 1 value for each param provided
                best_params = {k: v[0] for k, v in hyperparams_to_try.items()}
            log('After hyperparam_optimize, best params: ', logfile=logfile)
            log(best_params, logfile=logfile)
            model.model_params = best_params

        # Reset model state
        if hasattr(model, 'reset_parameters'):
            model.reset_parameters()
        else:
            log(f"WARNING: no reset_parameters method call for model {model}. Calling deepcopy instead", logfile=logfile)
            model = copy.deepcopy(model_cpy)
            # del model_cpy
        gc.collect()
        # todo: we are using val_set only for hyperparam tuning eval and then throwing it away! would be nicer to combine it into the training set to be used below
        # Training
        model.train(train_set) # [i for i,x in enumerate(train_set.data) if len(x) != len(set(x)) ] # check for repeating item examples

        # Prediction
        y_pred = model.predict(test_set)
        log(" TRAIN AND PREDICT COMPLETE \n TIME: {}".format(datetime.now().strftime("%Y-%m-%d-%H:%M")),
            logfile=logfile)
        # Sanity-fix #1, make sparse stuff dense, expect array
        if sp.issparse(y_pred):
            y_pred = y_pred.toarray()
        else:
            y_pred = np.asarray(y_pred)
        # Sanity-fix, remove predictions for already present items
        y_pred = remove_non_missing(y_pred, x_test, copy=False)

        # save model test predictions + actual test values + test inputs [may be useful to look at later]
        save_payload = {"test_set": test_set, "x_test": x_test, "y_pred": y_pred}
        save_object(save_payload, '{}_{}_res.pkl'.format(str(model)[0:64], c_fold))

        # reduce memory usage
        del test_set
        del train_set
        del val_set

        # Evaluate metrics
        results = evaluate(y_test, y_pred, METRICS)
        log("-" * 78, logfile=logfile)
        for metric, stats in zip(METRICS, results):
            log("* FOLD#{} {}: {} ({})".format(c_fold, metric, *stats), logfile=logfile)
            metrics_per_drop_per_model.append([c_fold, drop, str(model), metric, stats[0], stats[1]])
        log('=' * 78, logfile=logfile)

    # Return result metrics
    metrics_df = pd.DataFrame(metrics_per_drop_per_model,
                              columns=['fold', 'drop', 'model', 'metric', 'metric_val', 'metric_std'])
    return metrics_df



def hyperparam_optimize(model, train_set, val_set, y_val,
                        tunning_params={'prior': ['gauss'], 'gen_lr': [0.001], 'reg_lr': [0.001],
                                        'n_code': [10, 25, 50], 'n_epochs': [20, 50, 100],
                                        'batch_size': [100], 'n_hidden': [100], 'normalize_inputs': [True]},
                        metric='maf1@10', drop=0.5):
    # assert all(x in list(c_params.keys()) for x in list(tunning_params.keys()))
    # col - hyperparam name, row = specific combination of values to try
    exp_grid_n_combs = [len(x) for x in tunning_params.values()]
    exp_grid_cols = tunning_params.keys()
    l_rows = list(it.product(*tunning_params.values()))
    exp_grid_df = pd.DataFrame(l_rows, columns=exp_grid_cols)
    reses = []
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


# util
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


# Use only one of the vitals vars (heartrate), no text
def simplify_patients_dict(patients):
    # we may want to also predict the sequence in which icd codes were assigned
    # for procedural.
    # believe there is a temporal aspect, and relation with patient vitals and procedure
    # also, need to keep in mind that icd proc codes can repeat, diag are unique
    # note - icd codes in list are sorted by seq_num
    # (we only know if one icd proc/diag code came before another, but not when in relation to anything else)
    # in that case is it still useful to keep order in mind?
    # assume yes, but dont do anything with this info for now..
    keys_to_keep = ["hadm_id", "admission_type", "age", "ethnicity_grouped", "gender",
                  "icd9_code_d_lst", "icd9_code_p_lst", "icu_stay_start_lst", "icu_stay_stop_lst",
                  "los_hospital",
                  "los_icu_len", "los_icu_lst",
                  "seq_num_d_len", "seq_num_p_len",
                  "time_mins_lst",
                  "heartrate_min_lst", "heartrate_min_lst_delta", "heartrate_min_lst_max",
                  "heartrate_min_lst_mean", "heartrate_min_lst_min", "heartrate_min_lst_mm",
                  "heartrate_min_lst_sd", "heartrate_min_lst_slope"]
    keys_to_remove = [key for key in patients[0].keys() if key not in keys_to_keep]
    for i in range(0, len(patients)):
        for key in keys_to_remove:
            patients[i].pop(key)
    return patients


##### MAIN

# @param fold_index - run a specific fold of CV (-1 = run all folds)
# see lines at parser.add_argument for param details
def main(max_codes=100, min_count=10, drop=0.5, n_folds=5, model_idx=-1, outfile='out.csv', logfile='out.log', fold_index=-1):
    """ Main function for training and evaluating AAE methods on MIMIC data """
    # min_count = 10
    print('drop = {}; min_count = {}, max_codes = {}, n_folds = {}, model_idx = {}'.format(drop, min_count, max_codes, n_folds, model_idx))
    print("Loading data from", IN_DATA_PATH_DEMO_ICD_CODES)
    patients = load(IN_DATA_PATH_DEMO_ICD_CODES)
    note_embs = load(EMBEDDINGS_FILENAME)
    # patients = patients[0:100]
    patients = simplify_patients_dict(patients)
    # print("plt_pat_histograms_demog...")
    # plt_pat_histograms_demog(patients)
    # return None
    print("Unpacking MIMIC data...")
    note_embs_clean = {}
    for d in note_embs:
        note_embs_clean[d['hadm_id']] = d['txt_embedding']

    note_embs = note_embs_clean

    bags_of_patients, ids, side_info, d_icd_code_defs = unpack_patients(patients, icd_code_defs, note_embs)  # with conditions
    assert (len(set(ids)) == len(ids))
    print_icd_code_summary_statistics(d_icd_code_defs, logfile, side_info, patients)
    del patients

    # repeating_items = [i for i,v in enumerate(bags_of_patients) if len(v) - len(set(v)) != 0]
    # pd.Series(bags_of_patients[repeating_items[0]]).value_counts()

    bags = Bags(bags_of_patients, ids, side_info)  # with conditions, bags_of_patients may have repeating items!
    log("Whole dataset:", logfile=logfile)

    log(bags, logfile=logfile)



    log("drop = {}, min_count = {}".format(drop, min_count), logfile=logfile)
    sets_to_try = MODELS_WITH_HYPERPARAMS if model_idx < 0 else [MODELS_WITH_HYPERPARAMS[model_idx]]
    models_to_del = [i for i in range(len(MODELS_WITH_HYPERPARAMS)) if i != model_idx]
    for i in list(reversed(models_to_del)):
        del MODELS_WITH_HYPERPARAMS[i]


    for model, hyperparams_to_try in sets_to_try:
        if model.conditions is not None: # remove conditions that are not present in bags
            missing_from_bags = set(list(model.conditions)).difference(set(list(bags.owner_attributes.keys())))
            conds_nms_2keep = [condition for condition in model.conditions if str(condition) not in missing_from_bags]
            cond_items_2keep = [c_i for c_i in model.conditions.items() if c_i[0] in conds_nms_2keep]
            model.conditions = ConditionList(cond_items_2keep)

        # set some filenames
        indata_filenm = IN_DATA_PATH_DEMO_ICD_CODES.split("/")[-1][:-5]
        splitsets_fn = f"splitsets{indata_filenm}.pkl"
        c_fn = './{}_{}_{}.tsv'.format(outfile[:-4], str(model)[0:48], fold_index)

        # run the CV pipeline
        metrics_df = run_cv_pipeline(bags, drop, min_count, n_folds, logfile, model, hyperparams_to_try,
                                     split_sets_filename=splitsets_fn, fold_index=fold_index, max_codes=max_codes)

        # Store per-fold and pooled results
        for c_metric in set(metrics_df['metric'].to_list()): # pool results across folds (todo: assumes only 1 model is present in metrics_df!)
            map_rows = metrics_df.loc[metrics_df['metric'] == c_metric]
            mean_val = map_rows['metric_val'].mean()
            mean_sd = map_rows['metric_val'].std()
            mean_row = {'fold': 'Pooled', 'drop': '', 'model': '', 'metric': c_metric, 'metric_val': mean_val, 'metric_std': mean_sd}
            metrics_df = metrics_df.append(mean_row, ignore_index=True)

        metrics_df.to_csv(c_fn, sep='\t')
        print(f"Wrote results to {c_fn}")
        print("DONE")


def print_icd_code_summary_statistics(d_icd_code_defs, logfile, side_info, patients = None):
    all_codes = [y for x in patients for y in x['icd9_code_p_lst'] + x['icd9_code_d_lst']]
    all_codes = [c for c_list in list(side_info['icd9_code_d_lst'].values()) for c in c_list]
    all_codes += [c for c_list in list(side_info['icd9_code_p_lst'].values()) for c in c_list]
    t_codes = pd.value_counts(all_codes)
    filtered_t_codes = {category: count for category,count in t_codes.items() if count >= 200}
    filtered_t_codes = pd.Series(filtered_t_codes)
    x_values = np.arange(len(filtered_t_codes))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_values, filtered_t_codes, width = 1)
    ax.set_title('ICD9 code frequency')
    ax.set_ylabel('Counts')
    ax.set_xlabel('ICD codes')
    ax.set_xticks([])
    ax.set_xticklabels([])
    plt.show()
    plt.savefig('../../plt/icd9-code-freq-counts-300.png', dpi=300, bbox_inches='tight')
    # b_chart = filtered_t_codes.plot(kind='bar', width = 0.8)
    # b_chart


    n_codes_uniq = len(t_codes)
    n_codes_all = len(all_codes)
    code_counts = pd.value_counts(all_codes)
    # all_unique_codes = set(all_codes)
    # all_unique_code_defs = set([cd for cd_list in list(side_info['ICD9_defs_txt'].values()) for cd in cd_list])
    log("Total number of codes in current dataset = {}".format(n_codes_all), logfile=logfile)
    log("Total number of unique codes in current dataset = {}".format(n_codes_uniq), logfile=logfile)
    code_percentages = list(zip(code_counts, code_counts.index))
    code_percentages = [(val / n_codes_all, code) for val, code in code_percentages]
    code_percentages_accum = code_percentages
    for i in range(len(code_percentages)):
        if i > 0:
            code_percentages_accum[i] = (
                code_percentages_accum[i][0] + code_percentages_accum[i - 1][0], code_percentages_accum[i][1])
        else:
            code_percentages_accum[i] = (code_percentages_accum[i][0], code_percentages_accum[i][1])
    for i in range(len(code_percentages_accum)):
        c_code = code_percentages_accum[i][1]
        c_percentage = code_percentages_accum[i][0]
        c_def = d_icd_code_defs[c_code] if c_code in d_icd_code_defs.keys() else ''
        log("{}\t#{}\tcode: {}\t( desc: {})".format(c_percentage, i + 1, c_code, c_def), logfile=logfile)
        if c_percentage >= 0.5:
            log("first {} codes account for 50% of all code occurrences".format(i), logfile=logfile)
            log("Remaining {} codes account for remaining 50% of all code occurrences".format(n_codes_uniq - i),
                logfile=logfile)
            log("Last 1000 codes account for only {}% of data".format(
                (1 - code_percentages_accum[n_codes_uniq - 1000][0]) * 100), logfile=logfile)
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile',
                        help="File to write results to",
                        default='../results/test_run_{}.tsv'.format(datetime.now().strftime("%Y-%m-%d-%H:%M")))
    parser.add_argument('-l', '--logfile',
                        help="File to write logs to",
                        default='../run-logs/test_run_{}.log'.format(datetime.now().strftime("%Y-%m-%d-%H:%M")))
    parser.add_argument('-m', '--min-count', type=int,
                        default=50,
                        help="Minimum frequency count a code should have before it is used")
    parser.add_argument('-mc', '--max-codes', type=int,
                        default=100,
                        help="Use only top n most frequent codes")
    parser.add_argument('-dr', '--drop', type=float,
                        default=0.5,
                        help='Drop parameter - when between 0 and 1 behaves as percentage drop, when greater behaves as number of codes to drop.')
    parser.add_argument('-nf', '--n_folds', type=int,
                        help='Number of folds in cross-validation', default=5)
    parser.add_argument('-mn', '--model_name', type=str,
                        default="AAE-all-conds",
                        help=f'Name of model to use. Allowed values = {MODEL_NM2IDX.keys()}')
    parser.add_argument('-le', '--load_embeddings', type=int,
                        default=0,
                        help='Load w2v embeddings?',)
    parser.add_argument('-fi', '--fold_index', type=int,
                        default=-1,
                        help='Run a specific fold of cv. -1 to run all folds.')
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
        CONDITIONS_WITH_TEXT = CONDITIONS.append([('ICD9_defs_txt', PretrainedWordEmbeddingCondition(VECTORS) ) ])

    else:
        CONDITIONS_WITH_TEXT = CONDITIONS

    HPS_COUNTBASED = {"order": [1, 2, 3, 4, 5]}
    HPS_SVD = {"dims": [50, 100, 200, 500, 1000]}
    HPS_AE = {'lr': [0.001, 0.01],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
          'n_code': [100, 200],
          'n_epochs': [10, 20],
          'batch_size': [50, 100],
          'n_hidden': [200, 500],
          'normalize_inputs': [True]}
    HPS_AAE = {'prior': ['categorical'], # gauss, bernoulli
          'gen_lr': [0.01], # encoder/decoder LR
          'reg_lr': [0.001], # generator LR
          'disc_lr': [0.00005], # discriminator LR
          'n_code': [150], # n of neurons in the last layer of the encoder, and the first layer of the decoder (i.e., the bottleneck)
          'n_epochs': [70],
          'batch_size': [200],
          'n_hidden': [600], # place where magic happens
          'normalize_inputs': [True]}



    MODELS_WITH_HYPERPARAMS = [
        # *** BASELINES
        # Use no metadata (only item sets)
        (Countbased(), HPS_COUNTBASED), # 0
        # Use title (as defined in CONDITIONS above)
        (SVDRecommender(10, use_title=False), HPS_SVD),
        # *** BASELINES END
        # *** AEs
        (AAERecommender(adversarial=False, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=None, **ae_params), HPS_AE),
        (AAERecommender(adversarial=False, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=CONDITIONS, **ae_params), HPS_AE),
        (AAERecommender(adversarial=False, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=CONDITIONS_WITH_TEXT, **ae_params), HPS_AE),
        # *** DAEs
        (DAERecommender(conditions=None, **ae_params), HPS_AE),
        (DAERecommender(conditions=CONDITIONS, **ae_params), HPS_AE),
        (DAERecommender(conditions=CONDITIONS_WITH_TEXT, **ae_params), HPS_AE),
        # *** VAEs
        (VAERecommender(conditions=None, **vae_params), HPS_AE),
        (VAERecommender(conditions=CONDITIONS, **vae_params), HPS_AE),
        (VAERecommender(conditions=CONDITIONS_WITH_TEXT, **vae_params), HPS_AE),
        # *** AAEs
        (AAERecommender(adversarial=True, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=None, **ae_params), HPS_AAE),
        (AAERecommender(adversarial=True, prior='gauss', gen_lr=0.1, reg_lr=0.00001, conditions=CONDITIONS, **ae_params), HPS_AAE), # index = 12
        (AAERecommender(adversarial=True, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=CONDITIONS_WITH_TEXT, **ae_params), HPS_AAE),
    ]




    main(outfile=args.outfile, logfile=args.logfile, min_count=args.min_count, drop=args.drop, n_folds=args.n_folds, model_idx=MODEL_NM2IDX[args.model_name],
         fold_index=args.fold_index, max_codes=args.max_codes)

"""
Executable to run AAE on the MIMIC Dataset

Run via:

`python3 eval/mimic.py -m <min_count> -o logfile.txt`

"""
import argparse
import collections

from datetime import datetime
import numpy as np
import scipy.sparse as sp
# from numpy.distutils.command.config import config

from aaerec.datasets import Bags, corrupt_sets
from aaerec.transforms import lists2sparse
from aaerec.evaluation import remove_non_missing, evaluate
from aaerec.baselines import Countbased
from aaerec.svd import SVDRecommender
from aaerec.aae import AAERecommender, DecodingRecommender
from aaerec.vae import VAERecommender
from aaerec.dae import DAERecommender
from gensim.models.keyedvectors import KeyedVectors
from aaerec.condition import ConditionList, PretrainedWordEmbeddingCondition, CategoricalCondition, Condition, ContinuousCondition
from eval.fiv import load
from matplotlib import pyplot as plt
from sklearn import preprocessing
import itertools as it
import pandas as pd
import copy

# Set it to the Reuters RCV dataset
DATA_PATH = "../MIMIC/test_5k.json"
DEBUG_LIMIT = None
# These need to be implemented in evaluation.py
METRICS = ['mrr', 'map', 'f1']

# Set it to the word2vec-Google-News-corpus file TODO see if useful when using conditions
W2V_PATH = "/mnt/c/Development/github/Python/GoogleNews-vectors-negative300.bin.gz"
W2V_IS_BINARY = True
VECTORS = []
LOAD_EMBEDDINGS = False
if LOAD_EMBEDDINGS:
    print("Loading pre-trained embedding", W2V_PATH)
    VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)

# Hyperparameters TODO adapt them
ae_params = {
    'n_code': 50,
    'n_epochs': 100,
    # 'embedding': VECTORS,
    'batch_size': 100,
    'n_hidden': 100,
    'normalize_inputs': True,
}
vae_params = {
    'n_code': 50,
    # VAE results get worse with more epochs in preliminary optimization
    # (Pumed with threshold 50)
    'n_epochs': 50,
    'batch_size': 100,
    'n_hidden': 100,
    'normalize_inputs': True,
}
HYPERPARAMS_TO_TRY_OPTIM = {
    "<class 'aaerec.aae.AAERecommender'>" : {'prior': ['gauss'],
                                             'gen_lr': [0.0001, 0.001, 0.01], #[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
                                             'reg_lr': [0.001],
                                             'n_code': [50],
                                             'n_epochs': [30],
                                             'batch_size': [100],
                                             'n_hidden': [50],
                                             'normalize_inputs': [True]
                                             },
    "<class 'aaerec.aae.VAERecommender'>": {'prior': ['gauss'],
                                             'gen_lr': [0.0001, 0.001, 0.01], #[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
                                             'reg_lr': [0.001],
                                             'n_code': [50],
                                             'n_epochs': [30],
                                             'batch_size': [100],
                                             'n_hidden': [50],
                                             'normalize_inputs': [True]
                                             },
    "<class 'aaerec.aae.DAERecommender'>": {'prior': ['gauss'],
                                             'gen_lr': [0.0001, 0.001, 0.01], #[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
                                             'reg_lr': [0.001],
                                             'n_code': [50],
                                             'n_epochs': [30],
                                             'batch_size': [100],
                                             'n_hidden': [50],
                                             'normalize_inputs': [True]
                                             }

}

# Metadata to use TODO adapt them
# optional conditions (ICD9 codes not optional)
# todo: add _mean
CONDITIONS = ConditionList([
    ('gender', CategoricalCondition(embedding_dim=3, sparse=True, embedding_on_gpu=True)),
    ('ethnicity_grouped', CategoricalCondition(embedding_dim=7,  sparse=True, embedding_on_gpu=True)),
    ('admission_type', CategoricalCondition(embedding_dim=5, sparse=True, embedding_on_gpu=True)),
    ('los_hospital', ContinuousCondition(sparse=True)),
    ('age', ContinuousCondition(sparse=True)),
    ('seq_num_len', ContinuousCondition(sparse=True)),
    ('los_icu_lst_slope', ContinuousCondition(sparse=True)),
    ('heartrate_min_lst_slope', ContinuousCondition(sparse=True)),
    ('heartrate_max_lst_slope', ContinuousCondition(sparse=True)),
    ('heartrate_mean_lst_slope', ContinuousCondition(sparse=True)),
    ('sysbp_min_lst_slope', ContinuousCondition(sparse=True)),
    ('sysbp_max_lst_slope', ContinuousCondition(sparse=True)),
    ('sysbp_mean_lst_slope', ContinuousCondition(sparse=True)),
    ('diasbp_min_lst_slope', ContinuousCondition(sparse=True)),
    ('diasbp_max_lst_slope', ContinuousCondition(sparse=True)),
    ('diasbp_mean_lst_slope', ContinuousCondition(sparse=True)),
    ('meanbp_min_lst_slope', ContinuousCondition(sparse=True)),
    ('meanbp_max_lst_slope', ContinuousCondition(sparse=True)),
    ('meanbp_mean_lst_slope', ContinuousCondition(sparse=True)),
    ('resprate_min_lst_slope', ContinuousCondition(sparse=True)),
    ('resprate_max_lst_slope', ContinuousCondition(sparse=True)),
    ('resprate_mean_lst_slope', ContinuousCondition(sparse=True)),
    ('tempc_min_lst_slope', ContinuousCondition(sparse=True)),
    ('tempc_max_lst_slope', ContinuousCondition(sparse=True)),
    ('tempc_mean_lst_slope', ContinuousCondition(sparse=True)),
    ('spo2_min_lst_slope', ContinuousCondition(sparse=True)),
    ('spo2_max_lst_slope', ContinuousCondition(sparse=True)),
    ('spo2_mean_lst_slope', ContinuousCondition(sparse=True)),
    ('glucose_min_lst_slope', ContinuousCondition(sparse=True)),
    ('glucose_max_lst_slope', ContinuousCondition(sparse=True)),
    ('glucose_mean_lst_slope', ContinuousCondition(sparse=True)),
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
    ('los_icu_lst_sd', ContinuousCondition(sparse=True)),
    ('heartrate_min_lst_sd', ContinuousCondition(sparse=True)),
    ('heartrate_max_lst_sd', ContinuousCondition(sparse=True)),
    ('heartrate_mean_lst_sd', ContinuousCondition(sparse=True)),
    ('sysbp_min_lst_sd', ContinuousCondition(sparse=True)),
    ('sysbp_max_lst_sd', ContinuousCondition(sparse=True)),
    ('sysbp_mean_lst_sd', ContinuousCondition(sparse=True)),
    ('diasbp_min_lst_sd', ContinuousCondition(sparse=True)),
    ('diasbp_max_lst_sd', ContinuousCondition(sparse=True)),
    ('diasbp_mean_lst_sd', ContinuousCondition(sparse=True)),
    ('meanbp_min_lst_sd', ContinuousCondition(sparse=True)),
    ('meanbp_max_lst_sd', ContinuousCondition(sparse=True)),
    ('meanbp_mean_lst_sd', ContinuousCondition(sparse=True)),
    ('resprate_min_lst_sd', ContinuousCondition(sparse=True)),
    ('resprate_max_lst_sd', ContinuousCondition(sparse=True)),
    ('resprate_mean_lst_sd', ContinuousCondition(sparse=True)),
    ('tempc_min_lst_sd', ContinuousCondition(sparse=True)),
    ('tempc_max_lst_sd', ContinuousCondition(sparse=True)),
    ('tempc_mean_lst_sd', ContinuousCondition(sparse=True)),
    ('spo2_min_lst_sd', ContinuousCondition(sparse=True)),
    ('spo2_max_lst_sd', ContinuousCondition(sparse=True)),
    ('spo2_mean_lst_sd', ContinuousCondition(sparse=True)),
    ('glucose_min_lst_sd', ContinuousCondition(sparse=True)),
    ('glucose_max_lst_sd', ContinuousCondition(sparse=True)),
    ('glucose_mean_lst_sd', ContinuousCondition(sparse=True)),
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
    ('los_icu_lst_min', ContinuousCondition(sparse=True)),
    ('heartrate_min_lst_min', ContinuousCondition(sparse=True)),
    ('heartrate_max_lst_min', ContinuousCondition(sparse=True)),
    ('heartrate_mean_lst_min', ContinuousCondition(sparse=True)),
    ('sysbp_min_lst_min', ContinuousCondition(sparse=True)),
    ('sysbp_max_lst_min', ContinuousCondition(sparse=True)),
    ('sysbp_mean_lst_min', ContinuousCondition(sparse=True)),
    ('diasbp_min_lst_min', ContinuousCondition(sparse=True)),
    ('diasbp_max_lst_min', ContinuousCondition(sparse=True)),
    ('diasbp_mean_lst_min', ContinuousCondition(sparse=True)),
    ('meanbp_min_lst_min', ContinuousCondition(sparse=True)),
    ('meanbp_max_lst_min', ContinuousCondition(sparse=True)),
    ('meanbp_mean_lst_min', ContinuousCondition(sparse=True)),
    ('resprate_min_lst_min', ContinuousCondition(sparse=True)),
    ('resprate_max_lst_min', ContinuousCondition(sparse=True)),
    ('resprate_mean_lst_min', ContinuousCondition(sparse=True)),
    ('tempc_min_lst_min', ContinuousCondition(sparse=True)),
    ('tempc_max_lst_min', ContinuousCondition(sparse=True)),
    ('tempc_mean_lst_min', ContinuousCondition(sparse=True)),
    ('spo2_min_lst_min', ContinuousCondition(sparse=True)),
    ('spo2_max_lst_min', ContinuousCondition(sparse=True)),
    ('spo2_mean_lst_min', ContinuousCondition(sparse=True)),
    ('glucose_min_lst_min', ContinuousCondition(sparse=True)),
    ('glucose_max_lst_min', ContinuousCondition(sparse=True)),
    ('glucose_mean_lst_min', ContinuousCondition(sparse=True)),
    ('los_icu_lst_max', ContinuousCondition(sparse=True)),
    ('heartrate_min_lst_max', ContinuousCondition(sparse=True)),
    ('heartrate_max_lst_max', ContinuousCondition(sparse=True)),
    ('heartrate_mean_lst_max', ContinuousCondition(sparse=True)),
    ('sysbp_min_lst_max', ContinuousCondition(sparse=True)),
    ('sysbp_max_lst_max', ContinuousCondition(sparse=True)),
    ('sysbp_mean_lst_max', ContinuousCondition(sparse=True)),
    ('diasbp_min_lst_max', ContinuousCondition(sparse=True)),
    ('diasbp_max_lst_max', ContinuousCondition(sparse=True)),
    ('diasbp_mean_lst_max', ContinuousCondition(sparse=True)),
    ('meanbp_min_lst_max', ContinuousCondition(sparse=True)),
    ('meanbp_max_lst_max', ContinuousCondition(sparse=True)),
    ('meanbp_mean_lst_max', ContinuousCondition(sparse=True)),
    ('resprate_min_lst_max', ContinuousCondition(sparse=True)),
    ('resprate_max_lst_max', ContinuousCondition(sparse=True)),
    ('resprate_mean_lst_max', ContinuousCondition(sparse=True)),
    ('tempc_min_lst_max', ContinuousCondition(sparse=True)),
    ('tempc_max_lst_max', ContinuousCondition(sparse=True)),
    ('tempc_mean_lst_max', ContinuousCondition(sparse=True)),
    ('spo2_min_lst_max', ContinuousCondition(sparse=True)),
    ('spo2_max_lst_max', ContinuousCondition(sparse=True)),
    ('spo2_mean_lst_max', ContinuousCondition(sparse=True)),
    ('glucose_min_lst_max', ContinuousCondition(sparse=True)),
    ('glucose_max_lst_max', ContinuousCondition(sparse=True)),
    ('glucose_mean_lst_max', ContinuousCondition(sparse=True)),
    ('heartrate_min_lst_mm', ContinuousCondition(sparse=True)),
    ('heartrate_max_lst_mm', ContinuousCondition(sparse=True)),
    ('heartrate_mean_lst_mm', ContinuousCondition(sparse=True)),
    ('sysbp_min_lst_mm', ContinuousCondition(sparse=True)),
    ('sysbp_max_lst_mm', ContinuousCondition(sparse=True)),
    ('sysbp_mean_lst_mm', ContinuousCondition(sparse=True)),
    ('diasbp_min_lst_mm', ContinuousCondition(sparse=True)),
    ('diasbp_max_lst_mm', ContinuousCondition(sparse=True)),
    ('diasbp_mean_lst_mm', ContinuousCondition(sparse=True)),
    ('meanbp_min_lst_mm', ContinuousCondition(sparse=True)),
    ('meanbp_max_lst_mm', ContinuousCondition(sparse=True)),
    ('meanbp_mean_lst_mm', ContinuousCondition(sparse=True)),
    ('resprate_min_lst_mm', ContinuousCondition(sparse=True)),
    ('resprate_max_lst_mm', ContinuousCondition(sparse=True)),
    ('resprate_mean_lst_mm', ContinuousCondition(sparse=True)),
    ('tempc_min_lst_mm', ContinuousCondition(sparse=True)),
    ('tempc_max_lst_mm', ContinuousCondition(sparse=True)),
    ('tempc_mean_lst_mm', ContinuousCondition(sparse=True)),
    ('spo2_min_lst_mm', ContinuousCondition(sparse=True)),
    ('spo2_max_lst_mm', ContinuousCondition(sparse=True)),
    ('spo2_mean_lst_mm', ContinuousCondition(sparse=True)),
    ('glucose_min_lst_mm', ContinuousCondition(sparse=True)),
    ('glucose_max_lst_mm', ContinuousCondition(sparse=True)),
    ('glucose_mean_lst_mm', ContinuousCondition(sparse=True))
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

# Models without/with metadata (Reuters has only titles) TODO adapt them
MODELS = [
    # Use no metadata (only item sets)
    Countbased(),
    # SVDRecommender(10, use_title=False),
    # AAERecommender(adversarial=False, lr=0.001, **ae_params),
    # AAERecommender(adversarial=True, prior='gauss', gen_lr=0.001, reg_lr=0.001, **ae_params),
    # VAERecommender(conditions=None, **vae_params),
    # DAERecommender(conditions=None, **ae_params),
    # Use title (as defined in CONDITIONS above)
    # SVDRecommender(10, use_title=True),
    # AAERecommender(adversarial=True, prior='gauss', gen_lr=0.001, reg_lr=0.001, conditions=CONDITIONS, **ae_params),
    # AAERecommender(adversarial=False, conditions=CONDITIONS, lr=0.001, **ae_params),

    # DecodingRecommender(conditions=CONDITIONS, n_epochs=100, batch_size=100,
    #                     optimizer='adam', n_hidden=100, lr=0.001, verbose=True),
    # VAERecommender(conditions=CONDITIONS, **vae_params),
    # DAERecommender(conditions=CONDITIONS, **ae_params)
    # Put more here...
]

def prepare_evaluation(bags, test_size=0.1, n_items=None, min_count=None, drop=1):
    """
    Split data into train and dev set.
    Build vocab on train set and applies it to both train and test set.
    """
    # Split 10% validation data, one submission per day is too much.
    train_set, dev_set = bags.train_test_split(test_size=test_size)
    # Builds vocabulary only on training set
    # Limit of most frequent 50000 distinct items is for testing purposes
    vocab, __counts = train_set.build_vocab(max_features=n_items,
                                            min_count=min_count,
                                            apply=False)

    # Apply vocab (turn track ids into indices)
    train_set = train_set.apply_vocab(vocab)
    # Discard unknown tokens in the test set
    dev_set = dev_set.apply_vocab(vocab)

    # Drop one track off each playlist within test set
    print("Drop parameter:", drop)
    noisy, missing = corrupt_sets(dev_set.data, drop=drop)
    assert len(noisy) == len(missing) == len(dev_set)
    # Replace test data with corrupted data
    dev_set.data = noisy

    return train_set, dev_set, missing
def prepare_evaluation_kfold_cv(bags, n_folds=5, n_items=None, min_count=None, drop=1):
    """
    Split data into train and dev set.
    Build vocab on train set and applies it to both train and test set.
    """
    # Split 10% validation data, one submission per day is too much.
    train_sets, val_sets, test_sets = bags.create_kfold_train_validate_test(n_folds=n_folds)
    missings = []
    # Builds vocabulary only on training set
    # Limit of most frequent 50000 distinct items is for testing purposes
    #todo: ideally  you would want to build the vocab from both train and val sets
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

        # Drop one track off each playlist within test set
        print("Drop parameter:", drop)
        noisy, missing = corrupt_sets(test_set.data, drop=drop)
        assert len(noisy) == len(missing) == len(test_set)
        # Replace test data with corrupted data
        test_set.data = noisy
        train_sets[i] = train_set
        test_sets[i] = test_set
        val_sets[i] = val_set
        missings.append(missing)

    return train_sets, val_sets, test_sets, missings


def log(*print_args, logfile=None):
    """ Maybe logs the output also in the file `outfile` """
    if logfile:
        with open(logfile, 'a') as fhandle:
            print(*print_args, file=fhandle)
    print(*print_args)

def unpack_patients(patients):
    """
    Unpacks list of patients in a way that is compatible with our Bags dataset
    format. It is not mandatory that patients are sorted.
    """
    bags_of_codes, ids = [], []
    other_attributes = {'gender': {},
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
            other_attributes[c_var][c_hadm_id] = patient[c_var]
    # bag_of_codes and ids should have corresponding indices
    return bags_of_codes, ids, other_attributes


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
                        metric = 'f1', drop = 1):
        noisy, y_val = corrupt_sets(val_set.data, drop=drop)
        val_set.data = noisy
        c_params = model.model_params

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
        model_cpy = copy.deepcopy(model)
        for c_idx, c_row in exp_grid_df.iterrows():
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


def main(min_count = 50, n_folds = 1, drop = 0.1, outfile = ''):
    """Uncomment to generate heatmaps for correlations between variables"""
    outfile = '../test-run_{}.log'.format(datetime.now().strftime("%Y-%m-%d-%H:%M"))
    print("Loading data from", DATA_PATH)
    patients = load(DATA_PATH)
    print("Unpacking MIMIC data...")

    bags_of_patients, ids, side_info = unpack_patients(patients)  # with conditions
    assert(len(set(ids)) == len(ids))
    del patients
    bags = Bags(bags_of_patients, ids, side_info)  # with conditions

    log("Whole dataset:", logfile=outfile)
    log(bags, logfile=outfile)
    # create m table  for numpy.cov
    t_d = {}
    for k,v in side_info.items():
        if k == 'icd9_code_lst' or "_mm" in k:
            continue
        t_d[k] = list(v.values())

    m_df = pd.DataFrame(t_d)
    cat_cols = ['ethnicity_grouped', 'admission_type', 'gender']
    for col in cat_cols:
        m_df[col] = m_df[col].astype('category')
        m_df[col] = m_df[col].cat.codes

    # m_df = m_df.transpose()
    # c_m3x = np.cov(m_df.to_numpy())
    m_df = m_df.reindex(sorted(m_df.columns), axis=1)

    f = plt.figure(figsize=(64, 48))
    plt.matshow(m_df.corr(), fignum=f.number)
    plt.xticks(range(m_df.select_dtypes(['number']).shape[1]), m_df.select_dtypes(['number']).columns, fontsize=14, rotation=90)
    plt.yticks(range(m_df.select_dtypes(['number']).shape[1]), m_df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.title('Correlation Matrix', fontsize=16);
    plt.show()

    print(1)



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
            plt.savefig('../plots/drop-percentages/plot_{}.png'.format(c_model), bbox_inches='tight')
            plt.show()


def run_cv_pipeline(bags, drop, min_count, n_folds, with_hp_tun, outfile):
    metrics_per_drop_per_model = []
    train_sets, val_sets, test_sets, y_tests = prepare_evaluation_kfold_cv(bags, min_count=min_count, drop=drop,
                                                                           n_folds=n_folds)
    for c_fold in range(n_folds):
        log("FOLD = {}".format(c_fold), logfile=outfile)
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

        # THE GOLD (put into sparse matrix)
        y_test = lists2sparse(y_test, test_set.size(1)).tocsr(copy=False)
        # the known items in the test set, just to not recompute
        x_test = lists2sparse(test_set.data, test_set.size(1)).tocsr(copy=False)
        for model in MODELS:
            log('=' * 78, logfile=outfile)
            log(model, logfile=outfile)
            model_cpy = copy.deepcopy(model)
            if with_hp_tun and str(type(model)) in HYPERPARAMS_TO_TRY_OPTIM.keys():
                best_params, _, _ = hyperparam_optimize(model_cpy, train_set, val_set,
                                                        tunning_params=HYPERPARAMS_TO_TRY_OPTIM[str(type(model))],
                                                        drop=drop)
                log('After hyperparam_optimize, best params: ', logfile=outfile)
                log(best_params, logfile=outfile)
                model.model_params = best_params
            # Training
            model.train(train_set)
            # Prediction
            y_pred = model.predict(test_set)
            # Sanity-fix #1, make sparse stuff dense, expect array
            if sp.issparse(y_pred):
                y_pred = y_pred.toarray()
            else:
                y_pred = np.asarray(y_pred)
            # Sanity-fix, remove predictions for already present items
            y_pred = remove_non_missing(y_pred, x_test, copy=False)
            # Evaluate metrics
            results = evaluate(y_test, y_pred, METRICS)
            log("-" * 78, logfile=outfile)
            for metric, stats in zip(METRICS, results):
                log("* FOLD#{} {}: {} ({})".format(c_fold, metric, *stats), logfile=outfile)
                metrics_per_drop_per_model.append([c_fold, drop, str(model), metric, stats[0], stats[1]])
            log('=' * 78, logfile=outfile)
    metrics_df = pd.DataFrame(metrics_per_drop_per_model,
                              columns=['fold', 'drop', 'model', 'metric', 'metric_val', 'metric_std'])
    return metrics_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile',
                        help="File to store the results.")
    parser.add_argument('-m', '--min-count', type=int,
                        default=None,
                        help="Minimum count of items")
    parser.add_argument('--compute-mi', default=False,
                        action='store_true')
    parser.add_argument('-dr', '--drop', type=str,
                        help='Drop parameter', default="1")
    args = parser.parse_args()
    print(args)

    # Drop could also be a callable according to evaluation.py but not managed as input parameter
    try:
        drop = int(args.drop)
    except ValueError:
        drop = float(args.drop)

    main(outfile=args.outfile, min_count=args.min_count, drop=drop)

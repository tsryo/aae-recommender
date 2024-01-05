import datetime as dt
import pickle
import re

import pandas as pd
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt


def __try_log(msg, severity):
    time_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    str_out = '|%s| [%s]\t%s' % (severity, time_str, msg)
    print(str_out)

def try_log_trace(msg):
    __try_log(msg, 'TRACE')

def try_log_debug(msg):
    __try_log(msg, 'DEBUG')

def try_log_info(msg):
    __try_log(msg, 'INFO')

def try_log_warn(msg):
    __try_log(msg, 'WARN')

def try_log_error(msg):
    __try_log(msg, 'ERROR')

def try_log_severe(msg):
    __try_log(msg, 'SEVERE')



# plotting & printing
def log(*print_args, logfile=None):
    """ Maybe logs the output also in the file `outfile` """
    if logfile:
        with open(logfile, 'a') as fhandle:
            print(*print_args, file=fhandle)
    print(*print_args)



def print_plot_pat_dict(d1):
    # Assuming 'my_dict' is your dictionary
    for key, value in d1.items():
        if key in ['time_mins_lst', 'los_icu_lst']:
            # Skip the 'time' key
            continue
        if not re.match('.*mean_lst$', key):
            continue

        if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
            # Plot numeric lists as time series
            time_values = d1['time_mins_lst']
            plt.plot(time_values, value, label=key)
        elif isinstance(value, str):
            # Display whole strings
            print(f"{key}: {value}")
        elif isinstance(value, (int, float)):
            # Display plain numbers
            print(f"{key}: {value}")
        else:
            # Handle other data types as needed
            print(f"{key}: {type(value)}")

    plt.xlabel('Time (minutes)')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def plt_pat_histograms_demog(patients):
    for i in range(0, len(patients)):
        patient = patients[i]
        patients[i]['icd9_code_d_lst_len'] = len(patient['icd9_code_d_lst'])
        patients[i]['icd9_code_p_lst_len'] = len(patient['icd9_code_p_lst'])
    columns = list(patients[0].keys())
    str_cols = ['gender', 'ethnicity_grouped', 'admission_type', 'first_icu_stay', 'icd9_code_d_lst', 'icd9_code_p_lst']
    percent_missing_numeric = lambda x: len(np.where(np.isnan(x))[0]) / len(x)
    percent_missing_str = lambda x: sum([1 if i == 'nan' else 0 for i in x]) / len(x)
    missing_fn_mapper = {'str': percent_missing_str, 'num': percent_missing_numeric}
    for c_col in columns:
        col_type = 'num'
        print(c_col)
        c_vals = [patients[x][c_col] for x in range(0, len(patients))]
        if c_col == 'icd9_code_d_lst' or c_col == 'icd9_code_p_lst' or c_col == 'los_icu_lst':
            c_vals = list(np.concatenate(c_vals).flat)
        elif str.endswith(c_col, '_lst'):
            continue
        if c_col in str_cols:
            col_type = 'str'
            c_vals = [str(i) for i in c_vals]
        percent_missing = missing_fn_mapper[col_type](c_vals)

        if c_col == 'age':
            c_vals = np.multiply(np.round(np.divide(c_vals, 5)), 5)  # bin ages into 5-year groups
        n_uniq_vals = len(pd.Series(c_vals).value_counts())
        plt.figure(figsize=(7, 7))
        plt.grid(True, alpha=0.5, axis='y', which='both', linestyle='--', linewidth=2)
        plt.hist(c_vals, bins=min(32, n_uniq_vals), facecolor='#0d65f2', edgecolor='black', alpha=0.85, linewidth=1)

        plt.xlabel(c_col)
        plt.ylabel('Frequency')
        plt.title('Histogram of {} (missing = %{:.2f})'.format(c_col, percent_missing * 100))
        plt.savefig('../../plt/plots/demographics/histograms/{}.png'.format(c_col), bbox_inches='tight')
        plt.show()


# I/O and processing
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def normalize_conditional_data_bags(bags):
    for k in list(bags.owner_attributes.keys()):
        if k in ['note_embeddings', 'ICD9_defs_txt', 'gender', 'ethnicity_grouped', 'admission_type', 'icd9_code_d_lst', 'icd9_code_p_lst',
                 'los_icu_lst', 'time_mins_lst', 'icu_stay_start_lst', 'icu_stay_stop_lst']:
            continue
        c_vals = list(bags.owner_attributes[k].values())
        c_vals = np.nan_to_num(np.array(c_vals))
        c_vals = preprocessing.normalize([c_vals])[0].tolist()
        c_keys = list(bags.owner_attributes[k].keys())
        bags.owner_attributes[k] = {c_keys[i]: c_vals[i] for i in range(len(c_keys))}
    return bags

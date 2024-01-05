from collections.abc import Iterable
import pandas as pd
import json
import numpy as np
from CONSTANTS import IN_DATA_PATH_DEMO_ICD_CODES, ICD_CODE_DEFS_PATH, IN_DATA_PATH_VITALS, IN_DATA_PATH_DEMO_ICD_CODES_RAW
from irgan.utils import load
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from scipy.stats import rankdata
from utils.print_utils import try_log_info, try_log_debug
import re
sns.set_theme()

import warnings
warnings.filterwarnings("ignore")

# Declare constants
SEPARATOR = ","
ALL_TIMESERIES_MISSING_PLACEHOLDER = "*MISSING*"
SUBSAMPLE_HADM_IDS = 0.01 # debug , set to -1 otherwise
DEBUG_LST_HADMS = [] # any specific hadm_id you want to see getting parsed; overrides SUBSAMPLE_HADM_IDS
READM = True
PLOT_TIMESERIES_LINEPLOT = False
GEN_VITALS_SUMMARY_TABLE = False
READ_BUFFER_SIZE = 2048.0
table_cols = ['heartrate_min', 'heartrate_max', 'heartrate_mean', 'sysbp_min', 'sysbp_max',
       'sysbp_mean', 'diasbp_min', 'diasbp_max', 'diasbp_mean', 'meanbp_min',
       'meanbp_max', 'meanbp_mean', 'resprate_min', 'resprate_max',
       'resprate_mean', 'tempc_min', 'tempc_max', 'tempc_mean', 'spo2_min',
       'spo2_max', 'spo2_mean', 'glucose_min', 'glucose_max', 'glucose_mean']

pat_cols = ['heartrate_min_lst_mm', 'heartrate_max_lst_mm', 'heartrate_mean_lst_mm',
            'sysbp_min_lst_mm', 'sysbp_max_lst_mm', 'sysbp_mean_lst_mm', 'diasbp_min_lst_mm',
            'diasbp_max_lst_mm', 'diasbp_mean_lst_mm', 'meanbp_min_lst_mm', 'meanbp_max_lst_mm', 'meanbp_mean_lst_mm',
            'resprate_min_lst_mm', 'resprate_max_lst_mm', 'resprate_mean_lst_mm', 'tempc_min_lst_mm', 'tempc_max_lst_mm',
            'tempc_mean_lst_mm', 'spo2_min_lst_mm', 'spo2_max_lst_mm', 'spo2_mean_lst_mm', 'glucose_min_lst_mm', 'glucose_max_lst_mm', 'glucose_mean_lst_mm'
             ]

vitals_columns_to_aggr = ['heartrate_min', 'heartrate_max', 'heartrate_mean', 'sysbp_min', 'sysbp_max',
                          'sysbp_mean', 'diasbp_min', 'diasbp_max', 'diasbp_mean', 'meanbp_min', 'meanbp_max',
                          'meanbp_mean', 'resprate_min', 'resprate_max', 'resprate_mean', 'tempc_min', 'tempc_max',
                          'tempc_mean', 'spo2_min', 'spo2_max', 'spo2_mean', 'glucose_min', 'glucose_max',
                          'glucose_mean']
vitals_table = None
df_conditions = None
GLOBAL_MEANS = {}
GLOBAL_SDS = {}


## pre-processing functions start
def aggregate_vars_per_id(id_var, aggr_vars, df, drop_id_dups=True, drop_aggr_dups=False):
    lst_of_lsts = []
    for c_col in aggr_vars:
        try_log_debug("@aggregate_vars_per_id - Aggregating {0} values into lists per {1}".format(c_col, id_var))
        lst_name = '{0}_lst'.format(*[c_col])
        fn_to_apply = lambda x: list(x) if not drop_aggr_dups else list(set(x))
        df_c_col_lst = df.groupby(id_var)[c_col].apply(fn_to_apply).reset_index(name=lst_name)
        df = df.drop(c_col, axis=1)
        lst_of_lsts.append(df_c_col_lst)
    for c_lst in lst_of_lsts:
        df = df.merge(c_lst, on=id_var, how='left')

    if drop_id_dups:
        df = df.drop_duplicates(subset=id_var, keep="first")
    return df

def impute_timeseries_values(lst_vals, time_mins):
    nan_idxs = np.where(np.isnan(lst_vals))[0]
    if len(nan_idxs) == len(lst_vals):
        return [ALL_TIMESERIES_MISSING_PLACEHOLDER] * len(lst_vals)  # cant impute, set them to 0s for now and mark them later
    if len(nan_idxs) == 0:
        return lst_vals
    last_non_nan_idx = np.where(~np.isnan(lst_vals))[0][-1]
    first_non_nan_idx = np.where(~np.isnan(lst_vals))[0][0]
    last_non_nan_time = time_mins[last_non_nan_idx]
    first_non_nan_time = time_mins[first_non_nan_idx]

    for i in nan_idxs:
        imp_val = 0
        if i > last_non_nan_idx:
            imp_val = lst_vals[last_non_nan_idx]
        elif i < first_non_nan_idx:
            imp_val = lst_vals[first_non_nan_idx]
        else:
            next_non_nan_idx = np.where(~np.isnan(lst_vals[i + 1:]))[0][0] + i + 1
            prev_non_nan_idx = np.where(~np.isnan(lst_vals[:i]))[0][-1]
            c_nan_time = time_mins[i]
            next_non_nan_time = time_mins[next_non_nan_idx]
            prev_non_nan_time = time_mins[prev_non_nan_idx]
            time_dist_next = next_non_nan_time - c_nan_time
            time_dist_prev = c_nan_time - prev_non_nan_time
            total_time_diff = time_dist_prev +  time_dist_next
            val_weight_next = (total_time_diff - time_dist_next) / total_time_diff
            val_weight_prev = (total_time_diff-time_dist_prev)/total_time_diff

            imp_val = (lst_vals[prev_non_nan_idx]*val_weight_prev + lst_vals[next_non_nan_idx]*val_weight_next)
        lst_vals[i] = imp_val
    return lst_vals

def impute_ts_columns(cols, df):
    for col in cols:
        try_log_debug("@impute_ts_columns - Imputing series for column {}".format(col))
        for c_row in range(0,df.shape[0]):
            df[col].iloc[c_row] = impute_timeseries_values(df[col].iloc[c_row], df['time_mins_lst'].iloc[c_row])
    return df

def add_flag_missing_ts(cols, df):
    for col in cols:
        # _mm - missing marker
        df[col + "_mm"] = [1 if str(x[0]) == ALL_TIMESERIES_MISSING_PLACEHOLDER else 0 for x in df[col]]
        df[col] = [[0] * len(x) if str(x[0]) == ALL_TIMESERIES_MISSING_PLACEHOLDER else x for x in df[col]]
    return df.copy()

def add_ts_col_aggregates(series_columns_lst, aggr_fns_d, df):
    for k, v in aggr_fns_d.items():
        for c_col in series_columns_lst:
            aggr_col_nm = c_col + "_{}".format(k)
            try_log_debug("@add_ts_col_aggregates -> {} <- ; for column -> {} <- ;".format(k, c_col))
            df[aggr_col_nm] = [v(df.iloc[x]['time_mins_lst'], df.iloc[x][c_col]) if df.iloc[x][c_col][0] != ALL_TIMESERIES_MISSING_PLACEHOLDER else 0 for x in range(df.shape[0]) ]
    return df.copy()

def standardize_cap_and_normalize_cols(cols, df, sd_cap=10):
    for col in cols:
        c_mean = GLOBAL_MEANS[col]
        c_sd = GLOBAL_SDS[col]
        #standardize
        df[col] = [(x - c_mean) / c_sd for x in df[col]]
        #cap
        df[col] = [x if abs(x) <= sd_cap else sd_cap * (x / abs(x)) for x in df[col]]
        #normalize
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

#
# slope_fn = lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else np.mean(y)
slope_fn = lambda x,y: np.polyfit(x, y, 1)[0] if len(y) > 1 else np.mean(y) # x = time, y = value
# delta_fn = lambda x: sum([x[i - 1] - x[i] for i in range(1, len(x))]) / (len(x) - 1) if len(x) > 1 else np.mean(x)
delta_fn = lambda x,y: sum([(y[i - 1] - y[i]) / (x[i] - x[i - 1]) for i in range(1, len(y))]) / (len(y) - 1) if len(x) > 1 else np.mean(x) # x = time, y = value
mean_fn = lambda x,y: np.nanmean(y)  # x = nothing, y = value
sd_fn = lambda x,y: np.nanstd(y)  # x = nothing, y = value
min_fn = lambda x,y: np.nanmin(y)  # x = nothing, y = value
max_fn = lambda x,y: np.nanmax(y)  # x = nothing, y = value
aggr_fns_d = {"slope": slope_fn, "mean": mean_fn, "sd": sd_fn, "delta": delta_fn, "min": min_fn, "max": max_fn}


# Step 1 - read data from diagnoses, procedures, icu_staydetail, vitals.
if 1 == 1:
    try_log_info("***************** START Step 1 - read data from diagnoses, procedures, icu_staydetail, vitals. *****************")
    try_log_info("Reading data from {}".format(IN_DATA_PATH_DEMO_ICD_CODES_RAW))
    df = pd.read_csv(IN_DATA_PATH_DEMO_ICD_CODES_RAW, sep=SEPARATOR, index_col=False)
    uniq_hadm_ids_df = list(set(df['hadm_id'].values))
    try_log_info("N hosp admissions in {} (1) = {}".format(IN_DATA_PATH_DEMO_ICD_CODES_RAW, len(uniq_hadm_ids_df)))

    try_log_info("Reading data from {}".format(IN_DATA_PATH_VITALS))
    df_vitals = pd.read_csv(IN_DATA_PATH_VITALS, sep=SEPARATOR, index_col=False)
    uniq_hadm_ids_df_vitals = list(set(df_vitals['hadm_id'].values))
    try_log_info("N hosp admissions in {} (2) = {} ".format(IN_DATA_PATH_VITALS, len(uniq_hadm_ids_df_vitals)))


# Step 2 - Compute missing hadm_ids between (1) and (2) , Subsample dataset if specified
if 2 == 2:
    try_log_info("***************** START Step 2 - Compute missing hadm_ids between (1) and (2) , Subsample dataset if specified  *****************")
    hadms_missing_df = set(uniq_hadm_ids_df_vitals) - set(uniq_hadm_ids_df)
    try_log_info("{} hosp adms missing in (1) but present in (2)".format(len(hadms_missing_df)))
    hadms_missing_df_vitals = set(uniq_hadm_ids_df) - set(uniq_hadm_ids_df_vitals)
    try_log_info("{} hosp adms missing in (2) but present in (1)".format(len(hadms_missing_df_vitals)))
    ids_intersect = list(set(uniq_hadm_ids_df).intersection(set(uniq_hadm_ids_df_vitals)))
    n_ids = len(ids_intersect) # 56777

    if SUBSAMPLE_HADM_IDS != -1 and len(DEBUG_LST_HADMS) == 0:
        n_samples = int(n_ids * SUBSAMPLE_HADM_IDS)
        ids_intersect = ids_intersect[0:n_samples]
        try_log_info('sampling {} ids (out of {}) only'.format(n_samples, n_ids))
        df = df.loc[df['hadm_id'].isin(ids_intersect)]
        df_vitals = df_vitals.loc[df_vitals['hadm_id'].isin(ids_intersect)]
    else:
        try_log_debug("Skipping subsample....")
    if len(DEBUG_LST_HADMS) > 0:
        try_log_info('debugging {} ids (out of {}) only'.format(len(DEBUG_LST_HADMS), n_ids))
        df = df.loc[df['hadm_id'].isin(DEBUG_LST_HADMS)]
        df_vitals = df_vitals.loc[df_vitals['hadm_id'].isin(DEBUG_LST_HADMS)]

# Step 3 - Remove hadms that do not have vitals, set time_mins for vitals
if 3 == 3:
    try_log_info("***************** START Step 3 - Remove hadms that do not have vitals, set time_mins + icu_stay_start/stop for vitals  *****************")
    df_vitals = df_vitals.sort_values(['hadm_id', 'charttime'], ascending=True)
    all_vitals_hadm_ids = set(list(df_vitals['hadm_id']))
    df_vitals_new = None

    try_log_debug("Step 3 [1] setting time_mins & icu_stay_start/stop for vitals...")
    n_iters = len(list(all_vitals_hadm_ids))
    for c_idx in range(0, n_iters):
        hadm_id = list(all_vitals_hadm_ids)[c_idx]
        # log progress
        if c_idx != 0 and (c_idx+1) % round(n_iters*0.05) == 0:
            percent_done = round(((c_idx+1) / n_iters ) *100)
            # if percent_done % 5 != 0:
            percent_done = percent_done + (5-(percent_done % 5))
            try_log_debug("Progress ...{}%".format(percent_done))

        c_vitals = df_vitals.iloc[df_vitals['hadm_id'].values == hadm_id]
        ftime = c_vitals['charttime'].values[0]
        last_icu_stay_id = c_vitals['icustay_id'].values[0]
        ftime = datetime.strptime(ftime, '%Y-%m-%d %H:%M:%S')
        time_mins = [0]
        icu_stay_stop = [0]
        icu_stay_start = [1]
        for c_ts_idx in range(1, len(c_vitals['charttime'].values)):
            time_period = c_vitals['charttime'].values[c_ts_idx]
            c_icu_stay_id = c_vitals['icustay_id'].values[c_ts_idx]
            if last_icu_stay_id != c_icu_stay_id: # a new icu stay has started
                icu_stay_start.append(1)
                icu_stay_stop.pop()
                icu_stay_stop.append(1)
                icu_stay_stop.append(0)
                last_icu_stay_id = c_icu_stay_id
            else:
                icu_stay_stop.append(0)
                icu_stay_start.append(0)
            ctime = datetime.strptime(time_period, '%Y-%m-%d %H:%M:%S')
            c_time_in_mins = ctime - ftime
            c_time_in_mins = int(c_time_in_mins.total_seconds())/60
            time_mins.append(c_time_in_mins)

        icu_stay_stop.pop()
        icu_stay_stop.append(1)
        c_vitals['time_mins'] = time_mins
        c_vitals['icu_stay_start'] = icu_stay_start
        c_vitals['icu_stay_stop'] = icu_stay_stop
        df_vitals_new = pd.concat([df_vitals_new, c_vitals])
    df_vitals = df_vitals_new

    try_log_debug("Step 3 [2] - removing hadm_ids that do not have vitals...")
    df_new = df.loc[df['hadm_id'].isin(all_vitals_hadm_ids)]
    df = df.loc[df['hadm_id'].isin(all_vitals_hadm_ids)]
    try_log_info('*** END Step 3 -  Removed {} hadm_ids that that do not have vitals '.format(df.shape[0] - df_new.shape[0]))

# Step 3.1 - Remove all patients younger than 18
if 3.1 == 3.1:
    try_log_info("***************** START Step 3.1 - Remove all patients younger than 18 + those without any icd codes  *****************")
    df_new = df.loc[df['age'] >= 18] # remove all non-adult entries
    cnt_hadm_ids = len(list(set(df.hadm_id)))
    cnt_hadm_ids_new = len(list(set(df_new.hadm_id)))
    try_log_info('*** Step 3.1 [1] -  Removed {} hadm_ids that are under 18 years'.format(cnt_hadm_ids - cnt_hadm_ids_new))
    df = df_new

    # remove patients without ICD codes
    nhadms_before = len(set(df.hadm_id))
    df = df.dropna(subset=['icd9_code'])
    nhadms_after =  len(set(df.hadm_id))
    try_log_debug("*** Step 3.1 [2] Removed %d hamd_ids for not having any ICD codes" % (nhadms_before-nhadms_after))

    # for df_vitals
    all_hadm_ids = set(list(df['hadm_id']))
    df_vitals_new = df_vitals
    df_vitals_new = df_vitals_new.loc[df_vitals_new['hadm_id'].isin(all_hadm_ids)]
    df_vitals = df_vitals_new
    try_log_info("***************** END Step 3.1  *****************")



# Step 3.2 - round and decode age values (In MIMIC-III patients older than 89 are set to 300 (time shifted))
if 3.2 == 3.2:
    try_log_info("***************** START Step 3.2 - round and decode age values  *****************")
    df['age'] = round(df['age'], 0)
    df['age'] = [89 if age >= 89 else age for age in df['age']]
    try_log_info("***************** END Step 3.2 *****************")

# Step ** - Experimental: Visualize timeseries on a line-plot
if PLOT_TIMESERIES_LINEPLOT:
    try_log_info("***************** START Step ** - PLOT_TIMESERIES_LINEPLOT  *****************")
    random5_hadms = list(set(df_vitals_new['hadm_id'].tolist()))[100:105]
    vitals_5rand_hadms = df_vitals_new[df_vitals_new['hadm_id'].isin(random5_hadms)]

    plot_dir = '/mnt/c/Development/github/Python/plt/experiment/{}'
    plot_nm = plot_dir.format('timeseries_lineplot.png')
    vitals_5rand_hadms['hadm_id'] = rankdata(vitals_5rand_hadms['hadm_id'], method='dense')
    fig = sns.relplot(data=vitals_5rand_hadms, x="time_mins", y="heartrate_min", kind="line", hue ='hadm_id', size = 6, aspect = 2, palette = sns.color_palette("husl", 5))
    fig.savefig(plot_nm, dpi = 500)
    plt.show()
# Step ** - Generate summary table for patient vitals (vitals_table.csv)
if GEN_VITALS_SUMMARY_TABLE:
    try_log_info("***************** START Step ** - GEN_VITALS_SUMMARY_TABLE  *****************")
    patients = load(IN_DATA_PATH_DEMO_ICD_CODES)
    icd_code_defs = pd.read_csv(ICD_CODE_DEFS_PATH, sep='\t')
    all_icd_codes = [x['icd9_code_lst'] for x in patients]
    all_icd_codes = [y for x in all_icd_codes for y in x]
    rarest_icd_code_val_counts = pd.Series(all_icd_codes).value_counts()[-50:] # 50 most-rare
    rare_codes = None
    for c_icd_code in rarest_icd_code_val_counts.keys().tolist():
        is_diag = 'd_' in c_icd_code
        icd_diag_or_proc = 'DIAGNOSE' if is_diag else 'PROCEDURE'
        c_vitals = icd_code_defs[icd_code_defs['type'] == icd_diag_or_proc]
        c_vitals = c_vitals[c_vitals['icd9_code'] == c_icd_code[2:]] # remove the p_ or d_ prefix of icd_code
        rare_codes = pd.concat([c_vitals, rare_codes])
    np.where(np.isnan(all_icd_codes))
    np.nanmedian(all_icd_codes)
    np.nanquantile(all_icd_codes, 0.25)
    np.nanquantile(all_icd_codes, 0.75)

    for t_col in table_cols:
        c_vals = df_vitals[t_col].values
        c_row = {
            "n_measurements": len(c_vals)-len(np.where(np.isnan(c_vals))[0]),
            "med" : np.nanmedian(c_vals),
            "q1" : np.nanquantile(c_vals, q = 0.25),
            "q3" : np.nanquantile(c_vals, q=0.75),
            "vital": t_col,
        }
        c_row = pd.DataFrame(pd.Series(c_row))
        c_row = pd.DataFrame.transpose(c_row)
        vitals_table = pd.concat([vitals_table, c_row])
    vitals_table.to_csv('vitals_table.csv', sep = '\t')

    c_vitals.columns
# Step 4 - Calculate df_seq_num_len, drop icustay id columns
if 4 == 4:
    try_log_info("***************** START Step 4 - Calculate df_seq_num_len, drop icustay id columns  *****************")
    icd_all = list(df['icd9_code'].values)
    icd_diag = [x for x in icd_all if type(x) != float and x[0:2] == 'd_']
    icd_proc = [x for x in icd_all if type(x) != float and x[0:2] == 'p_']

    df = df.sort_values(['hadm_id'], ascending=True)
    df_p = df[df.icd9_code.str.startswith('p_', na=False)]
    df_p_seq_num_len = df_p.groupby('hadm_id')['seq_num'].apply(np.max).reset_index(name='seq_num_p_len')
    df_d = df[df.icd9_code.str.startswith('d_', na=False)]
    df_d_seq_num_len = df_d.groupby('hadm_id')['seq_num'].apply(np.max).reset_index(name='seq_num_d_len')


    df = df.merge(df_p_seq_num_len, on='hadm_id', how='left')
    df = df.merge(df_d_seq_num_len, on='hadm_id', how='left')

    df = df.drop(['first_icu_stay'], axis=1)
    df = df.drop(['icustay_id'], axis=1)
    # df = df.drop(['seq_num'], axis=1)
    df = df.drop_duplicates() # should drop nothing actually here... just in case
# Step 5 - Calculate global means/SDs of vitals
if 5 == 5:
    try_log_info("***************** START Step Step 5 - Calculate global means/SDs of vitals  *****************")
    for x in vitals_columns_to_aggr:
        GLOBAL_MEANS[x] = np.nanmean(df_vitals[x])
        GLOBAL_SDS[x] = np.nanstd(df_vitals[x])

    pats_cols_to_aggr = ['age', 'los_icu', 'seq_num_p_len', 'seq_num_d_len']
    for x in pats_cols_to_aggr:
        GLOBAL_MEANS[x] = np.nanmean(df[x])
        GLOBAL_SDS[x] = np.nanstd(df[x])
# Step 6-  iteratively load, parse, impute ts, and save records from csv to json format. Also set los_hospital to 0 where its negative
if 6 == 6:
    try_log_info("***************** START Step 6 - iteratively load, parse and save records from csv to json format  *****************")
    all_hadm_ids = list(set(df['hadm_id']))
    all_icds = df['icd9_code'].values
    n_iters_needed = np.ceil(len(all_hadm_ids)/READ_BUFFER_SIZE)
    df_cpy = df
    df_vitals_cpy = df_vitals
    df2 = df[['hadm_id', 'icd9_code']]
    df2 = df2.groupby(['hadm_id'])
    df2 = df2.count()
    icd_per_admission = df2.icd9_code.tolist()
    np.median(icd_per_admission) # 15
    np.percentile(icd_per_admission, [25, 75]) # 10, 21

    for i in range(int(n_iters_needed)):
        try_log_info("Iteration {} (out of {})".format(i+1, n_iters_needed))
        range_upper_bound = int((i*READ_BUFFER_SIZE + READ_BUFFER_SIZE)) if i != int(n_iters_needed)-1 else len(all_hadm_ids)
        hadm_ids = [all_hadm_ids[i] for i in list(range(int((i*READ_BUFFER_SIZE)), range_upper_bound))]
        df_vitals = df_vitals_cpy.loc[df_vitals_cpy['hadm_id'].isin(hadm_ids)]
        df = df_cpy.loc[df_cpy['hadm_id'].isin(hadm_ids)]
        df = df.drop(['subject_id', 'dod', 'admittime', 'dischtime', 'ethnicity', 'hospital_expire_flag', 'hospstay_seq', 'first_hosp_stay', 'intime', 'outtime'], axis=1)

        df = df.dropna(subset = ['icd9_code'])
        # - 10.07.23 - we need to process seq_num in a way that we preserve the order of each icd p_/d_ code
        # idea 1: just prefix the seq_num to the icd code - con:  need more post-processing later
        # idea 2: just have a seq_num_lst
        # aggregate_vars_per_id(id_var, aggr_vars, df, drop_id_dups=True, drop_aggr_dups=False
        pat_details_columns_to_aggr = ['icd9_code', 'seq_num', 'los_icu', 'icustay_seq']
        df_p = df[df['icd9_code'].str.startswith('p_')]
        xxx_p = aggregate_vars_per_id("hadm_id", pat_details_columns_to_aggr, df_p)
        # xxx_p.icustay_seq_lst = xxx_p.icustay_seq_lst.apply(lambda x: list(set(x)))
        xxx_p = xxx_p.rename(columns={'seq_num_lst': 'seq_num_p_lst'})
        xxx_p = xxx_p.rename(columns={'icd9_code_lst': 'icd9_code_p_lst'})

        df_d = df[df['icd9_code'].str.startswith('d_')]
        xxx_d = aggregate_vars_per_id("hadm_id", pat_details_columns_to_aggr, df_d)
        xxx_d = xxx_d.rename(columns={'seq_num_lst': 'seq_num_d_lst'})
        xxx_d = xxx_d.rename(columns={'icd9_code_lst': 'icd9_code_d_lst'})

        df = df.drop(['los_icu'], axis = 1)
        df = df.drop(['icustay_seq'], axis=1)
        df = df.drop(['seq_num'], axis=1)
        df = df.drop(['icd9_code'], axis=1)
        df = df.drop_duplicates()
        tmp_m = pd.merge(xxx_p[['hadm_id', 'icd9_code_p_lst', 'seq_num_p_lst', 'los_icu_lst', 'icustay_seq_lst']],
                         xxx_d[['hadm_id', 'icd9_code_d_lst', 'seq_num_d_lst', 'los_icu_lst', 'icustay_seq_lst']], how="outer", on="hadm_id")

        tmp_m['icustay_seq_lst_x'] = tmp_m['icustay_seq_lst_x'].apply(lambda x: x if type(x) == list else [] )
        tmp_m['icustay_seq_lst_y'] = tmp_m['icustay_seq_lst_y'].apply(lambda x: x if type(x) == list else [])
        tmp_m['icustay_seq_lst'] = tmp_m['icustay_seq_lst_x'] + tmp_m['icustay_seq_lst_y']

        tmp_m['los_icu_lst_x'] = tmp_m['los_icu_lst_x'].apply(lambda x: x if type(x) == list else [])
        tmp_m['los_icu_lst_y'] = tmp_m['los_icu_lst_y'].apply(lambda x: x if type(x) == list else [])
        tmp_m['los_icu_lst'] = tmp_m['los_icu_lst_x'] + tmp_m['los_icu_lst_y']

        tmp_m = tmp_m.drop(['los_icu_lst_x', 'los_icu_lst_y', 'icustay_seq_lst_x', 'icustay_seq_lst_y'], axis=1)
        # set(tmp_m.columns).intersection(set(df.columns))
        df = pd.merge(df, tmp_m, how = 'inner', on='hadm_id')

        df['seq_num_p_len'] = df['seq_num_p_len'].apply(lambda x: x if not np.isnan(x) else 0)
        df['seq_num_d_len'] = df['seq_num_d_len'].apply(lambda x: x if not np.isnan(x) else 0)

        df['icd9_code_p_lst'] = df['icd9_code_p_lst'].apply(lambda x: x if type(x) == list else [])
        df['icd9_code_d_lst'] = df['icd9_code_d_lst'].apply(lambda x: x if type(x) == list else [])

        df['seq_num_p_lst'] = df['seq_num_p_lst'].apply(lambda x: x if type(x) == list else [])
        df['seq_num_d_lst'] = df['seq_num_d_lst'].apply(lambda x: x if type(x) == list else [])

        # fix order of icu_los and de-dup based on icustay_seq_lst
        for j in range(0, df.shape[0]):
            # print(j)
            if (type(df.iloc[j].los_icu_lst) == float and np.isnan(df.iloc[j].los_icu_lst)) or\
                (type(df.iloc[j].icustay_seq_lst) == float and np.isnan(df.iloc[j].icustay_seq_lst)):
                df.at[j, 'los_icu_lst'] = []
                continue
            c_lst = [ x for _, x in sorted(zip(df.iloc[j].icustay_seq_lst, df.iloc[j].los_icu_lst))]
            c_seqs = [x for x, _ in sorted(zip(df.iloc[j].icustay_seq_lst, df.iloc[j].los_icu_lst))]
            c_uniqs = np.unique(c_seqs, return_index=True)[1]
            c_lst = [c_lst[k] for k in c_uniqs]
            c_seqs = [c_seqs[k] for k in c_uniqs]
            df.at[j, 'los_icu_lst'] = c_lst

        df['los_icu_len'] = df['los_icu_lst'].apply(lambda x: len(x))
        df = df.drop(['icustay_seq_lst'], axis = 1)

        df_vitals = df_vitals.drop(["subject_id", 'icustay_id'], axis=1)
        # for each patient (actually admission) get the lists of all icd9 codes, vitals, and icu_los;
        # standardize, generate aggregates, impute missing from series per patient, mark completely missing series with boolean features
        # vitals
        df_vitals = df_vitals.sort_values(['hadm_id', 'charttime'], ascending=True)
        df_vitals = df_vitals.drop(['charttime'], axis=1)
        df_vitals = standardize_cap_and_normalize_cols(vitals_columns_to_aggr, df_vitals, sd_cap=10)
        vitals_columns_aggregated = [x + "_lst" for x in vitals_columns_to_aggr]
        df_vitals = aggregate_vars_per_id("hadm_id", [*vitals_columns_to_aggr, 'time_mins', 'icu_stay_start', 'icu_stay_stop'], df_vitals)
        df_vitals = impute_ts_columns(vitals_columns_aggregated, df_vitals)
        df_vitals = add_ts_col_aggregates(vitals_columns_aggregated, aggr_fns_d, df_vitals)
        df_vitals = add_flag_missing_ts(vitals_columns_aggregated, df_vitals)

        assert all(df['hadm_id'].values == df_vitals['hadm_id'].values)
        # combine pat ditails and vitals
        df = pd.merge(df, df_vitals, how="inner", on="hadm_id")

        # Order items in icd9_code_p_lst and icd9_code_d_lst based on corresponding seq_nums
        # note: this will still leave dup codes (only procedural?) on a icd_code list per hadm_id, as long as they got assigned at different seq_nums
        n_iters = df.shape[0]
        for ii in range(0, n_iters):
            # print(ii)
            # log progress
            if ii != 0 and (ii + 1) % round(n_iters * 0.05) == 0:
                percent_done = round(((ii + 1) / n_iters) * 100)
                # if percent_done % 5 != 0:
                percent_done = percent_done + (5 - (percent_done % 5))
                try_log_debug("Progress ...{}%".format(percent_done))

            if type(df.iloc[ii].icd9_code_p_lst) == float and np.isnan(df.iloc[ii].icd9_code_p_lst):
                continue
            c_lst = [ x for _, x in sorted(zip(df.iloc[ii].seq_num_p_lst, df.iloc[ii].icd9_code_p_lst))]
            c_seqs = [x for x, _ in sorted(zip(df.iloc[ii].seq_num_p_lst, df.iloc[ii].icd9_code_p_lst))]
            c_uniqs = np.unique(c_seqs, return_index=True)[1]
            c_lst = [c_lst[j] for j in c_uniqs]
            c_seqs = [c_seqs[j] for j in c_uniqs]
            df.at[ii, 'icd9_code_p_lst'] = c_lst
            # df.iloc[ii].icd9_code_p_lst
            if type(df.iloc[ii].icd9_code_d_lst) == float and np.isnan(df.iloc[ii].icd9_code_d_lst):
                continue
            c_lst = [ x for _, x in sorted(zip(df.iloc[ii].seq_num_d_lst, df.iloc[ii].icd9_code_d_lst))]
            c_seqs = [x for x, _ in sorted(zip(df.iloc[ii].seq_num_d_lst, df.iloc[ii].icd9_code_d_lst))]
            c_uniqs = np.unique(c_seqs, return_index=True)[1]
            c_lst = [c_lst[j] for j in c_uniqs]
            c_seqs = [c_seqs[j] for j in c_uniqs]
            df.at[ii, 'icd9_code_d_lst'] = c_lst
        df = df.drop(['seq_num_d_lst'],axis=1)
        df = df.drop(['seq_num_p_lst'], axis=1)
        #re-order columns
        df = df[['hadm_id', 'age', 'gender', 'ethnicity_grouped', 'los_hospital', 'admission_type', 'seq_num_p_len',
                 'icd9_code_p_lst', 'seq_num_d_len', 'icd9_code_d_lst', 'los_icu_len', 'los_icu_lst',
                 'icu_stay_start_lst', 'icu_stay_stop_lst',
                 'heartrate_min_lst', 'heartrate_max_lst', 'heartrate_mean_lst',
                 'sysbp_min_lst', 'sysbp_max_lst', 'sysbp_mean_lst',
                 'diasbp_min_lst', 'diasbp_max_lst', 'diasbp_mean_lst',
                 'meanbp_min_lst', 'meanbp_max_lst', 'meanbp_mean_lst',
                 'resprate_min_lst', 'resprate_max_lst', 'resprate_mean_lst',
                 'tempc_min_lst', 'tempc_max_lst', 'tempc_mean_lst',
                 'spo2_min_lst', 'spo2_max_lst', 'spo2_mean_lst',
                 'glucose_min_lst', 'glucose_max_lst', 'glucose_mean_lst',
                 'time_mins_lst',
                 'heartrate_min_lst_slope', 'heartrate_max_lst_slope', 'heartrate_mean_lst_slope',
                 'sysbp_min_lst_slope', 'sysbp_max_lst_slope', 'sysbp_mean_lst_slope',
                 'diasbp_min_lst_slope', 'diasbp_max_lst_slope', 'diasbp_mean_lst_slope',
                 'meanbp_min_lst_slope', 'meanbp_max_lst_slope', 'meanbp_mean_lst_slope',
                 'resprate_min_lst_slope', 'resprate_max_lst_slope', 'resprate_mean_lst_slope',
                 'tempc_min_lst_slope', 'tempc_max_lst_slope', 'tempc_mean_lst_slope',
                 'spo2_min_lst_slope', 'spo2_max_lst_slope', 'spo2_mean_lst_slope',
                 'glucose_min_lst_slope', 'glucose_max_lst_slope', 'glucose_mean_lst_slope',
                 'heartrate_min_lst_mean', 'heartrate_max_lst_mean', 'heartrate_mean_lst_mean',
                 'sysbp_min_lst_mean', 'sysbp_max_lst_mean', 'sysbp_mean_lst_mean',
                 'diasbp_min_lst_mean', 'diasbp_max_lst_mean', 'diasbp_mean_lst_mean',
                 'meanbp_min_lst_mean', 'meanbp_max_lst_mean', 'meanbp_mean_lst_mean',
                 'resprate_min_lst_mean', 'resprate_max_lst_mean', 'resprate_mean_lst_mean',
                 'tempc_min_lst_mean', 'tempc_max_lst_mean', 'tempc_mean_lst_mean',
                 'spo2_min_lst_mean', 'spo2_max_lst_mean', 'spo2_mean_lst_mean',
                 'glucose_min_lst_mean', 'glucose_max_lst_mean', 'glucose_mean_lst_mean',
                 'heartrate_min_lst_sd', 'heartrate_max_lst_sd', 'heartrate_mean_lst_sd',
                 'sysbp_min_lst_sd', 'sysbp_max_lst_sd', 'sysbp_mean_lst_sd',
                 'diasbp_min_lst_sd', 'diasbp_max_lst_sd', 'diasbp_mean_lst_sd',
                 'meanbp_min_lst_sd', 'meanbp_max_lst_sd', 'meanbp_mean_lst_sd',
                 'resprate_min_lst_sd', 'resprate_max_lst_sd', 'resprate_mean_lst_sd',
                 'tempc_min_lst_sd', 'tempc_max_lst_sd', 'tempc_mean_lst_sd',
                 'spo2_min_lst_sd', 'spo2_max_lst_sd', 'spo2_mean_lst_sd',
                 'glucose_min_lst_sd', 'glucose_max_lst_sd', 'glucose_mean_lst_sd',
                 'heartrate_min_lst_delta', 'heartrate_max_lst_delta', 'heartrate_mean_lst_delta',
                 'sysbp_min_lst_delta', 'sysbp_max_lst_delta', 'sysbp_mean_lst_delta',
                 'diasbp_min_lst_delta', 'diasbp_max_lst_delta', 'diasbp_mean_lst_delta',
                 'meanbp_min_lst_delta', 'meanbp_max_lst_delta', 'meanbp_mean_lst_delta',
                 'resprate_min_lst_delta', 'resprate_max_lst_delta', 'resprate_mean_lst_delta',
                 'tempc_min_lst_delta', 'tempc_max_lst_delta', 'tempc_mean_lst_delta',
                 'spo2_min_lst_delta', 'spo2_max_lst_delta', 'spo2_mean_lst_delta',
                 'glucose_min_lst_delta', 'glucose_max_lst_delta', 'glucose_mean_lst_delta',
                 'heartrate_min_lst_min', 'heartrate_max_lst_min', 'heartrate_mean_lst_min',
                 'sysbp_min_lst_min', 'sysbp_max_lst_min', 'sysbp_mean_lst_min',
                 'diasbp_min_lst_min', 'diasbp_max_lst_min', 'diasbp_mean_lst_min',
                 'meanbp_min_lst_min', 'meanbp_max_lst_min', 'meanbp_mean_lst_min',
                 'resprate_min_lst_min', 'resprate_max_lst_min', 'resprate_mean_lst_min',
                 'tempc_min_lst_min', 'tempc_max_lst_min', 'tempc_mean_lst_min',
                 'spo2_min_lst_min', 'spo2_max_lst_min', 'spo2_mean_lst_min',
                 'glucose_min_lst_min', 'glucose_max_lst_min', 'glucose_mean_lst_min',
                 'heartrate_min_lst_max', 'heartrate_max_lst_max', 'heartrate_mean_lst_max',
                 'sysbp_min_lst_max', 'sysbp_max_lst_max', 'sysbp_mean_lst_max',
                 'diasbp_min_lst_max', 'diasbp_max_lst_max', 'diasbp_mean_lst_max',
                 'meanbp_min_lst_max', 'meanbp_max_lst_max', 'meanbp_mean_lst_max',
                 'resprate_min_lst_max', 'resprate_max_lst_max', 'resprate_mean_lst_max',
                 'tempc_min_lst_max', 'tempc_max_lst_max', 'tempc_mean_lst_max',
                 'spo2_min_lst_max', 'spo2_max_lst_max', 'spo2_mean_lst_max',
                 'glucose_min_lst_max', 'glucose_max_lst_max', 'glucose_mean_lst_max',
                 'heartrate_min_lst_mm', 'heartrate_max_lst_mm', 'heartrate_mean_lst_mm',
                 'sysbp_min_lst_mm', 'sysbp_max_lst_mm', 'sysbp_mean_lst_mm',
                 'diasbp_min_lst_mm', 'diasbp_max_lst_mm', 'diasbp_mean_lst_mm',
                 'meanbp_min_lst_mm', 'meanbp_max_lst_mm', 'meanbp_mean_lst_mm',
                 'resprate_min_lst_mm', 'resprate_max_lst_mm', 'resprate_mean_lst_mm',
                 'tempc_min_lst_mm', 'tempc_max_lst_mm', 'tempc_mean_lst_mm',
                 'spo2_min_lst_mm', 'spo2_max_lst_mm', 'spo2_mean_lst_mm',
                 'glucose_min_lst_mm', 'glucose_max_lst_mm', 'glucose_mean_lst_mm']]
        # df = df.drop(columns=['seq_num_p_lst', 'seq_num_d_lst'])

        # in some strange cases, los_hospital is recorded as negative, just set it to 0
        df.los_hospital = df.los_hospital.apply(lambda x : x if x >= 0 else 0)


        def round_floats(df,  n_digits = 4):
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = df[col].apply(lambda x: round(x, n_digits))
                elif isinstance(df[col].iloc[0], Iterable) and not isinstance(df[col].iloc[0], str):
                    df[col] = df[col].apply(
                        lambda lst: [round(val, n_digits) if isinstance(val, float) else val for val in lst])
            return df

        df1 = round_floats(df)
        # we can round up values up to 4 decimals without worry of information loss


        # from df to dict
        patients = df.T.to_dict()

        # serialize in json
        out_data_path = IN_DATA_PATH_DEMO_ICD_CODES_RAW[:-len('.csv')] + '_all_15Aug.json'
        try_log_info("Writing data to {}".format(out_data_path))
        with open(out_data_path, "a") as fp:
            for p in patients.values():
                fp.write(json.dumps(p) + "\n")



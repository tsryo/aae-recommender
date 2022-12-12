import pandas as pd
import json
import numpy as np

import matplotlib.pyplot as plt

SEPARATOR = ","
ALL_TIMESERIES_MISSING_PLACEHOLDER = "*MISSING*"
IN_DATA_PATH = "../../data/diagnoses_procedures_icd_icu_staydetail.csv"
IN_DATA_PATH_ICUSTAY_DETAIL = "../../data/diagnoses_procedures_icd_icu_staydetail.csv"
IN_DATA_PATH_VITALS = "../../data/vitals.csv"
READM = True

print("Reading data from {}".format(IN_DATA_PATH))
df = pd.read_csv(IN_DATA_PATH, sep=SEPARATOR, index_col=False)
df_vitals = pd.read_csv(IN_DATA_PATH_VITALS, sep=SEPARATOR, index_col=False)

df_conditions = None #pd.read_csv(IN_DATA_PATH_ICUSTAY_DETAIL, sep=SEPARATOR, index_col=False)
print(df.head())

#note: not all hadm_ids have vitals, remove them from our dataset
vital_hadm_ids = set(list(df_vitals['hadm_id']))
# todo: print how many you  remove
#print()
df = df.loc[df['hadm_id'].isin(vital_hadm_ids)]

READ_BUFFER_SIZE = 2048.0

vitals_columns_to_aggr = ['heartrate_min', 'heartrate_max', 'heartrate_mean', 'sysbp_min', 'sysbp_max',
                          'sysbp_mean', 'diasbp_min', 'diasbp_max', 'diasbp_mean', 'meanbp_min', 'meanbp_max',
                          'meanbp_mean', 'resprate_min', 'resprate_max', 'resprate_mean', 'tempc_min', 'tempc_max',
                          'tempc_mean', 'spo2_min', 'spo2_max', 'spo2_mean', 'glucose_min', 'glucose_max',
                          'glucose_mean']

df = df.sort_values(['hadm_id'], ascending=True)
df_seq_num_len = df.groupby('hadm_id')['seq_num'].apply(np.max).reset_index(name='seq_num_len')
df = df.merge(df_seq_num_len, on='hadm_id', how='left')
df = df.drop(['seq_num', 'first_icu_stay'], axis=1)
df = df.drop_duplicates()

GLOBAL_MEANS = {}
GLOBAL_SDS = {}
for x in vitals_columns_to_aggr:
    GLOBAL_MEANS[x] = np.nanmean(df_vitals[x])
    GLOBAL_SDS[x] = np.nanstd(df_vitals[x])

pats_cols_to_aggr = ['age', 'los_icu', 'seq_num_len']
for x in pats_cols_to_aggr:
    GLOBAL_MEANS[x] = np.nanmean(df[x])
    GLOBAL_SDS[x] = np.nanstd(df[x])


## pre-processing functions below
def aggregate_vars_per_id(id_var, aggr_vars, df, drop_id_dups=True, drop_aggr_dups=False):
    lst_of_lsts = []
    for c_col in aggr_vars:
        print("Aggregating {0} values into lists per {1}".format(c_col, id_var))
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


def impute_timeseries_values(lst_vals):
    nan_idxs = np.where(np.isnan(lst_vals))[0]
    if len(nan_idxs) == len(lst_vals):
        return [ALL_TIMESERIES_MISSING_PLACEHOLDER] * len(lst_vals)  # cant impute, set them to 0s for now and mark them later
    if len(nan_idxs) == 0:
        return lst_vals
    last_non_nan_idx = np.where(~np.isnan(lst_vals))[0][-1]
    first_non_nan_idx = np.where(~np.isnan(lst_vals))[0][0]
    for i in nan_idxs:
        imp_val = np.nan
        if i > last_non_nan_idx:
            imp_val = lst_vals[last_non_nan_idx]
        elif i < first_non_nan_idx:
            imp_val = lst_vals[first_non_nan_idx]
        else:
            next_non_nan_idx = np.where(~np.isnan(lst_vals[i + 1:]))[0][0] + i + 1
            prev_non_nan_idx = np.where(~np.isnan(lst_vals[:i]))[0][-1]
            imp_val = (lst_vals[prev_non_nan_idx] + lst_vals[next_non_nan_idx]) / 2
        lst_vals[i] = imp_val
    return lst_vals


def impute_timeseries_columns(cols, df):
    for col in cols:
        print("Imputing series for column {}".format(col))
        df[col] = [impute_timeseries_values(x) for x in df[col]]
    return df


def add_features_mark_missing_series(cols, df):
    for col in cols:
        # _mm - missing marker
        df[col + "_mm"] = [1 if str(x[0]) == ALL_TIMESERIES_MISSING_PLACEHOLDER else 0 for x in df[col]]
        df[col] = [[0] * len(x) if str(x[0]) == ALL_TIMESERIES_MISSING_PLACEHOLDER else x for x in df[col]]
    return df.copy()

def append_column_aggregates(series_columns_lst, aggr_fns_d, df):
    for k, v in aggr_fns_d.items():
        for c_col in series_columns_lst:
            aggr_col_nm = c_col + "_{}".format(k)
            print("append_column_aggregates for column {}".format(aggr_col_nm))
            df[aggr_col_nm] = [v(x) if x[0] != ALL_TIMESERIES_MISSING_PLACEHOLDER else 0 for x in df[c_col]]
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


# todo: consider using actual charttime differences between each measurement instance as X-axis
slope_fn = lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else np.mean(y)
delta_fn = lambda x: sum([x[i - 1] - x[i] for i in range(1, len(x))]) / (len(x) - 1) if len(x) > 1 else np.mean(x)
aggr_fns_d = {"slope": slope_fn, "mean": np.nanmean, "sd": np.std, "delta": delta_fn, "min": np.min, "max": np.max}
df = df.loc[df['age'] >= 18] # remove all non-adult entries
all_hadm_ids = list(set(df['hadm_id']))
n_iters_needed = np.ceil(len(all_hadm_ids)/READ_BUFFER_SIZE)

df_cpy = df
df_vitals_cpy = df_vitals

for i in range(int(n_iters_needed)):
    print("Iteration {} (out of {})".format(i, n_iters_needed))
    range_upper_bound = int((i*READ_BUFFER_SIZE + READ_BUFFER_SIZE)) if i != int(n_iters_needed)-1 else len(all_hadm_ids)
    hadm_ids = [all_hadm_ids[i] for i in list(range(int((i*READ_BUFFER_SIZE)), range_upper_bound))] # debug hack
    df_vitals = df_vitals_cpy.loc[df_vitals_cpy['hadm_id'].isin(hadm_ids)]
    df = df_cpy.loc[df_cpy['hadm_id'].isin(hadm_ids)]
    df = df.drop(['subject_id', 'icustay_id', 'dod', 'admittime', 'dischtime',
                  'ethnicity', 'hospital_expire_flag', 'hospstay_seq', 'first_hosp_stay', 'intime', 'outtime',
                  'icustay_seq'], axis=1)
    # drop unused columns
    df_vitals = df_vitals.drop(["subject_id", 'icustay_id'], axis=1)
    # round and decode age values
    df['age'] = round(df['age'], 0)
    df['age'] = [age - 210 if age > 289 else age for age in df['age']]

    # for each patient (actually admission) get the lists of all icd9 codes, vitals, and icu_los;
    # standardize, generate aggregates, impute missing from series per patient, mark completely missing series with boolean features
    if READM:
        # vitals
        df_vitals = df_vitals.sort_values(['hadm_id', 'charttime'], ascending=True)
        df_vitals = df_vitals.drop(['charttime'], axis=1)

        df_vitals = standardize_cap_and_normalize_cols(vitals_columns_to_aggr, df_vitals, sd_cap=10)
        vitals_columns_aggregated = [x + "_lst" for x in vitals_columns_to_aggr]
        df_vitals = aggregate_vars_per_id("hadm_id", vitals_columns_to_aggr, df_vitals)
        # todo:  remove cols with more than x% missing data
        df_vitals = impute_timeseries_columns(vitals_columns_aggregated, df_vitals)
        df_vitals = append_column_aggregates(vitals_columns_aggregated, aggr_fns_d, df_vitals)
        df_vitals = add_features_mark_missing_series(vitals_columns_aggregated, df_vitals)

        # patient details
        pat_details_columns_to_aggr = ['icd9_code', 'los_icu']
        #df = standardize_cap_and_normalize_cols(['age', 'los_icu', 'seq_num_len'], df, sd_cap=10)
        df = aggregate_vars_per_id("hadm_id", pat_details_columns_to_aggr, df, drop_aggr_dups=True)
        df = append_column_aggregates(['los_icu_lst'], aggr_fns_d, df)

        assert all(df['hadm_id'].values == df_vitals['hadm_id'].values)
        # combine pat ditails and vitals
        df = pd.merge(df, df_vitals, how="inner", on="hadm_id")
        print(df.head())
    else:
        df = df.groupby('hadm_id')['icd9_code'].apply(list).reset_index(name='icd9_code_lst')
        df = df.merge(df_conditions, on='hadm_id')
        print(df.head())

    # from df to dict
    patients = df.T.to_dict()

    # serialize in json
    out_data_path = IN_DATA_PATH[:-len('.csv')] + '_debug_2048_x28_12.json'
    print("Writing data to {}".format(out_data_path))
    with open(out_data_path, "a") as fp:
        for p in patients.values():
            fp.write(json.dumps(p) + "\n")


import pandas as pd
import json
import numpy as np
from CONSTANTS import DATA_PATH, ICD_CODE_DEFS_PATH, IN_DATA_PATH_VITALS, IN_DATA_PATH
from eval.fiv import load
import matplotlib.pyplot as plt

SEPARATOR = ","
ALL_TIMESERIES_MISSING_PLACEHOLDER = "*MISSING*"


READM = True
PLOT_FIGS = False
GEN_VITALS_SUMMARY_TABLE = False

print("Reading data from {}".format(IN_DATA_PATH))
df = pd.read_csv(IN_DATA_PATH, sep=SEPARATOR, index_col=False)
df_vitals = pd.read_csv(IN_DATA_PATH_VITALS, sep=SEPARATOR, index_col=False)



### plotting reasons...
from datetime import datetime
import seaborn as sns
sns.set_theme()



df_vitals = df_vitals.sort_values(['hadm_id', 'charttime'], ascending=True)
#note: not all hadm_ids have vitals, remove them from our dataset
vital_hadm_ids = set(list(df_vitals['hadm_id']))

df_vitals_new = None
for hadm_id in list(vital_hadm_ids):
    c_df = df_vitals.iloc[df_vitals['hadm_id'].values == hadm_id]
    ftime = c_df['charttime'].values[0]
    ftime = datetime.strptime(ftime, '%Y-%m-%d %H:%M:%S')
    time_mins = [0]
    for time_period in c_df['charttime'].values[1:]:
        ctime = datetime.strptime(time_period, '%Y-%m-%d %H:%M:%S')
        c_time_in_mins = ctime - ftime
        c_time_in_mins = int(c_time_in_mins.total_seconds())/60
        time_mins.append(c_time_in_mins)

    c_df['time_mins'] = time_mins
    df_vitals_new = pd.concat([df_vitals_new, c_df])

    # Create a visualization
    plot_dir = '/mnt/c/Development/github/Python/plt/experiment/{}'
    plot_nm = plot_dir.format('lineplot.png')
    #sns.lmplot(data=c_df, x="time_mins", y="heartrate_min" )
random10_hadms = list(set(df_vitals_new['hadm_id'].tolist()))[50:60]
c_df = df_vitals_new[df_vitals_new['hadm_id'].isin(random10_hadms)]

df_vitals = df_vitals_new
#sns.set(rc={'figure.figsize': (20, 10)})
from scipy.stats import rankdata
# c_df['hadm_id'] = rankdata(c_df['hadm_id'], method = 'dense')

if PLOT_FIGS:
    c_df['hadm_id'] = rankdata(c_df['hadm_id'], method='dense')
    fig = sns.relplot(data=c_df, x="time_mins", y="heartrate_min", kind="line", hue = 'hadm_id', size = 8, aspect = 2,  palette = sns.color_palette("husl", 10))

    fig.savefig(plot_nm, dpi = 500)
    plt.show()


## summarize idea - for each timeseries: fit a linear model on metric over time per patient
###  something like
## n_measurements, n_missings, median, Q1, Q3,
table_cols = ['heartrate_min', 'heartrate_max', 'heartrate_mean', 'sysbp_min', 'sysbp_max',
       'sysbp_mean', 'diasbp_min', 'diasbp_max', 'diasbp_mean', 'meanbp_min',
       'meanbp_max', 'meanbp_mean', 'resprate_min', 'resprate_max',
       'resprate_mean', 'tempc_min', 'tempc_max', 'tempc_mean', 'spo2_min',
       'spo2_max', 'spo2_mean', 'glucose_min', 'glucose_max', 'glucose_mean']
vitals_table = None

pat_cols = [ 'heartrate_min_lst_mm', 'heartrate_max_lst_mm', 'heartrate_mean_lst_mm',
            'sysbp_min_lst_mm', 'sysbp_max_lst_mm', 'sysbp_mean_lst_mm', 'diasbp_min_lst_mm',
            'diasbp_max_lst_mm', 'diasbp_mean_lst_mm', 'meanbp_min_lst_mm', 'meanbp_max_lst_mm', 'meanbp_mean_lst_mm',
            'resprate_min_lst_mm', 'resprate_max_lst_mm', 'resprate_mean_lst_mm', 'tempc_min_lst_mm', 'tempc_max_lst_mm',
            'tempc_mean_lst_mm', 'spo2_min_lst_mm', 'spo2_max_lst_mm', 'spo2_mean_lst_mm', 'glucose_min_lst_mm', 'glucose_max_lst_mm', 'glucose_mean_lst_mm'
             ]
if GEN_VITALS_SUMMARY_TABLE:
    patients = load(DATA_PATH)
    for p_col in pat_cols:
        vals = [x[p_col] for x in patients]
        print(p_col)
        print((sum(vals)/49002)*100)

    icd_code_defs = pd.read_csv(ICD_CODE_DEFS_PATH, sep='\t')
    ages = [x['icd9_code_lst'] for x in patients]
    ages = [y for x in ages for y in x]
    xxxx = pd.Series(ages).value_counts()[-50:]
    rare_codes = None
    for c_key in xxxx.keys().tolist():
        is_diag = 'd_' in c_key
        type_filter = 'DIAGNOSE' if is_diag else 'PROCEDURE'
        c_df = icd_code_defs[icd_code_defs['type'] == type_filter]
        c_df = c_df[c_df['icd9_code'] == c_key[2:]]
        rare_codes = pd.concat([c_df, rare_codes])

    np.where(np.isnan(ages))
    np.nanmedian(ages)
    np.nanquantile(ages,0.25)
    np.nanquantile(ages,0.75)

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

    c_df.columns


df_conditions = None
print(df.head())


icd_all = list(df['icd9_code'].values)
icd_diag = [x for x in icd_all if type(x) != float and x[0:2] == 'd_']
icd_proc = [x for x in icd_all if type(x) != float and x[0:2] == 'p_']
for x in icd_all:
    if type(x) == float:
        print(x)
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
        imp_val = np.nan
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

def impute_timeseries_columns(cols, df):
    for col in cols:
        print("Imputing series for column {}".format(col))
        for c_row in range(0,df.shape[0]):
            df[col].iloc[c_row] = impute_timeseries_values(df[col].iloc[c_row], df['time_mins_lst'].iloc[c_row])
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
all_icds  = df['icd9_code'].values
N_icd_diag_codes_all = [x for x in all_icds if type(x) != float and x[0:2] == 'd_']
len(N_icd_diag_codes_all)
len(set(N_icd_diag_codes_all))
icd_val_counts = pd.value_counts(all_icds)
icds_more_than50_occurs = [ x[1] for x in list(zip(icd_val_counts, icd_val_counts.index)) if x[0] >= 50]
all_icds_50 = [x for x in all_icds if x in icds_more_than50_occurs]

N_icd_proc_codes_all = [x for x in all_icds if x[0:2] == 'p_']
len(N_icd_proc_codes_all)
len(set(N_icd_proc_codes_all))
icd_val_counts = pd.value_counts(all_icds)
icds_more_than50_occurs = [ x for x in icd_val_counts.to_list() if x >= 50]



# Main scipt logic - iteratively load, parse and save records from csv to json format
# merges patient admisison vitals with icd code list + demographics data

n_iters_needed = np.ceil(len(all_hadm_ids)/READ_BUFFER_SIZE)

df_cpy = df
df_vitals_cpy = df_vitals
df2 = df[['hadm_id', 'icd9_code']]
df2 = df2.groupby(['hadm_id'])
df2 = df2.count()
icd_per_admission = df2.icd9_code.tolist()
np.median(icd_per_admission) # 15
np.percentile(icd_per_admission, [25, 75])

for i in range(int(n_iters_needed)):
    print("Iteration {} (out of {})".format(i, n_iters_needed))
    range_upper_bound = int((i*READ_BUFFER_SIZE + READ_BUFFER_SIZE)) if i != int(n_iters_needed)-1 else len(all_hadm_ids)
    hadm_ids = [all_hadm_ids[i] for i in list(range(int((i*READ_BUFFER_SIZE)), range_upper_bound))]
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
        df_vitals = aggregate_vars_per_id("hadm_id", [*vitals_columns_to_aggr, 'time_mins'], df_vitals)
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
    out_data_path = IN_DATA_PATH[:-len('.csv')] + '_all.json'
    print("Writing data to {}".format(out_data_path))
    with open(out_data_path, "a") as fp:
        for p in patients.values():
            fp.write(json.dumps(p) + "\n")


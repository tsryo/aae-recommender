#Emily add: comment: from CONSTANTS import IN_DATA_PATH_DEMO_ICD_CODES
from CONSTANTS import NDC_CODE_DEFS_PATH, LOAD_NDC_CODE_TEXT_DEFS
from utils.parse_noteevents import extract_attributes_from_patients_json
from collections.abc import Iterable
import pandas as pd
from irgan.utils import load
import json

# Emily vraag: heb niet IN_DATA_PATH_DEMO_ICD_CODES nodig, want die laad icd codes en patiente data die niet gerelateerd is aan medicatie
# icu stay staat namelijk wel in collums.. patient id (link medications to patients -> icd code?? waar vandaan)

# Define the input and output paths
prescriptions_path = '../../data/MIMIC/medication/prescriptions.csv'
out_data_path = '../../data/MIMIC/medication/patients_full.json'

# Read medication list
p_df = pd.read_csv(prescriptions_path, header=None)
p_df.columns = ['row_id', 'subject_id', 'hadm_id', 'icustay_id', 'startdate',
                'enddate', 'drug_type', 'drug', 'drug_name_poe', 'drug_name_generic',
                'formulary_drug_cd', 'gsn', 'ndc', 'prod_strength', 'dose_val_rx', 'dose_unit_rx',
                'form_val_disp', 'form_unit_disp', 'route']
# print('debug')
# cols_of_interest = ['hadm_id', 'ndc']

# Emily vraag: patient, is dat een list? checken
# Emily add: Load patient data, assuming that 'patients' is a list of patient records
patients = load(NDC_CODE_DEFS_PATH)

# Link medications to patients
for p in patients:
     c_rows = p_df[p_df['hadm_id'] == p['hadm_id']]
     c_rows = c_rows.sort_values(by='startdate')
    # list(c_rows['ndc'].values)
     p['ndc_list'] = list(c_rows['ndc'].values)

# Emily add: intialize dict text description of each NDC code (medication) 
if LOAD_NDC_CODE_TEXT_DEFS:
    ndc_code_defs = pd.read_csv(NDC_CODE_DEFS_PATH, sep='\t')
    d_ndc_code_defs = {}
    for _, row in ndc_code_defs.iterrows():
        ndc_code = row['ndc_code']
        if ndc_code in d_ndc_code_defs:
            print(f"{ndc_code} already in dict, prepending 0 to new key entry to prevent override")
            ndc_code = 'n0_' + ndc_code
        d_ndc_code_defs[ndc_code] = row['long_title']

# Emily add: Use extract_attributes_from_patients_json if needed
for p in patients:
    p['attributes'] = extract_attributes_from_patients_json(p)

# Emily add:
# Define imputation and aggregation functions
# If the medication data contains time series data with missing values this def helps to fill in those gaps
#def impute_timeseries_values(lst_vals, time_mins):
#    nan_idxs = np.where(np.isnan(lst_vals))[0]
#    if len(nan_idxs) == len(lst_vals):
#        return [ALL_TIMESERIES_MISSING_PLACEHOLDER] * len(lst_vals) # can't impute, set them to 0s for now and mark them later
#    if len(nan_idxs) == 0:
#        return lst_vals
#    last_non_nan_idx = np.where(~np.isnan(lst_vals))[0][-1]
#    first_non_nan_idx = np.where(~np.isnan(lst_vals))[0][0]
#    last_non_nan_time = time_mins[last_non_nan_idx]
#    first_non_nan_time = time_mins[first_non_nan_idx]

#    for i in nan_idxs:
#        imp_val = 0
#        if i > last_non_nan_idx:
#            imp_val = lst_vals[last_non_nan_idx]
#        elif i < first_non_nan_idx:
#            imp_val = lst_vals[first_non_nan_idx]
#        else:
#            next_non_nan_idx = np.where(~np.isnan(lst_vals[i + 1:]))[0][0] + i + 1
#            prev_non_nan_idx = np.where(~np.isnan(lst_vals[:i]))[0][-1]
#            c_nan_time = time_mins[i]
#            next_non_nan_time = time_mins[next_non_nan_idx]
#            prev_non_nan_time = time_mins[prev_non_nan_idx]
#            time_dist_next = next_non_nan_time - c_nan_time
#            time_dist_prev = c_nan_time - prev_non_nan_time
#            total_time_diff = time_dist_prev + time_dist_next
#            val_weight_next = (total_time_diff - time_dist_next) / total_time_diff
#            val_weight_prev = (total_time_diff - time_dist_prev) / total_time_diff

#            imp_val = (lst_vals[prev_non_nan_idx] * val_weight_prev + lst_vals[next_non_nan_idx] * val_weight_next)
#        lst_vals[i] = imp_val
#    return lst_vals

# Emily add:
# if i need to aggregate medication data (bijv. collecting all NDC codes per patient), this def can help group and aggregate this information effectively
# It will aggregates NDC lists per patient
#def aggregate_vars_per_id(id_var, aggr_vars, df, drop_id_dups=True, drop_aggr_dups=False):
#    lst_of_lsts = []
#    for c_col in aggr_vars:
#        lst_name = '{0}_lst'.format(c_col)
#        fn_to_apply = lambda x: list(x) if not drop_aggr_dups else list(set(x))
#        df_c_col_lst = df.groupby(id_var)[c_col].apply(fn_to_apply).reset_index(name=lst_name)
#        df = df.drop(c_col, axis=1)
#        lst_of_lsts.append(df_c_col_lst)
#    for c_lst in lst_of_lsts:
#        df = df.merge(c_lst, on=id_var, how='left')

#    if drop_id_dups:
#        df = df.drop_duplicates(subset=id_var, keep="first")
#    return df

# Example: Aggregate NDC codes per patient
# patients_df = pd.DataFrame(patients)
# aggregated_patients = aggregate_vars_per_id('hadm_id', ['ndc_list'], patients_df)

# when using the aggregated, this will be the new 'saving':
# with open(out_data_path, "a") as fp:
#     for p in aggregated_patients.to_dict(orient='records'):
#         fp.write(json.dumps(p) + "\n")

# Emily add: Saves the processed data to the specified output path
with open(out_data_path, "a") as fp:
    for p in patients:
        fp.write(json.dumps(p) + "\n")

# old version save the processed patients data
# p = patients[14]
# with open(out_data_path, "a") as fp:
#    for p in patients:
#        fp.write(json.dumps(p) + "\n")

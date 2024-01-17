from CONSTANTS import IN_DATA_PATH_DEMO_ICD_CODES
from utils.parse_noteevents import extract_attributes_from_patients_json
import pandas as pd
from irgan.utils import load
import json
# read medication list
#
p_df = pd.read_csv('../../data/MIMIC/medication/prescriptions.csv', header=None)
out_data_path = '../../data/MIMIC/medication/patients_full.json'
p_df.columns = ['row_id', 'subject_id', 'hadm_id', 'icustay_id', 'startdate',
                'enddate', 'drug_type', 'drug', 'drug_name_poe', 'drug_name_generic',
                'formulary_drug_cd', 'gsn', 'ndc', 'prod_strength', 'dose_val_rx', 'dose_unit_rx',
                'form_val_disp', 'form_unit_disp', 'route']
print('debug')
cols_of_interest = ['hadm_id', '']

#
patients = load(IN_DATA_PATH_DEMO_ICD_CODES)
for p in patients:
     c_rows = p_df[p_df['hadm_id'] == p['hadm_id']]
     c_rows = c_rows.sort_values(by='startdate')
     list(c_rows['ndc'].values)
     p['ndc_list'] = list(c_rows['ndc'].values)


p = patients[14]
with open(out_data_path, "a") as fp:
    for p in patients:
        fp.write(json.dumps(p) + "\n")

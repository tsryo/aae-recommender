# ICD9 codes prediction - SQL queries used for data extraction from MIMIC



## Extract icd codes plus patient demographics 
#### (used for creating csv referred by IN_DATA_PATH)
```
select 
pds_icd.subject_id,pds_icd.hadm_id,
pds_icd.seq_num,icd9_code,icustay_id,gender,dod,admittime,dischtime,los_hospital,age,ethnicity,ethnicity_grouped,admission_type,hospital_expire_flag,hospstay_seq,first_hosp_stay,intime,outtime,los_icu,icustay_seq,first_icu_stay
from 
	(
	select subject_id, hadm_id, seq_num, concat('d_', icd9_code) as icd9_code from 
	mimiciii.diagnoses_icd d
	union all (
			select subject_id, hadm_id, seq_num, concat('p_', icd9_code) as icd9_code 
			from mimiciii.procedures_icd
			  )
	) as pds_icd -- procedure and diagnose codes table 
	LEFT JOIN mimiciii.icustay_detail dmgrfx on dmgrfx.hadm_id = pds_icd.hadm_id
-- where dmgrfx.subject_id != pds_icd.subject_id	(sanity check, none match on hadm but NOT on sujectid)
where dischtime is not null -- expect 975761 rows
--limit 200 -- debug
;
```

## Extract vitals timeseries 
#### (used for creating csv referred by IN_DATA_PATH_VITALS)
```
select 
subject_id,hadm_id,icustay_id,
charttime,heartrate_min,heartrate_max,heartrate_mean,sysbp_min,sysbp_max,sysbp_mean,diasbp_min,
diasbp_max,diasbp_mean,meanbp_min,meanbp_max,meanbp_mean,resprate_min,resprate_max,resprate_mean,
tempc_min,tempc_max,tempc_mean,spo2_min,spo2_max,spo2_mean,glucose_min,glucose_max,glucose_mean
from
 mimiciii.vitals
-- limit 200 debug
;
/*where hadm_id in (select  hadm_id from  -- sanity check all rows from vitals should have icd code rows (they do!)
(
	select subject_id, hadm_id, seq_num, concat('d_', icd9_code) as icd9_code from 
	mimiciii.diagnoses_icd d
	union all (
			select subject_id, hadm_id, seq_num, concat('p_', icd9_code) as icd9_code 
			from mimiciii.procedures_icd
			  )
	) as pds_icd ) */	
```


## Extract data dictionary for icd codes 
#### (used for creating csv referred by ICD_CODE_DEFS_PATH )
```
select type, icd9_code, short_title, long_title
from 
	(
	select icd9_code, short_title, long_title, 'DIAGNOSE' as type from 
	mimiciii.d_icd_diagnoses d
	union all (
			select icd9_code, short_title, long_title, 'PROCEDURE' as type
			from mimiciii.d_icd_procedures
			  )
	) as d_pds_icd -- procedure and diagnose codes text descriptions table 
-- LIMIT 200 debug 
;
```

### Other example queries

#### Count admissions with at least one icd9 code
```
select count(DISTINCT hadm_id) from mimiciii.diagnoses_icd
```

#### Count patients with redmissions
```
select count(*) from mimiciii.icustay_detail where icustay_seq > 1
```

#### Extract ICD9 codes for all patients with redmissions
```
select DISTINCT di.subject_id, di.hadm_id, di.seq_num, di.icd9_code, id.hospstay_seq
from mimiciii.diagnoses_icd di, mimiciii.icustay_detail id 
where di.hadm_id = id.hadm_id
and di.subject_id in (
	select subject_id
	from mimiciii.icustay_detail
	where hospstay_seq > 1
)
order by di.subject_id, id.hospstay_seq, di.hadm_id, di.seq_num asc
```

#### Extract static features for all patients
```
select subject_id, hadm_id, age, gender, ethnicity, ethnicity_grouped, admission_type, hospstay_seq, icustay_seq 
from mimiciii.icustay_detail
order by subject_id, hadm_id asc
```

###### Notes:
`seq_num` from diagnoses_icd symbolizes priority of ICD9 code (with prio 1 meaning most important/significant code according to record)
Might be interesting to use it in prediction?
Question: is there a way to check the time a code was assigned? 

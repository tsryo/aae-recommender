import time
import Levenshtein
# Emily add: original:
#from CONSTANTS import IN_DATA_PATH_DEMO_ICD_CODES, IN_DATA_TEXT_PATH, EMBEDDINGS_FILENAME
# Emily add:
from CONSTANTS import (
    IN_DATA_PATH_DEMO_ICD_CODES,
    IN_DATA_TEXT_PATH,
    EMBEDDINGS_FILENAME,
    NDC_CODE_DEFS_PATH,
    LOAD_NDC_CODE_TEXT_DEFS
)
import itertools

from irgan.utils import load
import seaborn as sns
import re
from datetime import datetime
import json
import csv
import string
import torch.nn as nn
import spacy
import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import (
    AutoModel, AutoConfig,
    AutoTokenizer, logging
)
import torch
import torch.nn.functional as F
import gc; gc.enable()
# Emily add:
import os

sns.set_theme()

import warnings
warnings.filterwarnings("ignore")

RUN_STEP1 = False
RUN_STEP2 = False
RUN_STEP3 = False
RUN_STEP4 = False
RUN_STEP5 = False
RUN_STEP6 = True


# Declare constants
READ_BUFFER_SIZE_STEP1 = 50000
WRITE_BUFFER_SIZE_STEP1 = 10000
BATCHED_JSON_FILENAME = "../../data/MIMIC/hadm_ids_notes_batched.json"
BATCH_TO_HADM_ID_MAPPING = "../../data/MIMIC/batch-hadm-id-mapping.json"

WRITE_BUFFER_SIZE_STEP2 = 1000
WRITE_BUFFER_SIZE_STEP5 = 1000
CONCAT_JSON_FILENAME = "../../data/MIMIC/hadm_ids_notes_conc.json"
CONCAT_JSON_FILTERED_FILENAME = "../../data/MIMIC/hadm_ids_notes_conc_filtered.json"
CONCAT_JSON_PROCESSED_FILENAME = "../../data/MIMIC/hadm_ids_notes_conc_processed.json"


HADM_IDS_FROM_TEXT_MISSING_CODES_FILENAME = "../../data/MIMIC/hamd_ids_missing_codes.csv"


def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    formatted_time = ""
    # if hours > 0:
    formatted_time += f"{hours} hour{'s' if hours > 1 else ''} "
    # if minutes > 0:
    formatted_time += f"{minutes} minute{'s' if minutes > 1 else ''} "
    # if remaining_seconds > 0 or (hours == 0 and minutes == 0):
    formatted_time += f"{remaining_seconds} second{'s' if remaining_seconds > 1 else ''}"

    return formatted_time.strip()

def extract_date_in_milliseconds(text):
    milis = __extract_date_in_milliseconds_pat1(text)
    if milis is None:
        milis = __extract_date_in_milliseconds_pat2(text)
    if milis is None:
        milis = __extract_date_in_milliseconds_pat3(text)
    if milis is None:
        milis = __extract_date_in_milliseconds_pat4(text)
    if milis is None:
        return None
    return milis

def __extract_date_in_milliseconds_pat4(text):
    # Define the regex pattern to match the date
    date_pattern = r"\[\*\*Month/Day/Year\s+\*\*\] Date:\s+\[\*\*(\d{4}-\d{1,2}-\d{1,2})\*\*\]"

    # Find the date using the regex pattern
    match = re.search(date_pattern, text)

    if match:
        date_str = match.group(1)
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        timestamp_in_milliseconds = int(date_obj.timestamp() * 1000)
        return timestamp_in_milliseconds
    else:
        return None
def __extract_date_in_milliseconds_pat3(text):
    # Define the regex pattern to match the date
    date_pattern = r"Discharge Date:\s+\[\*\*(\d{4}-\d{1,2}-\d{1,2})\*\*\]"

    # Find the date using the regex pattern
    match = re.search(date_pattern, text)

    if match:
        date_str = match.group(1)
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        timestamp_in_milliseconds = int(date_obj.timestamp() * 1000)
        return timestamp_in_milliseconds
    else:
        return None

def __extract_date_in_milliseconds_pat2(text):
    pattern = r"Completed by:\[\*\*(\d{4}-\d{1,2}-\d{1,2})\*\*\]"
    match = re.search(pattern, text)

    if match:
        date_str = match.group(1)
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        milliseconds = int((date_obj - datetime(1970, 1, 1)).total_seconds() * 1000)
        return milliseconds

    return None

def __extract_date_in_milliseconds_pat1(text):
    date_pattern = r"\[\s*\*\*\s*(\d{4})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2})\s*\*\*\s*\]\s*(\d{1,2}:\d{2})"

    match = re.search(date_pattern, text)

    if match:
        year, month, day, time = match.groups()
        year, month, day = map(int, (year, month, day))
        hour, minute = map(int, time.split(':'))
        minute = min(59, minute) # ensure minute is a minute
        hour = min(23, hour) # ....
        date_time = datetime(year, month, day, hour, minute)
        timestamp_in_milliseconds = int(date_time.timestamp()) * 1000
        return timestamp_in_milliseconds
    else:
        return None

# Emily add: original:
#def read_file_in_chunks(filepath, chunk_size = 1000):
#    with open(filepath, 'r', encoding='utf-8') as file:
#        chunk = []
#        for line in file:
#            chunk.append(line)
#            if len(chunk) == chunk_size:
#                yield chunk
#                chunk = []
#        if chunk:
#            yield chunk  # Yield the last chunk if it's not full

# Emily add:
#def read_file_in_chunks(filepath, chunk_size = 1000):
#    full_path = os.path.abspath(filepath)
#    print(f"Trying to open file at: {full_path}")
#    try:
#        with open(filepath, 'r', encoding='utf-8') as file:
#            chunk = []
#        for line in file:
#            chunk.append(line)
#            if len(chunk) == chunk_size:
#                yield chunk
#                chunk = []
#        if chunk:
#            yield chunk  # Yield the last chunk if it's not full
#    except FileNotFoundError as e:
#        print(f"File Not Found Error: {e}")
#        print(f"Could not find file at: {full_path}")

#Emily add:
def read_file_in_chunks(filepath, batch_size):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            while True:
                lines = list(itertools.islice(file, batch_size))
                if not lines:
                    break
                yield lines
    except FileNotFoundError as e:
        print(f"File Not Found Error: {e}")
        print(f"Could not find file at: {filepath}")
        raise

def extract_attributes_from_patients_json(filepath, attrs):
    ret_o = {}
    patients = load(filepath)
    for attr in attrs:
        c_attr_vals = []
        for patient in patients:
            c_attr_vals.append(patient[attr])
        ret_o[attr] = c_attr_vals
    if len(attrs) == 1:
        return list(ret_o.values())[0] # horrible python syntax, surely there is a better way?
    return ret_o


def invert_dictionary(input_dict):
    inverted_dict = {}

    for key, value_list in input_dict.items():
        for value in value_list:
            if value not in inverted_dict:
                inverted_dict[value] = [key]
            else:
                inverted_dict[value].append(key)

    return inverted_dict


def read_specific_lines(filepath, line_numbers):
    selected_lines = []

    with open(filepath, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, start=0):
            if i in line_numbers:
                selected_lines.append(line)
                if len(selected_lines) == len(line_numbers):
                    break

    return selected_lines


def remove_punctuation(input_string):
    # Create a translation table to remove punctuation
    translator = str.maketrans("", "", string.punctuation)

    # Use the translation table to remove punctuation
    cleaned_string = input_string.translate(translator)

    return cleaned_string


def replace_numbers_with_token(input_string, token="<NUM>"):
    # Define a regular expression pattern to match numbers
    number_pattern = r'\b\d+\b'

    # Use re.sub() to replace numbers with the token
    replaced_string = re.sub(number_pattern, token, input_string)

    return replaced_string


def remove_numbers_from_words(input_string):
    # Define a regular expression pattern to match words containing numbers
    word_with_numbers_pattern = r'\b([a-zA-Z]+)\d+([a-zA-Z]*)\b'

    # Use re.sub() to replace words with numbers within them
    cleaned_string = re.sub(word_with_numbers_pattern, '', input_string)

    return cleaned_string


def remove_strings(input_string, strings_to_remove):
    for string in strings_to_remove:
        input_string = input_string.replace(string, '')

    return input_string

start_time_seconds = int(time.time())

# Step 1 - read data from patient details and noteevents.
if RUN_STEP1:
    print(f"***************** START Step 1 - read {IN_DATA_TEXT_PATH}, parse and store into batched json list {BATCHED_JSON_FILENAME}. *****************")
    regex_hadm = re.compile(r'\d+,\d+,\d+,\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,.*,"')
    notes_per_hadm_id = {}
    hadm_ids_per_batch = {}
    #n_lines = 91691298 #count_lines_in_file(IN_DATA_TEXT_PATH) # i.e. 91.69 million
    lines_read = 0
    # read & process batches of text
    # init state
    leftover_str = ''
    batch_write_counter = 0
    for batch in read_file_in_chunks(IN_DATA_TEXT_PATH, READ_BUFFER_SIZE_STEP1):
        # if batch_write_counter > 3: # debug
        #     break
        c_str = ''.join(batch)
        c_str = leftover_str + c_str # append to front what was not yet processed from last batch
        lines_read = lines_read + len(batch)
        print(f"read {lines_read} lines")

        c_start_pos = 0
        c_end_pos = 0
        looking_for_end_match = False
        looking_for_start_match = True
        match_positions = [match.start() for match in regex_hadm.finditer(c_str)]
        for c_match_pos in match_positions:
            if looking_for_start_match:
                c_start_pos = c_match_pos
                looking_for_start_match = False
                looking_for_end_match = True
                continue
            if looking_for_end_match:
                c_end_pos = c_match_pos
                looking_for_end_match = False
            if not looking_for_start_match and not looking_for_end_match:
                # process str ...
                c_str_match = c_str[c_start_pos:c_end_pos]
                c_note_start = c_str_match.index("\"")
                # get hadm_id from header
                c_header = c_str_match[:c_note_start]
                header_splt = c_header.split(',')
                hadm_id = int(header_splt[2])
                # get note
                c_note = c_str_match[(c_note_start+1):]
                c_date_millis = extract_date_in_milliseconds(c_note)
                if c_date_millis is None:
                    continue
                # save note
                if hadm_id in notes_per_hadm_id.keys(): # is there a note for this hadm_id already?
                    prev_notes = notes_per_hadm_id[hadm_id]
                    if c_date_millis in prev_notes.keys(): # is there a note from this timestamp already?
                        print(f"Found note with same timestamp {c_date_millis} for hadm_id {hadm_id}")
                        p_note = notes_per_hadm_id[hadm_id][c_date_millis]
                        distance = Levenshtein.distance(p_note, c_note)
                        if distance > 20: # some threshold for detecting duplicates
                            print(f"multiple non-duplicate notes found for the same hadm_id and same date! {hadm_id}")
                            # take the longer one (assume more rich data)
                            if len(c_note) > len(p_note):
                                notes_per_hadm_id[hadm_id][c_date_millis] = c_note
                        # else:
                        #     print("Ignoring duplicate....")
                    else:
                        notes_per_hadm_id[hadm_id][c_date_millis] = c_note
                else:
                    notes_per_hadm_id[hadm_id] = {c_date_millis: c_note}
                # reset state
                looking_for_end_match = True
                c_start_pos = c_end_pos # start from where we left off in next match
        # end loop for match_positions
        # when you finish finding matches, remember which parts of the string are leftover and add them to the next batch
        leftover_str = c_str[c_end_pos:]

        # perform batch writing to file every 2 hadm_ids or so
        if len(notes_per_hadm_id.keys()) >= WRITE_BUFFER_SIZE_STEP1:
            with open(BATCHED_JSON_FILENAME, "a") as json_file:
                json.dump(notes_per_hadm_id, json_file)
                json_file.write("\n")
            hadm_ids_per_batch[batch_write_counter] = list(notes_per_hadm_id.keys())
            notes_per_hadm_id = {}
            batch_write_counter = batch_write_counter + 1

    # write last batch
    if len(notes_per_hadm_id.keys()) != 0:
        with open(BATCHED_JSON_FILENAME, "a") as json_file:
            json.dump(notes_per_hadm_id, json_file)
        notes_per_hadm_id = {}
        batch_write_counter = batch_write_counter + 1

    # write batch to hadm id mapping
    with open(BATCH_TO_HADM_ID_MAPPING, "a") as json_file:
        json.dump(hadm_ids_per_batch, json_file)


    print(f"Done after {batch_write_counter} batches")

if RUN_STEP2:
    print(f"Read {BATCHED_JSON_FILENAME} , merge multiple entries per hadm_id and concatenate notes")
    batches_per_hadm_id = {}
    hadm_ids_per_batch = {}
    print(f"Reading {BATCH_TO_HADM_ID_MAPPING}...")
    with open(BATCH_TO_HADM_ID_MAPPING, "r") as json_file:
        hadm_ids_per_batch = json.load(json_file)
    print("Inverting dict ...")
    batches_per_hadm_id = invert_dictionary(hadm_ids_per_batch)
    # start doing this per hadm_id, if needed make it per batch of hadm_ids ...
    # concated_notes_d = {}
    n_iters_left = len(batches_per_hadm_id.keys())


    # do batching ..
    all_hadm_ids = list(batches_per_hadm_id.keys())
    n_hadm_ids = len(all_hadm_ids)
    start_batch_idx = 0
    end_batch_idx = WRITE_BUFFER_SIZE_STEP2

    while 1 == 1:
        c_hadm_ids = all_hadm_ids[start_batch_idx:end_batch_idx]
        c_notes_per_hadm_id_d = {}
        concated_notes_d = {}
        for c_hadm_id in c_hadm_ids:
            c_notes_per_hadm_id_d[c_hadm_id] = {}
        print(f"Num of iters left: {n_hadm_ids - start_batch_idx}")
        # rewrite below for loop but for batches now..
        c_lines = list(set([ y for x in c_hadm_ids for y in batches_per_hadm_id[x] ]))
        c_lines = [int(x) for x in c_lines]
        c_lines_str = read_specific_lines(BATCHED_JSON_FILENAME, c_lines)

        # get all notes for a hadm_id to be under the same key in the dict
        for c_line_str in c_lines_str:
            c_line_d = json.loads(c_line_str)
            for c_hadm_id in c_hadm_ids:
                if str(c_hadm_id) in c_line_d.keys():
                    c_notes_per_hadm_id_d[c_hadm_id].update(c_line_d[str(c_hadm_id)]) # k = timestamp ; v = t

        #concatenate notes per hadm_id ordered by ts
        for c_hadm_id in c_hadm_ids:
            c_notes_d = c_notes_per_hadm_id_d[c_hadm_id]
            c_keys = list(c_notes_d.keys())
            c_concatenated_note = ''
            num_vals = [int(x) for x in c_keys]
            for c_ts in sorted(num_vals):
                c_note = c_notes_d[str(c_ts)]
                c_note = c_note[:-1]
                c_concatenated_note = c_concatenated_note + c_note + "****NOTE_END****"
            concated_notes_d[c_hadm_id] = c_concatenated_note

        # set idxs for next batch, reset any var state
        new_start_batch_idx = min(end_batch_idx+1, n_hadm_ids)
        end_batch_idx = min(end_batch_idx + WRITE_BUFFER_SIZE_STEP2, n_hadm_ids)

        break_cond = new_start_batch_idx == start_batch_idx

        # write to file
        # if len(concated_notes_d.keys()) >= WRITE_BUFFER_SIZE_STEP2 or break_cond:
        with open(CONCAT_JSON_FILENAME, "a") as json_file:
            json.dump(concated_notes_d, json_file)
            json_file.write("\n")

        if break_cond:
            break
        else:
            start_batch_idx = new_start_batch_idx

    print(f"Done step 2. Written to {CONCAT_JSON_FILENAME}")

if RUN_STEP3:
    print(f"Read {CONCAT_JSON_FILENAME} , record entries for which we have no icd_codes in {IN_DATA_PATH_DEMO_ICD_CODES}")
    all_hadm_ids_text = []
    all_hadm_ids_icd_codes = []
    # get hadm_ids from patients
    all_hadm_ids_icd_codes = extract_attributes_from_patients_json(IN_DATA_PATH_DEMO_ICD_CODES, ['hadm_id'])
    for c_chunk in read_file_in_chunks(CONCAT_JSON_FILENAME, chunk_size=10):
    # first3 = read_specific_lines(CONCAT_JSON_FILENAME, line_numbers=[0,1,2])
        c_chunk = [json.loads(x) for x in c_chunk]
        for x in c_chunk:
            all_hadm_ids_text = all_hadm_ids_text + list(x.keys())

    all_hadm_ids_text = [int(x) for x in all_hadm_ids_text]
    hadm_ids_missing_text = list(set(all_hadm_ids_icd_codes).difference(all_hadm_ids_text))
    hadm_ids_missing_codes = list(set(all_hadm_ids_text).difference(all_hadm_ids_icd_codes))

    with open(HADM_IDS_FROM_TEXT_MISSING_CODES_FILENAME, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(hadm_ids_missing_codes)
    print(f"Wrote hadm_ids from notes that are  missing codes in {HADM_IDS_FROM_TEXT_MISSING_CODES_FILENAME}")

if RUN_STEP4:
    print(f"Read {CONCAT_JSON_FILENAME} , remove entries for which we have no icd_codes in {IN_DATA_PATH_DEMO_ICD_CODES} [recorded in {HADM_IDS_FROM_TEXT_MISSING_CODES_FILENAME}]")
    # read HADM_IDS_FROM_TEXT_MISSING_CODES_FILENAME into list
    hadm_ids_to_remove = read_specific_lines(HADM_IDS_FROM_TEXT_MISSING_CODES_FILENAME, line_numbers=[0])
    hadm_ids_to_remove = hadm_ids_to_remove[0]
    hadm_ids_to_remove = hadm_ids_to_remove.strip().split(',')
    # hadm_ids_to_remove = [int(x) for x in hadm_ids_to_remove]

    # read CONCAT_JSON_FILENAME row by row
    for c_line in read_file_in_chunks(CONCAT_JSON_FILENAME, chunk_size=1):
        c_line = c_line[0]
        # in current row , parse as dict
        c_line = json.loads(c_line)
        # remove entries that are present in HADM_IDS_FROM_TEXT_MISSING_CODES_FILENAME
        c_to_remove = [key for key in hadm_ids_to_remove if key in c_line]
        for key in c_to_remove:
            c_line.pop(key, None)
        # write current row to new file CONCAT_JSON_FILTERED_FILENAME
        # make it so that each line is one dict and it has only one hadm_id in it
        list_of_hadm_id_dicts = []
        for hadm_id in c_line.keys():
            list_of_hadm_id_dicts.append({int(hadm_id): c_line[hadm_id]})

        with open(CONCAT_JSON_FILTERED_FILENAME, "a") as json_file:
            for c_d in list_of_hadm_id_dicts:
                json.dump(c_d, json_file)
                json_file.write("\n")
    print(f"Wrote output to {CONCAT_JSON_FILTERED_FILENAME}")

if RUN_STEP5:
    print(f"Read {CONCAT_JSON_FILTERED_FILENAME} and process text to prepare for model embedding, write results to {CONCAT_JSON_PROCESSED_FILENAME}")
    nlp = spacy.load('en_core_web_sm')
    nlp.Defaults.stop_words.add("patient")
    nlp.Defaults.stop_words.add("hospital")
    nlp.Defaults.stop_words.add("mg")
    nlp.Defaults.stop_words.add("day")
    nlp.Defaults.stop_words.add("qd")
    nlp.Defaults.stop_words.add("show")
    nlp.Defaults.stop_words.add("follow")
    nlp.Defaults.stop_words.add("note")
    nlp.Defaults.stop_words.add("q")
    nlp.Defaults.stop_words.add("time")
    write_entries = []

    # read notes per hadm_id
    batch_write_counter = 0
    for c_d in read_file_in_chunks(CONCAT_JSON_FILTERED_FILENAME, chunk_size=1):
        c_d = c_d[0]
        c_d = json.loads(c_d)
        c_hadm_id = list(c_d.keys())[0]
        c_txt = c_d[c_hadm_id]
        c_txt_orig = c_d[c_hadm_id]
        # print("0. Remove note end tokens")
        c_txt = remove_strings(c_txt, ["****NOTE_END****"])
        # print("1.Lowercase")
        c_txt = c_txt.lower()
        # print("2.Remove punctuation and special chars")
        c_txt = remove_punctuation(c_txt)
        # print("3.Handle numerals (within words)")
        c_txt = remove_numbers_from_words(c_txt)
        # print("4.Tokenize")
        c_txt = nlp(c_txt)
        # print("5.Lemmatize and remove stop words")
        c_txt = [token.lemma_ for token in c_txt if token.lemma_ not in nlp.Defaults.stop_words]
        # print("5.Handle numerals (tokenize)")
        c_txt = ['<NUM>' if token.isdigit() else token for token in c_txt]
        # print("8.Remove non-alphanum tokens")
        c_txt = [token for token in c_txt if any(c.isalnum() for c in token)]
        # prepare to store
        c_txt = '\t'.join(c_txt)
        c_entry = {'hadm_id': c_hadm_id, 'eventnotes': c_txt}
        write_entries.append(c_entry)
        # write in batches
        if len(write_entries) >= WRITE_BUFFER_SIZE_STEP5:
            with open(CONCAT_JSON_PROCESSED_FILENAME, "a") as json_file:
                for x in write_entries:
                    json.dump(x, json_file)
                    json_file.write('\n')

            write_entries = []
            batch_write_counter = batch_write_counter + 1
            print(f"batch {batch_write_counter} done")

    # write last batch
    if len(write_entries) != 0:
        with open(CONCAT_JSON_PROCESSED_FILENAME, "a") as json_file:
            for x in write_entries:
                json.dump(x, json_file)
                json_file.write('\n')
        batch_write_counter = batch_write_counter + 1
        print(f"batch {batch_write_counter} done")

    print(f"Completed after {batch_write_counter} batches")






    print(f"Wrote output to {CONCAT_JSON_PROCESSED_FILENAME}")

if RUN_STEP6:
    class AttentionPooling(nn.Module):
        def __init__(self, num_layers, hidden_size, hiddendim_fc, device=None):
            super(AttentionPooling, self).__init__()
            self.num_hidden_layers = num_layers
            self.hidden_size = hidden_size
            self.hiddendim_fc = hiddendim_fc
            self.dropout = nn.Dropout(0.1)
            self.device = device
            q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
            self.q = nn.Parameter(torch.from_numpy(q_t)).float()
            w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
            self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()

        def forward(self, all_hidden_states):
            hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                         for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
            hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
            out = self.attention(hidden_states)
            out = self.dropout(out)
            return out

        def attention(self, h):
            print(f"h.device = {h.device}")
            print(f"self.q.device = {self.q.device}")
            v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
            v = F.softmax(v, -1)
            v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
            v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
            return v


    print(f"Read {CONCAT_JSON_PROCESSED_FILENAME} and try  to learn representation via transformer [bert]")
    logging.set_verbosity_error()
    logging.set_verbosity_warning()

    batch_size = 1000
    batch_counter = 0

    # load pre-trained transformer model
    model_name = 'roberta-base'  # 'bert-base-uncased'
    config = AutoConfig.from_pretrained(model_name)
    config.update({'output_hidden_states': True})
    max_seq_length = 256

    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
    model = AutoModel.from_pretrained(model_name, config=config)
    # model.to(device)
    print(f'loaded model done')
    for in_lines in read_file_in_chunks(CONCAT_JSON_PROCESSED_FILENAME, batch_size):
        write_out = []
        print(f"Batch {batch_counter} start...")

        # read and parse text into tokens
        # in_lines = read_specific_lines(CONCAT_JSON_PROCESSED_FILENAME, line_numbers=list(range(0, 10)))  # all 46189
        # print(f'read input done')
        x1 = []
        for c_line in in_lines:
            c_line_d = json.loads(c_line.strip())
            hadm_id = c_line_d['hadm_id']
            c_line_d['eventnotes'] = c_line_d['eventnotes'].split('\t')
            x1.append(c_line_d)
        # print(f'massage input done')
        # get GPU if possible
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        texts = [' '.join(x['eventnotes']) for x in x1]
        features = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_seq_length,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        # print(f'tokenizer init done')
        with torch.no_grad():
            outputs = model(features['input_ids'], features['attention_mask'])
        print(f'run model done')
        all_hidden_states = torch.stack(outputs[2])

        hiddendim_fc = 128
        pooler = AttentionPooling(config.num_hidden_layers, config.hidden_size, hiddendim_fc)
        # pooler.to(device)
        # print(f'load attention pooling done')
        attention_pooling_embeddings = pooler(all_hidden_states)
        print(f'pooling done')
        # logits = nn.Linear(hiddendim_fc, 1)(attention_pooling_embeddings)  # regression head


        # print(f'write to file start')
        #  how do we link the 100 embeddings back to their hadm_ids?
        xxx = attention_pooling_embeddings.detach().numpy()
        for i in range(0, xxx.shape[0]):
            write_out.append({"hadm_id": x1[i]['hadm_id'], "txt_embedding": list(xxx[i,])})

        # write to file
        with open(EMBEDDINGS_FILENAME, "a") as json_file:
            for x in write_out:
                x['txt_embedding'] = [round(float(y), 4) for y in x['txt_embedding']]
                json.dump(x, json_file)
                json_file.write('\n')

        print(f"Wrote to {EMBEDDINGS_FILENAME}")
        gc.collect()
        # EMBEDDINGS_FILENAME
        # print(f'Hidden States Output Shape: {all_hidden_states.detach().numpy().shape}')
        # print(f'Attention Pooling Output Shape: {attention_pooling_embeddings.detach().numpy().shape}')
        # print(f'Logits Shape: {logits.detach().numpy().shape}')

        batch_counter += 1

# Emily vraag: error: features not defined?   define it appropriately, due to the error it is not defined before  
  
    del config, model, tokenizer, features
    gc.collect()


print("DONE DONE")
end_time_seconds = int(time.time())
print(f"execution took {format_time(end_time_seconds-start_time_seconds)}")





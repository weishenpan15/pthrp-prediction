import numpy as np
import argparse
import os.path as osp
import time
from copy import deepcopy
import json
from sklearn.linear_model import LinearRegression
from utils import split_line_str
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--train_feature_set',   default="wusm", type=str, help='name of training data')
parser.add_argument('--train_data_dir',   default="./wusm_data", type=str, help='path of testing data')
parser.add_argument('--data_dir',   default="./wcm_data", type=str, help='path of testing data')
parser.add_argument('--dataset',   default="wcm", type=str, help='name of testing data')
args = parser.parse_args()

train_feature_set = args.train_feature_set
train_data_dir = args.train_data_dir
dataset = args.dataset
data_dir = args.data_dir
non_found_set = set()
normal_range_file = "normal_region"
assay_map_file = "assay_map"
if dataset == "wusm":
    filename = "AACC_Kaggle_2022_PTHrP_input_data.txt"
    encoding = None
    sep = "\t"
elif dataset == "wcm":
    filename = "1_q_res.csv"
    encoding = None
    sep = ","
elif dataset == "mda":
    filename = "full output file-rev3.csv"
    encoding = None
    sep = ","
else:
    raise NotImplementedError

# Lab tests selected for the training WUSM dataset based on our screening criterion
if train_feature_set == "wusm":
    sel_assay = ['ALT', 'AST', 'Albumin', 'Alk Phos', 'Anion Gap', 'BUN', 'Baso Abs', 'Baso Pct', 'Bili Totl',
                 'CO2 Totl', 'Ca Ionized', 'Calcium', 'Chloride', 'Creatinine', 'Eos Abs', 'Eos Pct', 'Glucose', 'Hct',
                 'Hgb', 'INR', 'ImmGran Abs', 'ImmGran Pct', 'Lymph Pct', 'Lymphocyte Abs', 'MCH', 'MCHC', 'MCV', 'MPV',
                 'Magnesium', 'Mono Abs', 'Mono Pct', 'NRBC Abs Auto', 'Neut Abs', 'Neut Pct', 'PT', 'PTH Intact',
                 'Phos Plas', 'Plt', 'Potassium Plas', 'Protein Plas', 'RBC', 'RDW CV', 'RDW SD', 'Sodium', 'TSH',
                 'Vit D 25 OH', 'WBC', 'aPTT']
elif train_feature_set == "wcm":
    sel_assay = ['ALT', 'AST', 'Albumin', 'Alk Phos', 'Anion Gap', 'BUN', 'BUN_Creat Ratio', 'Baso Abs', 'Baso Pct',
                 'Bili Direct', 'Bili Indirect', 'Bili Totl', 'CO2 Totl', 'Calcium', 'Calcium Level with PTH',
                 'Chloride', 'Creatinine', 'Eos Abs', 'Eos Pct', 'Globulin', 'Glu P', 'Glucose', 'Hct', 'Hgb', 'INR',
                 'Instr WBC', 'Lymph Pct', 'Lymphocyte Abs', 'MCH', 'MCHC', 'MCV', 'MPV', 'Magnesium', 'Mono Abs',
                 'Mono Pct', 'Neut Abs', 'Neut Pct', 'PT', 'PTH Intact', 'Phos Plas', 'Plt', 'Potassium Plas',
                 'Protein Plas', 'RBC', 'RBC Ur', 'RDW CV', 'Sodium', 'Spec Grav Ur', 'TSH', 'Vit D 25 OH', 'WBC',
                 'WBC Ur', 'aPTT', 'pH, Ur']
elif train_feature_set == "mda":
    sel_assay = ['ALT', 'AST', 'Albumin', 'Alk Phos', 'Anion Gap', 'BUN', 'Baso Abs', 'Baso Pct', 'Bili Direct',
                 'Bili Indirect', 'Bili Totl', 'CO2 Totl', 'Ca Ionized', 'Calcium', 'Chloride', 'Creatinine', 'Eos Abs',
                 'Eos Pct', 'Glucose', 'Glucose rPOC', 'Hct', 'Hgb', 'IG Abs', 'IGRE %', 'INR', 'INRBC', 'LDH',
                 'Lymph Pct', 'Lymphocyte Abs', 'MCH', 'MCHC', 'MCV', 'MPV', 'Magnesium', 'Mono Abs', 'Mono Pct',
                 'Neut Abs', 'Neut Pct', 'POC Clean Dev', 'PT', 'PTH Intact', 'Phos Plas', 'Plt', 'Potassium Plas',
                 'Protein Plas', 'RBC', 'RBC Ur', 'RDW CV', 'RDW SD', 'Sodium', 'Spec Grav Ur', 'T4 Free', 'TSH',
                 'UA Glucose', 'UA Ketones', 'UA Protein', 'Vit D 25 OH', 'WBC', 'WBC Ur', 'Washu', 'aPTT', 'pH, Ur']
else:
    raise NotImplementedError

def load_range_dict(file):
    fin = open(file, "r")
    dict_str = fin.read()
    range_dict = json.loads(dict_str)
    freq_range_dict = {}
    for assay in range_dict:
        tmp_cnt, tmp_range = [], []
        for range_str in range_dict[assay]:
            low_str, high_str = range_str.split("@")
            try:
                low_float, high_float = float(low_str), float(high_str)
                tmp_cnt.append(range_dict[assay][range_str])
                tmp_range.append((low_float, high_float))
            except:
                continue
        # Find the most frequent normal range for each assay
        if len(tmp_cnt) > 0:
            tmp_cnt = np.array(tmp_cnt)
            max_idx = tmp_cnt.argmax()
            freq_range_dict[assay] = tmp_range[max_idx]
    return freq_range_dict

def load_assay_map(file):
    assay_map = {}
    with open(file, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            l_items = line.strip().split("@")
            assay_map[l_items[0]] = l_items[1]
    return assay_map

# Load the mapping information of the assays
assay_map = load_assay_map(osp.join(data_dir, assay_map_file))
assay_map_train = load_assay_map(osp.join(train_data_dir, assay_map_file))

# Read the information of normal range of each assay
freq_range_dict = load_range_dict(osp.join(data_dir, normal_range_file))
freq_range_dict_train = load_range_dict(osp.join(train_data_dir, normal_range_file))

# Make the available normal ranges consistent
for key_train in freq_range_dict_train:
    uni_name = assay_map_train[key_train]
    keys_dataset = [tmp_key for tmp_key in assay_map if assay_map[tmp_key] == uni_name]
    for key_dataset in keys_dataset:
        if key_dataset not in freq_range_dict:
            freq_range_dict[key_dataset] = freq_range_dict_train[key_train]

del_keys = []
for key in freq_range_dict:
    uni_name = assay_map[key]
    keys_train = [tmp_key for tmp_key in assay_map_train if assay_map_train[tmp_key] == uni_name]
    for key_train in keys_train:
        if key_train not in freq_range_dict_train:
            del_keys.append(key)
for del_key in del_keys:
    del freq_range_dict[del_key]

# Read the label information
with open(osp.join(data_dir, 'label_dict'), "r") as fin:
    f_str = fin.read()
    label_dict = json.loads(f_str)

print("Selected assays")
for item in sel_assay:
    print(item)

patient_feat_dict = defaultdict(lambda: defaultdict(lambda: dict()))
# Load the measurement history of each patient within the observation window
print("Processing lines:")
with open(osp.join(data_dir, filename), "r", encoding=encoding) as fin:
    lines = fin.readlines()
    info = lines[0].strip().split(sep)
    for (l_idx, l_str) in enumerate(lines[1:]):
        line_results = split_line_str(l_str, dataset)

        pid, timestr, assay, value = line_results['pid'], line_results['time_str'], line_results['task_assay'], line_results['value_str']

        assay_ori = deepcopy(assay)
        if assay in assay_map:
            assay = assay_map[assay]
        if assay in sel_assay:
            try:
                val = float(value.replace('"', '').replace("'","").replace(' ','').replace('>', '').replace('<', ''))
                # Normalize with normal range
                if assay_ori in freq_range_dict:
                    try:
                        low_str, high_str = line_results['low_str'], line_results['high_str']
                        low_float, high_float = float(low_str), float(high_str)
                    except:
                        low_float, high_float = freq_range_dict[assay_ori][0], freq_range_dict[assay_ori][1]
                    val = (val - low_float) / (high_float - low_float)
                else:
                    # print("Alert: range not found", assay, assay_ori)
                    if "Pct" in assay:
                        # print(val)
                        val = val / 100
                    # non_found_set.add(assay)
                patient_feat_dict[pid][assay][timestr] = val
            except:
                continue
        if l_idx % 100000 == 0:
            print("{}/{}".format(l_idx, len(lines)))

# Extract features with the measurement history
sel_assay.sort()

train_data, test_data = [], []
train_y, test_y = [], []
train_pid, test_id = [], []
feat_name_map = {}

for assay in sel_assay:
    for type in ['latest', 'mean', 'min', 'max', 'rate']:
        feat_name_map[assay + "_" + type] = len(feat_name_map)

# Generate the names of the features
for pid in label_dict:
    tmp_data = np.zeros(len(sel_assay) * 5)
    for idx in range(len(tmp_data)):
        tmp_data[idx] = np.nan
    if pid in patient_feat_dict:
        tmp_feature_dict = patient_feat_dict[pid]
        label_timestr = label_dict[pid]['time_str']
        label_struct_time = time.strptime(label_timestr, "%Y-%m-%d %H:%M:%S")
        label_time = time.mktime(label_struct_time)

        # For each assay, calculate the statistics in the observation window
        for assay in tmp_feature_dict:
            timestr_assay = [tmp_key for tmp_key in tmp_feature_dict[assay]]
            timestr_assay.sort()
            assay_array, delta_time_array = [], []
            for tmp_timestr in timestr_assay:
                tmp_struct_time = time.strptime(tmp_timestr, "%Y-%m-%d %H:%M:%S")
                tmp_time = time.mktime(tmp_struct_time)
                delta_time = label_time - tmp_time
                if delta_time > 0 and delta_time <= 365 * 24 * 60 * 60:
                    assay_array.append(tmp_feature_dict[assay][tmp_timestr])
                    delta_time_array.append(delta_time / (24 * 60 * 60))

            assay_array, delta_time_array = np.array(assay_array), np.array(delta_time_array)
            if len(assay_array) > 0:
                for type in ['latest', 'mean', 'min', 'max', 'rate']:
                    if type == 'latest':
                        tmp_data[feat_name_map[assay + "_" + type]] = assay_array[-1]
                    elif type == 'mean':
                        tmp_data[feat_name_map[assay + "_" + type]] = assay_array.mean()
                    elif type == 'min':
                        tmp_data[feat_name_map[assay + "_" + type]] = assay_array.min()
                    elif type == 'max':
                        tmp_data[feat_name_map[assay + "_" + type]] = assay_array.max()
                    elif type == 'rate':
                        if len(assay_array) > 1:
                            lm = LinearRegression()
                            lm.fit(delta_time_array[:, None], assay_array[:, None])
                            tmp_data[feat_name_map[assay + "_" + type]] = lm.coef_

    if label_dict[pid]['if_test']:
        test_data.append(tmp_data)
        test_y.append(label_dict[pid]['label'])
        test_id.append(pid)
    else:
        train_data.append(tmp_data)
        train_y.append(label_dict[pid]['label'])
        train_pid.append(pid)

print("Feature list:")
feat_name_list = ['' for _ in range(len(feat_name_map))]
for key in feat_name_map:
    feat_name_list[feat_name_map[key]] = key
    print("{},{}".format(feat_name_map[key], key))
# print(non_found_set)

train_data, train_y, train_pid, test_data, test_y, test_id = np.array(train_data), np.array(train_y), np.array(train_pid), np.array(test_data), np.array(test_y), np.array(test_id)
# Save the data
np.savez(osp.join(data_dir, "dataset_preprocessed_train_w_{}".format(train_feature_set)), train_data = train_data, train_y = train_y, train_pid = train_pid, test_data = test_data, test_y = test_y, test_id = test_id,feat_name_list = feat_name_list, feat_name_map = feat_name_map)

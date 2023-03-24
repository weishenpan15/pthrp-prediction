import argparse
import numpy as np
import json
import os.path as osp
import time
import re
from collections import defaultdict
from utils import convert_time_str, split_line_str
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',   default="wcm", type=str, help='Dataset')
parser.add_argument('--data_dir',   default="wcm_data", type=str, help='Path of Training Data')
args = parser.parse_args()

dataset = args.dataset
data_dir = args.data_dir
if dataset == "wusm":
    assay_map_file = "assay_map"
    filename = "AACC_Kaggle_2022_PTHrP_input_data.txt"
    encoding = None
    sep = "\t"
elif dataset == "wcm":
    assay_map_file = "assay_map"
    filename = "1_q_res.csv"
    sep = ","
    encoding = None
elif dataset == "mda":
    assay_map_file = "assay_map"
    filename = "full output file-rev3.csv"
    sep = ","
    encoding = None
else:
    raise NotImplementedError

# Read the label and time of the PTHrP tests
if dataset == "wusm":
    label_dict = {}
    fins = ["AACC_Kaggle_2022_PTHrP_train_results.txt", "AACC_Kaggle_2022_PTHrP_test_results_masked.txt"]
    for fin in fins:
        with open(osp.join(data_dir, fin), "r") as fin_dir:
            lins = fin_dir.readlines()
            info = lins[0].strip().split("\t")
            for (l_idx, lin) in enumerate(lins[1:]):
                l_items = lin.strip().split("\t")
                pid, timestr = l_items[0], l_items[13].replace('"', '')
                med_service, encounter_type = l_items[10].replace('"', ''), l_items[11].replace('"', '')
                if l_items[6] == '"Normal"':
                    label_dict[pid] = {'label': 0, 'time_str': timestr}
                elif l_items[6] == '"Abnormal"':
                    label_dict[pid] = {'label': 1, 'time_str': timestr}
                else:
                    label_dict[pid] = {'label': None, 'time_str': timestr}
elif dataset == "wcm" or dataset == "mda":
    if dataset == "wcm":
        assay_set = set()
        reg = re.compile(r'"(.*?)"')
        label_dict_wtime = {}
        with open(osp.join(data_dir, "PTHRP File 7-27-22 pk.csv"), "r", encoding='utf-8') as fin:
            lines = fin.readlines()
            info = lines[0].strip().split(",")
            for in_line in lines[1:]:
                new_in_line = in_line.strip()
                line_info = new_in_line.split(",")

                if len(line_info) != 24:
                    re_result = re.findall(reg, new_in_line)
                    if len(re_result) > 0:
                        for tmp_result in re_result:
                            new_in_line = new_in_line.replace(tmp_result, tmp_result.replace(',', ';'))
                        new_in_line = new_in_line.replace('"', '')
                    line_info = new_in_line.split(",")

                if len(line_info) == 24:
                    p_id = line_info[12]
                    time_str, med_service = line_info[0], line_info[22]
                    value_str, label_str = line_info[3], line_info[4]

                    if p_id not in label_dict_wtime:
                        label_dict_wtime[p_id] = {}

                    new_time_str = convert_time_str(time_str)

                    if label_str == 'H':
                        label_dict_wtime[p_id][new_time_str] = {'label': 1, 'time_str': new_time_str}
                    else:
                        label_dict_wtime[p_id][new_time_str] = {'label': 0, 'time_str': new_time_str}
    else:
        pid_dict = {}
        label_dict_wtime = {}
        reg = re.compile(r'"(.*?)"')
        name_set = set()
        with open(osp.join(data_dir, "full output file-rev3.csv"), "r", encoding='utf-8') as fin:
            lines = fin.readlines()
            info = lines[0].strip().split(",")
            for in_line in lines[1:]:
                new_in_line = in_line.strip()
                line_info = new_in_line.split(",")

                if len(line_info) != 27:
                    re_result = re.findall(reg, new_in_line)
                    if len(re_result) > 0:
                        for tmp_result in re_result:
                            new_in_line = new_in_line.replace(tmp_result, tmp_result.replace(',', ';'))
                        new_in_line = new_in_line.replace('"', '')
                    line_info = new_in_line.split(",")

                if len(line_info) == 27 and line_info[11] == 'PTH Peptide-Mayo':
                    p_id = line_info[0]
                    time_str = line_info[17] + " " + line_info[18]
                    value_str, label_str = line_info[19], line_info[22]

                    if p_id not in label_dict_wtime:
                        label_dict_wtime[p_id] = {}

                    new_time_str = convert_time_str(time_str)
                    if len(label_str) > 0:
                        label_dict_wtime[p_id][new_time_str] = {'label': 1, 'time_str': new_time_str}
                    else:
                        label_dict_wtime[p_id][new_time_str] = {'label': 0, 'time_str': new_time_str}
    label_dict = {}
    for p_id in label_dict_wtime:
        time_str_list, time_list = [], []
        for tmp_time_str in label_dict_wtime[p_id].keys():
            label_struct_time = time.strptime(tmp_time_str, "%Y-%m-%d %H:%M:%S")
            label_time = time.mktime(label_struct_time)
            time_str_list.append(tmp_time_str)
            time_list.append(label_time)

        if len(time_str_list) > 1:
            argmin_idx = np.argmin(np.array(time_list))
            label_dict[p_id] = label_dict_wtime[p_id][time_str_list[argmin_idx]]
        else:
            label_dict[p_id] = label_dict_wtime[p_id][time_str_list[0]]
else:
    raise NotImplementedError

# Split the data samples into train and test sets
if dataset == "wusm":
    test_label_dict = {}
    with open(osp.join(data_dir, "submission_key.csv"), "r") as fin:
        inlines = fin.readlines()
        for inline in inlines[1:]:
            line_item = inline.strip().split(",")
            test_label_dict[line_item[0]] = int(line_item[1])
    for pid in label_dict:
        if label_dict[pid]['label'] is None:
            label_dict[pid]['label'] = test_label_dict[pid]
            label_dict[pid]['if_test'] = True
        else:
            label_dict[pid]['if_test'] = False
else:
    all_pids, all_labels = [], []
    for pid in label_dict:
        all_pids.append(pid)
        all_labels.append(label_dict[pid]['label'])
    all_pids, all_labels = np.array(all_pids), np.array(all_labels)
    train_pids, test_pids, train_labels, test_labels = train_test_split(all_pids, all_labels,test_size=0.2, stratify = all_labels, random_state = 0, shuffle=True)
    for pid in label_dict:
        if pid in test_pids:
            label_dict[pid]['if_test'] = True
        else:
            label_dict[pid]['if_test'] = False

# Write the train-test split
if dataset == "wusm":
    label_dict_str = json.dumps(label_dict)
    with open(osp.join(data_dir, "label_dict"), "w") as fin:
        fin.write(label_dict_str)
else:
    label_dict_str = json.dumps(label_dict)
    with open(osp.join(data_dir, "label_dict"), "w") as fin:
        fin.write(label_dict_str)

# Load the mapping information of the assays
assay_map = {}
with open(osp.join(data_dir, assay_map_file), "r") as fin:
    lines = fin.readlines()
    for line in lines:
        l_items = line.strip().split("@")
        assay_map[l_items[0]] = l_items[1]

# Count the normal ranges for the assays and save
normal_region_dict = defaultdict(lambda: defaultdict(lambda: 0))
print("Processing lines:")
with open(osp.join(data_dir, filename), "r", encoding=encoding) as fin:
    lines = fin.readlines()
    info = lines[0].strip().split(sep)
    for (l_idx, l_str) in enumerate(lines[1:]):
        line_results = split_line_str(l_str, dataset)
        pid, event_timestr, task_assay, high_str, low_str = line_results['pid'], line_results['time_str'], line_results['task_assay'], line_results['high_str'], line_results['low_str']
        if task_assay in assay_map:
            try:
                event_struct_time = time.strptime(event_timestr, "%Y-%m-%d %H:%M:%S")
                event_time = time.mktime(event_struct_time)

                label_timestr = label_dict[pid]['time_str']
                event_struct_time = time.strptime(label_timestr, "%Y-%m-%d %H:%M:%S")
                label_time = time.mktime(event_struct_time)
            except:
                continue

            delta_time = label_time - event_time
            if delta_time > 0 and delta_time <= 365 * 24 * 60 * 60:
                region_str = low_str + "@" + high_str
                normal_region_dict[task_assay][region_str] += 1

        if l_idx % 100000 == 0:
            print("{}/{}".format(l_idx, len(lines)))

        normal_region_json_str = json.dumps(normal_region_dict)
        with open(osp.join(data_dir, "normal_region"), "w") as fin:
            fin.write(normal_region_json_str)

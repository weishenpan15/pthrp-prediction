from scipy import interpolate
import numpy as np
import re

def fit_curve(curve_fpr, curve_tpr):
    xnew=np.linspace(0,1,101)
    kind = 'linear'
    f=interpolate.interp1d(curve_fpr,curve_tpr,kind=kind)
    ynew=f(xnew)

    return xnew, ynew

def delete_extra_zero(str):
    if "." in str:
        str = str.rstrip('0')
        if str.endswith('.'):
            return str.rstrip('.')
        else:
            return str
    else:
        return str

def split_line_str(line_str , dataset):
    if dataset == "wusm":
        line_info = line_str.strip().split("\t")
        results = {'pid': line_info[0], 'time_str': line_info[12].replace('"', ''), 'task_assay': line_info[2].replace('"', ''), 'value_str':line_info[3].replace('"', ''),
                   'low_str': delete_extra_zero(line_info[5].replace('"', '')), 'high_str': delete_extra_zero(line_info[6].replace('"', ''))}
    elif dataset == "wcm":
        reg = re.compile(r"""["'](.*?)["']""")
        new_in_line = line_str.strip()
        line_info = new_in_line.split(",")
        if len(line_info) != 13:
            re_result = re.findall(reg, new_in_line)
            if len(re_result) > 0:
                for tmp_result in re_result:
                    new_in_line = new_in_line.replace(tmp_result, tmp_result.replace(',', ';'))
                new_in_line = new_in_line.replace('"', '')
            line_info = new_in_line.split(",")
        results = {'pid': line_info[7], 'time_str': convert_time_str(line_info[6]), 'task_assay': line_info[12], 'value_str': line_info[11], "high_str": line_info[0], "low_str": line_info[1]}
    elif dataset == "mda":
        reg = re.compile(r'"(.*?)"')
        new_in_line = line_str.strip()
        line_info = new_in_line.split(",")
        if len(line_info) != 27:
            re_result = re.findall(reg, new_in_line)
            if len(re_result) > 0:
                for tmp_result in re_result:
                    new_in_line = new_in_line.replace(tmp_result, tmp_result.replace(',', ';'))
                new_in_line = new_in_line.replace('"', '')
            line_info = new_in_line.split(",")
        if len(line_info) == 28:
            print(len(line_info))

        high_str, low_str = "", ""
        if len(line_info[21]) > 0:
            reg_normal_leq = re.compile(r'(?<=Normal: <=)\d+\.?\d*')
            reg_normal_leq_result = re.findall(reg_normal_leq, new_in_line)

            reg_normal_l = re.compile(r'(?<=Normal: <)\d+\.?\d*')
            reg_normal_l_result = re.findall(reg_normal_l, new_in_line)
            if len(reg_normal_l_result) > 0 or len(reg_normal_leq_result) > 0:
                if len(reg_normal_leq_result) > 0:
                    if abs(float(reg_normal_leq_result[0])) > 10e-10:
                        high_str = reg_normal_leq_result[0]
                        low_str = "0.0"
                else:
                    if abs(float(reg_normal_l_result[0])) > 10e-10:
                        high_str = reg_normal_l_result[0]
                        low_str = "0.0"
            else:
                reg_high = re.compile(r'(?<=High: )\d+\.?\d*')
                reg_high_result = re.findall(reg_high, new_in_line)
                reg_low = re.compile(r'(?<=Low: )\d+\.?\d*')
                reg_low_result = re.findall(reg_low, new_in_line)
                if len(reg_high_result) > 0 and len(reg_low_result) > 0:
                    high_str = reg_high_result[0]
                    low_str = reg_low_result[0]


        results = {'pid': line_info[0], 'time_str': convert_time_str(line_info[17] + " " + line_info[18]), 'task_assay': line_info[11],
                   'value_str': line_info[19], 'high_str': high_str, 'low_str': low_str}
    else:
        raise NotImplementedError

    return results

def convert_time_str(time_str):
    time_info_0 = time_str.split(" ")
    time_info_1 = time_info_0[0].split("/")
    time_info_2 = time_info_0[1].split(":")
    if len(time_info_1[2]) < 4:
        time_info_1[2] = "20" + time_info_1[2]
    if len(time_info_1[0]) < 2:
        time_info_1[0] = "0" + time_info_1[0]
    if len(time_info_1[1]) < 2:
        time_info_1[1] = "0" + time_info_1[1]

    if len(time_info_2[0]) < 2:
        time_info_2[0] = "0" + time_info_2[0]
    if len(time_info_2[1]) < 2:
        time_info_2[1] = "0" + time_info_2[1]
    time_info_2.append("00")
    return "{}-{}-{} {}:{}:{}".format(time_info_1[2], time_info_1[0], time_info_1[1], time_info_2[0], time_info_2[1], time_info_2[2])

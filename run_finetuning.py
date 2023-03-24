import argparse
import pickle
import os.path as osp
from copy import deepcopy

import random
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb

from utils import *

np.random.seed(0)
random.seed(0)


def finetune_evaluate_model(old_model, tr_data, tr_y, val_data, val_y, params):
    kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    epoch_list, auroc_list = [], []
    for epoch in range(0, 1000, 50):
        epoch_list.append(epoch)
        auroc_list.append([])

    for tr_idx, val_idx in kf.split(tr_data, tr_y):
        sub_tr_data, sub_val_data = tr_data[tr_idx], tr_data[val_idx]
        sub_tr_y, sub_val_y = tr_y[tr_idx], tr_y[val_idx]
        for n_idx, n_estimater in enumerate(range(0, 1000, 50)):
            params['n_estimators'] = n_estimater
            model = xgb.XGBClassifier(**params)
            model.fit(sub_tr_data, sub_tr_y, xgb_model=old_model)
            pred_score_val_new = model.predict_proba(sub_val_data)[:, 1]
            score_new = roc_auc_score(sub_val_y, pred_score_val_new)
            auroc_list[n_idx].append(score_new)

    aver_auroc_list = []
    for tmp_list in auroc_list:
        aver_auroc_list.append(np.array(tmp_list).mean())

    best_idx = np.argmax(aver_auroc_list)
    best_epoch = epoch_list[best_idx]
    # print("Best Epoch: ", best_epoch)

    params['n_estimators'] = best_epoch
    model = xgb.XGBClassifier(**params)
    model.fit(tr_data, tr_y, xgb_model=old_model)
    result, curve_result = evaluate_model(model, val_data, val_y)

    return result, curve_result, model


def preprocess_data(train_data, test_data, train_y, feat_name_list):
    # Select the features by FDR test
    new_feat_list, sel_idx = [], []
    idx_list, p_list = [], []
    for f_idx in range(train_data.shape[1]):
        feat = train_data[:, f_idx]
        s_samples = np.isnan(feat)
        s_feat = feat[1 - s_samples > 0]
        s_y = train_y[1 - s_samples > 0]
        if s_feat.std() < 1e-6:
            continue
        p_feat, n_feat = s_feat[s_y == 1], s_feat[s_y == 0]
        if len(p_feat) == 0 or len(n_feat) == 0:
            continue
        U1, p = mannwhitneyu(p_feat, n_feat)
        idx_list.append(f_idx)
        p_list.append(p)

    p_list = np.array(p_list)
    sel_result, p_list_cor = fdrcorrection(p_list, alpha=0.05)

    for (idx, p_result) in enumerate(sel_result):
        if sel_result[idx]:
            sel_idx.append(idx_list[idx])
            new_feat_list.append(feat_name_list[idx_list[idx]])
    feat_name_list = deepcopy(new_feat_list)
    print("Feature size after feature selection by statistic testing:", len(sel_idx))
    train_data = train_data[:, np.array(sel_idx)]
    test_data = test_data[:, np.array(sel_idx)]

    # Impute the missing values
    imputer = SimpleImputer(strategy = 'median')
    imputer.fit(train_data)
    train_data = imputer.fit_transform(train_data)
    test_data = imputer.transform(test_data)

    # Z-score normalization
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    return train_data, test_data, feat_name_list


def preprocess_data_finetune(train_data, cite_train_data, test_data, train_y, feat_name_list):
    new_feat_list, sel_idx = [], []
    idx_list, p_list = [], []
    for f_idx in range(train_data.shape[1]):
        feat = train_data[:, f_idx]
        s_samples = np.isnan(feat)
        s_feat = feat[1 - s_samples > 0]
        s_y = train_y[1 - s_samples > 0]
        if s_feat.std() < 1e-6:
            # print("{},{},{}".format(feat_name_list[f_idx], 0, 1))
            continue
        p_feat, n_feat = s_feat[s_y == 1], s_feat[s_y == 0]
        if len(p_feat) == 0 or len(n_feat) == 0:
            # print("{},{},{}".format(feat_name_list[f_idx], 0, -1))
            continue
        U1, p = mannwhitneyu(p_feat, n_feat)
        # print("{},{},{}".format(feat_name_list[f_idx].replace(",", ";"), U1, p))

        idx_list.append(f_idx)
        p_list.append(p)

    p_list = np.array(p_list)
    sel_result, p_list_cor = fdrcorrection(p_list, alpha=0.05)

    for (idx, p_result) in enumerate(sel_result):
        if sel_result[idx]:
            sel_idx.append(idx_list[idx])
            new_feat_list.append(feat_name_list[idx_list[idx]])

    feat_name_list = deepcopy(new_feat_list)
    print("Current feature size after feature selection by statistic testing:", len(sel_idx))
    train_data = train_data[:, np.array(sel_idx)]
    cite_train_data = cite_train_data[:, np.array(sel_idx)]
    test_data = test_data[:, np.array(sel_idx)]

    # Impute and normalize data
    imputer= SimpleImputer(strategy='median')
    imputer.fit(train_data)
    train_data = imputer.fit_transform(train_data)
    for f_idx in range(test_data.shape[1]):
        if np.isnan(cite_train_data[:, f_idx]).mean() < 1.0:
            tmp_imputer = SimpleImputer(strategy='median')
            tmp_val_mat = cite_train_data[:, f_idx][:, np.newaxis]
            tmp_test_mat = test_data[:, f_idx][:, np.newaxis]
            tmp_imputer.fit(tmp_val_mat)
            tmp_val_mat = tmp_imputer.transform(tmp_val_mat)
            tmp_test_mat = tmp_imputer.transform(tmp_test_mat)
            cite_train_data[:, f_idx] = tmp_val_mat[:, 0]
            test_data[:, f_idx] = tmp_test_mat[:, 0]
        else:
            cite_train_data[:, f_idx] = np.median(train_data[:, f_idx])
            test_data[:, f_idx] = np.median(train_data[:, f_idx])

    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    cite_train_data = scaler.transform(cite_train_data)
    test_data = scaler.transform(test_data)

    return train_data, cite_train_data, test_data, feat_name_list


def evaluate_model(model, val_data, val_y):
    pred_score_xgb_val = model.predict_proba(val_data)[:, 1]
    pred_score_val = pred_score_xgb_val

    curve_fpr, curve_tpr, curve_thresholds = roc_curve(val_y, pred_score_val, pos_label=1)

    # Determine the cut-off threshold under 0.9 sensitivity
    y_pred_prob = pred_score_val
    sort_index = np.argsort(y_pred_prob)[::-1]
    y_pre_prob_sorted, y_sorted = y_pred_prob[sort_index], val_y[sort_index]
    all_sens, all_thres = [], []
    for idx in range(len(y_pre_prob_sorted)):
        tmp_thre = y_pre_prob_sorted[idx]
        y_pre = np.zeros_like(val_y)
        y_pre[y_pre_prob_sorted > tmp_thre] = 1
        tn, fp, fn, tp = confusion_matrix(y_sorted, y_pre).ravel()
        sens = tp / (tp + fn)  # recall of the postive class
        all_sens.append(sens)

    all_sens = np.array(all_sens)
    thre_idx = np.argmin(np.abs(all_sens - 0.9))
    thre = y_pre_prob_sorted[thre_idx]

    pred_y_val = (pred_score_val > thre)

    precision = precision_score(val_y, pred_y_val)
    tnr = recall_score(1 - val_y, 1 - pred_y_val)
    auroc = roc_auc_score(val_y, pred_score_val)

    return {'auroc':auroc, 'precision under 0.9 sensitivity':precision, 'specificity under 0.9 sensitivity': tnr}, {'curve_fpr':curve_fpr, 'curve_tpr': curve_tpr, 'curve_thresholds': curve_thresholds}


def train_evaluate_model(tr_data, tr_y, val_data, val_y, params):
    model = xgb.XGBClassifier(**params)

    model.fit(tr_data, tr_y)
    result, curve_result = evaluate_model(model, val_data, val_y)

    return result, curve_result, model


parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir',   default="./wusm_data", type=str, help='path of training data')
parser.add_argument('--train_dataset',   default="wusm", type=str, help='name of training data')
parser.add_argument('--test_data_dir',   default="./wcm_data", type=str, help='path of testing data')
parser.add_argument('--test_dataset',   default="wcm", type=str, help='name of testing data')
parser.add_argument('--sample_size',   default=100, type=int, help='sample size for finetuning and re-training')
args = parser.parse_args()

train_file_name = "dataset_preprocessed_train_w_{}.npz".format(args.train_dataset)

# Read the optimal hyperparameters
with open("./hyperparameters/params-xgb-cv-auroc-{}.pickle".format(train_file_name.replace(".", "_")), 'rb') as f:
    result_dict = pickle.load(f)
    xgb_params0 = result_dict['best_para']
    print("Model parameters: ")
    print(result_dict['best_para'])
    xgb_params0['random_state'] = 0

# Load Data:
data_dict = np.load(osp.join(args.train_data_dir, train_file_name), allow_pickle=True)
train_data, train_y = data_dict['train_data'], data_dict['train_y']
feat_name_list = list(data_dict['feat_name_list'])
print("Feature size Before feature selection by statistic testing:", train_data.shape[1])

test_file_name = "dataset_preprocessed_train_w_{}.npz".format(args.train_dataset)
data_dict = np.load(osp.join(args.test_data_dir, test_file_name), allow_pickle=True)
cite_train_data, cite_train_y = data_dict['train_data'], data_dict['train_y']
test_data, test_y = data_dict['test_data'], data_dict['test_y']
feat_name_list_test = list(data_dict['feat_name_list'])

# Train model on training data and evaluate on test data
train_data0, test_data0, _ = preprocess_data(train_data, test_data, train_y, feat_name_list)
_, _, old_model = train_evaluate_model(train_data0, train_y, test_data0, test_y, xgb_params0)

cite_train_idx = [tmp_idx for tmp_idx in range(cite_train_data.shape[0])]
aver_result_retrain = {}
aver_result_finetune = {}
sub_size = args.sample_size
for s in range(10):
    print("Run:", s)
    sub_cite_train_data, _, sub_val_y, _ = train_test_split(cite_train_data, cite_train_y, test_size = 1 - (sub_size / cite_train_data.shape[0]), stratify=cite_train_y, random_state=s, shuffle=True)

    _, sub_cite_train_data1, test_data1, _ = preprocess_data_finetune(train_data, sub_cite_train_data,
                                                                                      test_data, train_y,
                                                                                      feat_name_list)
    # Since the samples are few, set all the split threshold to 1 and lower the learning rate
    xgb_params1 = deepcopy(xgb_params0)
    xgb_params1['subsample'] = 1
    xgb_params1['colsample_bytree'] = 1
    xgb_params1['min_child_weight'] = 1
    xgb_params1['learning_rate'] = xgb_params1['learning_rate'] / 10
    test_result, _, model = finetune_evaluate_model(old_model, sub_cite_train_data1, sub_val_y, test_data1, test_y, xgb_params1)
    for result_key in test_result:
        if result_key not in aver_result_finetune:
            aver_result_finetune[result_key] = []
        aver_result_finetune[result_key].append(test_result[result_key])

    for result_key in test_result:
        print("### Subset Finetune Test with Size {}: {},{:.3f}".format(sub_size, result_key, test_result[result_key]))

print("### Subset Finetune Test:", sub_size)
for result_key in aver_result_finetune:
    print("{},{:.3f} \pm {:.3f}".format(result_key, np.array(aver_result_finetune[result_key]).mean(), np.array(aver_result_finetune[result_key]).std()))





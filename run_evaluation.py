import argparse
import pickle
import os.path as osp
from copy import deepcopy

import random
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
import shap

from utils import *

np.random.seed(0)
random.seed(0)

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

def preprocess_data_with_cite_specific_data(train_data, cite_train_data, test_data, train_y, feat_name_list):
    # Select the features by FDR test based on training information
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

    cite_train_data = cite_train_data[:, np.array(sel_idx)]
    test_data = test_data[:, np.array(sel_idx)]

    imputer = SimpleImputer(strategy = 'median')
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
    scaler.fit(cite_train_data)
    cite_train_data = scaler.transform(cite_train_data)
    test_data = scaler.transform(test_data)


    return cite_train_data, test_data, feat_name_list


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
parser.add_argument('--cv_seed',   default=None, type=int, help='random seed for cross-validation')
parser.add_argument('--test_mode',   default=1, type=int, help='0 for directly transport, 1 for re-training', choices=[0, 1])
args = parser.parse_args()
cv_seed = args.cv_seed

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

# Train and evaluate model on the training data with stratified 5-fold cross-validation
aver_result = {}
k_cnt = 0
kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
for tr_idx, val_idx in kf.split(train_data, train_y):
    tr_data, tr_val_data = deepcopy(train_data[tr_idx]), deepcopy(train_data[val_idx])
    tr_y, tr_val_y = train_y[tr_idx], train_y[val_idx]
    tr_data, tr_val_data, feat_name_list_new = preprocess_data(tr_data, tr_val_data, tr_y, feat_name_list)
    result, curve_result, _ = train_evaluate_model(tr_data, tr_y, tr_val_data, tr_val_y, xgb_params0)
    print("CV Fold {}: ".format(k_cnt))
    for result_key in result:
        print("{},{:.3f}".format(result_key, result[result_key]))

    for result_key in result:
        if result_key not in aver_result:
            aver_result[result_key] = []
        aver_result[result_key].append(result[result_key])
    k_cnt += 1

print("CV Average:")
for result_key in aver_result:
    print("{},{:.3f} \pm {:.3f}".format(result_key, np.array(aver_result[result_key]).mean(), np.array(aver_result[result_key]).std()))

# Train model on training data and evaluate on test data
if args.test_mode == 0:
    train_data, test_data, feat_name_list = preprocess_data(train_data, test_data, train_y, feat_name_list)
    test_result, test_curve_result, model = train_evaluate_model(train_data, train_y, test_data, test_y, xgb_params0)
else:
    cite_train_data, test_data, feat_name_list = preprocess_data_with_cite_specific_data(train_data, cite_train_data, test_data, train_y, feat_name_list)
    test_result, test_curve_result, model = train_evaluate_model(cite_train_data, cite_train_y, test_data, test_y, xgb_params0)

print("Test:")
for result_key in test_result:
    print("{},{:.3f}".format(result_key, test_result[result_key]))

# Plot the AUROC curve
sns.set(style="darkgrid")
plt.figure(figsize=(6,6))
x, y = fit_curve(test_curve_result["curve_tpr"], test_curve_result["curve_fpr"])
plt.plot(y ,x, "-", color = "tab:blue", label = "(AUROC: {:.3f})".format(test_result['auroc']))
plt.plot(y[90],x[90],'o', color = "tab:blue")
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.legend(loc = 'lower right', prop={'size': 15})
plt.show()

# Calculate Shapley values
if args.test_mode == 0:
    explainer = shap.Explainer(model)
    shap_values = explainer(np.concatenate((train_data, test_data)))
    shap_values.feature_names = feat_name_list
    shap.summary_plot(shap_values, max_display=20)
else:
    explainer = shap.Explainer(model)
    shap_values = explainer(np.concatenate((cite_train_data, test_data)))
    shap_values.feature_names = feat_name_list
    shap.summary_plot(shap_values, max_display=20)



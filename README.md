# Pthrp Prediction across Multiple Clinical Centers

This is the public code of the submiited manuscript "Generalizability of a Machine Learning Model for Improving Utilization of Parathyroid Hormone-Related Peptide Testing Across Multiple Clinical Centers".
The following process shows the setting of training model on dataset from Washington University School of Medicine in St. Louis (WUSM) as shown in the main context of the manuscript. For other settings, please change the parameter accordingly.

## 0. Create Environment and Install Libs:

```
conda env create -f env.yaml
```

## 1. Preprocess the Data:
Preprocess the data to collect the time of PTHrP and normal ranges of the lab tests for the following feature extraction. 
Note: The data comes from Washington University School of Medicine in St. Louis (WUSM), Weill Cornell Medicine (WCM, New York) and University of Texas M.D. Anderson Cancer Center (MDA, Houston), which can not be public released. 
The data of WUSM is from the AACC-2022 Competition: https://www.kaggle.com/competitions/aacc-2022-predicting-pthrp-results. It should be applied and used according to the data policy of the competition.
The data files should be put into the directories accordingly. And the code to parse the information may need to be re-wrote according to the format of the data files. It is assumed that the data files of each dataset is saved in ./wusm_data, ./wcm_data and ./mda_data respectively.
```
python preprocess_data.py --dataset wusm --data_dir ./wusm_data
python preprocess_data.py --dataset wcm --data_dir ./wcm_data
python preprocess_data.py --dataset mda --data_dir ./mda_data
```

## 2. Extract the Features:
Extract the features for each dataset
Note: the training dataset must be indicated explicitly, because the lab tests used in each training datase are different according to our selection criterion. The mapping relation of the lab test names is obtained by consulting clinician and should be saved in the file "assay_map", in which each line indicate the name in one dataset and the unified name seperated by "@". 
```
python extract_features.py --dataset wusm --data_dir ./wusm_data --train_feature_set wusm --train_data_dir ./wusm_data
python extract_features.py --dataset wcm --data_dir ./wcm_data --train_feature_set wusm --train_data_dir ./wusm_data
python extract_features.py --dataset mda --data_dir ./mda_data --train_feature_set wusm --train_data_dir ./wusm_data
```

## 3. Run the Experiments:
Train model on WUSM and directly transport to in-cite test set, WCM and MDA
```
python run_evaluation.py --train_dataset wusm --train_data_dir ./wusm_data --test_dataset wusm --test_data_dir ./wusm_data --test_mode 0
python run_evaluation.py --train_dataset wusm --train_data_dir ./wusm_data --test_dataset wcm --test_data_dir ./wcm_data --test_mode 0
python run_evaluation.py --train_dataset wusm --train_data_dir ./wusm_data --test_dataset mda --test_data_dir ./mda_data --test_mode 0
```

Re-train models on WCM and MDA based on the features and hyperparameters on WUSM:
```
python run_evaluation.py --train_dataset wusm --train_data_dir ./wusm_data --test_dataset wcm --test_data_dir ./wcm_data --test_mode 1
python run_evaluation.py --train_dataset wusm --train_data_dir ./wusm_data --test_dataset mda --test_data_dir ./mda_data --test_mode 1
```
Note: for re-building the model, the process should be repeated from Step 1 and change the training dataset.

Train a model on WUSM and finetune it on WCM with a few samples (controlled by --sameple_size)
```
python run_finetuning.py --train_dataset wusm --train_data_dir ./wusm_data --test_dataset wcm --test_data_dir ./wcm_data --sample_size 100
```

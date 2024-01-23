# %%
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc, confusion_matrix, \
     RocCurveDisplay, precision_score, recall_score, average_precision_score, PrecisionRecallDisplay, precision_recall_curve
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel
from adrd.utils.misc import get_and_print_metrics_multitask
from adrd.utils.misc import get_metrics, print_metrics
import numpy as np
from tqdm import tqdm
import json
# from adrd.data import _conf
import adrd.utils.misc
import torch
import os
from icecream import ic

ic.disable()

from torchvision import transforms

import monai
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    HistogramNormalized,
    RandSpatialCropSamplesd,
    RandSpatialCropd,
    CenterSpatialCropd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    Resized,
)

# %%
radiologist_features = ['ID', 'cdr_CDRGLOB', 'his_NACCAGE', 'his_SEX', 'his_RACE', 'his_RACESEC', 'his_RACETER', 'his_EDUC', 'mri_zip', 'NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']
mri_fea = ['ID', 'cdr_CDRGLOB', 'mri_zip', 'NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']
# basedir="."
basedir=".."


fname = 'nacc_test_with_np_cli'
# fname = 'nacc_neuropath_test'
# fname = 'clinician_review_cases_test'
# fname = 'radiologist_review_cases_test'
# fname = 'adni_revised_labels'
# fname = 'adni_neuropath'
# fname = 'fhs_converted_6months_cleaned'
save_path = f'{basedir}/model_predictions_after_corr_stripped/'
dat_file = f'{basedir}/data/train_vld_test_split_updated/{fname}.csv'
cnf_file = f'{basedir}/dev/data/toml_files/default_conf_new.toml'
# ckpt_path = '/home/skowshik/ADRD_repo/pipeline_v1_main/adrd_tool/dev/ckpt/final/new_embeddings_current_best_model_correction_bs128.pt'
ckpt_path = '/data_1/skowshik/ckpts_backbone_swinunet/new_embeddings_current_best_model_correction.pt'
emb_path = '/data_1/dlteif/SwinUNETR_MRI_stripped_emb/'
# emb_path = '/data_1/dlteif/SwinUNETR_MRI_stripped_emb_ADNI/'
# emb_path = '/data_1/ahangar/FHS_MRI_Embeddings_NEW/'

dat_file = pd.read_csv(dat_file)
radiologist_features = [fea for fea in radiologist_features if fea in dat_file.columns]
if 'radiologist' in fname:
    dat_file = dat_file[radiologist_features]
print(dat_file)

# img_net="DenseNetEMB"
# img_mode=1
# mri_type='SEQ'

img_net="SwinUNETREMB"
img_mode=1
mri_type="ALL"

if 'fhs' in fname.lower():
    # dat_file = dat_file[dat_file['neuropath_avail'] == 1].reset_index(drop=True)
    print(len(dat_file))
    print(len(dat_file[~dat_file['ID'].isna()]))
    dat_file['ID'] = 'FHS_' + dat_file['ID']
    print(dat_file['ID'])

#%%
# Select only 3D MRIs
other_path = '/SeaExpCIFS/Raw_MRIs/ALL_nii'
other_3d_mris = set()
for cohort in os.listdir(other_path):
    print('cohort: ', cohort)
    for mri in tqdm(os.listdir(f'{other_path}/{cohort}')):
        if ('stanford' in cohort.lower()) or ('oasis' in cohort.lower()):
            other_3d_mris.add(mri)
            continue
            
        if mri.endswith('json'):
            continue
        
        json_name = mri.replace('.nii', '.json')
        json_file = f'{other_path}/{cohort}/{json_name}'
        if not os.path.exists(json_file):
            continue
        with open(json_file, 'r') as fi:
            data = json.load(fi)
            if 'MRAcquisitionType' not in data or data['MRAcquisitionType'] == '2D':
                continue
        other_3d_mris.add(mri)

#%%
# other_3d_mris = None

# %%

if not os.path.exists(save_path):
    os.makedirs(save_path)


# initialize datasets
seed = 0
print('Done.\nLoading testing dataset ...')
dat_tst = CSVDataset(dat_file=dat_file, cnf_file=cnf_file, mode=0, img_mode=img_mode, mri_type=mri_type, other_3d_mris=other_3d_mris, emb_path=emb_path, stripped=True)

print('Done.')

# %%
# load saved Transformer
device = 'cuda:2'
img_dict = {'img_net': 'SwinUNETREMB', 'img_size': 128, 'patch_size': 16, 'imgnet_ckpt': ckpt_path, 'imgnet_layers': 4, 'train_imgnet': False}
mdl = ADRDModel.from_ckpt(ckpt_path, device=device, img_dict=img_dict)
print("loaded")

#%%
def minmax_normalized(x, keys=["image"]):
    for key in keys:
        eps = torch.finfo(torch.float32).eps
        x[key] = torch.nn.functional.relu((x[key] - x[key].min()) / (x[key].max() - x[key].min() + eps))
    return x

flip_and_jitter = monai.transforms.Compose([
        monai.transforms.RandAxisFlipd(keys=["image"], prob=0.5),
        transforms.RandomApply(
            [
                monai.transforms.RandAdjustContrastd(keys=["image"], gamma=(-0.3,0.3)), # Random Gamma => randomly change contrast by raising the values to the power log_gamma 
                monai.transforms.RandBiasFieldd(keys=["image"]), # Random Bias Field artifact
                monai.transforms.RandGaussianNoised(keys=["image"]),

            ],
            p=0.4
        ),
    ])

# Custom transformation to filter problematic images
class FilterImages:
    def __init__(self, dat_type):
        # self.problematic_indices = []
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                CropForegroundd(keys=["image"], source_key="image"),
                monai.transforms.RandScaleCropd(keys=["image"], roi_scale=0.7, max_roi_scale=1, random_size=True, random_center=True),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=128),
                flip_and_jitter,
                monai.transforms.RandGaussianSmoothd(keys=["image"], prob=0.5),
                minmax_normalized,
            ]            
        )
        
        self.vld_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                CropForegroundd(keys=["image"], source_key="image"),
                # CenterSpatialCropd(keys=["image"], roi_size=(args.img_size,)*3),
                Resized(keys=["image"], spatial_size=(128*2,)*3),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=128),
                minmax_normalized,
            ]
        )
        
        if dat_type == 'trn':
            self.transforms = self.train_transforms
        else:
            self.transforms = self.vld_transforms

    def __call__(self, data):
        image_data = data["image"]
        try:
            # check = nib.load(image_data).get_fdata()
            # print(len(check.shape))
            # if len(check.shape) > 3:
            #     return None
            
            return self.transforms(data)
        except Exception as e:
            print(f"Error processing image: {image_data}{e}")
            return None
        
# tst_filter_transform = FilterImages(dat_type='tst')
tst_filter_transform = None
    
#%%

# loading testing predictions
scores, scores_proba, y_pred = mdl.predict(x=dat_tst.features, _batch_size=64, img_transform=tst_filter_transform)

#%%
def save_predictions(dat_tst, scores_proba, scores, save_path=None, filename=None, if_save=True):
    y_true = [{k:int(v) if v is not None else np.NaN for k,v in entry.items()} for entry in dat_tst.labels]
    mask = [{k:1 if v is not None else 0 for k,v in entry.items()} for entry in dat_tst.labels]

    y_true_ = {f'{k}_label': [smp[k] for smp in y_true] for k in y_true[0] if k in dat_file.columns}
    scores_proba_ = {f'{k}_prob': [round(smp[k], 3) if isinstance(y_true[i][k], int) else np.NaN for i, smp in enumerate(scores_proba)] for k in scores_proba[0] if k in dat_file.columns}
    scores_ = {f'{k}_logit': [round(smp[k], 3) if isinstance(y_true[i][k], int) else np.NaN for i, smp in enumerate(scores)] for k in scores[0] if k in dat_file.columns}
    ids = dat_file['ID']

    y_true_df = pd.DataFrame(y_true_)
    scores_df = pd.DataFrame(scores_)
    scores_proba_df = pd.DataFrame(scores_proba_)
    if 'cdr_CDRGLOB' in dat_file:
        cdr = dat_file['cdr_CDRGLOB']
        cdr_df = pd.DataFrame(cdr)
    id_df = pd.DataFrame(ids)
    if 'fhs' in fname:
        fhsid = ids = dat_file[['id', 'idtype', 'framid']]
        fhsid_df = pd.DataFrame(fhsid)
        if 'cdr_CDRGLOB' in dat_file:
            df = pd.concat([fhsid_df, id_df, y_true_df, scores_proba_df, cdr_df], axis=1)
        else:
            df = pd.concat([fhsid_df, id_df, y_true_df, scores_proba_df], axis=1)
    else:
        if 'cdr_CDRGLOB' in dat_file:
            df = pd.concat([id_df, y_true_df, scores_proba_df, cdr_df], axis=1)
        else:
            df = pd.concat([id_df, y_true_df, scores_proba_df], axis=1)
        
    if if_save:
        df.to_csv(save_path + filename, index=False)
    return df


save_predictions(dat_tst, scores_proba, scores, save_path, f'{fname}_stripped_prob_bs128.csv', if_save=True)

# %%
from matplotlib import rc, rcParams
rc('axes', linewidth=1)
rc('font', size=18)
plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.serif'] = ['Arial']

def read_csv(filename):
    return pd.read_csv(filename)

def get_classification_report(y_true, y_pred, features, output_dict=True):
    report = classification_report(
        y_true,
        y_pred,
        output_dict=output_dict,
        target_names=features
    )
    return report

def plot_classification_report(report, filepath, format):
    figure(figsize=(10, 8), dpi=100)
    cls_report_plot = sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
    # plt.show()
    cls_report_plot.figure.savefig(filepath, format=format, dpi=300, bbox_inches='tight')

# Confusion Matrix

def confusion_matrix_2_2(y_true, y_pred, labels=[0,1]):
    return confusion_matrix(y_true, y_pred, labels=labels)

def plot_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=10):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=True, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("CM for class - " + class_label)

# def confusion_matrix_3_3(y_true, y_pred):
#     return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

def multilabel_cm(y_true, y_pred):
    return multilabel_confusion_matrix(y_true, y_pred)

def plot_multilabel_cm(y_true, y_pred, features, filepath, format, class_names=[0,1]):
    cm = multilabel_cm(y_true, y_pred)
    fig, ax = plt.subplots(4, 4, figsize=(14, 14), dpi=100)

    for axes, cfs_matrix, label in zip(ax.flatten(), cm, features):
        plot_confusion_matrix(cfs_matrix, axes, label, class_names)

    fig.tight_layout()
    # plt.show()
    fig.figure.savefig(filepath, format=format, dpi=300, bbox_inches='tight')

# AUC ROC

def roc_auc_scores(y_true, y_pred, features):
    # n_classes = y_true.shape[1]

    tpr = dict()
    fpr = dict()
    auc_scores = dict()
    thresholds = dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], thresholds[i] = roc_curve(y_true=y_true[:, i], y_score=y_pred[:, i], pos_label=1, drop_intermediate=False)
    #     auc_scores[i] = auc(fpr[i], tpr[i])
        
    for i, fea in enumerate(features):
        fpr[fea], tpr[fea], thresholds[fea] = roc_curve(y_true=y_true[:, i], y_score=y_pred[:, i], pos_label=1, drop_intermediate=False)
        auc_scores[fea] = auc(fpr[fea], tpr[fea])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true.ravel(), y_pred.ravel()
    )
    auc_scores["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[fea] for fea in features]))
    mean_tpr = np.zeros_like(all_fpr)
    for i, fea in enumerate(features):
        mean_tpr += np.interp(all_fpr, fpr[fea], tpr[fea])
    mean_tpr /= len(features)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    auc_scores["macro"] = auc(fpr["macro"], tpr["macro"])

    # Compute weighted-average ROC curve and ROC area
    support = np.sum(y_true, axis=0)
    weights = support / np.sum(support)
    print(len(weights))
    weighted_tpr = np.zeros_like(all_fpr)
    for i, fea in enumerate(features):
        weighted_tpr += weights[i] * np.interp(all_fpr, fpr[fea], tpr[fea])
    fpr["weighted"] = all_fpr
    tpr["weighted"] = weighted_tpr
    auc_scores["weighted"] = auc(fpr["weighted"] , tpr["weighted"])  

    return fpr, tpr, auc_scores, thresholds

def generate_roc(y_true, y_pred, features, figname='Average_ROC_curves'):
    fpr, tpr, auc_scores, _ = roc_auc_scores(y_true=y_true, y_pred=y_pred, features=features)
    # n_classes = y_true.shape[1]
    
    lw = 2
    # text_size = 22
    # legend_properties = {'size': text_size}

    # # Individual ROC curves
    # plt.figure()
    
    # colors = ['darkorange', 'steelblue', 'green', "yellow", "red", "black", "brown", "pink", "cyan", "purple", "lime", "aqua", "gold", "skyblue"] # set the colors for each class
    # fpr_value = np.linspace(0, 1, 100)
    # for i, color in zip(range(n_classes), colors):
    #     interp_tpr = np.interp(fpr_value, fpr[i], tpr[i])
    #     plt.plot(fpr_value, interp_tpr, color=color, lw=lw/2, alpha=0.8,
    #             label='{0} (AUC = {1:0.2f})'.format(features[i], auc_scores[i]))

    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curves')
    # # plt.show()
    # plt.savefig(save_path + "ROC_curves.pdf", format='pdf', dpi=300, bbox_inches='tight')

    # Average ROC curves
    colors_ = [(30/255, 136/255, 229/255, 1.0), (255/255, 193/255, 7/255, 1.0), (216/255, 27/255, 96/255, 1.0)]
    
    sensitivity_comb = np.array(list(tpr['micro']) + list(tpr['macro']) + list(tpr['weighted']))
    specificity_comb = np.array(list(1 - fpr['micro']) + list(1 - fpr['macro']) + list(1 - fpr['weighted']))
    plt_style = np.array(['micro (AUC = {0:0.2f})'.format(auc_scores["micro"])] * len(tpr['micro']) 
                         + ['macro (AUC = {0:0.2f})'.format(auc_scores["macro"])] * len(tpr['macro']) 
                         + ['weighted (AUC = {0:0.2f})'.format(auc_scores["weighted"])] * len(tpr['weighted']))
    df_roc = pd.DataFrame({'Specificity': specificity_comb, 'Sensitivity': sensitivity_comb, 'AUC ROC': plt_style})
    # print(df)
    
    fig = plt.figure(figsize=(6, 6), dpi=300)
    sns.lineplot(data=df_roc, x='Specificity', y='Sensitivity', style='AUC ROC', hue='AUC ROC', palette=colors_, linewidth=lw)
    
    # plt.setp(plt.spines.values(), color='w')
    plt.axhline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axhline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.0, linestyle='-', color='k', lw=1, zorder=1)
    plt.axhline(0.0, linestyle='-', color='k', lw=1, zorder=1)

    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.title('')
    plt.legend(loc='lower right') #, prop=legend_properties)
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path + f"{figname}.pdf", format='pdf', dpi=300, bbox_inches='tight')

# P-R curve

def precision_recall(y_true, y_pred, features):
    # Compute the precision-recall curve and average precision for each class
    # n_classes = y_true.shape[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i, fea in enumerate(features):
        precision[fea], recall[fea], _ = precision_recall_curve(y_true[:, i],
                                                            y_pred[:, i])
        precision[fea], recall[fea] = precision[fea][::-1], recall[fea][::-1]
        average_precision[fea] = average_precision_score(y_true[:, i], y_pred[:, i])

    # Compute the micro-average precision-recall curve and average precision
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),
        y_pred.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_pred,
                                                        average="micro")

    # Compute the macro-average precision-recall curve and average precision
    mean_recall = np.unique(np.concatenate([recall[fea] for i, fea in enumerate(features)]))
    # mean_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(mean_recall)
    for i, fea in enumerate(features):
        mean_precision += np.interp(mean_recall, recall[fea], precision[fea])
    mean_precision /= len(features)
    recall["macro"] = mean_recall
    precision["macro"] = mean_precision

    average_precision["macro"] = average_precision_score(y_true, y_pred,
                                                        average="macro")

    # Compute the weighted-average precision-recall curve and average precision

    support = np.sum(y_true, axis=0)
    weights = support / np.sum(support)
    weighted_precision = np.zeros_like(mean_recall)
    for i, fea in enumerate(features):
        weighted_precision += weights[i] * np.interp(mean_recall, recall[fea], precision[fea])
    recall["weighted"] = mean_recall
    precision["weighted"] = weighted_precision
    average_precision["weighted"] = average_precision_score(y_true, y_pred,
                                                                average="weighted")

    return precision, recall, average_precision


def generate_pr(y_true, y_pred, features, figname='Average_PR_curves'):
    precision, recall, average_precision = precision_recall(y_true=y_true, y_pred=y_pred, features=features)
    # n_classes = y_true.shape[1]
    lw = 2
    # text_size = 22
    # legend_properties = {'size': text_size}

    # # Plot the precision-recall curves for all classes, micro-average, macro-average, and weighted-average
    # plt.figure()

    # colors = ['darkorange', 'steelblue', 'green', "yellow", "red", "black", "brown", "pink", "cyan", "purple", "lime", "aqua", "gold", "skyblue"] # set the colors for each class
    # mean_recall = np.linspace(0, 1, 100)
    # for i, color in zip(range(n_classes), colors):
    #     interp_precision = np.interp(mean_recall, recall[i], precision[i])
    #     plt.plot(mean_recall, interp_precision, color=color, lw=lw/2, alpha=0.8,
    #     label='{0} (AP = {1:0.2f})'.format(features[i], average_precision[i]))

    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall curve')
    # plt.legend(loc='lower right')
    # plt.savefig(save_path + "PR_curves.pdf", format='pdf', dpi=300, bbox_inches='tight')
    # # plt.show()
    
    colors_ = [(30/255, 136/255, 229/255, 1.0), (255/255, 193/255, 7/255, 1.0), (216/255, 27/255, 96/255, 1.0)]
    
    precision_comb = np.array(list(precision['micro']) + list(precision['macro']) + list(precision['weighted']))
    recall_comb = np.array(list(recall['micro']) + list(recall['macro']) + list(recall['weighted']))
    plt_style = np.array(['micro (AP = {0:0.2f})'.format(average_precision["micro"])] * len(precision['micro']) 
                         + ['macro (AP = {0:0.2f})'.format(average_precision["macro"])] * len(precision['macro']) 
                         + ['weighted (AP = {0:0.2f})'.format(average_precision["weighted"])] * len(precision['weighted']))
    df_pr = pd.DataFrame({'Recall': recall_comb, 'Precision': precision_comb, 'AUC PR': plt_style})
    # print(df)
    
    fig = plt.figure(figsize=(6, 6), dpi=300)
    sns.lineplot(data=df_pr, x='Recall', y='Precision', style='AUC PR', hue='AUC PR', palette=colors_, linewidth=lw)
    
    
    
    plt.axhline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axhline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.0, linestyle='-', color='k', lw=1, zorder=1)
    plt.axhline(0.0, linestyle='-', color='k', lw=1, zorder=1)


    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('')
    plt.legend(loc='lower right') #, prop=legend_properties)
    plt.tight_layout()
    plt.savefig(save_path + f"{figname}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    # plt.show()

def save_performance_report(met, filepath):
    figure(figsize=(24, 20), dpi=300)
    met_df = pd.DataFrame(met).transpose()
    report_plot = sns.heatmap(met_df, annot=True)

    # plt.show()
    report_plot.figure.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')


# %%
# Generate performance report
# list-of-dict to dict-of-list
y_true = [{k:int(v) if v is not None else 0 for k,v in entry.items()} for entry in dat_tst.labels]
mask = [{k:1 if v is not None else 0 for k,v in entry.items()} for entry in dat_tst.labels]

y_true_dict = {k: [smp[k] for smp in y_true] for k in y_true[0]}
y_pred_dict = {k: [smp[k] for smp in y_pred] for k in y_pred[0]}
scores_proba_dict = {k: [smp[k] for smp in scores_proba] for k in scores_proba[0]}
mask_dict = {k: [smp[k] for smp in mask] for k in mask[0]}

met = {}
for k in dat_tst.label_modalities:
    print('Performance metrics of {}'.format(k))
    metrics = get_metrics(np.array(y_true_dict[k]), np.array(y_pred_dict[k]), np.array(scores_proba_dict[k]), np.array(mask_dict[k]))
    print_metrics(metrics)

    met[k] = metrics
    met[k].pop('Confusion Matrix')

save_performance_report(met, save_path + fname + '.pdf')

#%%
# Generate classification report and multilabel confusion matrix

y_true_ = [{k:int(v) if v is not None else np.NaN for k,v in entry.items()} for entry in dat_tst.labels]
y_pred_ = [{k:int(smp[k]) for k in smp.keys() if (k in y_true_[i] and isinstance(y_true_[i][k], int))} for i, smp in enumerate(y_pred)]
scores_proba_ = [{k:smp[k] for k in smp.keys() if (k in y_true_[i] and isinstance(y_true_[i][k], int))} for i, smp in enumerate(scores_proba)]

y_true_ = np.array([list(y_true_[k].values()) for k in range(len(y_true_))])
y_pred_ = np.array([list(y_pred_[k].values()) for k in range(len(y_pred_))])
scores_proba_ = np.array([list(scores_proba_[k].values()) for k in range(len(scores_proba_))])

classification_report_ = get_classification_report(y_true_, y_pred_, list(dat_tst.label_modalities.keys()))
plot_classification_report(classification_report_, filepath=save_path + "classification_report.pdf", format='pdf')
plot_multilabel_cm(y_true_, y_pred_, list(dat_tst.label_modalities.keys()), filepath=save_path + "multilabel_confusion_matrix.pdf", format='pdf')


#%%
# Generate AUC-ROC curves and AUC-PR curves from saved predictions 
import pandas as pd
import numpy as np
labels = ['NC', 'MCI', 'DE'] 
basedir = '../model_predictions_after_corr_stripped'
nacc = pd.read_csv(f'{basedir}/nacc_test_with_np_cli_stripped_prob_bs128.csv')
adni = pd.read_csv(f'{basedir}/adni_revised_labels_stripped_prob_bs128.csv')
fhs = pd.read_csv(f'{basedir}/fhs_converted_6months_cleaned_stripped_prob_bs128.csv')

df = pd.concat([nacc, adni, fhs], axis=0).reset_index()

y_true_ =  np.array(df[[f'{lab}_label' for lab in labels]])
scores_proba_ = np.array(df[[f'{lab}_prob' for lab in labels]])
print(scores_proba_.shape)

generate_roc(y_true_, scores_proba_, labels, figname=f'fig2g')
generate_pr(y_true_, scores_proba_, labels, figname=f'fig2h')

labels_de =['AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']
y_true_ =  np.array(nacc[[f'{lab}_label' for lab in labels_de]])
scores_proba_ =  np.array(nacc[[f'{lab}_prob' for lab in labels_de]])
print(scores_proba_.shape)

generate_roc(y_true_, scores_proba_, labels_de, figname=f'fig4k')
generate_pr(y_true_, scores_proba_, labels_de, figname=f'fig4l')


# %%
# Generate predictions for different missingness patterns
from itertools import product
import pandas as pd
import numpy as np
basedir=".."
df = pd.read_csv(f'{basedir}/data/train_vld_test_split_updated/nacc_test_with_np_cli.csv')
groups = np.array(['updrs', 'npiq', 'gds', 'faq', 'bat', 'img_MRI'])
groups_col = np.array([f'mask_{grp}' for grp in groups])

# load saved Transformer
device = 'cuda:0'
img_dict = {'img_net': 'SwinUNETREMB', 'img_size': 128, 'patch_size': 16, 'imgnet_ckpt': ckpt_path, 'imgnet_layers': 4, 'train_imgnet': False}
mdl = ADRDModel.from_ckpt(ckpt_path, device=device, img_dict=img_dict)
print("loaded")

combinations = list(product([0, 1], repeat=len(groups)))
print(combinations)
print(len(df.columns))

#%%
combination_dfs = []
# Print the combinations
for i, combo in tqdm(enumerate(combinations)):
    fea_mask = groups[np.where(np.array(combo) == 1)]
    sub_df = df.drop([col for col in df.columns if any(col.startswith(grp) for grp in fea_mask)], axis=1)
    
    print(combo)
    print("columns: ", len(sub_df.columns))
    # break

    if 'img_MRI' in fea_mask:
        img_mode = -1
    else:
        img_mode = 1
        
    
    dat_tst = CSVDataset(dat_file=sub_df, cnf_file=cnf_file, mode=0, img_mode=img_mode, mri_type=mri_type, other_3d_mris=None, emb_path=emb_path, stripped=True)
    
    y_true = [{k:int(v) if v is not None else np.NaN for k,v in entry.items()} for entry in dat_tst.labels]
    mask = [{k:1 if v is not None else 0 for k,v in entry.items()} for entry in dat_tst.labels]
    
    scores, scores_proba, y_pred = mdl.predict(x=dat_tst.features, _batch_size=64, img_transform=None)
    preds = save_predictions(dat_tst, y_true, scores_proba, scores, if_save=False)
    combo_df = pd.concat([preds, pd.DataFrame([combo] * len(preds), columns=groups_col)], axis=1)
    # print(combo_df)
    # break
    combination_dfs.append(combo_df)
    # break

combined = pd.concat(combination_dfs, axis=0)
combined.to_csv(save_path + 'fig_2c_combined.csv', index=False)
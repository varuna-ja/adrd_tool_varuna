# %%
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc, confusion_matrix, RocCurveDisplay, precision_score, recall_score, average_precision_score, PrecisionRecallDisplay, precision_recall_curve
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# from adrd import data
# from data import CSVDataset
from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel, DynamicCalibratedClassifier, StaticCalibratedClassifier
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
# basedir="."
basedir="/home/skowshik/ADRD_repo/pipeline_v1_main/adrd_tool"
# fname = 'merged_train'
fname = 'nacc_test_with_np_cli'
# fname = 'adni_revised_labels'
# fname = 'nacc_neuropath_test'
# fname = 'adni_neuropath'
# fname = 'fhs_clinical'
# fname = 'fhs_neuropath'
# fname = 'clinician_review_cases_test'
# fname = 'radiologist_review_cases_test'
# save_path = '/home/skowshik/ADRD_repo/pipeline_v1_main/adrd_tool/radiologist_plot/'
save_path = f'{basedir}/model_predictions_new_emb/'
dat_file = f'{basedir}/data/training_cohorts/train_vld_test_split_updated/{fname}.csv'
dat_file = pd.read_csv(dat_file)
# dat_file = dat_file[~dat_file['bat_NACCMMSE'].isna()]
radiologist_features = [fea for fea in radiologist_features if fea in dat_file.columns]

# dat_file = dat_file[radiologist_features]
print(dat_file)
cnf_file = f'{basedir}/dev/data/toml_files/default_conf.toml'
ckpt_path = '/data_1/skowshik/ckpts_backbone_swinunet/new_embeddings_current_best_model.pt'
# ckpt_path = f'{basedir}/dev/ckpt/backbone_densenet/ckpt_using_densenet_emb_with_rankingloss_0005_lr5ene4_bs128_AUPR.pt'
# emb_path = '/data_1/skowshik/DenseNet_emb/'
# emb_path = '/data_1/dlteif/SwinUNETR_MRI_emb/' 
# emb_path = '/data_1/dlteif/SwinUNETR_MRI_emb_ADNI/'
# emb_path = '/data_1/dlteif/SwinUNETR_MRI_emb_FHS_Neuropath/'
# emb_path = '/data_1/ahangar/FHS_Clinical/SwinUNETR_MRI_emb_FHS/'
emb_path = '/data_1/dlteif/SwinUNETR_MRI_stripped_emb/'

# img_net="DenseNetEMB"
# img_mode=1
# mri_type='SEQ'

img_net="SwinUNETREMB"
img_mode=1
mri_type="ALL"

#%%
# other_path = '/projectnb/ivc-ml/dlteif/Raw_MRIs'
# other_path = '/SeaExpCIFS/Raw_MRIs/ALL_nii'
# other_3d_mris = set()
# for cohort in os.listdir(other_path):
#     for mri in tqdm(os.listdir(f'{other_path}/{cohort}')):
#         if ('stanford' in cohort.lower()) or ('oasis' in cohort.lower()) or ('fhs' in cohort.lower()):
#             other_3d_mris.add(mri)

#             continue
            
#         if mri.endswith('json'):
#             continue
        
#         json_name = mri.replace('.nii', '.json')
#         json_file = f'{other_path}/{cohort}/{json_name}'
#         if not os.path.exists(json_file):
#             continue
#         with open(json_file, 'r') as fi:
#             data = json.load(fi)
#             if 'MRAcquisitionType' not in data or data['MRAcquisitionType'] == '2D':
#                 continue
#         other_3d_mris.add(mri)
other_3d_mris = None     
def read_csv(filename):
    return pd.read_csv(filename)

#%%
# len(other_3d_mris)

# %%
# if __name__ == '__main__':
    
if not os.path.exists(save_path):
    os.makedirs(save_path)


# initialize datasets
seed = 0
print('Done.\nLoading testing dataset ...')
dat_tst = CSVDataset(dat_file=dat_file, cnf_file=cnf_file, mode=0, img_mode=img_mode, mri_type=mri_type, other_3d_mris=other_3d_mris, emb_path=emb_path, stripped=True)

print('Done.')

# print(set(dat_trn.ids).intersection(set(dat_tst.ids)))
# %%
# load saved Transformer
device = 'cuda:0'
img_dict = {'img_net': 'SwinUNETREMB', 'img_size': 128, 'patch_size': 16, 'imgnet_ckpt': ckpt_path, 'imgnet_layers': 4, 'train_imgnet': False}
mdl = ADRDModel.from_ckpt(ckpt_path, device=device, img_dict=img_dict)
print("loaded")

# #%%
# print('Replacing T1-weighted MRIs by embeddings ...')
# with torch.set_grad_enabled(False):
#     for smp in tqdm(dat_tst.features):
#         for src_k in smp:
#             if src_k.startswith('img_MRI'):
#                 img = torch.tensor(np.expand_dims(smp[src_k], axis=1), device=device)
#                 print(img.size())
#                 vec = mdl.net_.modules_emb_src.img_MRI_1(img)
#                 # mdl.net_.img_model(img)
#                 print(vec.shape)
#                 print(vec)
#                 break
#         # break
            

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

# %%
# test_transforms = None

# loading testing predictions
# scores = mdl.predict_logits(dat_tst.features, img_transform=test_transforms)
# scores_proba = mdl.predict_proba(dat_tst.features, _batch_size=64, img_transform=test_transforms)
scores, scores_proba, y_pred = mdl.predict(x=dat_tst.features, _batch_size=128, img_transform=tst_filter_transform)


# %%
def save_predictions(dat_tst, y_true, scores_proba, scores, save_path=None, filename=None, if_save=True):
    y_true = [{k:int(v) if v is not None else np.NaN for k,v in entry.items()} for entry in dat_tst.labels]
    mask = [{k:1 if v is not None else 0 for k,v in entry.items()} for entry in dat_tst.labels]

    y_true_ = {f'{k}_label': [smp[k] for smp in y_true] for k in y_true[0] if k in dat_file.columns}
    scores_proba_ = {f'{k}_prob': [round(smp[k], 3) if isinstance(y_true[i][k], int) else np.NaN for i, smp in enumerate(scores_proba)] for k in scores_proba[0] if k in dat_file.columns}
    scores_ = {f'{k}_logit': [round(smp[k], 3) if isinstance(y_true[i][k], int) else np.NaN for i, smp in enumerate(scores)] for k in scores[0] if k in dat_file.columns}
    cdr = dat_file['cdr_CDRGLOB']
    ids = dat_file['ID']

    y_true_df = pd.DataFrame(y_true_)
    scores_df = pd.DataFrame(scores_)
    scores_proba_df = pd.DataFrame(scores_proba_)
    cdr_df = pd.DataFrame(cdr)
    id_df = pd.DataFrame(ids)
    df = pd.concat([id_df, y_true_df, scores_proba_df, cdr_df], axis=1)
    if if_save:
        df.to_csv(save_path + filename, index=False)
    return df

y_true = [{k:int(v) if v is not None else np.NaN for k,v in entry.items()} for entry in dat_tst.labels]
mask = [{k:1 if v is not None else 0 for k,v in entry.items()} for entry in dat_tst.labels]
save_predictions(dat_tst, y_true, scores_proba, scores, save_path, fname + '_stripped_prob.csv', if_save=False)


#%%


# %%
from itertools import product
basedir="/home/skowshik/ADRD_repo/pipeline_v1_main/adrd_tool"
df = pd.read_csv(f'{basedir}/data/training_cohorts/train_vld_test_split_updated/nacc_test_with_np_cli.csv')
groups = np.array(['updrs', 'npiq', 'gds', 'faq', 'bat', 'img_MRI'])
groups_col = np.array([f'mask_{grp}' for grp in groups])

# load saved Transformer
device = 'cuda:0'
img_dict = {'img_net': 'SwinUNETREMB', 'img_size': 128, 'patch_size': 16, 'imgnet_ckpt': ckpt_path, 'imgnet_layers': 4, 'train_imgnet': False}
mdl = ADRDModel.from_ckpt(ckpt_path, device=device, img_dict=img_dict)
print("loaded")

combinations = list(product([0, 1], repeat=len(groups)))
print(len(combinations))
print(len(df.columns))
combination_dfs = []
# Print the combinations
for i, combo in enumerate(combinations):
    fea_mask = groups[np.where(np.array(combo) == 1)]
    sub_df = df.drop([col for col in df.columns if any(col.startswith(grp) for grp in fea_mask)], axis=1)
    
    print(combo)
    print(len(sub_df.columns))
    # break

    if 'img_MRI' in fea_mask:
        img_mode = -1
    else:
        img_mode = 1
        
    
    dat_tst = CSVDataset(dat_file=sub_df, cnf_file=cnf_file, mode=0, img_mode=img_mode, mri_type=mri_type, other_3d_mris=None, emb_path=emb_path)
    
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

# %%

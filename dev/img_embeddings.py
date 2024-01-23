#%%
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc, confusion_matrix, \
     RocCurveDisplay, precision_score, recall_score, average_precision_score, PrecisionRecallDisplay, precision_recall_curve
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from data.imaging_data import CSVDataset
from adrd.model import ImagingModel
import torch
from icecream import ic, install
install()
ic.configureOutput(includeContext=True)
ic.disable()
import argparse
from tqdm import tqdm
import json
import os
from torchvision import transforms
from adrd.utils.misc import get_and_print_metrics_multitask
from adrd.utils.misc import get_metrics, print_metrics
import adrd.utils.misc
import random

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


#%%
# define variables
prefix = '/home/skowshik/ADRD_repo/pipeline_v1_main/adrd_tool/data/training_cohorts/train_vld_test_split_updated'

# dat_files = [f'{prefix}/merged_train.csv', f'{prefix}/merged_vld.csv', f'{prefix}/nacc_test_with_np_cli.csv']
dat_files = [f'{prefix}/adni_revised_labels.csv']
# ckpt_path = '/data_1/skowshik/densenet_ckpt/bn_size_2/ckpt_densenet_other_3way_config2.pt'
ckpt_path = '/data_1/skowshik/densenet_ckpt/bn_size_4/ckpt_densenet_flair_3way_config2.pt'
emb_type = 'FLAIR'
save_path = '/data_1/skowshik/DenseNet_emb/'
img_backend='DenseNet'
img_size = 128
batch_size = 32

#%%
# load the MRIs
avail_cohorts = {}
with open('/home/skowshik/ADRD_repo/pipeline_v1/clinician_review/mri_3d.json') as json_data:
    mri_json = json.load(json_data)
mri_emb_dict = {}
for mag, seq in mri_json.items():
    for seq_name, mri_name in tqdm(seq.items()):
        if emb_type.lower() in ['t1', 't2', 'flair']:
            if emb_type.lower() not in seq_name.lower():
                continue
        else:
            if ('t1' in seq_name.lower()) or ("t2" in seq_name.lower()) or ("flair" in seq_name.lower()):
                continue
        for name, pairs in mri_name.items():
            for pair in pairs:
                # print(mri_pair)
                mri = pair['mri']
                if ('localizer' in mri.lower()) | ('LOC' in mri) | ('calibration' in mri.lower()) | ('gre_field_mapping' in mri.lower()):
                    continue
                zip_name = name[:-2] + '.zip'
                if zip_name in mri_emb_dict:
                    mri_emb_dict[zip_name].append(mri)
                else:
                    mri_emb_dict[zip_name] = [mri]
                    
                if 'NACC' in avail_cohorts:
                    avail_cohorts['NACC'] += 1
                else:
                    avail_cohorts['NACC'] = 1
                # except:
                #     continue



# %%
other_path = '/SeaExpCIFS/Raw_MRIs/ALL_nii'
cnt = 0
n = ''
for cohort in os.listdir(other_path):
    for mri in tqdm(os.listdir(f'{other_path}/{cohort}')):
            
        if mri.endswith('json'):
            continue
        
        json_name = mri.replace('.nii', '.json')
        json_file = f'{other_path}/{cohort}/{json_name}'
        if ('stanford' not in cohort.lower()) and ('oasis' not in cohort.lower()):
            if not os.path.exists(json_file):
                continue
            with open(json_file, 'r') as fi:
                data = json.load(fi)
                if 'MRAcquisitionType' not in data or data['MRAcquisitionType'] == '2D':
                    continue
                
        if ('localizer' in mri.lower()) | ('LOC' in mri) | ('calibration' in mri.lower()) | ('gre_field_mapping' in mri.lower()):
            continue
        
        if ("t1" in emb_type.lower()) or ("t2" in emb_type.lower()) or ("flair" in emb_type.lower()):
            # if emb_type.lower() == 't1' and  'stanford' in cohort.lower():
            #     pass
            if (emb_type.lower() not in mri.lower()):
                if emb_type.lower() == 't1':
                    if ('t1' not in mri.lower()) and ('mprage' not in mri.lower()) and ('bravo' not in mri.lower()) and ('spgr' not in mri.lower()) and ('fspgr' not in mri.lower()) and ('mp-rage' not in mri.lower()):
                        continue 
                else:
                    continue
            elif emb_type.lower() == 't2':
                if 'flair' in mri.lower():
                    continue
        else:
            if ('t1' in mri.lower()) or ('mprage' in mri.lower()) or ('bravo' in mri.lower()) or ('spgr' in mri.lower()) or ('fspgr' in mri.lower()) or ('mp-rage' in mri.lower()) or ("t2" in mri.lower()) or ("flair" in mri.lower()):
                continue
            
        if (mri.lower().startswith('adni')) or (mri.lower().startswith('nifd')) or (mri.lower().startswith('4rtni')):
            name = '_'.join(mri.split('_')[:4])
        elif (mri.lower().startswith('aibl')) or (mri.lower().startswith('sub')) or (mri.lower().startswith('ppmi')):
            name =  '_'.join(mri.split('_')[:2])
        elif mri.lower().startswith('stanford')  or 'stanford' in cohort.lower():
            # name = mri.split('.')[0]
            # mri = mri
            name = mri.split('.')[0].split('_')[0] + '_' + mri.split('.')[0].split('_')[2]
        else:
            continue
        
        if mri.lower().startswith('sub'):
            dict_name = 'OASIS'
        else:
            dict_name = name.split('_')[0]
            
        if dict_name in avail_cohorts:
            avail_cohorts[dict_name] += 1
        else:
            avail_cohorts[dict_name] = 1
        
        if name in mri_emb_dict:
            mri_emb_dict[name].append(f'{other_path}/{cohort}/{mri}')
        else:
            mri_emb_dict[name] = [f'{other_path}/{cohort}/{mri}']
        n = name if len(mri_emb_dict[name]) > cnt else n
        cnt = max(cnt, len(mri_emb_dict[name]))

  #%%
print(f"AVAILABLE {emb_type} MRI Cohorts: ", avail_cohorts)
if 'NACC' not in avail_cohorts:
    print('NACC MRIs not available')
    
#%%
# for k,v in mri_emb_dict.items():
#     if 'stanford' in k.lower():
#         print(k)


#%%
def minmax_normalized(x, keys=["image"]):
    for key in keys:
        eps = torch.finfo(torch.float32).eps
        x[key] = torch.nn.functional.relu((x[key] - x[key].min()) / (x[key].max() - x[key].min() + eps))
    return x

# Custom transformation to filter problematic images
class FilterImages:
    def __init__(self, dat_type):
        
        self.tst_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                CropForegroundd(keys=["image"], source_key="image"),
                # CenterSpatialCropd(keys=["image"], roi_size=(img_size,)*3),
                Resized(keys=["image"], spatial_size=(img_size*2,)*3),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=img_size),
                minmax_normalized,
            ]
        )
        
        self.transforms = self.tst_transforms

    def __call__(self, data):
        # print(data)
        # data = self.transforms(data)
        # return data
        # print(data['image'].shape)
        try:
            data = self.transforms(data)
            # print(data.shape)
            if data['image'].shape[0] != 1:
                return None
            return data
        except Exception as e:
            # print(f"Error processing image: {image_data}{e}")
            return None
        
tst_filter_transform = FilterImages(dat_type='tst')

# print(tst_filter_transform({'image': '/SeaExpCIFS/Raw_MRIs/ALL_nii/STANFORD_nii/3.nii'})['image'])

#%%
# load the model and the dataloader
device = 'cuda'
# img_dict = {'img_net': 'ViTAutoEnc', 'img_size': 128, 'patch_size': 16, 'imgnet_ckpt': '/data_1/checkpoint.pth', 'imgnet_layers': 3, 'train_imgnet': False}
mdl = ImagingModel.from_ckpt(ckpt_path, device=device, img_backend=img_backend, load_from_ckpt=True)
print("loaded")
# print(mdl.net_.features[0].weight)

torch.set_grad_enabled(False)
mdl.net_.eval()
# mdl.to(device)
#%%

seed = 0
label_names = ['NC', 'MCI', 'DE']
# initialize datasets
avail_cohorts = {}
for dat_file in dat_files:
    print(dat_file)
    dat = CSVDataset(dat_file=dat_file, label_names=label_names, mri_emb_dict=mri_emb_dict)
    print("Loading Testing dataset ... ")
    tst_list, df = dat.get_features_labels(mode=3)

    logits: list[dict[str, float]] = []
    img_embeddings = None
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, fea_dict in tqdm(enumerate(tst_list)):
        image = fea_dict['image']
        f = image.split('/')
        # print(f)
        
        emb_path = save_path + '@'.join(f[5:]).replace('.nii', '.npy')
        if '@' in emb_path:
            cohort_name = 'NACC'
        else:
            # print(f)
            cohort_name = f[4].split('_')[0]
            
        if cohort_name in avail_cohorts:
            avail_cohorts[cohort_name] += 1
        else:
            avail_cohorts[cohort_name] = 1
        # break
        if os.path.exists(emb_path):
            continue
        print('save_path:' + emb_path)
        print('img_path:' + image)
        
        
        try:
            mri = tst_filter_transform({'image': image})['image']
            print(mri.shape)
            outputs = torch.nn.Sequential(*list(mdl.net_.features.children()))(mri.unsqueeze(0).to(mdl.device))
            outputs = torch.flatten(outputs, 1)
            # outputs = outputs.squeeze(0)
            print(outputs.shape)
            # break
            try:
                np.save(emb_path, outputs)
            except:
                np.save(save_path + '@'.join([f[5], f[-1]]).replace('.nii', '.npy'), outputs)
            print("saved")
        except:
            continue
        # break
    # break
print(avail_cohorts)
# #%%
# np.load('/data_1/skowshik/DenseNet_t1_emb/NACC106212_128401136192134176253428321101154621488411ni@1.2.840.113619.2.134.1762534283.2110.1154621488.411@1.2.840.113619.2.134.1762534283.2110.1154621488.518@1.2.840.113619.2.134.1762534283.2110.1154621488.518.npy').shape

 #%%
# ldr_tst = mdl._init_test_dataloader(batch_size=16, tst_list=tst_list, img_tst_trans=tst_filter_transform)
# scores, scores_proba, y_pred = mdl.predict(ldr_tst)

# #%%

# def save_performance_report(met, filepath):
#     figure(figsize=(24, 20), dpi=300)
#     met_df = pd.DataFrame(met).transpose()
#     report_plot = sns.heatmap(met_df, annot=True)

#     # plt.show()
#     report_plot.figure.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')

# # list-of-dict to dict-of-list
# y_true = [{k:int(v) if v is not None else 0 for k,v in entry.items()} for entry in dat.labels]
# mask = [{k:1 if v is not None else 0 for k,v in entry.items()} for entry in dat.labels]
# y_true_dict = {k: [smp[k] for smp in y_true] for k in y_true[0]}
# y_pred_dict = {k: [smp[k] for smp in y_pred] for k in y_pred[0]}
# scores_proba_dict = {k: [smp[k] for smp in scores_proba] for k in scores_proba[0]}
# mask_dict = {k: [smp[k] for smp in mask] for k in mask[0]}

# met = {}
# for k in label_names:
#     # if k in ['NC', 'MCI', 'DE']:
#     #     continue
#     print('Performance metrics of {}'.format(k))
#     metrics = get_metrics(np.array(y_true_dict[k]), np.array(y_pred_dict[k]), np.array(scores_proba_dict[k]), np.array(mask_dict[k]))
#     print_metrics(metrics)

#     met[k] = metrics
#     met[k].pop('Confusion Matrix')

# save_performance_report(met, f'/home/skowshik/ADRD_repo/pipeline_v1/adrd_tool/dev/visualization_figures/img_model/config2_3way/performane_report_{emb_type}.svg')

        
    
    
# %%
import torch
from adrd.nn import DenseNet

from torchvision import transforms
import random

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

img_size = 128
device = 'cuda'

def minmax_normalized(x, keys=["image"]):
    for key in keys:
        eps = torch.finfo(torch.float32).eps
        x[key] = torch.nn.functional.relu((x[key] - x[key].min()) / (x[key].max() - x[key].min() + eps))
    return x

# Custom transformation to filter problematic images
class FilterImages:
    def __init__(self, dat_type):
        
        self.tst_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                CropForegroundd(keys=["image"], source_key="image"),
                # CenterSpatialCropd(keys=["image"], roi_size=(img_size,)*3),
                Resized(keys=["image"], spatial_size=(img_size*2,)*3),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=img_size),
                minmax_normalized,
            ]
        )
        
        self.transforms = self.tst_transforms

    def __call__(self, data):
        # print(data)
        # data = self.transforms(data)
        # return data
        # print(data['image'].shape)
        try:
            data = self.transforms(data)
            # print(data.shape)
            if data['image'].shape[0] != 1:
                return None
            return data
        except Exception as e:
            # print(f"Error processing image: {image_data}{e}")
            return None
        
tst_filter_transform = FilterImages(dat_type='tst')

state_dict = torch.load('/data_1/skowshik/densenet_ckpt/bn_size_4/ckpt_densenet_t1_3way_config2.pt')
state_dict.pop('optimizer')
state_dict.pop('scaler')
state_dict.pop('label_distribution')
net_ = DenseNet( 
                tgt_modalities = state_dict.pop('tgt_modalities'),
                load_from_ckpt=True
            ).to(device)
net_.load_state_dict(state_dict)
torch.set_grad_enabled(False)
net_.eval()

mri = '/SeaExpCIFS/Raw_MRIs/ALL_nii/OASIS_nii/sub-OAS30001_ses-d3132_T1w.nii'
img_mri = tst_filter_transform({"image": mri})['image']
output = torch.nn.Sequential(*list(net_.features.children()))(img_mri.unsqueeze(0).to(device))
# output
# %%

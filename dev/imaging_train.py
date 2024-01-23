#%%
# import sys
# sys.path.append('..')
import pandas as pd

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
import nibabel as nib

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

# dat_file_all = '/home/skowshik/data/new_nacc_unique.csv'
# dat_file = '/home/skowshik/ADRD_repo/adrd_tool/data/nacc_new/new_nacc_processed_revised_labels.csv'
# dat_file = '/home/skowshik/ADRD_repo/adrd_tool/data/nacc_new/merged_data_nacc_nifd_stanford_adni_aibl_rtni_ppmi_revised_labels.csv'
# cnf_file = '/home/skowshik/ADRD_repo/adrd_tool/adrd/dev/data/toml_files/default_merged_nacc_nifd_stanford_adni_aibl_rtni_ppmi_revised_labels.toml'
# cnf_file = '/home/skowshik/ADRD_repo/adrd_tool/adrd/dev/data/toml_files/default_nacc_revised_labels.toml'

def parser():
    parser = argparse.ArgumentParser("Transformer pipeline", add_help=False)

    # Set Paths for running SSL training
    parser.add_argument('--data_path', default='/home/skowshik/ADRD_repo/adrd_tool/data/nacc_new/new_nacc_processed_revised_labels.csv', type=str,
        help='Please specify path to the entire data.')
    parser.add_argument('--ckpt_path', default='/home/skowshik/ADRD_repo/adrd_tool/adrd/dev/ckpt/revised_labels/ckpt.pt', type=str,
        help='Please specify the ckpt path')
    parser.add_argument('--num_epochs', default=256, type=int,
        help='Please specify the number of epochs')
    parser.add_argument('--batch_size', default=128, type=int,
        help='Please specify the batch size')
    parser.add_argument('--lr', default=1e-4, type=float,
        help='Please specify the learning rate')
    parser.add_argument('--gamma', default=2, type=float,
        help='Please specify the gamma value for the focal loss')
    parser.add_argument('--img_size', default=128, type=int,
        help='Please specify the img_size')
    parser.add_argument('--emb_type', default='T1', type=str,
        help='Please specify the embedding type [T1, T2, FLAIR, SWI, DWI, OTHER, ALL]')
    parser.add_argument('--backend', default='C3D', type=str, choices=['C3D', 'DenseNet'],
        help='Please specify the backend from [C3D, DenseNet]')
    parser.add_argument('--wandb_', action="store_true", help="Set to True to init wandb logging.")
    args = parser.parse_args()
    return args

# if not os.path.exists(save_path):
#     os.makedirs(save_path)
args = parser()

print(args.emb_type)
#%%
# emb_type = 'T1'
with open('./clinician_review/mri_3d.json') as json_data:
    mri_json = json.load(json_data)
mri_emb_dict = {}
for mag, seq in mri_json.items():
    for seq_name, mri_name in tqdm(seq.items()):
        if args.emb_type != 'ALL':
            if args.emb_type.lower() in ['t1', 't2', 'flair']:
                if args.emb_type.lower() not in seq_name.lower():
                    continue
            else:
                if ('t1' in seq_name.lower()) or ("t2" in seq_name.lower()) or ("flair" in seq_name.lower()):
                    continue
        print(seq_name)
        for name, pairs in mri_name.items():
            for pair in pairs:
                # print(mri_pair)
                mri = pair['mri']
                if ('localizer' in mri.lower()) | ('localiser' in mri.lower()) | ('LOC' in mri) | ('calibration' in mri.lower()) | ('gre_field_mapping' in mri.lower()):
                    continue
                zip_name = name[:-2] + '.zip'
                
                if zip_name in mri_emb_dict:
                    mri_emb_dict[zip_name].append(mri)
                else:
                    mri_emb_dict[zip_name] = [mri]

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
                
        if ('localizer' in mri.lower()) | ('localiser' in mri.lower()) | ('LOC' in mri) | ('calibration' in mri.lower()) | ('gre_field_mapping' in mri.lower()):
                continue
        
        if args.emb_type != 'ALL':
            if ("t1" in args.emb_type.lower()) or ("t2" in args.emb_type.lower()) or ("flair" in args.emb_type.lower()):
                if (args.emb_type.lower() not in mri.lower()):
                    if args.emb_type.lower() == 't1':
                        if ('t1' not in mri.lower()) and ('mprage' not in mri.lower()) and ('bravo' not in mri.lower()) and ('spgr' not in mri.lower()) and ('fspgr' not in mri.lower()) and ('mp-rage' not in mri.lower()):
                            continue 
                    else:
                        continue
                elif args.emb_type.lower() == 't2':
                    if 'flair' in mri.lower():
                        continue
            else:
                if ('t1' in mri.lower()) or ('mprage' in mri.lower()) or ('bravo' in mri.lower()) or ('spgr' in mri.lower()) or ('fspgr' in mri.lower()) or ('mp-rage' in mri.lower()) or ("t2" in mri.lower()) or ("flair" in mri.lower()):
                    continue
            
        if (mri.lower().startswith('adni')) or (mri.lower().startswith('nifd')) or (mri.lower().startswith('4rtni')):
            name = '_'.join(mri.split('_')[:4])
        elif (mri.lower().startswith('aibl')) or (mri.lower().startswith('sub')) or (mri.lower().startswith('ppmi')):
            name =  '_'.join(mri.split('_')[:2])
        elif mri.lower().startswith('stanford') or 'stanford' in cohort.lower():
            name = mri.split('.')[0]
        else:
            continue
        
        if name in mri_emb_dict:
            mri_emb_dict[name].append(f'{other_path}/{cohort}/{mri}')
        else:
            mri_emb_dict[name] = [f'{other_path}/{cohort}/{mri}']
        # print(mri)

#%%
# initialize datasets
# label_names = ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']
label_names = ['NC', 'MCI', 'DE']
seed = 0
dat = CSVDataset(dat_file=args.data_path, label_names=label_names, mri_emb_dict=mri_emb_dict)
print("Loading training dataset ... ")
trn_list, trn_df = dat.get_features_labels(mode=0)
print("Done.\nLoading Validation dataset ...")
vld_list, vld_df = dat.get_features_labels(mode=1)
print("Done.\nLoading entire dataset ...")
# tst_list = dat.get_features_labels(mode=2)
total_dat_list, total_df = dat.get_features_labels(mode=3)
print("Done.")
# print(f'intersection: {set(dat_trn.ids).intersection(set(dat_tst.ids))}')

# label_fractions = dat.label_fractions
# labels_ = dat.label_names

# df = pd.read_csv(args.data_path)



# getting label fractions

label_distribution = {}
for label in label_names:
    label_distribution[label] = dict(total_df[label].value_counts())
    
label_fractions = {}
for label in label_names:
    try:
        label_fractions[label] = total_df[label].value_counts()[1] / len(total_df)
    except:
        label_fractions[label] = 0.3

print(label_fractions)
print(label_distribution)


# print(dat_trn.feature_modalities)
# print(dat_trn.label_modalities)

# initialize and save Transformer
mdl = ImagingModel(
    tgt_modalities = label_names,
    label_fractions = label_fractions,
    num_epochs = args.num_epochs,
    batch_size = args.batch_size, # 64, 
    lr = args.lr,
    weight_decay = 0.01,
    gamma = args.gamma,
    bn_size = 4,
    # criterion = 'Balanced Accuracy',
    criterion = 'AUC (ROC)',
    device = 'cuda',
    cuda_devices = [0,2],
    ckpt_path = args.ckpt_path,
    load_from_ckpt = True,
    save_intermediate_ckpts = True,
    data_parallel = True,
    verbose = 4,
    img_backend = args.backend,
    label_distribution = label_distribution,
    wandb_ = args.wandb_,
    # k = 5,
    _amp_enabled = False
)



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

# train_transforms = Compose(
#     [
#         LoadImaged(keys=["image"]),
#         EnsureChannelFirstd(keys=["image"]),
#         Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
#         CropForegroundd(keys=["image"], source_key="image"),
#         monai.transforms.RandScaleCropd(keys=["image"], roi_scale=0.7, max_roi_scale=1, random_size=True, random_center=True),
#         monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=args.img_size),
#         flip_and_jitter,
#         monai.transforms.RandGaussianSmoothd(keys=["image"], prob=0.5),
#         minmax_normalized,
#     ]            
# )


# vld_transforms = Compose(
#     [
#         LoadImaged(keys=["image"]),
#         EnsureChannelFirstd(keys=["image"]),
#         Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
#         CropForegroundd(keys=["image"], source_key="image"),
#         # CenterSpatialCropd(keys=["image"], roi_size=(args.img_size,)*3),
#         Resized(keys=["image"], spatial_size=(args.img_size*2,)*3),
#         monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=args.img_size),
#         minmax_normalized,
#     ]
# )

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
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=args.img_size),
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
                Resized(keys=["image"], spatial_size=(args.img_size*2,)*3),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=args.img_size),
                minmax_normalized,
            ]
        )
        
        if dat_type == 'trn':
            self.transforms = self.train_transforms
        else:
            self.transforms = self.vld_transforms

    def __call__(self, data):
        try:
            image_data = data["image"]
            check = nib.load(image_data).get_fdata()
            if len(check.shape) > 3:
                return None
            return self.transforms(data)
        except Exception as e:
            # print(f"Error processing image: {image_data}{e}")
            return None
        
trn_filter_transform = FilterImages(dat_type='trn')
vld_filter_transform = FilterImages(dat_type='vld')

mdl.fit(trn_list, vld_list, img_train_trans=trn_filter_transform, img_vld_trans=vld_filter_transform)

import torch
import json
import argparse
import os
import monai
import pandas as pd
import numpy as np

from dev.data.dataset_csv import CSVDataset
from adrd.model import ADRDModel
from tqdm import tqdm
from collections import defaultdict
# change the train, vld and test file paths, new cnf_file path


basedir = '.'
# data_path="/home/varunaja/mri_pet/adrd_tool_varuna/adrd_transformer/data/adni_data_single.csv" # path to the data file before train val test split
# train_path="/home/varunaja/mri_pet/adrd_tool_varuna/adrd_transformer/data/adni_train_split_single.csv"
# vld_path="/home/varunaja/mri_pet/adrd_tool_varuna/adrd_transformer/data/adni_val_split_single.csv"
# test_path="/home/varunaja/mri_pet/adrd_tool_varuna/adrd_transformer/data/a4_test_split_single.csv"
# cnf_file="/home/varunaja/mri_pet/adrd_tool_varuna/adrd_transformer/meta_files/ab_tau_config_finetune.toml"
# orig_ckpt_path = '/data_1/skowshik/ckpts_backbone_swinunet/ckpt_without_imaging.pt'
# new_ckpt_path = f'{basedir}/dev/ckpt/model_ckpt_finetune_no_rl.pt'

data_path="/home/varunaja/mri_pet/adrd_tool_varuna/adrd_transformer/data/oasis_data_single.csv" # path to the data file before train val test split
train_path="/home/varunaja/mri_pet/adrd_tool_varuna/adrd_transformer/data/oasis_train_split_single.csv"
vld_path="//home/varunaja/mri_pet/adrd_tool_varuna/adrd_transformer/data/lucky_adni_val_split_single.csv"
test_path="/home/varunaja/mri_pet/adrd_tool_varuna/adrd_transformer/data/a4_test_split_single.csv"
cnf_file="/home/varunaja/mri_pet/adrd_tool_varuna/adrd_transformer/meta_files/oasis_config.toml"
orig_ckpt_path = '/data_1/skowshik/ckpts_backbone_swinunet/ckpt_without_imaging.pt'
new_ckpt_path = f'{basedir}/dev/ckpt/model_ckpt_finetune_oasis.pt'

# no need to change these as they will not be used with non-imaging model
emb_path = '/data_1/dlteif/SwinUNETR_MRI_stripped_MNI_emb/' 
nacc_mri_info = "dev/nacc_mri_3d.json"
other_mri_info = "dev/other_3d_mris.json"

img_net="NonImg"
img_mode=-1
mri_type="SEQ"

# these are labels to remove from the model's state dictionary
labels_to_remove = ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']

# add the new labels
new_labels = ['amy_label', 'tau_label']
train_path
state_dict = torch.load(orig_ckpt_path, map_location=torch.device('cpu'))
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
else:
    src_modalities = state_dict.pop('src_modalities')
    tgt_modalities = state_dict.pop('tgt_modalities')
    if 'label_distribution' in state_dict:
        label_distribution = state_dict.pop('label_distribution')
    if 'optimizer' in state_dict:
        optimizer = state_dict.pop('optimizer')
    d_model = state_dict.pop('d_model')
    nhead = state_dict.pop('nhead')
    num_encoder_layers = state_dict.pop('num_encoder_layers')
    num_decoder_layers = state_dict.pop('num_decoder_layers')
    if 'epoch' in state_dict.keys():
        start_epoch = state_dict.pop('epoch')
    img_net = state_dict.pop('img_net')
    imgnet_layers = state_dict.pop('imgnet_layers')
    img_size = state_dict.pop('img_size')
    patch_size = state_dict.pop('patch_size')
    imgnet_ckpt = state_dict.pop('imgnet_ckpt')
    train_imgnet = state_dict.pop('train_imgnet')
    if 'scaler' in state_dict and state_dict['scaler']:
        state_dict.pop('scaler')

# initialize datasets
seed = 0
stripped = '_stripped_MNI'
print("Loading training dataset ... ")
dat_trn = CSVDataset(dat_file=train_path, cnf_file=cnf_file, mode=0, img_mode=img_mode, mri_type=mri_type, arch=img_net, emb_path=emb_path, nacc_mri_info=nacc_mri_info, other_3d_mris=other_mri_info, transforms=None, stripped=stripped)
print("Done.\nLoading Validation dataset ...")
dat_vld = CSVDataset(dat_file=vld_path, cnf_file=cnf_file, mode=1, img_mode=img_mode, mri_type=mri_type, arch=img_net, emb_path=emb_path, nacc_mri_info=nacc_mri_info, other_3d_mris=other_mri_info, transforms=None, stripped=stripped)
# print("Done.\nLoading testing dataset ...")
# dat_tst = CSVDataset(dat_file=test_path, cnf_file=cnf_file, mode=2, img_mode=img_mode, mri_type=mri_type, arch=img_net, emb_path=emb_path, nacc_mri_info=nacc_mri_info, other_mri_info=other_mri_info, transforms=None, stripped=stripped)
# print("Done.")
df = pd.read_csv(data_path)

label_distribution = {}
for label in new_labels:
    label_distribution[label] = dict(df[label].value_counts())
label_fractions = dat_trn.label_fractions

print(label_fractions)
print(label_distribution)

num_epochs = 255
batch_size = 64
lr = 1e-3
weight_decay = 0.005
gamma = 2
fusion_stage = 'middle'
load_from_ckpt = False
save_intermediate_ckpts = True
ranking_loss = True
train_imgnet = False

# initialize Transformer
mdl = ADRDModel(
    src_modalities = dat_trn.feature_modalities,
    tgt_modalities = dat_trn.label_modalities,
    label_fractions = label_fractions,
    d_model = d_model,
    nhead = nhead,
    num_epochs = num_epochs,
    batch_size = batch_size, 
    lr = lr,
    weight_decay = weight_decay,
    gamma = gamma,
    criterion = 'MCC',
    device = 'cpu',
    cuda_devices = [1,2],
    img_net = img_net,
    imgnet_layers = imgnet_layers,
    img_size = img_size,
    fusion_stage= fusion_stage,
    imgnet_ckpt = imgnet_ckpt,
    patch_size = patch_size,
    ckpt_path = new_ckpt_path,
    train_imgnet = train_imgnet,
    load_from_ckpt = load_from_ckpt,
    save_intermediate_ckpts = save_intermediate_ckpts,
    data_parallel = False,
    verbose = 4,
    wandb_ = 1,
    label_distribution = label_distribution,
    ranking_loss = ranking_loss,
    _amp_enabled = False,
    _dataloader_num_workers = 1,
)
# Copy the saved model weights to the new model state dictionary
new_mdl_state_dict = mdl.net_.state_dict()
for key in state_dict.keys():
    if key in labels_to_remove or key == 'emb_aux':
        continue
    if key in new_mdl_state_dict:
        new_mdl_state_dict[key] = state_dict[key]

# Load the updated state dictionary into the new model
mdl.net_.load_state_dict(new_mdl_state_dict)
# Train the model
mdl.fit(dat_trn.features, dat_vld.features, dat_trn.labels, dat_vld.labels, img_train_trans=None, img_vld_trans=None, img_mode=img_mode)

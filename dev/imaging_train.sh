#!/bin/bash

# run this script from adrd_tool/

conda activate adrd

# install the package
# cd adrd_tool
pip install -e .

# define the variables
prefix="/home/skowshik/ADRD_repo/pipeline_v1_main/adrd_tool"
data_path="${prefix}/data/training_cohorts/train_vld_test_split_updated/merged_train.csv"
ckpt_path="${prefix}/dev/ckpt/densenet/ckpt_densenet_flair_bn_size_4_2layer_downsamp.pt"
# backend=C3D
backend=DenseNet
emb_type=FLAIR
# run train.py
python dev/imaging_train.py --data_path $data_path --ckpt_path $ckpt_path --num_epochs 256 --batch_size 32 --lr 1e-4 --gamma 2 --emb_type $emb_type --img_size 128 --backend $backend --wandb_


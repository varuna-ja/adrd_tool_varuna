
# import sys
# sys.path.append('../')
from dataset_csv import CSVDataset
from adrd.model import Transformer

# dat_file = '/home/varunaja/mri_pet/adrd_transformer/data/adni_data.csv'
dat_file = '/home/varunaja/mri_pet/adrd_tool/adrd_transformer/data/adni_data.csv'
cnf_file = '/home/varunaja/mri_pet/adrd_tool/adrd_transformer/scripts/cx_config.toml'

# initialize datasets
seed = 0
print('Loading training dataset ... ')
dat_trn = CSVDataset(
    dat_file, cnf_file,
    mode=0, split=[8, 2], seed=seed
)

# print(dat_trn.feature_modalities)
# print('Done.\nLoading testing dataset ...')
# dat_tst = CSVDataset(mode=1, split=[8, 2], seed=seed)
print('Done.')
# print(dat_trn.features[0])
# print(dat_trn.feature_modalities)

# initialize and save Transformer
mdl = Transformer(
    src_modalities = dat_trn.feature_modalities,
    tgt_modalities = dat_trn.label_modalities,
    d_model = 256,
    nhead = 1,
    num_layers=1,
    # num_encoder_layers = 2,
    # num_decoder_layers = 2,
    num_epochs = 128,
    batch_size = 128,  # 64
    # batch_size = 16 * 4,
    # batch_size_multiplier = 4, 
    lr = 1e-3,
    weight_decay = 1e-2,
    # beta = 0.9999,
    beta = 0.9999,
    gamma = 2.0,
    # scale = 1e4,
    criterion = 'Loss',
    device = 'cpu',
    verbose = 4,
    # _device_ids = [0, 1, 2, 3],
    # _dataloader_num_workers = 1,
    # _amp_enabled = True,
)
mdl.fit(dat_trn.features, dat_trn.labels)
mdl.save('./ckpt_nonimg_062623.pt')
# %%

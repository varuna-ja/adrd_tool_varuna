from time import time
from copy import deepcopy
import pandas as pd
import numpy as np
import toml
from functools import cached_property
from tqdm import tqdm

# dat_file = '/home/cxue2/Dropbox/adrd_tool/dev/data/nacc.csv'
# cnf_file = '/home/cxue2/Dropbox/adrd_tool/dev/data/default.toml'

# dat_file = '/home/cxue2/Dropbox/adrd_tool/dev/data/new_nacc_processed_revised_labels.csv'
# cnf_file = '/home/cxue2/Dropbox/adrd_tool/dev/data/default_nacc_revised_labels.toml'

value_mapping = {
    'his_SEX':          {'female': 0, 'male': 1},
    'his_HISPANIC':     {'no': 0, 'yes': 1, 1: 0, 2: 1, 3: 1, '1': 0, '2': 1, '3': 1, '1.0': 0, '2.0': 1, '3.0': 1},
    'his_NACCNIHR':     {'whi': 0, 'blk': 1, 'asi': 2, 'ans': 2, 'ind': 3, 'haw': 4, 'mul': 5, 'mix': 5},
    # 'his_RACE':         {'whi': 0, 'blk': 1, 'asi': 2, 'ans': 2,, 'ind': 3, 'haw': 4 'oth': 5, 'mix': 5},
    'his_RACE':         {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'oth': 5},
    'his_RACESEC':      {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'oth': 5},
    'his_RACETER':      {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'oth': 5},
    # 'race':         {'ans': 0, 'blk': 1, 'haw': 2, 'ind': 3, 'whi': 4, 'mix': 5},
    'his_TOBAC100': {0: 0, 1: 1, 2: 1, 3: 1}
}

class CSVDataset:

    def __init__(self, dat_file, cnf_file, mode=0, split=(1,), seed=3227):
        ''' ... '''
        # load configuration file
        self.cnf = toml.load(cnf_file)

        # load data csv
        df_raw = pd.read_csv(dat_file)
        df = df_raw.copy(deep=True)

        # load T1 mris
        if 'img_MRI_T1' in self.feature_modalities:
            list_mri = []
            for i in tqdm(df.index):
                # filepath_mri = df_raw.loc[i]['path'] + df_raw.loc[i]['filename']
                # print('/data_1/NACC_ALL/npy/' + df_raw.loc[i]['img_MRI_T1'])
                if pd.isna(df_raw.loc[i]['filename']):
                    list_mri.append(None)
                else:
                    filepath_mri = df_raw.loc[i]['path'] + df_raw.loc[i]['filename']
                    list_mri.append(np.load(filepath_mri, mmap_mode='r')[np.newaxis, :])
            # df.insert(len(features) - 1, 'img_MRI_T1', list_mri)
            df['img_MRI_T1'] = list_mri
        
        # filter dataframe by modes
        print('Out of {} samples, '.format(len(df)), end='')
        msk = self._get_mask_mode(df, mode, split, seed)
        df = df[msk]
        print('{} are selected for mode {}.'.format(len(df), mode))

        # check feature availability in data file
        print('Out of {} features in configuration file, '.format(len(self.cnf['feature'])), end='')
        tmp = [fea for fea in self.cnf['feature'] if fea not in df.columns]
        print('{} are unavailable in data file.'.format(len(tmp)))

        # check label availability in data file
        print('Out of {} labels in configuration file, '.format(len(self.cnf['label'])), end='')
        tmp = [lbl for lbl in self.cnf['label'] if lbl not in df.columns]
        print('{} are unavailable in data file.'.format(len(tmp)))

        # get feature and label names
        features = list(self.cnf['feature'].keys())
        labels = list(self.cnf['label'].keys())

        # omit features that are not present in dat_file
        features = [fea for fea in features if fea in df.columns]

        # drop columns that are not present in configuration
        df = df[features + labels]

        # drop rows where ALL features are missing
        df_fea = df[features]
        df_fea = df_fea.dropna(how='all')
        print('Out of {} samples, {} are dropped due to complete feature missing.'.format(len(df), len(df) - len(df_fea)))
        df = df[df.index.isin(df_fea.index)]

        # drop rows where ANY labels are missing
        df_lbl = df[labels]
        df_lbl = df_lbl.dropna(how='all')
        print('Out of {} samples, {} are dropped due to complete label missing.'.format(len(df), len(df) - len(df_lbl)))
        df = df[df.index.isin(df_lbl.index)]

        # some of the feature values need to be adjusted
        for name in features + labels:
            if name in value_mapping:
                col = df[name].to_list()
                try:
                    col = [value_mapping[name][s] if not pd.isnull(s) else None for s in col]
                except KeyError as err:
                    print(err, name)
                    exit()
                df[name] = col
        
        # change np.nan to None
        df.replace({np.nan: None}, inplace=True)

        # done for df
        self.df = df

        # construct dictionaries for features and labels
        self.features, self.labels = [], []
        keys = df.columns.values.tolist()
        for i in range(len(df)):
            vals = df.iloc[i].to_list()
            self.features.append(dict(zip(keys[:len(features)], vals[:len(features)])))
            self.labels.append(dict(zip(keys[len(features):], vals[len(features):])))

        # test: remove if None
        for i in range(len(self.features)):
            for k, v in list(self.features[i].items()):
                if v is None:
                    self.features[i].pop(k)

    def __len__(self):
        ''' ... '''
        return len(self.df)

    def __getitem__(self, idx):
        ''' ... '''
        return self.features[idx], self.labels[idx]

    def _get_mask_mode(self, df, mode, split, seed):
        ''' ... '''
        # normalize split into ratio
        ratio = np.array(split) / np.sum(split)
        
        # list of modes for all samples
        arr = []
        
        # 0th ~ (N-1)th modes
        for i in range(len(split) - 1):
            arr += [i] * round(ratio[i] * len(df))
        
        # last mode
        arr += [len(split) - 1] * (len(df) - len(arr))
        
        # random shuffle
        # random seed will be fixed before shuffle and reset right after
        arr = np.array(arr)
        np.random.seed(seed)
        np.random.shuffle(arr)
        np.random.seed(int(1000 * time()) % 2 ** 32)
        
        # generate mask
        msk = (arr == mode).tolist()
        
        return msk

    @cached_property
    def feature_modalities(self):
        ''' ... '''
        return self.cnf['feature']

    @cached_property
    def label_modalities(self):
        ''' ... '''
        return self.cnf['label']


if __name__ == '__main__':
    # load dataset
    dset = CSVDataset(mode=0, split=[8, 2])
    # print(dset[0])

    print(dset.df)
    print(dset.df.iloc[:, [28]].unique())
    # print(dset.feature_modalities)

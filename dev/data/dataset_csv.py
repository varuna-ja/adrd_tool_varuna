#%%
from time import time
from copy import deepcopy
import pandas as pd
import numpy as np
import toml
from tqdm import tqdm
# from adrd.nn import ImageModel
from icecream import ic
import torch
from .MRI_load import load_mris
import ast
import os
import glob
import nibabel as nib
import json
from collections import defaultdict
import random
#%%


# value_mapping = {
#     'his_SEX':          {'female': 0, 'male': 1},
#     'his_HISPANIC':     {'no': 0, 'yes': 1},
#     'his_NACCNIHR':     {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'mul': 5},
#     'his_RACE':         {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'oth': 5},
#     'his_RACESEC':      {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'oth': 5},
#     'his_RACETER':      {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'oth': 5},
# }

# label_names = ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']

value_mapping = {}

label_names = ['amy_label', 'tau_label']


# To use on ivc-ml scc
# nacc_mri_info = "/projectnb/ivc-ml/dlteif/NACC_raw/mri_3d.json"
# other_path = '/projectnb/ivc-ml/dlteif/Raw_MRIs'

# uncomment this to use on echo
# nacc_mri_info = "../clinician_review/mri_3d.json"
# other_path = '/SeaExpCIFS/Raw_MRIs/ALL_nii'
other_path = '/SeaExpCIFS/ADNI_MRI/raw'

class CSVDataset:

    def __init__(self, dat_file, cnf_file, mode=0, img_mode=0, dat_trn=None, mri_type='ALL', other_3d_mris=None, arch=None, emb_path='/data_1/dlteif/SwinUNETR_MRI_stripped_emb/', nacc_mri_info = "./clinician_review/mri_3d.json", transforms=None, stripped=True):
        ''' ... '''
        # load data csv
        if isinstance(dat_file, str):
            print(dat_file)
            df = pd.read_csv(dat_file)
        else:
            df = dat_file
            
        # load configuration file
        self.cnf = toml.load(cnf_file)
        
        def apply_mri_zip(row):
            if 'mri_zip' in row:
                if pd.isna(row['mri_zip']):
                    return row['ID']
                else:
                    return row['mri_zip']
            else:
                return row['ID']
                
        def fill_nan_features(row, fea):
            if isinstance(row[fea], list):
                return pd.Series(row[fea])
            else:
                return pd.Series([np.NaN] * 150)
                
        

        avail_cohorts = set()
        # df['mri_zip'] = df.apply(apply_mri_zip, axis=1)
        cnt = 0
        
        
        # f = open('notexists.txt', 'w')
        if img_mode in [0,2]:
            with open(nacc_mri_info) as json_data:
                mri_json = json.load(json_data)
            self.all_seq_dict = {}
            for seq_type in ['T1', 'T2', 'FLAIR', 'OTHER']:
                mri_dict = {}
                for mag, seq in mri_json.items():
                    for seq_name, mri_name in tqdm(seq.items()):
                        
                        if mri_type != "ALL":
                            if seq_type != 'OTHER':
                                if seq_type.lower() not in seq_name.lower():
                                    continue
                            else:
                                if ('t1' in seq_name.lower() and 'MT1' not in mri) or ("t2" in mri.lower()) or ("flair" in mri.lower()):
                                    continue
                                
                        for name, pairs in mri_name.items():
                            for pair in pairs:
                                # print(mri_pair)
                                mri = pair['mri']
                                if stripped and not '_stripped.nii' in mri:
                                    if not os.path.exists(mri.replace('.nii', '_stripped.nii')):
                                        continue
                                    mri = mri.replace('.nii', '_stripped.nii')
                                if ('localizer' in mri.lower()) | ('localiser' in mri.lower()) | ('LOC' in mri) | ('calibration' in mri.lower()) | ('field_mapping' in mri.lower()) | (mri.lower()[:-4].endswith('_ph')) | ('seg' in mri.lower()):
                                    continue
                                zip_name = name[:-2] + '.zip'
                                if zip_name in mri_dict:
                                    mri_dict[zip_name].append(mri)
                                else:
                                    mri_dict[zip_name] = [mri]
                                # if not os.path.exists(mri):
                                #     f.write(mri+'\n')
                if len(mri_dict) != 0:
                    avail_cohorts.add('NACC')
                                
                # other cohorts
                for cohort in os.listdir(other_path):
                    if os.path.isfile(f'{other_path}/{cohort}'):
                        continue
                    for mri in tqdm(os.listdir(f'{other_path}/{cohort}')):
                        if mri.endswith('json'):
                            continue
                        
                        if other_3d_mris is not None and mri not in other_3d_mris:
                            continue
                                
                        if stripped and not '_stripped.nii' in mri:
                            if not os.path.exists(mri.replace('.nii', '_stripped.nii')):
                                continue
                            mri = mri.replace('.nii', '_stripped.nii')
                        
                        if ('localizer' in mri.lower()) | ('localiser' in mri.lower()) |  ('LOC' in mri) | ('calibration' in mri.lower()) | ('field_mapping' in mri.lower()) | (mri.lower()[:-4].endswith('_ph')) | ('seg' in mri.lower()):
                            continue
                        
                        if mri_type != "ALL":
                            if seq_type != 'OTHER':
                                if (seq_type.lower() not in mri.lower()):
                                    if seq_type.lower() == 't1':
                                        if ('t1' not in mri.lower()) and ('mprage' not in mri.lower()) and ('bravo' not in mri.lower()) and ('spgr' not in mri.lower()) and ('fspgr' not in mri.lower()) and ('mp-rage' not in mri.lower()): 
                                            continue 
                                    else:
                                        continue
                                
                                elif seq_type.lower() == 't2':
                                    if 'flair' in mri.lower():
                                        continue
                                    
                            else:
                                if (('t1' in mri.lower()) and ('MT1' not in mri)) or ('mprage' in mri.lower()) or ('bravo' in mri.lower()) or ('spgr' in mri.lower()) or ('fspgr' in mri.lower()) or ('mp-rage' in mri.lower()) or ("t2" in mri.lower()) or ("flair" in mri.lower()):
                                    continue
                            
                        if (mri.lower().startswith('adni')) or (mri.lower().startswith('nifd')) or (mri.lower().startswith('4rtni')):
                            name = '_'.join(mri.split('_')[:4])
                        elif (mri.lower().startswith('aibl')) or (mri.lower().startswith('sub')) or (mri.lower().startswith('ppmi')):
                            name =  '_'.join(mri.split('_')[:2])
                        elif mri.lower().startswith('stanford') or 'stanford' in cohort.lower():
                            if 't1' in mri.lower():
                                name = mri.split('.')[0].split('_')[0] + '_' + mri.split('.')[0].split('_')[2]
                            else:
                                name = mri.split('.')[0]
                        else:
                            continue
                        
                        if mri.lower().startswith('sub'):
                            avail_cohorts.add('OASIS')
                        else:
                            avail_cohorts.add(name.split('_')[0])
                        
                        if name in mri_dict:
                            mri_dict[name.replace('_stripped', '')].append(f'{other_path}/{cohort}/{mri}')
                        else:
                            mri_dict[name.replace('_stripped', '')] = [f'{other_path}/{cohort}/{mri}']
                        
                        # if not os.path.exists(f'{other_path}/{cohort}/{mri}'):
                        #     f.write(f'{other_path}/{cohort}/{mri}\n')
                            
                if mri_type == "ALL":
                    self.all_seq_dict[mri_type] = mri_dict
                    break
                            
                self.all_seq_dict[seq_type] = mri_dict
        #%%
        elif img_mode == 1:
            cnt = 0
            mri_embs = os.listdir(emb_path)
            
            with open(nacc_mri_info) as json_data:
                mri_json = json.load(json_data)
            nacc_mri_names_dict = {}
            for mag, seq in mri_json.items():
                for seq_name, mri_name in tqdm(seq.items()):
                    print(seq_name)
                    for name, pairs in mri_name.items():
                        for pair in pairs:
                            mri = pair['mri']
                            if seq_name in ['t1', 't2', 'flair']:
                                nacc_mri_names_dict[mri.split('/')[-1]] = seq_name
                            else:
                                nacc_mri_names_dict[mri.split('/')[-1]] = 'other'
            # print(nacc_mri_names_dict)
            # print(nacc_mri_names_dict['1.2.840.113713.11.1.0.0.0.1391475871.14308.531399760.2.nii'])
            self.all_seq_dict = {}
            avail_cohorts = set()
                
            for seq_type in ['T1', 'T2', 'FLAIR', 'OTHER']:
                mri_dict = {}
                for mri in mri_embs:
                    if stripped and not '_stripped.npy' in mri:
                        if not os.path.exists(mri.replace('.npy', '_stripped.npy')):
                            continue
                        mri = mri.replace('.npy', '_stripped.npy')
                        
                    if ('localizer' in mri.lower()) | ('localiser' in mri.lower()) |  ('LOC' in mri) | ('calibration' in mri.lower()) | ('field_mapping' in mri.lower()) | (mri.lower()[:-4].endswith('_ph')) | ('seg' in mri.lower()):
                        continue
                    
                    if 'stanford' not in mri.lower() and '@' not in mri and other_3d_mris is not None and not mri.replace('_stripped.npy', '.npy').replace('.npy', '.nii') in other_3d_mris:
                        continue
                    
                    if '@' in mri:
                        if mri.split('@')[-1].replace('_stripped.npy', '.npy').replace('.npy', '.nii') not in nacc_mri_names_dict:
                            # print(mri)
                            continue
                    
                    if mri_type != "ALL":
                        if seq_type != 'OTHER':
                            if '@' in mri:
                                if nacc_mri_names_dict[mri.split('@')[-1].replace('_stripped.npy', '.npy').replace('.npy', '.nii')] != seq_type.lower():
                                    continue
                                elif 't1' in mri.lower() and 'MT1' in mri:
                                    continue
                            elif (seq_type.lower() not in mri.lower()):
                                if seq_type.lower() == 't1':
                                    if ('stanford' not in mri.lower()) and ('t1' not in mri.lower()) and ('mprage' not in mri.lower()) and ('bravo' not in mri.lower()) and ('spgr' not in mri.lower()) and ('fspgr' not in mri.lower()) and ('mp-rage' not in mri.lower()) and ('mp_rage' not in mri.lower()):
                                        continue
                                else:
                                    continue
                            
                            elif seq_type.lower() == 't2':
                                if 'flair' in mri.lower():
                                    continue
                        else:
                            if '@' in mri:
                                if nacc_mri_names_dict[mri.split('@')[-1].replace('_stripped.npy', '.npy').replace('.npy', '.nii')] != "other":
                                    continue
                            elif ('stanford' in mri.lower()) or ('t1' in mri.lower()) or ('mprage' in mri.lower()) or ('bravo' in mri.lower()) or ('spgr' in mri.lower()) or ('fspgr' in mri.lower()) or ('mp-rage' in mri.lower()) or ('mp_rage' in mri.lower()) or ("t2" in mri.lower()) or ("flair" in mri.lower()):
                                continue
                            elif ("@" in mri) and (nacc_mri_names_dict[mri.split('@')[-1].replace('_stripped.npy', '.npy').replace('.npy', '.nii')] in ['t1', 't2', 'flair']):
                                continue
                            
                            
                    name = ''
                    if '@' in mri:
                        name = mri.split('@')[0][:-2] + '.zip'
                        avail_cohorts.add('NACC')
                        # continue
                        # pass
                    else:
                        if (mri.lower().startswith('adni')) or (mri.lower().startswith('nifd')) or (mri.lower().startswith('4rtni')):
                            name = '_'.join(mri.split('_')[:4])
                        elif (mri.lower().startswith('aibl')) or (mri.lower().startswith('sub')) or (mri.lower().startswith('ppmi')):
                            name =  '_'.join(mri.split('_')[:2])
                        elif mri.lower().startswith('stanford'):
                            if 't1' in mri.lower():
                                name = mri.split('.')[0].split('_')[0] + '_' + mri.split('.')[0].split('_')[2]
                            else:
                                name = mri.split('.')[0]
                        elif mri.lower().startswith('fhs'):
                            name_list = mri.split('_')
                            indices = [index for index, item in enumerate(name_list) if 'nifti' in item.lower()]
                            name = '_'.join(mri.split('_')[:indices[0] + 1])
                        elif mri.lower().startswith('bmc'):
                            name = '_'.join(mri.split('_')[:2])
                        else:
                            continue
                        
                        if mri.lower().startswith('sub'):
                            avail_cohorts.add('OASIS')
                        else:
                            avail_cohorts.add(name.split('_')[0])
                        # pass
                    # if (('t2' in name.lower()) & ('flair' in name.lower())) | ('swi' not in name.lower()) & ('t2' not in name.lower()) & ('dti' not in name.lower()) & ('dwi' not in name.lower()):
                    name = name.replace('_stripped', '')
                    if name in mri_dict:
                        mri_dict[name].append(mri)
                    else:
                        mri_dict[name] = [mri]
                    cnt += 1
                        
                if mri_type == "ALL":
                    self.all_seq_dict[mri_type] = mri_dict
                    break
                self.all_seq_dict[seq_type] = mri_dict
        
        print("AVAILABLE MRI Cohorts: ", avail_cohorts)
        if 'NACC' not in avail_cohorts:
            print('NACC MRIs not available')
        print(f"Avail mris: {cnt}")
        
        if img_mode in [0,1,2]:
            if mri_type == "ALL":
                print(len(df[~df['mri_zip'].isna()]))
                print(len(self.all_seq_dict[mri_type]))
                df['filename'] = df['mri_zip'].map(self.all_seq_dict[mri_type])
                
                print(f"{len(df[~df[f'filename'].isna()])} patients with MRI found")
                
                split_features = df.apply(fill_nan_features, fea=f'filename', axis=1)
                split_features.columns = [f"img_MRI_{i+1}" for i in range(len(split_features.columns))]
                df = pd.concat([df, split_features], axis=1)                
            
                
                df[['ID', 'mri_zip', 'filename']].to_csv('./mri_check.csv', index=False)
                
                ffff = list(df['filename'])
                p = []
                for x in ffff:
                    if isinstance(x, list):
                        for li in x:
                            if li in p:
                                print(li)
                                print("duplicate")
                            p.append(li)
                            # if li.replace('_stripped.npy', '.npy').replace('.npy', '.nii') not in other_3d_mris:
                            #     print(li)
                # print("Testing: ", p)
                

                df = df.dropna(axis=1, how='all')

                if dat_trn is not None:
                    # print(dat_trn.columns)
                    columns_to_drop = [col for col in df.columns if col.startswith('img') and col not in dat_trn.columns]
                    df = df.drop(columns=columns_to_drop)
                    

                for i in range(len(split_features.columns)):
                    self.cnf['feature'][f'img_MRI_{i+1}'] = {'type': 'imaging', 'shape': "################ TO_FILL_MANUALLY ################"}
            else:
                for k, v in self.all_seq_dict.items():
                    print(k,v)
                    df[f'filename_{k}'] = df['mri_zip'].map(v)
                    
                    print(f"{len(df[~df[f'filename_{k}'].isna()])} patients with {k} MRI found")
                    
                    split_features = df.apply(fill_nan_features, fea=f'filename_{k}', axis=1)
                    split_features.columns = [f"img_MRI_{k}_{i+1}" for i in range(len(split_features.columns))]
                    df = pd.concat([df, split_features], axis=1)

                    df = df.dropna(axis=1, how='all')

                    if dat_trn is not None:
                        columns_to_drop = [col for col in df.columns if col.startswith('img') and col not in dat_trn.columns]
                        df = df.drop(columns=columns_to_drop)
                        

                    for i in range(len(split_features.columns)):
                        self.cnf['feature'][f'img_MRI_{k}_{i+1}'] = {'type': 'imaging', 'shape': "################ TO_FILL_MANUALLY ################"}   
        
        if 'ID' in df.columns:
            self.ids = list(df['ID'])

        df.reset_index(drop=True, inplace=True)
        print('{} are selected for mode {}.'.format(len(df), mode))

        # check feature availability in data file
        print('Out of {} features in configuration file, '.format(len(self.cnf['feature'])), end='')
        tmp = [fea for fea in self.cnf['feature'] if fea not in df.columns]
        print('{} are unavailable in data file.'.format(tmp))

        # check label availability in data file
        print('Out of {} labels in configuration file, '.format(len(self.cnf['label'])), end='')
        tmp = [lbl for lbl in self.cnf['label'] if lbl not in df.columns]
        print('{} are unavailable in data file.'.format(len(tmp)))

        self.cnf['feature'] = {k:v for k,v in self.cnf['feature'].items() if k in df.columns}
        self.cnf['label'] = {k:v for k,v in self.cnf['label'].items() if k in df.columns}

        # get feature and label names
        features = list(self.cnf['feature'].keys())
        labels = list(self.cnf['label'].keys())

        # omit features that are not present in dat_file
        features = [fea for fea in features if fea in df.columns]

        # mri
        img_fea_to_pop = []
        total = 0
        total_cohorts = {}
        for fea in self.cnf['feature'].keys():
            # print('fea: ', fea)
            if self.cnf['feature'][fea]['type'] == 'imaging':
                print('imaging..')
                if img_mode == -1:
                    # to train non imaging model
                    img_fea_to_pop.append(fea)
                elif img_mode == 0:
                    print("fea: ", fea)
                    filenames = df[fea].dropna().to_list()
                    with open('notexists.txt', 'a') as f:
                        for fn in filenames:
                            if fn is None:
                                continue
                            if not os.path.exists(fn):
                                f.write(fn+'\n')

                # load MRI embeddings 
                elif img_mode == 1:
                    print("fea: ", fea)
                    filenames = df[fea].to_list()
                    
                    if len(df[~df[fea].isna()]) == 0:
                        continue
                    # print(fea)
                        
                    npy = []
                    n = 0
                    for fn in tqdm(filenames):
                        try:
                            path_fn = emb_path + fn
                            # print('path_fn: ', path_fn)
                            data = np.load(path_fn, mmap_mode='r')
                            if np.isnan(data).any():
                                npy.append(None)
                                continue
                            
                            if 'swinunet' in emb_path.lower():
                                if len(data.shape) < 5:
                                    data = np.expand_dims(data, axis=0)
                                    
                            # print(data)
                            
                            # data = data.flatten()
                            # if len(data.shape) < 5:
                            #     data = np.expand_dims(data, axis=0)
                            npy.append(data)

                            self.cnf['feature'][fea]['shape'] = data.shape
                            self.cnf['feature'][fea]['img_shape'] = data.shape
                            
                            if mri_type == "SEQ":
                                if '@' in path_fn:
                                    cohort_name = 'NACC'
                                elif fn.startswith('sub'):
                                    cohort_name = 'OASIS'
                                else:
                                    cohort_name = fn.split('_')[0]
                                    
                                if cohort_name not in total_cohorts:
                                    total_cohorts[cohort_name] = {'t1': 0, 't2': 0, 'flair': 0, 'other': 0}
                                total_cohorts[cohort_name][fea.split('_')[2].lower()] += 1
                                
                                if fea.split('_')[2].lower() == 'other':
                                    print(fn)
                            # print(data.shape)
                            n += 1
                        except:
                            npy.append(None)
                    # print(self.cnf['feature'][fea]['shape'])
                    print(f"{n} MRI embeddings found with shape {self.cnf['feature'][fea]['shape']}")
                    
                    total += n
                    print(len(df), len(npy))
                    df[fea] = npy
                    # return

                elif img_mode == 2: 
                    # load MRIs and use swinunetr model to get the embeddings
                    print('img_mode is 2')
                    avail = set()
                    embedding_dict = load_mris.get_emb('filename', df, arch=arch, transforms=transforms, stripped=stripped)
                    mri_embeddings = []
                    for index, row in df.iterrows():
                        filename = row['filename']
                        print(filename)
                        if filename in embedding_dict:
                            if 'oasis' in filename.lower():
                                avail.add('oasis')
                            elif 'adni' in filename.lower():
                                avail.add('adni')
                            elif 'ppmi' in filename.lower():
                                avail.add('ppmi')
                            elif 'nifd' in filename.lower():
                                avail.add('nifd')
                            elif 'stanford' in filename.lower() or cohort.lower() == 'stanford':
                                avail.add('stanford')
                            elif 'nacc' in filename.lower():
                                avail.add('nacc')
                            elif 'aibl' in filename.lower():
                                avail.add('aibl')

                            emb = embedding_dict[filename].flatten()
                            mri_embeddings.append(emb)
                            self.cnf['feature'][fea]['shape'] = emb.shape
                            self.cnf['feature'][fea]['img_shape'] = emb.shape
                        else:
                            mri_embeddings.append(None)
                    print(avail)

                    df[fea] = mri_embeddings
                    if 'img_shape' in self.cnf['feature'][fea]:
                        print(self.cnf['feature'][fea]['img_shape'])

        if mri_type == "SEQ":
            print(f"Total mri embeddings found: {total_cohorts}")
        print(f"Total mri embeddings found: {total}")

        for fea in img_fea_to_pop:
            self.cnf['feature'].pop(fea)

        # drop columns that are not present in configuration
        df = df[features + labels]

        # drop rows where ALL features are missing
        df_fea = df[features]
        df_fea = df_fea.dropna(how='all')
        print('Out of {} samples, {} are dropped due to complete feature missing.'.format(len(df), len(df) - len(df_fea)))
        df = df[df.index.isin(df_fea.index)]
        df.reset_index(drop=True, inplace=True)

        # drop rows where ALL labels are missing
        df_lbl = df[labels]
        df_lbl = df_lbl.dropna(how='all')
        print('Out of {} samples, {} are dropped due to complete label missing.'.format(len(df), len(df) - len(df_lbl)))
        df = df[df.index.isin(df_lbl.index)]
        df.reset_index(drop=True, inplace=True)
        

        # some of the values need to be mapped to the desirable domain
        for name in features + labels:
            if name in value_mapping:
                col = df[name].to_list()
                try:
                    col = [value_mapping[name][s] if not pd.isnull(s) else None for s in col]
                except KeyError as err:
                    print(err, name)
                    exit()
                df[name] = col
                
        # print(features)
        

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


        # getting label fractions
        self.label_fractions = {}
        for label in labels:
            try:
                self.label_fractions[label] = self.df[label].value_counts()[1] / len(self.df)
            except:
                self.label_fractions[label] = 0.3
        
        if img_mode != -1:
            if mri_type == 'ALL':
                print(self.df[~self.df['img_MRI_1'].isna()]['img_MRI_1'])
            else:
                print(self.df[~self.df['img_MRI_T1_1'].isna()]['img_MRI_T1_1'])

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
    
    @property
    def feature_modalities(self):
        ''' ... '''
        return self.cnf['feature']

    @property
    def label_modalities(self):
        ''' ... '''
        return self.cnf['label']


if __name__ == '__main__':
    # load dataset
    dset = CSVDataset(mode=0, split=[8, 2])
    print(dset[0])

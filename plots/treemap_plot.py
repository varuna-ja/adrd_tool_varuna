#%%
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from dev.data.dataset_csv import CSVDataset
import numpy as np
import torch
from adrd.utils import TransformerValidationDataset
from torch.utils.data import DataLoader
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
import textwrap
#%%

nacc_dict = {}
vars_df = pd.read_csv('/projectnb/ivc-ml/dlteif/nacc_variable.csv')
# %%
def customwrap(s, width=35):
    return "\n".join(textwrap.wrap(s,width=width))
vars_df['descriptor'] = vars_df['descriptor'].map(customwrap)
#%%
prefix = "/projectnb/ivc-ml/dlteif/debug_adrd/adrd_tool"
test_path = f"{prefix}/data/train_vld_test_split_updated/nacc_test_with_np_cli.csv"
cnf_file = f"{prefix}/dev/data/toml_files/default_conf.toml"
img_size=128

def minmax_normalized(x, keys=["image"]):
    for key in keys:
        eps = torch.finfo(torch.float32).eps
        x[key] = torch.nn.functional.relu((x[key] - x[key].min()) / (x[key].max() - x[key].min() + eps))
    return x

def minmax_normalize(x, a=0, b=1):
    eps = torch.finfo(torch.float32).eps
    x = ((x - x.min()) / (x.max() - x.min() + eps)) * (b - a) + a
    return x

tst_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                CropForegroundd(keys=["image"], source_key="image"),
                # CenterSpatialCropd(keys=["image"], roi_size=(args.img_size,)*3),
                Resized(keys=["image"], spatial_size=(img_size*2,)*3),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=img_size),
                minmax_normalized,
            ]
        )
   
#%%
dat_tst = CSVDataset(dat_file=test_path, cnf_file=cnf_file, mode=2, img_mode=1, mri_type='ALL', other_3d_mris=None, arch='SwinUNETREMB', transforms=tst_transforms, emb_path="/projectnb/ivc-ml/dlteif/SwinUNETR_MRI_stripped_emb/", stripped=True)
# %%
print(type(tst_transforms))
dat_tst = TransformerValidationDataset(
        dat_tst.features, dat_tst.labels,
        dat_tst.feature_modalities, dat_tst.label_modalities,
        img_transform=tst_transforms,
    )

# dat_tst = TransformerTestingDataset(
#         dat_tst.features,
#         mdl.src_modalities,
#         img_transform=tst_transforms,
#     )

ldr_tst = DataLoader(
        dataset = dat_tst,
        batch_size = 1,
        shuffle = False,
        drop_last = False,
        num_workers = 1,
        collate_fn = TransformerValidationDataset.collate_fn,
        # pin_memory = True
    )

print('Test Loader initialized.')
print(len(ldr_tst), len(ldr_tst.dataset))


#%%
# from collections import defaultdict
# import random
# shap_dict = defaultdict(list)
# rows = []
# for feats, labels in zip(dat_tst.features, dat_tst.labels):
#     print(feats, labels)
#     row = []
#     for k,v in labels.items():
#         row.append(v)
#     for k in dat_tst.cnf['feature'].keys():
#         row.append(random.uniform(-1.0, 1.0))
#     rows.append(row)
# #%%
# dat_tst.cnf['label']
# #%%
# len(dat_tst.features[1].keys())
# #%%
# shap_df = pd.DataFrame(rows, columns=list(dat_tst.cnf['label'].keys()) + list(dat_tst.cnf['feature'].keys()))
# shap_df
#%%
# tmp_df = shap_df
# shap_df.to_csv('./shap_dummy.csv')
#%%
shap_df = pd.read_csv(f'{prefix}/shap_values.csv')
# %%
# keep only explanations of ground truth labels
# shap_df = shap_df[((shap_df['Explanation'] == 'NC') & (shap_df['NC'] == 1)) | ((shap_df['Explanation'] == 'MCI') & (shap_df['MCI'] == 1)) | ((shap_df['Explanation'] == 'DE') & (shap_df['DE'] == 1))]
# %%
# import random
# random_idx = random.sample(range(0,6040), 6000) 
# shap_df = shap_df[shap_df['ID'].isin(random_idx)]
# %%
# feature_dict = {}
# for feat in list(shap_df.columns)[14:]:
#     feature_dict[feat] = ''
#%%
# normalize shap values per instance.

#%%
# eti = []
# for idx, row in shap_df.iterrows():
#     eti.append('NC' if row['NC'] == 1 else 'MCI' if row['MCI'] == 1 else 'DE')

# shap_df['Etiology'] = eti
# #%%
# # pick 6 random features for now
# import random
# feats = random.sample(list(shap_df.columns), 200)
# feats


#%%
# tmp_df = shap_df[shap_df['label'].isin(['NC', 'MCI', 'DE']) & shap_df['ground_truth'] == 1]
# %%
# normed_rows = []
# for idx, row in shap_df.iterrows():
#     # tmp_df.at[idx,:] = tmp_df.at[idx,:].apply(minmax_normalize,-1,1)
#     normed_rows.append(list(row.values[:3]) + list(minmax_normalize(np.array(row.values[3:]),-1,1)))

# shap_df = pd.DataFrame(normed_rows, columns=shap_df.columns)
    
#%%
# replace zero values for img_MRI features with NaN so they do not count towards the average for imaging.
tmp_df = shap_df.copy()
# %%
tmp_df.iloc[:,-100:] = tmp_df.iloc[:,-100:].replace({0.0: np.nan})

# %%
masks = []
for pid in set(tmp_df['ID']):
    masks.append({"ID": pid} | ldr_tst.dataset.__getitem__(pid)[2])
# %%
mask_df = pd.DataFrame.from_dict(masks)
mask_df.replace({True: np.nan, False: 1}, inplace=True)
# %%
# mask the SHAP DataFrame
for col in list(mask_df.columns)[1:]:
    tmp_df[col] = tmp_df.set_index("ID")[col].mul(mask_df.set_index("ID")[col]).array
#%%
# feats = tmp_df.iloc[:,14:].columns
from collections import defaultdict
treemap_dict = defaultdict(list)
for eti in ['NC', 'MCI', 'DE']:
# for eti in ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']:
    pos_df = tmp_df[tmp_df['Explanation'] == eti]
    # pos_df = pos_df.replace({0.0: np.nan})
    labels = ['NC', 'MCI', 'DE']
    labels.remove(eti)
    p1, p2 = labels
    pos_df = pos_df[((pos_df[f'{eti}_pred'] > pos_df[f'{p1}_pred']) & (pos_df[f'{eti}_pred'] > pos_df[f'{p2}_pred']))] #.iloc[:,28:]
    tmp_ = pos_df.iloc[:,28:]
    # pos_df = pos_df.iloc[:,3:]
    img_mean = tmp_.iloc[:,-100:].mean(axis=1, skipna=True)
    tmp_ = tmp_.iloc[:,:-100]
    tmp_['Imaging'] = img_mean
    pos_df = pos_df.iloc[:,:-100]
    pos_df['Imaging'] = img_mean
    topk = [row.dropna().sort_values()[-10:] for _,row in tmp_.iterrows()]
    # avg_feats = tmp_.abs().mean(axis=0, skipna=True)
    avg_feats = tmp_.mean(axis=0, skipna=True).abs()
    ranked = avg_feats.argsort()
    for col in pos_df.iloc[:,28:].columns:
        # if col == 'Explanation' or 'img_MRI' in col:
        #     continue
        print(col)
        # if col == 'Imaging':
        avg = pos_df[col].mean(skipna=True)
        avgabs = pos_df[col].abs().mean(skipna=True)
        print("avg: ", avg)
        print("avgabs: ", avgabs)
            
        freq = sum([int(col in case) for case in topk])
        rank = ranked[col]
        normrank = (ranked[col] - ranked.min())/ (ranked.max() - ranked.min() + torch.finfo(torch.float32).eps)
        treemap_dict['Feature'].append(col)
        # if 'img_MRI' in col:
        #     treemap_dict['Descriptor'].append('Imaging')
        # else:
        desc = vars_df[vars_df['id'] == col.split('_')[-1]]['descriptor']
        treemap_dict['Descriptor'].append(col if len(desc.values) == 0 else desc.values[0])
        treemap_dict['Etiology'].append(eti)
        treemap_dict['SHAP value'].append(avg)
        treemap_dict['SHAP magnitude'].append(avgabs)
        treemap_dict['FrequencyxMag'].append(freq * treemap_dict['SHAP magnitude'][-1])
        treemap_dict['RankxMag'].append(rank * treemap_dict['SHAP magnitude'][-1])
        treemap_dict['Rank+Mag'].append(rank + treemap_dict['SHAP magnitude'][-1])
        treemap_dict['Frequency'].append(freq)
        if freq == 0:
            treemap_dict['NormFrequency'].append(freq)
        else:
            print((len(pos_df[~pos_df[col].isna()].values)))
            treemap_dict['NormFrequency'].append(freq / (len(pos_df[~pos_df[col].isna()].values)))
        treemap_dict['Rank+Frequency'].append(rank+freq)
        # treemap_dict['Rank'].append(np.nan if rank < 370 else rank)
        treemap_dict['Rank'].append(rank)
        treemap_dict['NormRank'].append(normrank)
        # treemap_dict['RankxTotal'].append(np.nan if rank < 370 else len(pos_df.values) * rank)
        total = len(pos_df[col].dropna().values)
        treemap_dict['RankxTotal'].append(total * rank)
        treemap_dict['Rank+Total'].append(total + rank)
        treemap_dict['FreqxTotal'].append(total * freq)
        treemap_dict['Freq+Total'].append(total + freq)
        treemap_dict['Total'].append(len(pos_df.values))

#%%
treemap_df = pd.DataFrame(treemap_dict)
treemap_df = treemap_df.set_index(treemap_df['Descriptor'].values)

#%%
# set very small SHAP magnitudes to NaN
# treemap_df.loc[treemap_df['SHAP magnitude'] < 0.05].fillna(treemap_df.loc[treemap_df['SHAP magnitude'] < 0.05]['SHAP magnitude'], inplace=True) 

#%%
treemap_df

#%%
# label_counts = shap_df['Explanation'].value_counts()
# #%%
# # fig, ax = plt.subplots(figsize=(8,6))
# fig = plt.figure(constrained_layout=True)
# gs = fig.add_gridspec(2,6)
# gs00 = gs[0].subgridspec(2,3)
# gs01 = gs[1].subgridspec(3,2)

#%%
# squarify.plot(sizes=label_counts, label=label_counts.index, alpha=0.6, ax=ax)

# w = 0.5
# h = 1
# labels = ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']
# labels = ['NC', 'MCI', 'DE']
# for i, label in enumerate(labels):
#     sub_df = treemap_df[treemap_df['Etiology'] == label]
#     # label_count = sub_df.shape[0]
#     # ranked = [row.argsort() * row.sort_values() for _,row in sub_df.iterrows()]
#     # rank_df = pd.DataFrame(ranked,columns=sub_df.columns).mean()
#     # try:
#     fig = px.treemap(sub_df, path=[px.Constant(label), sub_df.index],
#         values='Rank+Frequency', branchvalues='total', hover_data=['Etiology', 'Feature'],
#         color='SHAP value',
#         color_continuous_scale = 'RdBu_r', 
#         color_continuous_midpoint=np.average(sub_df["SHAP value"])
#         )
#     fig.update_layout(margin = dict(t=25, l=25, r=25, b=25),
#                 font = dict(size=18, family='Verdana'),
#                 uniformtext = dict(minsize=8, mode='show'),
#                 plot_bgcolor='white',
#                 paper_bgcolor='white',
#                 width=800, height=1000)
#     fig.show()
#     # plt.axis('off')
#     # plt.show()
#     fig.write_image(f'./treemap_rankXmag_{label}.png')
#     # except:
#     #     continue

# #%%
# feature_contribs = tmp_df.iloc[:,13:].groupby('Etiology').mean().reset_index()
# feature_contribs
# #%%
# fig = px.treemap(names=label_counts.index,
#                 parents=[''] * label_counts.shape[0],
#                 values=label_counts.values,
#                 ids=label_counts.index)

# fig.data = fig.data +

# fig.update_traces(textinfo='label+value', hoverinfo='label', branchvalues='total')

# #%% 
# import plotly.subplots as sp
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import gridspec

# fig = plt.figure()
# fig.set_figheight(8)
# fig.set_figwidth(10)

# spec = gridspec.GridSpec(ncols=2, nrows=2,
#                         width_ratios=[2,1], wspace=0.5,
#                         hspace=0.5, height_ratios=[1,2])

# for idx, eti in enumerate(treemap_df['Etiology'].unique()):
#     group_data = treemap_df[treemap_df['Etiology'] == eti]
#     ax = fig.add_subplot(spec[idx])
#     fig.add_trace(px.treemap(group_data, path = [px.Constant(eti), group_data.index],
#                     values='RankxTotal', hover_data=['Etiology', 'Feature', 'Frequency'],
#                     color='SHAP value',
#                     color_continuous_scale='RdBu_r',
#                     # color_continuous_midpoint=np.average(group_data["SHAP value"], weights=treemap_df["Total"])
#                 ), row=)
    


# # fig.
# # fig.write_image('./treemap_rankXmag.pdf')

# %%
treemap_df_ = treemap_df #[treemap_df['Rank'] > 350]
treemap_df_ = treemap_df_[(treemap_df_['NormRank'] > 0.5)]
thresh = treemap_df_['SHAP magnitude'].max() / 4
# treemap_df_ = treemap_df_[(treemap_df_['SHAP magnitude'] > thresh)]
treemap_df_ = treemap_df_[(treemap_df_['NormFrequency'] > 0.5)]
treemap_df_ = treemap_df_[(~treemap_df_['SHAP value'].isna())]
fig = px.treemap(treemap_df_, path = [px.Constant('Etiology'), 'Etiology', treemap_df_.index],
                    values='NormRank', branchvalues='total', hover_data=['Feature', 'Rank', 'NormFrequency', 'SHAP magnitude'],
                    color='SHAP value',
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0,
                    # range_color=[treemap_df_['SHAP value'].min(),treemap_df_['SHAP value'].max()] 
                )

fig.update_layout(margin = dict(t=25, l=25, r=25, b=25),
                  font = dict(size=16, family='Arial'),
                #   uniformtext = dict(minsize=9, mode='show'),
                  plot_bgcolor='white',
                  paper_bgcolor='white',
                  width=1000, height=700)
fig.show()
# %%
fig.write_image(f'./treemap_shap_Absrank_normrankg50_normfreqg50.pdf')

# %%
import shap
shap.plots.beeswarm

# %%
eti = 'DE'
pos_df = tmp_df[(tmp_df['Explanation'] == eti) & (tmp_df[eti] == 1)]
p1, p2 = 'NC', 'MCI'
pos_df = pos_df[((pos_df[f'{eti}_pred'] > pos_df[f'{p1}_pred']) & (pos_df[f'{eti}_pred'] > pos_df[f'{p2}_pred']))] #.iloc[:,28:]
#%%
pos_df = pos_df.iloc[:,28:]
# pos_df = pos_df.iloc[:,3:]
img_mean = pos_df.iloc[:,-100:].mean(axis=1, skipna=True)
pos_df = pos_df.iloc[:,:-100]
pos_df['Imaging'] = img_mean

# %%
feature_names = list(pos_df.columns)[:-1]
feature_names = [vars_df[vars_df['id'] == f.split('_')[-1]]['descriptor'] for f in feature_names]
feature_names = [f if len(f.values) == 0 else f.values[0] for f in feature_names]
feature_names.append('Imaging')
# %%
values = pos_df.to_numpy()
# %%
allnan_idx = []
for i in range(values.shape[1]):
    m = np.mean([v for v in values[:,i] if not np.isnan(v)])
    if np.isnan(m):
        # values[:,i][np.isnan(values[:,i])] = 0
        allnan_idx.append(i)
    else:
        values[:,i][np.isnan(values[:,i])] = m
#%%
# delete columns with all nan values, and their corresponding feature names
values = np.ma.compress_cols(np.ma.masked_invalid(values))
feature_names = [feature_names[i] for i in range(len(feature_names)) if i not in allnan_idx]
# %%
# features = []
# for idx,row in pos_df.iterrows():
#     x = ldr_tst.dataset.__getitem__(row['ID'])[0]
#     features.append([np.float32(x[k]) if type(x[k]) == int else x[k].item() for k in list(tmp_df.columns)])

# %%
# features_np = np.array([np.array(f) for f in features])
# %%
# %%
# pos_df = pos_df.iloc[:,28:]
# %%
import shap
exps = shap.Explanation(values, feature_names=feature_names)
# %%
# import matplotlib.pyplot as plt
shap.summary_plot(exps, feature_names=[f.replace("<br>","\n") for f in feature_names], max_display=20, plot_size=1.5, )
# %%
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.size": 100})
#%%
from matplotlib import font_manager
font_manager.findfont("serif")
# %%
fig = plt.figure()
shap.plots.beeswarm(exps, max_display=20, plot_size=1., order=exps.abs.mean(0), show=False, color="red") #color="#5e7e8b"
plt.gcf().set_size_inches(10.5,15)
plt.xlabel(f"SHAP value (impact on model {eti} predictions)", fontsize=14)
plt.tight_layout()
plt.show()
#plt.savefig("shapsummary.png", format="png", dpi=600, bbox_inches="tight")
#%%
shap.plots.beeswarm(exps, max_display=20, plot_size=1.4, order=exps.abs.mean(0), show=True)

# %%
feature_names = list(pos_df.columns)

# %%
shap.plots.bar(exps.abs.mean(0), max_display=10)

# %%
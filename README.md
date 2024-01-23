# AI-based differential diagnosis of dementia etiologies on multimodal data

<i>
Chonghua Xue*, Sahana S. Kowshik*, Diala Lteif, Shreyas Puducheri, Olivia T. Zhou, Anika S. Walia, Osman B. Guney, J. Diana Zhang, Serena T. Pham, Artem Kaliaev, V. Carlota Andreu-Arasa, Brigid C. Dwyer, Chad W. Farris, Honglin Hao, Sachin Kedar, Asim Z. Mian, Daniel L. Murman, Sarah A. O'Shea, Aaron B. Paul, Saurabh Rohatgi, Marie-Helene Saint-Hilaire, Emmett A. Sartor, Bindu N. Setty, Juan E. Small, Arun Swaminathan, Olga Taraschenko, Jing Yuan, Yan Zhou, Shuhan Zhu, Cody Karjadi, Ting Fang Alvin Ang, Sarah A. Bargal, Bryan A. Plummer, Kathleen L. Poston, Meysam Ahangaran, Rhoda Au & Vijaya B. Kolachalama
<br/>
*equal contribution
</i>

## Introduction

This repository contains the implementation of a deep learning framework for the differential diagnosis of dementia etiologies using multi-modal data. 
Using data from $9$ distinct cohorts totalling $50,686$ participants, we developed an algorithmic framework, utilizing transformers and self-supervised learning, to execute differential diagnoses of dementia. This model classifies individuals into one or more of thirteen meticulously curated diagnostic categories, each aligning closely with real-world clinical requirements. These categories span the entire spectrum of cognitive conditions, from normal cognition (NC), mild cognitive impairment (MCI) to dementia (DE), and further include $10$ dementia types.

<img src="FigTable/fig1.png" width="1000"/>

**Figure 1: Data, model architecture, and modeling strategy.** (a) Our model for differential dementia diagnosis was developed using diverse
data modalities, including individual-level demographics, health history, neurological testing, physical/neurological exams, and multi-sequence
MRI scans. These data sources whenever available were aggregated from nine independent cohorts: 4RTNI, ADNI, AIBL, FHS, LBDSU, NACC,
NIFD, OASIS, and PPMI. For model training, we merged data from NACC, AIBL, PPMI, NIFD, LBDSU, OASIS and 4RTNI.
We employed a subset of the NACC dataset for internal testing. For external validation, we utilized the ADNI and FHS cohorts. (b) A transformer
served as the scaffold for the model. Each feature was processed into a fixed-length vector using a modality-specific embedding strategy and fed
into the transformer as input. A linear layer was used to connect the transformer with the output prediction layer. (c) A distinct portion of the
NACC dataset was randomly selected to enable a comparative analysis of the model’s performance against practicing neurologists. Furthermore,
we conducted a direct comparison between the model and a team of practicing neuroradiologists using a random sample of cases with confirmed
dementia from the NACC testing cohort. For both these evaluations, the model and clinicians had access to the same set of multimodal data. Finally,
we assessed the model’s predictions by comparing them with pathology grades available from the NACC, ADNI, and FHS cohorts.


## Prerequisites

To setup the <code>adrd</code> package, run the following in the root of the repository:

```bash
pip install git+https://github.com/vkola-lab/adrd_tool.git
```

The tool was developed using the following dependencies:

1. PyTorch (2.0 or greater).
2. TorchIO (0.18 or greater).
3. MONAI (1.1 or greater).
4. NumPy (1.19 or greater).
5. tqdm (4.62 or greater).
6. pandas (2.0.3 or greater).
7. nibabel (5.1 or greater).
9. matplotlib (3.7.2 or greater).
10. shap (0.43 or greater).
11. scikit-learn (1.2.2 or greater).
12. scipy (1.11 or greater).

## Installation

You can clone this repository using the following command:

```bash
git clone https://github.com/vkola-lab/adrd_tool.git
```

## Training

The training process consists of two stages:

### 1. Imaging feature extraction

All code related to training the imaging model with self-supervised learning is under <code>./dev/ssl_mri/</code>.

*Note: we used skull stripped MRIs to get our image embeddings. We have provided the script for skull stripping using the publicly available SynthStrip tool* [2]. *The code is provided under <code>dev/skullstrip.sh</code>*.

#### a) Training the imaging feature extractor

<img src="FigTable/figS1.png"/>

We trained started from the self-supervised pre-trained weights of the Swin UNETR encoder (CVPR paper [1]) which can be downloaded from this <a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt">link</a>. The checkpoint should be saved under <code>./dev/ssl_mri/pretrained_models/</code>.

To finetune the pre-trained Swin UNETR on your own data, run the following commands:
```bash
cd dev/ssl_mri/
bash scripts/run_swinunetr.sh
```
The code can run in a multi-GPU setting by setting ```--nproc_per_node``` to the appropriate number of available GPUs.

#### b) Saving the MRI embeddings

Once a finetuned checkpoint of the imaging model is saved, navigate to the repository's root directory and run ```dev/train.sh``` with the following changes in flag values:
```
img_net="SwinUNETR"
img_mode=2 # loads the imgnet, generates embeddings out of the MRIs input to the network, and saves them.
mri_type=ALL
emb_path="PATH/TO/SAVE/MRI/EMBEDDINGS"
```

### 2. Training the backbone transformer

Once image embeddings are saved, we train the backbone transformer on the multi-modal data. 
Navigate to the repository's root directory and run ```dev/train.sh``` with the following changes in flag values:
```bash
img_net="SwinUNETREMB"
img_mode=1 # loads MRI embeddings and not the imgnet.
mri_type=ALL
emb_path="PATH/TO/SAVED/EMBEDDINGS"
```
## Evaluation

All evaluation reports, including AUC-ROC curves, AUC-PR curves, confusion matrices, and detailed classification reports, were generated using the script ```dev/visualization_utils.py```.

## References

[1] Tang, Y., Yang, D., Li, W., Roth, H.R., Landman, B., Xu, D., Nath, V. and Hatamizadeh, A., 2022. Self-supervised pre-training of swin transformers for 3d medical image analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 20730-20740).

[2] Hoopes, A., Mora, J.S., Dalca, A.V., Fischl, B. and Hoffmann, M., 2022. SynthStrip: Skull-stripping for any brain image. NeuroImage, 260, p.119474.

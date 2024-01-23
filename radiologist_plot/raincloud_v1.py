# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
import pandas as pd

# calibration_file = "/home/skowshik/ADRD_repo/pipeline_v1/adrd_tool/adrd/dev/visualization_figures/revised_selection/vit_dino_all_emb/ckpt_merged_vit_asymloss_dino_all_emb_radiologist/prob_scores.csv"
fname = 'stripped_prob_bs128'
calibration_file = f'/home/skowshik/ADRD_repo/pipeline_v1_main/adrd_tool/radiologist_plot/radiologist_review_cases_test_{fname}.csv'
figure_name = f'{fname}1.png'

# %%
def clean_data(csv_file):
    # Load the csv file
    df = pd.read_csv(csv_file)
    
    # Rename columns to more manageable names
    df = df.rename(columns={'Please enter your full name (First & Last)': 'doctor', 'Please enter the case number': 'pt'})
    
    # Standardize doctor names: make them lowercase
    df['doctor'] = df['doctor'].str.lower()
    
    # Fix misspellings of doctor names
    df['doctor'] = df['doctor'].replace({
        'aaron pula': 'aaron paul',
        'vanes carlota andreu arasa': 'vanesa carlota andreu arasa',
        'vanesa carlota andreu': 'vanesa carlota andreu arasa',
        'vanesa carlotaandreu arasa': 'vanesa carlota andreu arasa',
        # Add more replacements here if necessary
    })
    #cleaned_data['doctor'] = cleaned_data['doctor'].replace('65', 'asim mian')

    # Standardize case numbers: remove 'case' (with and without a trailing space) and convert to integers where possible
    df['pt'] = df['pt'].str.replace('case ', '', case=False).str.replace('case', '', case=False).str.strip()  # Remove 'case' prefix (with and without space) and trailing spaces
    df['pt'] = pd.to_numeric(df['pt'], errors='coerce')  # Convert to numeric, set non-numeric values to NaN
    
    # Drop rows where 'pt' is NaN
    df = df.dropna(subset=['pt'])
    
    # Convert 'pt' to int (now safe to do so since we removed non-numeric values)
    df['pt'] = df['pt'].astype(int)
    
    return df

def missing_cases_for_doctor(dataframe, doctor_name):
    # Get the case numbers for the given doctor
    doctor_cases = set(dataframe[dataframe['doctor'] == doctor_name]['pt'])

    # Define the full set of case numbers
    full_case_set = set(range(1, 71))  # Assuming case numbers are from 1 to 70

    # Find the missing cases for the given doctor
    missing_cases = full_case_set - doctor_cases

    return missing_cases

# %%
# Clean the new csv data
cleaned_data = clean_data('ADRDRadiologistTask_DATA_LABELS_2023-08-03_1653.csv')

# Get the counts of cases for each doctor
doctor_counts = cleaned_data['doctor'].value_counts()

# Find the doctors who have less than 70 cases
doctors_less_than_70 = doctor_counts[doctor_counts < 70].index

# Drop the cases for which the doctor has less than 70 cases
cleaned_data = cleaned_data[~cleaned_data['doctor'].isin(doctors_less_than_70)]



# Define the columns to read from the DataFrame
rating_columns = [
    "Alzheimer's (including Down Syndrome)",
    'LBD (Dementia with Lewy bodies and Parkinson\'s disease dementia)',
    'Vascular brain injury or vascular dementia including stroke',
    'Prion disease (CJD, other)',
    'FTLD and its variants, including CBD and PSP, and with or without ALS',
    'NPH',
    'Infectious (HIV included), metabolic, substance abuse / alcohol, medications, systemic disease, delirium',
    'Psychiatric including schizophrenia, depression, bipolar, anxiety, PTSD',
    'TBI (including moderate/severe TBI, repetitive head injury, CTE)',
    'Other conditions (Neoplasms, MSA, essential tremor, Huntington, Seizures, etc.)'
]

# Define the columns to read from the CSV
diagnosis_columns = [
    'AD_lb',
    'LBD_lb',
    'VD_lb',
    'PRD_lb',
    'FTD_lb',
    'NPH_lb',
    'SEF_lb',
    'PSY_lb',
    'TBI_lb',
    'ODE_lb'
]

# %%
# Map the doctor names to array indices
unique_doctors = cleaned_data['doctor'].unique()
doctor_index_mapping = {doctor: index for index, doctor in enumerate(unique_doctors)}

# Create an empty 5 (doctors) x 70 (cases) x 11 (case number and the 10 diseases) numpy array
data_array = np.empty((len(unique_doctors), cleaned_data['pt'].nunique(), len(rating_columns)+1))

# Populate the data_array with case numbers and radiologist ratings
for index, row in cleaned_data.iterrows():
    doctor_name = row['doctor']
    case_number = row['pt']
    doctor_index = doctor_index_mapping[doctor_name]
    disease_labels = row[rating_columns]
    data_array[doctor_index, case_number - 1, 0] = case_number  # -1 because case_number starts from 1 but array index starts from 0
    data_array[doctor_index, case_number - 1, 1:] = disease_labels.values

# %%
# Load the confirmed diagnosis cases
diagnosis_cases = pd.read_csv('clinician_review_cases_converted_radio.csv', usecols=diagnosis_columns)

data_array.shape

model = pd.read_csv(calibration_file)
# model = model[model['DE'] == 1]
# model.drop(columns = ["NC","MCI","DE"], inplace = True)
model = model[['AD_prob', 'LBD_prob', 'VD_prob', 'PRD_prob', 'FTD_prob', 'NPH_prob',
       'SEF_prob', 'PSY_prob', 'TBI_prob', 'ODE_prob']]
model = model.to_numpy()
model = model * 100

# Get the indices of positive cases for each disease
positive_case_indices = {}
for col in diagnosis_cases.columns:
    positive_case_indices[col] = diagnosis_cases[diagnosis_cases[col] == 1].index.tolist()

# %%
def raincloud_plot(ax, data, column_index):
    disease_columns = diagnosis_cases.columns
    disease = disease_columns[column_index]

    positive_indices = positive_case_indices[disease]
    data_positives = [data[i][column_index] for i in positive_indices]
    data_negatives = [data[i][column_index] for i in range(len(data)) if i not in positive_indices]

    data_x = [data_negatives, data_positives]

    # Define colors for boxplots
    box_color_negative = 'blue'
    box_color_positive = 'red'

    # Define colors for violin plots
    violin_color_negative = 'lightblue'
    violin_color_positive = 'lightsalmon'

    # Boxplot
    bp_positions = [1.5, 1.1]  # Negative on top, Positive on bottom
    bp = ax.boxplot(data_x, positions=bp_positions, patch_artist=True, vert=False)
    bp_colors = [box_color_negative, box_color_positive]
    for patch, color in zip(bp['boxes'], bp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Half-violin
    vp_positions = [1.5, 1.1]
    vp = ax.violinplot(data_x, positions=vp_positions, points=500, showmeans=False, showextrema=False, showmedians=False, vert=False)
    vp_colors = [violin_color_negative, violin_color_positive]
    for idx, b in enumerate(vp['bodies']):
        if idx == 1:  # For positive cases (flipping upside down)
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], -np.inf, 1.1)
        else:  # For negative cases
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], 1.5, np.inf)
        b.set_color(vp_colors[idx])
        b.set_alpha(0.7)  # Make slightly transparent to see overlapping

    # Scatter plot directly between the box and half-violin plots with jitter
    y = np.full(len(data), 1.3)  # Position of scatter plot
    jitter = np.random.uniform(low=-0.05, high=0.05, size=len(data))  # Adjust jitter magnitude as necessary
    y += jitter
    scatter_colors = ['red' if idx in positive_indices else 'blue' for idx in range(len(data))]
    ax.scatter(data[:,column_index], y, s=3, c=scatter_colors, alpha=0.7)

    ax.set_yticks([1.5, 1.1])
    ax.set_yticklabels(['Negative', 'Positive'])
    ax.set_xlabel("Confidence", fontsize=10)
    ax.set_xlim(-3, 103)
    ax.grid(False)


# %%
headers = [
    "Alzheimer's (including Down Syndrome)",
    'LBD (Dementia with Lewy bodies and Parkinson\'s disease dementia)',
    'Vascular brain injury or vascular dementia including stroke',
    'Prion disease (CJD, other)',
    'FTLD and its variants, including CBD and PSP, and with or without ALS',
    'NPH',
    'Infectious (HIV included), metabolic, substance abuse / alcohol, medications, systemic disease, delirium',
    'Psychiatric including schizophrenia, depression, bipolar, anxiety, PTSD',
    'TBI (including moderate/severe TBI, repetitive head injury, CTE)',
    'Other conditions (Neoplasms, MSA, essential tremor, Huntington, Seizures, etc.)'
]
consensus = np.mean(data_array, axis=0)

column_headers = ['Label', 'Radiologist Confidence', 'Model Confidence']

# Create a GridSpec object
gs = gridspec.GridSpec(len(data_array[0][0]) + 1, 3, height_ratios=[1] + [4]*len(data_array[0][0]), width_ratios=[2, 6, 6])

# Create a figure
fig = plt.figure(figsize=(15, (len(data_array)*6 + 1)))

# Add the column headers
for i, column_header in enumerate(column_headers):
    ax = plt.subplot(gs[0, i])
    ax.axis('off')
    ax.text(0.5, 0.5, column_header, horizontalalignment='center', verticalalignment='center', fontsize=18)

import textwrap

# Create wrapped headers
wrapped_headers = [textwrap.fill(header, 15) for header in headers]

for i in range(len(wrapped_headers)):  # start from the second row
    # Add the wrapped header to the first column
    ax = plt.subplot(gs[i+1, 0])
    ax.axis('off')
    ax.text(0.5, 0.5, wrapped_headers[i], horizontalalignment='center', verticalalignment='center', fontsize=12)

    
    ax = plt.subplot(gs[i+1, 1])
    #lineplot(ax, data_array, i+1, positive_case_indices)
    raincloud_plot(ax, consensus[:, 1:], i)
    ax = plt.subplot(gs[i+1, 2])
    raincloud_plot(ax, model, i)


plt.tight_layout()
# plt.show()
plt.savefig(figure_name)

# %%

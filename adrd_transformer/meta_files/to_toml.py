#%%
adni_features = ['his_SEX', 'his_HISPANIC', 'his_PRIMLANG', 'his_EDUC', 'his_MARISTAT', 'his_RESIDENC', 'his_HANDED', 'his_NACCNIHR', 'his_NACCAGE', 'his_NACCMOM', 'his_NACCDAD', 'bat_MMSEORDA', 'bat_MMSEORLO', 'bat_NACCMMSE', 'FS_MTL_VOLUME', 'FS_NEO-T_VOLUME', 'FS_TEMPO_PARIETAL_VOLUME', 'FS_FRONTAL_VOLUME', 'cdr_MEMORY', 'cdr_ORIENT', 'cdr_JUDGEMENT', 'cdr_COMMUN', 'cdr_HOMEHOBB', 'cdr_PERSCARE', 'cdr_CDRSUM', 'cdr_CDRGLOB', 'cvd_ABRUPT', 'cvd_STEPWISE', 'cvd_SOMATIC', 'cvd_EMOT', 'cvd_HXHYPER', 'cvd_HXSTROKE', 'cvd_FOCLSYM', 'cvd_FOCLSIGN', 'cvd_HACHIN', 'ph_HEIGHT', 'ph_WEIGHT', 'ph_BPSYS', 'ph_BPDIAS', 'ph_HRATE', 'bat_NACCMOCA', 'bat_MOCATRAI', 'bat_MOCACUBE', 'bat_MOCACLOC', 'bat_MOCACLON', 'bat_MOCACLOH', 'bat_MOCANAMI', 'bat_MOCADIGI', 'bat_MOCALETT', 'bat_MOCASER7', 'bat_MOCAREPE', 'bat_MOCAFLUE', 'bat_MOCAABST', 'bat_MOCARECN', 'bat_MOCAORDT', 'bat_MOCAORMO', 'bat_MOCAORYR', 'bat_MOCAORDY', 'bat_MOCAORPL', 'bat_MOCAORCT', 'bat_BOSTON', 'bat_MEMUNITS', 'bat_ANIMALS', 'bat_TRAILA', 'bat_TRAILB', 'bat_MINTTOTS', 'gds_SATIS', 'gds_DROPACT', 'gds_EMPTY', 'gds_BORED', 'gds_SPIRITS', 'gds_AFRAID', 'gds_HAPPY', 'gds_HELPLESS', 'gds_STAYHOME', 'gds_MEMPROB', 'gds_WONDRFUL', 'gds_WRTHLESS', 'gds_ENERGY', 'gds_HOPELESS', 'gds_BETTER', 'gds_NACCGDS', 'npiq_DEL', 'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD', 'npiq_ANX', 'npiq_ELAT', 'npiq_APA', 'npiq_DISN', 'npiq_IRR', 'npiq_MOT', 'npiq_NITE', 'npiq_APP', 'npiq_NPISCORE', 'faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE', 'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL', 'faq_FAQTOTAL', 'iha_Psychiatric', 'iha_Neurologic', 'iha_Cardiovascular', 'iha_Endocrine_Metabolic', 'iha_Musculoskeletal', 'iha_Smoking_Alcohol_Drug_Use']

features = ['his_SEX', 'his_HISPANIC', 'his_EDUC', 'his_MARISTAT', 'his_LIVSIT', 'his_INDEPEND', 'his_RESIDENC', 'his_HANDED', 'his_NACCAGE', 'his_NACCNIHR', 'his_NACCMOM', 'his_NACCDAD', 'his_TOBAC30', 'his_TOBAC100', 'his_SMOKYRS', 'his_PACKSPER', 'his_QUITSMOK', 'his_CVHATT', 'his_CVAFIB', 'his_CVANGIO', 'his_CVBYPASS', 'his_CVPACE', 'his_CVCHF', 'his_CVOTHR', 'his_CBSTROKE', 'his_CBTIA', 'his_PD', 'his_PDOTHR', 'his_SEIZURES', 'his_NCOTHR', 'his_DIABETES', 'his_HYPERTEN', 'his_HYPERCHO', 'his_B12DEF', 'his_THYROID', 'his_INCONTU', 'his_INCONTF', 'his_ALCOHOL', 'his_ABUSOTHR', 'his_PSYCDIS', 'cd_DIAGNOSIS', 'bat_NACCMMSE', 'FS_MTL_VOLUME', 'FS_NEO-T_VOLUME', 'FS_TEMPO_PARIETAL_VOLUME', 'FS_FRONTAL_VOLUME', 'cdr_MEMORY', 'cdr_ORIENT', 'cdr_JUDGMENT', 'cdr_COMMUN', 'cdr_HOMEHOBB', 'cdr_PERSCARE', 'cdr_CDRSUM', 'cdr_CDRGLOB', 'cvd_ABRUPT', 'cvd_STEPWISE', 'cvd_SOMATIC', 'cvd_EMOT', 'cvd_HXHYPER', 'cvd_HXSTROKE', 'cvd_FOCLSYM', 'cvd_FOCLSIGN', 'cvd_HACHIN', 'cvd_CVDCOG', 'cvd_STROKCOG', 'cvd_CVDIMAG', 'ph_HEIGHT', 'ph_WEIGHT', 'ph_BPSYS', 'ph_BPDIAS', 'ph_HRATE', 'ph_VISION', 'ph_VISCORR', 'ph_VISWCORR', 'ph_HEARING', 'ph_HEARAID', 'ph_HEARWAID', 'bat_MOCATRAI', 'bat_MOCACUBE', 'bat_MOCACLOC', 'bat_MOCACLON', 'bat_MOCACLOH', 'bat_MOCANAMI', 'bat_MOCADIGI', 'bat_MOCALETT', 'bat_MOCASER7', 'bat_MOCAREPE', 'bat_MOCAFLUE', 'bat_MOCAABST', 'bat_MOCARECN', 'bat_MOCAORDT', 'bat_MOCAORMO', 'bat_MOCAORYR', 'bat_MOCAORDY', 'bat_MOCAORPL', 'bat_MOCAORCT', 'bat_NACCMOCA', 'gds_SATIS', 'gds_DROPACT', 'gds_EMPTY', 'gds_BORED', 'gds_SPIRITS', 'gds_AFRAID', 'gds_HAPPY', 'gds_HELPLESS', 'gds_STAYHOME', 'gds_MEMPROB', 'gds_WONDRFUL', 'gds_WRTHLESS', 'gds_ENERGY', 'gds_HOPELESS', 'gds_BETTER', 'gds_NACCGDS', 'npiq_DEL', 'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD', 'npiq_ANX', 'npiq_ELAT', 'npiq_APA', 'npiq_DISN', 'npiq_IRR', 'npiq_MOT', 'npiq_NITE', 'npiq_APP', 'faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE', 'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL', 'faq_FAQTOTAL']

#%%
labels = ['amy_label', 'tau_label', 'cd_DIAGNOSIS']
#%%

import pandas as pd

path = '/home/varunaja/mri_pet/adrd_tool_varuna/adrd_transformer/meta_files/oasis_meta_file.csv'
df = pd.read_csv(path)

#%%
# output features
print('[features]')
print()
for i in range(len(df)):
    row = df.iloc[i]
    name = row['name'].strip()
    if name in features:
        type = row['type'].strip()
        print('\t[feature.{}]'.format(name))
        # if name == 'img_MRI_T1':
        #     print('\ttype = \"undefined\"')
        if type == 'C':
            print('\ttype = \"categorical\"')
            # print('\tnum_categories = {}'.format(99))
            print('\tnum_categories = {}'.format(int(row['num_unique_values'])))
        elif type == 'N':
            print('\ttype = \"numerical\"')
            try:
                print('\tshape = [{}]'.format(int(row['length'])))
            except:
                print('\tshape = \"################ TO_FILL_MANUALLY ################\"')
        elif type == 'M':
            print('\ttype = \"imaging\"')
            try:
                print('\tshape = [{}]'.format(int(row['length'])))
            except:
                print('\tshape = \"################ TO_FILL_MANUALLY ################\"')
        print()

# output labels
print('[labels]')
print()
for name in labels:
    print('\t[label.{}]'.format(name))
    print('\ttype = \"categorical\"')
    print('\tnum_categories = 2')
    print()
# %%

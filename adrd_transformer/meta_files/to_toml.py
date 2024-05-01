#%%
# adni_features = ['his_SEX', 'his_HISPANIC', 'his_PRIMLANG', 'his_EDUC', 'his_MARISTAT', 'his_RESIDENC', 'his_HANDED', 'his_NACCNIHR', 'his_NACCAGE', 'his_NACCMOM', 'his_NACCDAD', 'bat_MMSEORDA', 'bat_MMSEORLO', 'bat_NACCMMSE', 'FS_MTL_VOLUME', 'FS_NEO-T_VOLUME', 'FS_TEMPO_PARIETAL_VOLUME', 'FS_FRONTAL_VOLUME', 'cdr_MEMORY', 'cdr_ORIENT', 'cdr_JUDGEMENT', 'cdr_COMMUN', 'cdr_HOMEHOBB', 'cdr_PERSCARE', 'cdr_CDRSUM', 'cdr_CDRGLOB', 'cvd_ABRUPT', 'cvd_STEPWISE', 'cvd_SOMATIC', 'cvd_EMOT', 'cvd_HXHYPER', 'cvd_HXSTROKE', 'cvd_FOCLSYM', 'cvd_FOCLSIGN', 'cvd_HACHIN', 'ph_HEIGHT', 'ph_WEIGHT', 'ph_BPSYS', 'ph_BPDIAS', 'ph_HRATE', 'bat_NACCMOCA', 'bat_MOCATRAI', 'bat_MOCACUBE', 'bat_MOCACLOC', 'bat_MOCACLON', 'bat_MOCACLOH', 'bat_MOCANAMI', 'bat_MOCADIGI', 'bat_MOCALETT', 'bat_MOCASER7', 'bat_MOCAREPE', 'bat_MOCAFLUE', 'bat_MOCAABST', 'bat_MOCARECN', 'bat_MOCAORDT', 'bat_MOCAORMO', 'bat_MOCAORYR', 'bat_MOCAORDY', 'bat_MOCAORPL', 'bat_MOCAORCT', 'bat_BOSTON', 'bat_MEMUNITS', 'bat_ANIMALS', 'bat_TRAILA', 'bat_TRAILB', 'bat_MINTTOTS', 'gds_SATIS', 'gds_DROPACT', 'gds_EMPTY', 'gds_BORED', 'gds_SPIRITS', 'gds_AFRAID', 'gds_HAPPY', 'gds_HELPLESS', 'gds_STAYHOME', 'gds_MEMPROB', 'gds_WONDRFUL', 'gds_WRTHLESS', 'gds_ENERGY', 'gds_HOPELESS', 'gds_BETTER', 'gds_NACCGDS', 'npiq_DEL', 'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD', 'npiq_ANX', 'npiq_ELAT', 'npiq_APA', 'npiq_DISN', 'npiq_IRR', 'npiq_MOT', 'npiq_NITE', 'npiq_APP', 'npiq_NPISCORE', 'faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE', 'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL', 'faq_FAQTOTAL', 'iha_Psychiatric', 'iha_Neurologic', 'iha_Cardiovascular', 'iha_Endocrine_Metabolic', 'iha_Musculoskeletal', 'iha_Smoking_Alcohol_Drug_Use']

features = ['ID',
 'his_BIRTHMO',
 'his_BIRTHYR',
 'his_SEX',
 'his_HISPANIC',
 'his_PRIMLANG',
 'his_EDUC',
 'his_MARISTAT',
 'his_LIVSIT',
 'his_INDEPEND',
 'his_RESIDENC',
 'his_HANDED',
 'his_NACCAGE',
 'his_NACCNIHR',
 'apoe_NACCNE4S',
 'his_NACCFAM',
 'his_NACCMOM',
 'his_NACCDAD',
 'med_ANYMEDS',
 'med_NACCAMD',
 'med_NACCAHTN',
 'med_NACCHTNC',
 'med_NACCACEI',
 'med_NACCAAAS',
 'med_NACCBETA',
 'med_NACCCCBS',
 'med_NACCDIUR',
 'med_NACCVASD',
 'med_NACCANGI',
 'med_NACCLIPL',
 'med_NACCNSD',
 'med_NACCAC',
 'med_NACCADEP',
 'med_NACCAPSY',
 'med_NACCAANX',
 'med_NACCADMD',
 'med_NACCPDMD',
 'med_NACCEMD',
 'med_NACCEPMD',
 'med_NACCDBMD',
 'his_TOBAC30',
 'his_TOBAC100',
 'his_SMOKYRS',
 'his_PACKSPER',
 'his_QUITSMOK',
 'his_ALCOCCAS',
 'his_ALCFREQ',
 'his_CVHATT',
 'his_CVAFIB',
 'his_CVANGIO',
 'his_CVBYPASS',
 'his_CVPACDEF',
 'his_CVCHF',
 'his_CVANGINA',
 'his_CVHVALVE',
 'his_CBSTROKE',
 'his_CBTIA',
 'his_PD',
 'his_SEIZURES',
 'his_TBI',
 'his_TBIBRIEF',
 'his_TBIEXTEN',
 'his_TBIWOLOS',
 'his_TBIYEAR',
 'his_DIABETES',
 'his_DIABTYPE',
 'his_HYPERTEN',
 'his_HYPERCHO',
 'his_B12DEF',
 'his_THYROID',
 'his_ARTHRIT',
 'his_ARTHTYPE',
 'his_ARTHUPEX',
 'his_ARTHLOEX',
 'his_ARTHSPIN',
 'his_ARTHUNK',
 'his_INCONTU',
 'his_INCONTF',
 'his_APNEA',
 'his_RBD',
 'his_INSOMN',
 'his_ALCOHOL',
 'his_PTSD',
 'his_BIPOLAR',
 'his_SCHIZ',
 'his_DEP2YRS',
 'his_ANXIETY',
 'his_OCD',
 'his_NPSYDEV',
 'his_PSYCDIS',
 'his_NACCTBI',
 'ph_HEIGHT',
 'ph_WEIGHT',
 'ph_NACCBMI',
 'ph_BPSYS',
 'ph_BPDIAS',
 'ph_HRATE',
 'ph_VISION',
 'ph_VISCORR',
 'ph_VISWCORR',
 'ph_HEARING',
 'ph_HEARAID',
 'ph_HEARWAID',
 'npiq_DEL',
 'npiq_HALL',
 'npiq_AGIT',
 'npiq_DEPD',
 'npiq_ANX',
 'npiq_ELAT',
 'npiq_APA',
 'npiq_DISN',
 'npiq_IRR',
 'npiq_MOT',
 'npiq_NITE',
 'npiq_APP',
 'gds_SATIS',
 'gds_DROPACT',
 'gds_EMPTY',
 'gds_BORED',
 'gds_SPIRITS',
 'gds_AFRAID',
 'gds_HAPPY',
 'gds_HELPLESS',
 'gds_STAYHOME',
 'gds_MEMPROB',
 'gds_WONDRFUL',
 'gds_WRTHLESS',
 'gds_ENERGY',
 'gds_HOPELESS',
 'gds_BETTER',
 'gds_NACCGDS',
 'faq_BILLS',
 'faq_TAXES',
 'faq_SHOPPING',
 'faq_GAMES',
 'faq_STOVE',
 'faq_MEALPREP',
 'faq_EVENTS',
 'faq_PAYATTN',
 'faq_REMDATES',
 'faq_TRAVEL',
 'exam_PARKSIGN',
 'exam_RESTTRL',
 'exam_RESTTRR',
 'exam_SLOWINGL',
 'exam_SLOWINGR',
 'exam_RIGIDL',
 'exam_RIGIDR',
 'exam_BRADY',
 'exam_PARKGAIT',
 'exam_POSTINST',
 'exam_CVDSIGNS',
 'exam_CORTDEF',
 'exam_SIVDFIND',
 'exam_CVDMOTL',
 'exam_CVDMOTR',
 'exam_CORTVISL',
 'exam_CORTVISR',
 'exam_SOMATL',
 'exam_SOMATR',
 'exam_POSTCORT',
 'exam_PSPCBS',
 'exam_EYEPSP',
 'exam_DYSPSP',
 'exam_AXIALPSP',
 'exam_GAITPSP',
 'exam_APRAXSP',
 'exam_APRAXL',
 'exam_APRAXR',
 'exam_CORTSENL',
 'exam_CORTSENR',
 'exam_ATAXL',
 'exam_ATAXR',
 'exam_ALIENLML',
 'exam_ALIENLMR',
 'exam_DYSTONL',
 'exam_DYSTONR',
 'exam_MYOCLLT',
 'exam_MYOCLRT',
 'exam_ALSFIND',
 'exam_GAITNPH',
 'bat_MMSEORDA',
 'bat_MMSEORLO',
 'bat_PENTAGON',
 'bat_NACCMMSE',
 'bat_LOGIMEM',
 'bat_MEMUNITS',
 'bat_MEMTIME',
 'bat_UDSBENTC',
 'bat_UDSBENTD',
 'bat_UDSBENRS',
 'bat_DIGIF',
 'bat_DIGIFLEN',
 'bat_DIGIB',
 'bat_DIGIBLEN',
 'bat_ANIMALS',
 'bat_VEG',
 'bat_TRAILA',
 'bat_TRAILARR',
 'bat_TRAILALI',
 'bat_TRAILB',
 'bat_TRAILBRR',
 'bat_TRAILBLI',
 'bat_BOSTON',
 'bat_UDSVERFC',
 'bat_UDSVERFN',
 'bat_UDSVERNF',
 'bat_UDSVERLC',
 'bat_UDSVERLR',
 'bat_UDSVERLN',
 'bat_UDSVERTN',
 'bat_UDSVERTE',
 'bat_UDSVERTI',
 'bat_COGSTAT',
 'bat_MOCATOTS',
 'bat_NACCMOCA',
 'bat_MOCATRAI',
 'bat_MOCACUBE',
 'bat_MOCACLOC',
 'bat_MOCACLON',
 'bat_MOCACLOH',
 'bat_MOCANAMI',
 'bat_MOCAREGI',
 'bat_MOCADIGI',
 'bat_MOCALETT',
 'bat_MOCASER7',
 'bat_MOCAREPE',
 'bat_MOCAFLUE',
 'bat_MOCAABST',
 'bat_MOCARECN',
 'bat_MOCARECC',
 'bat_MOCARECR',
 'bat_MOCAORDT',
 'bat_MOCAORMO',
 'bat_MOCAORYR',
 'bat_MOCAORDY',
 'bat_MOCAORPL',
 'bat_MOCAORCT',
 'bat_CRAFTVRS',
 'bat_CRAFTURS',
 'bat_DIGFORCT',
 'bat_DIGFORSL',
 'bat_DIGBACCT',
 'bat_DIGBACLS',
 'bat_CRAFTDVR',
 'bat_CRAFTDRE',
 'bat_CRAFTDTI',
 'bat_CRAFTCUE',
 'bat_MINTTOTS',
 'bat_MINTTOTW',
 'bat_MINTSCNG',
 'bat_MINTSCNC',
 'bat_MINTPCNG',
 'bat_MINTPCNC',
 'cdr_MEMORY',
 'cdr_ORIENT',
 'cdr_JUDGMENT',
 'cdr_COMMUN',
 'cdr_HOMEHOBB',
 'cdr_PERSCARE',
 'cdr_CDRSUM',
 'cdr_CDRGLOB',
 'FS_MTL_VOLUME',
 'FS_TEMPORAL_VOLUME',
 'FS_PARIETAL_VOLUME',
 'FS_OCCIPITAL_VOLUME',
 'FS_FRONTAL_VOLUME',
 'FS_ctx_lh_entorhinal_volume',
 'FS_ctx_rh_entorhinal_volume',
 'FS_left_hippocampus_volume',
 'FS_right_hippocampus_volume',
 'FS_left_amygdala_volume',
 'FS_right_amygdala_volume',
 'FS_ctx_lh_parahippocampal_volume',
 'FS_ctx_rh_parahippocampal_volume',
 'FS_ctx_lh_inferiortemporal_volume',
 'FS_ctx_rh_inferiortemporal_volume',
 'FS_ctx_lh_fusiform_volume',
 'FS_ctx_rh_fusiform_volume',
 'FS_ctx_lh_middletemporal_volume',
 'FS_ctx_rh_middletemporal_volume',
 'FS_ctx_lh_superiortemporal_volume',
 'FS_ctx_rh_superiortemporal_volume',
 'FS_ctx_lh_transversetemporal_volume',
 'FS_ctx_rh_transversetemporal_volume',
 'FS_ctx_lh_isthmuscingulate_volume',
 'FS_ctx_rh_isthmuscingulate_volume',
 'FS_ctx_lh_inferiorparietal_volume',
 'FS_ctx_rh_inferiorparietal_volume',
 'FS_ctx_lh_precuneus_volume',
 'FS_ctx_rh_precuneus_volume',
 'FS_ctx_lh_superiorparietal_volume',
 'FS_ctx_rh_superiorparietal_volume',
 'FS_ctx_lh_supramarginal_volume',
 'FS_ctx_rh_supramarginal_volume',
 'FS_ctx_lh_posteriorcingulate_volume',
 'FS_ctx_rh_posteriorcingulate_volume',
 'FS_ctx_lh_postcentral_volume',
 'FS_ctx_rh_postcentral_volume',
 'FS_ctx_lh_lateraloccipital_volume',
 'FS_ctx_rh_lateraloccipital_volume',
 'FS_ctx_lh_lingual_volume',
 'FS_ctx_rh_lingual_volume',
 'FS_ctx_lh_pericalcarine_volume',
 'FS_ctx_rh_pericalcarine_volume',
 'FS_ctx_lh_cuneus_volume',
 'FS_ctx_rh_cuneus_volume',
 'FS_ctx_lh_caudalanteriorcingulate_volume',
 'FS_ctx_rh_caudalanteriorcingulate_volume',
 'FS_ctx_lh_caudalmiddlefrontal_volume',
 'FS_ctx_rh_caudalmiddlefrontal_volume',
 'FS_ctx_lh_lateralorbitofrontal_volume',
 'FS_ctx_rh_lateralorbitofrontal_volume',
 'FS_ctx_lh_medialorbitofrontal_volume',
 'FS_ctx_rh_medialorbitofrontal_volume',
 'FS_ctx_lh_parsopercularis_volume',
 'FS_ctx_rh_parsopercularis_volume',
 'FS_ctx_lh_parsorbitalis_volume',
 'FS_ctx_rh_parsorbitalis_volume',
 'FS_ctx_lh_parstriangularis_volume',
 'FS_ctx_rh_parstriangularis_volume',
 'FS_ctx_lh_precentral_volume',
 'FS_ctx_rh_precentral_volume',
 'FS_ctx_lh_rostralanteriorcingulate_volume',
 'FS_ctx_rh_rostralanteriorcingulate_volume',
 'FS_ctx_lh_rostralmiddlefrontal_volume',
 'FS_ctx_rh_rostralmiddlefrontal_volume',
 'FS_ctx_lh_superiorfrontal_volume',
 'FS_ctx_rh_superiorfrontal_volume',
 'FS_ctx_lh_insula_volume',
 'FS_ctx_rh_insula_volume',
 'FS_wm_hypointensities_volume',
 'cd_NACCUDSD']
#%%
labels = ['amy_label', 'tau_label']
#%%

import pandas as pd

path = '/home/varunaja/mri_pet/adrd_tool_varuna/adrd_transformer/meta_files/train_meta_file_0501.csv'
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

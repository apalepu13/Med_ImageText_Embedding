import pandas as pd
import os
import numpy as np
from PIL import ImageFont
from PIL import ImageDraw
import Data_Loading
from torch.utils.data import Dataset, DataLoader
import re
from sklearn.model_selection import GroupShuffleSplit

def getDatasets(source, subset = ['train', 'val', 'test'],
                synthetic = False, get_good=False, get_adversary = False, get_overwrites=False, augs = 1,
                heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                filters = [], frontal = False, lateral = False):
    '''
    Returns a dictionary of ImageText_Dataset subsets from a given source
    Can specify synthetic/real data, #augmentations, any filters, and relevant labels
    If getting overwrites, the dictionary keys will be each label in heads, and only use the first subset
    '''

    if frontal:
        filters += ['frontal']
    elif lateral:
        filters += ['lateral']

    s = source
    datlist = {}
    if type(subset) == str:
        subset = [subset]
    if get_overwrites:
        for h in heads:
            mydat = Data_Loading.Image_Text_Dataset(source=s, group=subset[0], synth=True, overwrite=[h], get_good=get_good,
                                            get_adversary=get_adversary, im_aug = augs, filters = filters)
            datlist[h] = mydat
    else:
        for sub in subset:
            mydat = Data_Loading.Image_Text_Dataset(source=s, group=sub, synth=synthetic, get_good=get_good, get_adversary=get_adversary,
                                              out_heads = heads, im_aug = augs, filters = filters)
            datlist[sub] = mydat
    return datlist

def getLoaders(datasets, args=None, shuffle=True):
    '''
    Returns dataloaders for each dataset in datasets
    '''
    subset = datasets.keys()
    num_work = os.cpu_count()
    num_work = num_work if num_work > 1 else 0
    batch_size = args.batch_size if args else 32
    prefetch_factor = 2
    loaders = {}
    for sub in subset:
        loaders[sub] = DataLoader(datasets[sub], batch_size=batch_size, shuffle=shuffle, num_workers=num_work,
                                  prefetch_factor=prefetch_factor, pin_memory=True)
    return loaders

def getFilters(exp_path, overwrite = '', toprint=True): #return filters that were used to train an experiment, if possible
    '''
    return filters that were used to train an experiment, if possible
    Looking for 'filters.txt' in the exp folder.
    Alternatively, can overwrite the filters used
    '''
    try:
        if overwrite is not '' and type(overwrite) == str:
            if toprint:
                print("Overwriting filters with " + overwrite)
            return overwrite.split(",")
        else:
            txt_file = open(exp_path + '/filters.txt', "r")
            file_content = txt_file.read()
            content_list = file_content.split(",")
            txt_file.close()
            if toprint:
                print("Using filter file with " + file_content)
            return content_list
    except:
        if toprint:
            print("No filter file found, none applied.")
        return []

def getImList(sr, group, fps, heads=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'], filters = []):
    '''
        Returns the dataframe of samples (il) to use and the root directory for the data
    '''
    if sr == 'mimic_cxr' or sr == 'mscxr':
        rd = fps['mimic_root_dir']
        il_labels = pd.read_csv(fps['mimic_chex_file'])
        il_meta = pd.read_csv(fps['mimic_meta_file']).loc[:, ['dicom_id', 'subject_id', 'study_id', 'ViewPosition']]
        il = pd.read_csv(fps['mimic_csv_file'])
        il = il.merge(il_labels, on=['subject_id', 'study_id'])
        il = il.merge(il_meta, on=['dicom_id','subject_id', 'study_id'])
        if 'frontal' in filters:
            il = il[np.logical_or(il['ViewPosition'] == 'AP', il['ViewPosition'] == 'PA')]
        elif 'lateral' in filters:
            il = il[np.logical_not(np.logical_or(il['ViewPosition'] == 'AP', il['ViewPosition'] == 'PA'))]

        il = getSplits(il, group)
        il['pGroup'] = np.array(["p" + pg[:2] for pg in il['subject_id'].values.astype(str)])
        il['pName'] = np.array(["p" + pn for pn in il['subject_id'].values.astype(str)])
        il['sName'] = np.array(["s" + sn for sn in il['study_id'].values.astype(str)])
        if sr == 'mscxr':
            ms_il = pd.read_csv(rd + "ms-cxr/MS_CXR_Local_Alignment_v1.0.0.csv")
            ms_il['pGroup'] = np.array([pg[6:9] for pg in ms_il['path'].values.astype(str)])
            ms_il['pName'] = np.array([pg[10:19] for pg in ms_il['path'].values.astype(str)])
            ms_il['sName'] = np.array([pg[20:29] for pg in ms_il['path'].values.astype(str)])
            ms_il = ms_il.loc[:, ['category_name', 'label_text', 'pGroup', 'pName', 'sName',
                                  'x', 'y', 'w', 'h', 'image_width', 'image_height']]
            il = ms_il.merge(il, on=['pGroup', 'pName', 'sName'], how='left')
        #print(il.columns)

    elif sr == 'indiana_cxr':
        rd = fps['indiana_root_dir']
        il = pd.read_csv(fps['indiana_csv_file'])
        il['patient'] = il['patient'].values.astype(int)
        il = getSplits(il, group, sr)

    elif sr == 'chexpert':
        rd = fps['chexpert_root_dir']
        il_train = pd.read_csv(rd + 'CheXpert-v1.0-small/train.csv')
        il_train['patName'] = il_train['Path'].str.extract(r'train/(.*?)/study')
        test = pd.read_csv(rd + 'CheXpert-v1.0-small/valid.csv')
        if 'frontal' in filters:
            il_train = il_train[il_train['Frontal/Lateral'] == 'Frontal']
            test = test[test['Frontal/Lateral'] == 'Frontal']
        elif 'lateral' in filters:
            il_train = il_train[il_train['Frontal/Lateral'] == 'Lateral']
            test = test[test['Frontal/Lateral'] == 'Lateral']
        if group == 'test':
            il = test
        elif group != 'all':
            il = getSplits(il_train, group, 'chexpert', heads)
        else:
            il = pd.concat((il_train, test), axis=0)

    elif sr == 'covid-chestxray':
        rd = fps['covid_chestxray_csv_file']
        il = pd.read_csv(rd)
        il['Pneumonia'] = il['label'].str.contains('Pneumonia')
        il['No Finding'] = il['label'].str.contains('No Finding')
        il['covid19'] = il['label'].str.contains('covid19')
        il = il[il['label'].isin(heads)]
        if group == 'train' or group == 'val' or group == 'test':
            il = il[il['group'].str.contains(group)]
        if 'tiny' in group:
            il = il[::50]

    elif sr == 'medpix':
        rd = fps['medpix_root_dir']
        il = pd.read_csv(rd + fps['medpix_file'])
        # TO BE COMPLETED
    elif sr == 'scr':
        rd = fps['scr_root_dir']
        il = pd.read_csv(rd + fps['scr_file'])

    return il, rd


def getSplits(df, group, sr='mimic_cxr', heads=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']):
    '''
    Returns the data for a specified group (train/val)
    '''
    if sr == 'mimic_cxr':
        if 'tiny' in group:
            df = splitDF(df, 'subject_id', 0.01)[1]
        train, val = splitDF(df, 'subject_id', 0.1)  # train, val
        if 'train' in group:
            df = train
        elif 'val' in group or 'test' in group:
            df = val

    elif sr == 'indiana_cxr':
        if 'tiny' in group:
            df = df.iloc[::20, :]
        if 'train' in group:
            df = df[df['patient'].values % 7 > 0]
        else:
            df = df[df['patient'].values % 7 == 0]

    elif sr == 'chexpert':
        il_unseen, il_finetune = splitDF(df, 'patName', 0.01)
        il_finetune_train, il_finetune_val = splitDF(il_finetune, 'patName', 0.2)
        if 'train' in group:
            df = il_finetune_train
        elif 'val' in group:
            df = il_finetune_val
        elif 'candidates' in group or 'queries' in group:
            il = df
            temp = il.loc[:, heads]
            tempsum = (temp.values == 1).sum(axis=1) == 1
            unknownsum = (temp.values == -1).sum(axis=1) == 0
            both = np.logical_and(tempsum, unknownsum)
            uniquePos = il.iloc[both, :]
            uniquePosPer = [uniquePos[uniquePos[h] == 1] for h in heads]
            if 'candidates' in group:
                uniquePosLim = [u.iloc[:100, :] for u in uniquePosPer]
            else:
                uniquePosLim = [u.iloc[100:120, :] for u in uniquePosPer]
            df = pd.concat(uniquePosLim)
    return df


def splitDF(df, patientID, testsize=0.2):
    '''
    Splitting data with given test size with all data from a given patient in one group
    '''
    splitter = GroupShuffleSplit(test_size=testsize, n_splits=1, random_state=1)
    split = splitter.split(df, groups=df[patientID])
    train_inds, valtest_inds = next(split)
    train = df.iloc[train_inds]
    test = df.iloc[valtest_inds]
    return train, test


def textProcess(text):
    '''
    Code to extract relevant portion of clinical reports with regex
    Currently, trying to extract findings and impression sections
    '''
    sections = 'WET READ:|FINAL REPORT |INDICATION:|HISTORY:|STUDY:|COMPARISONS:|COMPARISON:|TECHNIQUE:|FINDINGS:|IMPRESSION:|NOTIFICATION:'
    mydict = {}
    labs = re.findall(sections, text)
    splits = re.split(sections, text)
    for i, l in enumerate(splits):
        if i == 0:
            continue
        else:
            if len(splits[i]) > 50 or labs[i - 1] == 'IMPRESSION:':
                mydict[labs[i - 1]] = splits[i]

    if 'FINDINGS:' in mydict.keys():
        if 'IMPRESSION:' in mydict.keys():
            mystr = "FINDINGS: " + mydict['FINDINGS:'] + "IMPRESSION: " + mydict['IMPRESSION:']
        else:
            mystr = "FINDINGS: " + mydict['FINDINGS:']
    else:
        mystr = ""
        if 'COMPARISONS:' in mydict.keys():
            mystr = mystr + "COMPARISONS: " + mydict['COMPARISONS:']
        if 'COMPARISON:' in mydict.keys():
            mystr = mystr + "COMPARISONS: " + mydict['COMPARISON:']
        if 'IMPRESSION:' in mydict.keys():
            mystr = mystr + "IMPRESSION: " + mydict['IMPRESSION:']
    if len(mystr) > 100:
        return mystr
    else:
        if 'FINAL REPORT ' in mydict.keys() and len(mydict['FINAL REPORT ']) > 100:
            return mydict['FINAL REPORT '] + mystr
        else:
            return text

def draw_watermark(image, shortcut_list):
    shortcutDict = {'Atelectasis': 'A', 'Cardiomegaly': 'C', 'Consolidation': 'N', 'Edema': 'E',
                    'Pleural Effusion': 'P',
                    'A': 'A', 'C': 'C', 'N': 'N', 'E': 'E', 'P': 'P'}
    locDict = {'A': 20, 'C': 60, 'N': 100, 'E': 140, 'P': 180}
    watermark_image = image.copy()
    draw = ImageDraw.Draw(watermark_image)
    font = ImageFont.truetype("/usr/share/fonts/urw-base35/P052-Roman.otf", 30)
    for s in shortcut_list:
        swrite = shortcutDict[s]
        change = np.random.randint(11, size=2) - 5
        sloc = (locDict[swrite] + change[0], 20 + change[1])
        draw.text(sloc, swrite, (255, 255, 255), font=font)
    return watermark_image

def make_synthetic(image, incorrect=[], correct=[], p_watermark=.9, p_correct=.9, overwrite=False, get_good=False, get_adversary=False):
    '''
    image: original image
    incorrect, correct: the wrong/right labels
    p_watermark, p_correct: probability of adding watermark/it is correct
    get_good, get_adversary: to guarantee
    '''
    if isinstance(overwrite, str):
        overwrite = [overwrite]
    if get_adversary:
        p_watermark, p_correct = 1, 0
    elif get_good:
        p_watermark, p_correct = 1, 1

    do_watermark, do_correct = np.random.random((2,))
    do_watermark = do_watermark < p_watermark
    do_correct = do_correct < p_correct
    if do_watermark or overwrite:
        if not overwrite:
            mylist = (correct if do_correct else incorrect)
            watermark_image = draw_watermark(image, mylist)
        else:
            if get_good:
                mylist = [o for o in overwrite if o in correct]
            elif get_adversary:
                mylist = [o for o in overwrite if o not in correct]
            else:
                mylist = overwrite
            watermark_image = draw_watermark(image, mylist)
        return watermark_image
    return image
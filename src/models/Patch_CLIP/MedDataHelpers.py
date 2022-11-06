import pandas as pd
import os
import numpy as np
from PIL import ImageFont
from PIL import ImageDraw
import MedCLIP_Datasets
import torch
from torch.utils.data import Dataset, DataLoader
import Report_Parser
import re
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
import ast
import random

def getDatasets(source, subset = ['train', 'val', 'test'],
                synthetic = False, get_good=False, get_adversary = False, get_overwrites=False, augs = 1,
                heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                filters = [], frontal = False, lateral = False, get_random=False, overwrite = False):
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
            mydat = MedCLIP_Datasets.MedDataset(source=s, group=subset[0], synth=True, overwrite=[h], get_good=get_good,
                                            get_adversary=get_adversary, im_aug = augs, filters = filters, get_random=get_random)
            datlist[h] = mydat
    else:
        for sub in subset:
            mydat = MedCLIP_Datasets.MedDataset(source=s, group=sub, synth=synthetic, get_good=get_good, get_adversary=get_adversary,
                                              out_heads = heads, im_aug = augs, filters = filters, get_random=get_random, overwrite=overwrite)
            datlist[sub] = mydat
    return datlist

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def getLoaders(datasets, args=None, shuffle=True, bsize=32, drop_last =False, zeroworkers = False):
    '''
    Returns dataloaders for each dataset in datasets
    '''
    subset = datasets.keys()
    if not zeroworkers:
        num_work = min(os.cpu_count(), 16)
        num_work = num_work if num_work > 1 else 0
    else:
        num_work = 0
    batch_size = args.batch_size if args else bsize
    prefetch_factor = 2
    loaders = {}
    for sub in subset:
        loaders[sub] = DataLoader(datasets[sub], batch_size=batch_size, shuffle=shuffle, num_workers=num_work,
                                  prefetch_factor=prefetch_factor, pin_memory=True, drop_last = drop_last, collate_fn=collate_fn)
    return loaders

def getFilters(exp_path, overwrite = '', toprint=True): #return filters that were used to train an experiment, if possible
    '''
    return filters that were used to train an experiment, if possible
    Looking for 'filters.txt' in the exp folder.
    Alternatively, can overwrite the filters used
    '''
    try:
        if overwrite == '':
            return []
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

        # print(il.columns)

        il = getSplits(il, group)
        il['pGroup'] = np.array(["p" + pg[:2] for pg in il['subject_id'].values.astype(str)])
        il['pName'] = np.array(["p" + pn for pn in il['subject_id'].values.astype(str)])
        il['sName'] = np.array(["s" + sn for sn in il['study_id'].values.astype(str)])
        if 'findings' in filters:
            il['text'] = il.apply(lambda row: Report_Parser.parse_report(row, rd, findings_only=True), axis=1)
            il = il[il['text'] != ""]
        elif 'impression' in filters:
            il['text'] = il.apply(lambda row: Report_Parser.parse_report(row, rd, impression_only=True), axis=1)
            il = il[il['text'] != ""]
        else:
            il['text'] = il.apply(lambda row: Report_Parser.parse_report(row, rd, impression_only=True), axis=1)
            il['findings'] = il.apply(lambda row: Report_Parser.parse_report(row, rd, findings_only=True), axis=1)
            il['impression'] = il['text']
            il = il[il['findings'] != ""]
            il = il[il['impression'] != ""]


        if sr == 'mscxr':
            ms_il = pd.read_csv(rd + "ms-cxr/MS_CXR_Local_Alignment_v1.0.0.csv")
            ms_il['pGroup'] = np.array([pg[6:9] for pg in ms_il['path'].values.astype(str)])
            ms_il['pName'] = np.array([pg[10:19] for pg in ms_il['path'].values.astype(str)])
            ms_il['sName'] = np.array([pg[20:29] for pg in ms_il['path'].values.astype(str)])
            ms_il = ms_il.loc[:, ['category_name', 'label_text', 'pGroup', 'pName', 'sName',
                                  'x', 'y', 'w', 'h', 'image_width', 'image_height']]
            il = ms_il.merge(il, on=['pGroup', 'pName', 'sName'], how='left')


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

    elif sr == 'chextest':
        rd = fps['chexpert_root_dir']
        il = pd.read_csv(rd + 'test_set/groundtruth.csv')
        rad1 = pd.read_csv(rd + 'test_set/radiologists/benchmark_radiologists/bc4.csv')
        rad2 = pd.read_csv(rd + 'test_set/radiologists/benchmark_radiologists/bc6.csv')
        rad3 = pd.read_csv(rd + 'test_set/radiologists/benchmark_radiologists/bc8.csv')
        il = il.merge(rad1, on=['Study'], suffixes = (None, "_rad1"))
        il = il.merge(rad2, on=['Study'], suffixes=(None, "_rad2"))
        il = il.merge(rad3, on=['Study'], suffixes=(None, "_rad3"))
        print(il.shape)
        print(il.head())

        il["Study"] = il["Study"].apply(lambda x: "" + x[14:])
        il["fullstudypath"] = rd + 'test_set/' + il["Study"]

        df = pd.DataFrame({'fullstudypath': pd.Series(dtype='str'),
                           'im_path': pd.Series(dtype='str')})
        for fsp in il["fullstudypath"].values:
            ims = os.listdir(fsp)
            for im in ims:
                df = df.append({'fullstudypath': fsp, 'im_path': fsp + "/" + im}, ignore_index=True)
        il = il.merge(df, on=['fullstudypath'], how='right')

        return il, rd

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

    elif sr == 'padchest':
        rd = fps['padchest_root_dir']
        il = pd.read_csv(fps['padchest_file'], compression='gzip', header=0)
        il['im_path'] = rd + il['ImageDir'].astype(str) + '/' + il['ImageID']
        if 'frontal' in filters:
            il = il[il['Projection'].str.contains('AP|PA')]
        elif 'lateral' in filters:
            il = il[~il['Projection'].str.contains('AP|PA')]

        il = il[il['MethodLabel'] == 'Physician']
        if 'tiny' in group:
            il = il[::10]

        il = il.loc[:, ['im_path', 'Labels']]
        labellist = il.pop('Labels').values.astype(str)
        outlist = []
        for i, l in enumerate(labellist):
            try:
                outlist.append(ast.literal_eval(l))
            except:
                outlist.append([])

        print(type(outlist[0]))
        mlb = MultiLabelBinarizer(sparse_output=False)
        toadd = pd.DataFrame(
            mlb.fit_transform(outlist),
            index=il.index,
            columns=mlb.classes_)
        print(toadd.sum())

        high_importance = ['COPD signs', 'endotracheal tube', 'pleural effusion', 'pulmonary edema', 'heart insufficiency',
                           'pulmonary fibrosis', 'cardiomegaly', 'vascular redistribution', 'consolidation', 'hilar congestion',
                           'pulmonary mass', 'cavitation', 'alveolar pattern', 'calcified pleural thickening', 'lung metastasis',
                           'emphysema', 'interstitial pattern', 'costophrenic angle blunting','tuberculosis','atelectasis',
                           'reticular interstitial pattern','pneumonia','lobar atelectasis','normal', 'pleural thickening',
                           'reticulonodular interstitial pattern','infiltrates','hypoexpansion','hypoexpansion basal',
                           'humeral fracture','pneumothorax','multiple nodules','hyperinflated lung', 'bronchiectasis',
                           'adenopathy','mediastinal enlargement','laminar atelectasis','vertebral compression','rib fracture',
                           'tuberculosis sequelae', 'hilar enlargement', 'tracheal shift', 'mediastinal mass', 'central vascular redistribution',
                           'vertebral fracture','superior mediastinal enlargement','vascular hilar enlargement','nodule',
                           'air trapping','bullas', 'ground glass pattern', 'calcified adenopathy', 'minor fissure thickening',
                           'unchanged', 'clavicle fracture','pseudonodule','end on vessel']
        print(len(high_importance))
        for h in high_importance:
            try:
                assert h in toadd.columns
            except:
                print(h)
        toadd = toadd.loc[:, high_importance]
        #toadd = toadd.loc[:, toadd.sum() >=50]
        print(toadd.head())
        il = il.join(toadd)
        print(il.head())



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
        elif 'med' in group:
            df = splitDF(df, 'subject_id', 0.3)[1]

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
        il_unseen, il_finetune = splitDF(df, 'patName', 0.1)
        il_finetune_train, il_finetune_val = splitDF(il_finetune, 'patName', 0.1)
        if 'train' in group:
            df = il_finetune_train
        elif 'val' in group:
            df = il_finetune_val
        elif 'candidates' in group or 'queries' in group:
            il = df
            il = il.drop_duplicates(subset=['patName'])
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
    if testsize == 1.0:
        return df, df
    splitter = GroupShuffleSplit(test_size=testsize, n_splits=1, random_state=1)
    split = splitter.split(df, groups=df[patientID])
    train_inds, valtest_inds = next(split)
    train = df.iloc[train_inds]
    test = df.iloc[valtest_inds]
    return train, test


def textProcess(text):
    parsed = Report_Parser.parse_report(text)
    if 'findings' in parsed:
        return parsed['findings']
    else:
        return "NO FINDINGS"

def draw_watermark(image, shortcut_list, shufflewatermarks=False):
    shortcutDict = {'Atelectasis': 'A', 'Cardiomegaly': 'C', 'Consolidation': 'N', 'Edema': 'E',
                    'Pleural Effusion': 'P',
                    'A': 'A', 'C': 'C', 'N': 'N', 'E': 'E', 'P': 'P'}
    locDict = {'A': 20, 'C': 60, 'N': 100, 'E': 140, 'P': 180}
    watermark_image = image.copy()
    draw = ImageDraw.Draw(watermark_image)
    font = ImageFont.truetype("/usr/share/fonts/urw-base35/P052-Roman.otf", 30)
    for s in shortcut_list:
        swrite = shortcutDict[s] if not shufflewatermarks else random.choice(['A', 'C', 'N', 'E', 'P'])
        change = np.random.randint(11, size=2) - 5
        sloc = (locDict[swrite] + change[0], 20 + change[1])
        draw.text(sloc, swrite, (255, 255, 255), font=font)
    return watermark_image

def make_synthetic(image, incorrect=[], correct=[], p_watermark=.9, p_correct=.9, overwrite=False, get_good=False, get_adversary=False, get_random=False):
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
    elif get_random:
        p_watermark, p_correct = 0.5, 1

    do_watermark, do_correct = np.random.random((2,))
    do_watermark = do_watermark < p_watermark
    do_correct = do_correct < p_correct
    if do_watermark or overwrite:
        if not overwrite:
            mylist = (correct if do_correct else incorrect)
            watermark_image = draw_watermark(image, mylist, shufflewatermarks = get_random)
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
import argparse
import pandas as pd
import numpy as np
import os
from PIL import Image
import shutil
from multiprocessing import Process

def findBeginning(im_list, rgb_root_dir, num = 0):
    idx = int(np.floor(im_list.shape[0]/2))
    ims = im_list.iloc[idx, :]
    img_name = os.path.join(rgb_root_dir, ims['pGroup'], ims['pName'], ims['sName'], ims['dicom_id'] + '.jpg')
    lower = -1
    upper = im_list.shape[0]
    while upper > lower + 1:
        if os.path.exists(img_name):
            lower = idx
            idx = int(np.floor(lower + (upper-lower)/2))
        else:
            upper = idx
            idx = int(np.floor(lower + (upper - lower)/2))
        ims = im_list.iloc[idx, :]
        img_name = os.path.join(rgb_root_dir, ims['pGroup'], ims['pName'], ims['sName'], ims['dicom_id'] + '.jpg')
    print(lower)
    print(im_list.shape[0])
    return lower + num

def worker(i):
    mimic_root_dir = '../../../../../mimic_cxr/'
    rgb_root_dir = '/n/scratch_gpu/users/a/anp2971/mimic_cxr_col/'

    im_list = pd.read_csv('../../../data/mimic-cxr-2.0.0-split.csv')
    im_list['pGroup'] = np.array(["p" + pg[:2] for pg in im_list['subject_id'].values.astype(str)])
    im_list['pName'] = np.array(["p" + pn for pn in im_list['subject_id'].values.astype(str)])
    im_list['sName'] = np.array(["s" + sn for sn in im_list['study_id'].values.astype(str)])


    begin = 0 + i
    print("begin", begin)
    for idx in np.arange(begin, im_list.shape[0], ncores):
        ims = im_list.iloc[idx, :]
        img_name = os.path.join(mimic_root_dir, ims['pGroup'], ims['pName'], ims['sName'], ims['dicom_id'] + '.jpg')
        im = Image.open(img_name)
        im = im.convert("RGB")
        img_save_name = os.path.join(rgb_root_dir, ims['pGroup'], ims['pName'], ims['sName'], ims['dicom_id'] + '.jpg')
        src_name_txt = os.path.join(mimic_root_dir, ims['pGroup'], ims['pName'], ims['sName'], ims['sName'] + '.txt')
        dest_name_txt = os.path.join(rgb_root_dir, ims['pGroup'], ims['pName'], ims['sName'], ims['sName'] + '.txt')
        try:
            os.makedirs(os.path.dirname(dest_name_txt), exist_ok=False)
            shutil.copyfile(src_name_txt, dest_name_txt)
            im.save(img_save_name)
        except:
            shutil.copyfile(src_name_txt, dest_name_txt)
            im.save(img_save_name)


if __name__ == '__main__':
    ncores = 20
    processes = []
    for i in range(ncores):
        process = Process(target=worker, args=(i,))
        processes.append(process)
        process.start()

    for proc in processes:
        proc.join()

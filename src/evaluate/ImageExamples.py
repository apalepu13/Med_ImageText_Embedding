import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/joint_embedding_model/')
from HelperFunctions import *

resultDir = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/ImageExamples/'

heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
dataset = 'mimic_cxr'
get_good = False
get_bad = True
num = 50

dat = getDatasets(source=dataset, subset='test',synthetic = (get_good or get_bad), get_adversary=get_bad,get_good=get_good, get_text=False)
[DL] = getLoaders(dat, subset= ['test'], shuffle=False, bs=1)

for i, res in enumerate(DL):
    if i >= num:
        break
    else:
        if dataset == 'mimic_cxr':
            im1, im2, df, study = res
            imgs = [im1, im2]
        else:
            im1, df = res
            imgs = [im1]

        labstr = ""
        for h in heads:
            labs = df[h]
            if labs[0] == 1:
                labstr += h
        if labstr == "":
            continue
        labstr += "_"


        print(i)
        for j, img in enumerate(imgs):
            myfig, myax = plt.subplots(1, 1, figsize=(7, 7))
            img[:, 0, :, :] = (img[:, 0, :, :] * .229) + .485
            img[:, 1, :, :] = (img[:, 1, :, :] * .224) + .456
            img[:, 2, :, :] = (img[:, 2, :, :] * .225) + .406
            img = img.permute(0, 2, 3, 1).squeeze()
            myax.imshow(img)
            imgname = "Img" + str(i) + "_" + ("" if (get_good or get_bad) else "real_")
            if (get_good or get_bad):
                imgname = imgname + ("good_" if get_good else "bad_")

            imgname += labstr
            plt.savefig(resultDir + imgname, bbox_inches='tight')
            break







import pickle5 as pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from functools import reduce

import numpy as np
modelpaths = ['/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp2/',
               '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp3/']
sub = ['all', 'all']
name = ['reg. PITER', 'unreg. PITER', 'chexzero']
colors = ['r', 'b']

fullpaths = [modelpaths[i] + 'predictions/metricsresults.pickle' for i in range(len(sub))]
fullpaths = fullpaths + ['/n/data2/hms/dbmi/beamlab/anil/CheXzero/metricsresults.pickle']

radpath = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp2/predictions/radiologistresults.pickle'
with open(radpath, 'rb') as handle:
    rad1MCC, rad2MCC, rad3MCC, rad1F1, rad2F1, rad3F1 =  pickle.load(handle)


heads = ['Total', 'Cardiomegaly', 'Edema','Consolidation', 'Atelectasis', 'Pleural Effusion']
fig, ax = plt.subplots(6, 2, figsize=(9, 20))
plt.subplots_adjust(hspace=0.7)


for j, h in enumerate(heads):
    ax[j, 0].scatter(0, rad1MCC[h], color='k')
    ax[j, 0].scatter(1, rad2MCC[h], color='k')
    ax[j, 0].scatter(2, rad3MCC[h],color='k')
    ax[j, 1].scatter(0, rad1F1[h],color='k')
    ax[j, 1].scatter(1, rad2F1[h],color='k')
    ax[j, 1].scatter(2, rad3F1[h],color='k')
    if j == 0:
        ax[j, 0].set_title(h + ' MCC', size=22)
        ax[j, 1].set_title(h + ' F1', size=22)
    else:
        ax[j, 0].set_title(h + ' MCC', size=18)
        ax[j, 1].set_title(h + ' F1', size=18)
    ax[j, 0].set_ylabel("MCC", size=14)
    ax[j, 1].set_ylabel("F1", size=14)
    ax[j,0].axvline(2.5, 0, 1)
    ax[j,1].axvline(2.5, 0, 1)




for i in range(len(fullpaths)):
    with open(fullpaths[i], 'rb') as handle:
        aucs, MCC, F1 = pickle.load(handle)
        if i == 1:
            MCCme = MCC
            F1me = F1
    for j, h in enumerate(heads):
        ax[j, 0].scatter(i + 3, MCC[h])
        ax[j, 1].scatter(i + 3, F1[h])

for j, h in enumerate(heads):
    ax[j, 0].set_xticklabels(['','Rad 1', 'Rad2', 'Rad3'] + name, rotation=25, ha='right', size=14)
    ax[j, 1].set_xticklabels(['','Rad 1', 'Rad2', 'Rad3'] + name, rotation=25, ha='right', size=14)
    ax[j, 0].set_xlim((-0.5, 5.5))
    ax[j, 1].set_xlim((-0.5, 5.5))
    ax[j, 0].set_ylim((MCCme[h]-0.2, MCCme[h] + 0.2))
    ax[j, 1].set_ylim((F1me[h]-0.2, F1me[h] + 0.2))

plt.savefig('/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/classification_scores_chexpert.png', bbox_inches='tight')

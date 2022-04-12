import sys
sys.path.insert(0, '../../models/joint_embedding_model/')
import pandas as pd
import numpy as np
from Data_Loading import *
import torch
print("CUDA Available: " + str(torch.cuda.is_available()))
from Transformer import *
from Pretraining import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
mimic_dat = getDatasets(source='m', subset = ['test'], get_text=False)
[mimic_loader] = getLoaders(mimic_dat, subset=['test'])

outDF = []
print("Hi")
for i, (im1, im2, df, text) in enumerate(mimic_loader):
    dfvals = np.array([df[h].numpy() for h in heads]).T
    dfvals = (dfvals == 1).astype(int)
    dfsums = np.sum(dfvals, axis=1)
    for r in np.arange(dfvals.shape[0]):
        if dfsums[r] == 1:
            head_ind = np.argwhere(dfvals[r, :])[0]
            myhead = heads[head_ind][0]
            outDF.append([myhead, text[r]])
df = pd.DataFrame(outDF, columns=['Variable', 'Text'])
df.to_csv('/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/mimic_label_queries.csv')
print(df.shape)
print(np.unique(df['Variable'].values, return_counts=True))



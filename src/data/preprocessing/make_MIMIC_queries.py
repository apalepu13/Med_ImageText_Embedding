import sys
sys.path.insert(0, '../../models/Patch_CLIP/')
import pandas as pd
import numpy as np
import MedDataHelpers
import torch
print("CUDA Available: " + str(torch.cuda.is_available()))

# Device configuration
multilabel=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion', 'Pneumonia',
                  'No Finding', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Other',
                  'Pneumonia','Pneumothorax', 'Support Devices'])
mimic_dat = MedDataHelpers.getDatasets(source='m', subset = ['val'], heads=heads, filters=['frontal'])
mimic_loader = MedDataHelpers.getLoaders(mimic_dat)
mimic_loader = mimic_loader['val']

outDF = []
print("Hi")
if multilabel:
    for i, (im1, im2, df, text) in enumerate(mimic_loader):
        dfvals = np.array([df[h].numpy() for h in heads]).T
        dfvals = (dfvals == 1).astype(int)
        for r in np.arange(dfvals.shape[0]):
            myrow = [dfvals[r, k] for k in np.arange(dfvals.shape[1])]
            myrow.append(text[r])
            outDF.append(myrow)

    df = pd.DataFrame(outDF, columns=list(heads).append('Description'))
    df.to_csv('/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/mimic_label_queries_multilabel.csv')
    print(df.shape)
    print(df.head())
else:
    for i, samples in enumerate(mimic_loader):
        text = samples['texts']
        df = samples['labels']
        dfvals = np.array([df[h].numpy() for h in heads]).T
        dfvals = (dfvals == 1).astype(int)
        dfsums = np.sum(dfvals, axis=1)
        for r in np.arange(dfvals.shape[0]):
            if dfsums[r] != 1:
                continue
            head_ind = np.argwhere(dfvals[r, :])[0]
            for myh in head_ind:
                myhead = heads[myh]
                outDF.append([myhead, text[r]])

    df = pd.DataFrame(outDF, columns=['Variable', 'Text'])
    df.to_csv('/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/mimic_label_queries_tinytrain.csv')
    print(df.shape)
    print(np.unique(df['Variable'].values, return_counts=True))



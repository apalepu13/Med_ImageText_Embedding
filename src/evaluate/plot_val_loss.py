import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

exp = 'exp5'
folder = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/' + exp + '/'
mod = ['je_model-' + str(i*2) + '.pt' for i in np.arange(50)]
v = []
for i,m in enumerate(mod):
    try:
        loadpath = folder + m
        checkpoint = torch.load(loadpath, map_location=torch.device('cpu'))
        v.append(checkpoint['val_loss'])
        final = i
    except:
        break

plt.figure(figsize = (8,8))
plt.plot(np.arange(final+1)*2, np.array(v))
plt.ylim(0, 12)
plt.xlabel("Number of Epochs", size=18)
plt.ylabel("Validation Loss", size=18)
plt.title("Validation Loss during training", size = 20)
plt.savefig('/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/training/' + exp + '_training.png', bbox_inches='tight')



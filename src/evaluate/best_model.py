import os
import regex as re
import torch

modelDir = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/'
model = 'vision_model/vision_VIT_real'
#model = 'je_model/synth/exp2/'
mDir = modelDir + model

all_files = os.listdir(os.path.join(mDir))
modlist = [file for file in all_files if 'model' in file]

best = ["Hi", 10000000000000000000]
for m in modlist:
    checkpoint = torch.load(os.path.join(mDir, m))
    vloss = checkpoint['val_loss']
    if vloss < best[1]:
        best[0] = m
        best[1] = vloss

print(best)




import torch
import os
import re
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/Patch_CLIP/')
print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

resultsdir = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/training/'
path = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp6/'
all_files = os.listdir(path)
all_models = np.array([f for f in all_files if 'je_model' in f])
all_mod_numbers = np.array([int(re.search(r'\d+', model).group()) for model in all_models])
mod_order = np.argsort(all_mod_numbers)
all_models = all_models[mod_order]
all_mod_numbers = all_mod_numbers[mod_order]
plt.figure(figsize = (6, 6))
train_clips, val_clips, train_pens, val_pens = [], [], [], []
for i, model in enumerate(all_models):
    print(all_mod_numbers[i])
    checkpoint = torch.load(path + model, map_location=device)
    train_losses = checkpoint['train_losses'].cpu()
    val_losses = checkpoint['val_losses'].cpu()
    print(val_losses)
    train_clips.append(torch.mean(train_losses[:3]))
    train_pens.append(torch.mean(train_losses[3:]))
    val_clips.append(torch.mean(val_losses[:3]))
    val_pens.append(torch.mean(val_losses[3:]))

plt.plot(all_mod_numbers, train_clips,'ro', label="train_clip")
plt.plot(all_mod_numbers, val_clips,'bo', label="val clip")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train/val CLIP loss over time")
plt.legend()
plt.savefig(resultsdir + path[-5:-1] + "traincurves.png", bbox_inches='tight')

plt.figure(figsize = (6, 6))
plt.plot(all_mod_numbers, train_pens,'r*', label="train penalty")
plt.plot(all_mod_numbers, val_pens,'b*', label="val penalty")
plt.xlabel("Epochs")
plt.ylabel("penalty")
plt.title("Train/val attention penalty over time")
plt.legend()
plt.savefig(resultsdir + path[-5:-1] + "penaltycurves.png", bbox_inches='tight')





import torch
import os
import re
import matplotlib.pyplot as plt
import sys
import numpy as np
import pickle
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/Patch_CLIP/')
print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

cutoff = 30
resultsdir = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/training/'
#path = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp1/'
path = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/synth_cxr_cnn/exp1/'
finetuned = False
cnn = "cnn" in path
synth = "synth" in path
all_files = os.listdir(path)
all_models = np.array([f for f in all_files if 'je_model' in f])
all_mod_numbers = np.array([int(re.search(r'\d+', model).group()) for model in all_models])
mod_order = np.argsort(all_mod_numbers)
all_models = all_models[mod_order]
all_mod_numbers = all_mod_numbers[mod_order]
all_models = all_models[all_mod_numbers <= cutoff]
all_mod_numbers = all_mod_numbers[all_mod_numbers <= cutoff]
plt.figure(figsize = (6, 6))


if finetuned:
    load_file = path + 'finetune_losses.pickle'
    with open(load_file, 'rb') as handle:
        train_losses, val_losses = pickle.load(handle)

    epoch_nums = np.arange(len(train_losses))
    plt.plot(epoch_nums, train_losses, 'ro', label="finetune train")
    plt.plot(epoch_nums, val_losses, 'bo', label="finetune val")
    plt.xlabel("Epochs")
    plt.ylabel("Avg BCE Loss")
    plt.ylim(0.5, 2)
    origtitle = "Train/val loss after finetuning"
    origtitle = "CNN " + origtitle if cnn else "CLIP " + origtitle
    origtitle = "Synthetically trained " + origtitle if synth else "Real trained " + origtitle
    plt.title(origtitle)
    plt.legend()
    plt.savefig(resultsdir + path[-5:-1] + origtitle + "traincurves.png", bbox_inches='tight')
    sys.exit()

elif not cnn:
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
    plt.ylabel("CLIP Loss (Batch size = 32)")
    plt.ylim(0.5, 2)
    plt.title("Train/val CLIP loss over time")
    plt.legend()
    if synth:
        plt.savefig(resultsdir + path[-5:-1] + "synthtraincurves.png", bbox_inches='tight')
    else:
        plt.savefig(resultsdir + path[-5:-1] + "traincurves.png", bbox_inches='tight')

    plt.figure(figsize = (6, 6))
    plt.plot(all_mod_numbers, train_pens,'r*', label="train penalty")
    plt.plot(all_mod_numbers, val_pens,'b*', label="val penalty")
    plt.xlabel("Epochs")
    plt.ylabel("penalty")
    plt.title("Train/val attention penalty over time")
    plt.legend()
    if "synth" in path:
        plt.savefig(resultsdir + path[-5:-1] + "synthpenaltycurves.png", bbox_inches='tight')
    else:
        plt.savefig(resultsdir + path[-5:-1] + "penaltycurves.png", bbox_inches='tight')
else:
    train_loss, val_loss = [], []
    for i, model in enumerate(all_models):
        print(all_mod_numbers[i])
        checkpoint = torch.load(path + model, map_location=device)
        train_losses = checkpoint['train_loss'].detach().cpu().numpy()
        val_losses = checkpoint['val_loss']
        print(val_losses)
        train_loss.append(train_losses)
        val_loss.append(val_losses)

    plt.plot(all_mod_numbers, train_loss, 'ro', label="train cnn")
    plt.plot(all_mod_numbers, val_loss, 'bo', label="val cnn")
    plt.xlabel("Epochs")
    plt.ylabel("Avg BCE Loss")
    plt.ylim(0, 1)
    plt.title("Train/val Classification loss over time")
    plt.legend()
    if synth:
        plt.savefig(resultsdir + path[-5:-1] + "synthcnntraincurves.png", bbox_inches='tight')
    else:
        plt.savefig(resultsdir + path[-5:-1] + "traincnncurves.png", bbox_inches='tight')






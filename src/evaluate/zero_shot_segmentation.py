import argparse
import sys
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/Patch_CLIP/')
import torch
import CLIP_Embedding
import MedDataHelpers
import utils
import numpy as np
import matplotlib.pyplot as plt

print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def plot_original_image(sample, ax, heads, alpha=1.0, index=0, title=True):
    image = sample['images'][0]
    if not title:
        ax.imshow(image.permute(0, 2, 3, 1)[0, :, :, :].squeeze(), alpha=alpha)
    else:
        ax.imshow(utils.normalize(image), alpha=alpha)
        labels = [h for h in heads if sample['labels'][h] == 1]
        ax.set_title("Image " + str(index) + "\nOriginal x-ray: " + str(labels))

def plot_lung_mask(sample, ax):
    lung_mask = sample['lung_mask']
    inf_mask = sample['inf_mask']
    total_mask = lung_mask + 2 * inf_mask
    ax.imshow(total_mask.squeeze(), cmap='plasma')
    ax.set_title("Lung mask")

def get_heats(im_sims, heads):
    heats = {}
    for i, h in enumerate(heads):
        heatmap_res = im_sims[:, :, i].squeeze() #N P
        heatmap_res = autoscale(heatmap_res)
        heatmap_res = torch.reshape(heatmap_res, (heatmap_res.shape[0], 1, int(np.sqrt(heatmap_res.shape[1])), int(np.sqrt(heatmap_res.shape[1])))) #N 1 p p
        heats[h] = torch.nn.functional.interpolate(heatmap_res, 224) #N 1 224 224
    return heats

def get_avg_ig(index, heads, heats):
    igs = []
    for h in heads:
        igs.append(heats['No Finding'][index, :, :, :])
    igs = torch.stack(igs).mean(dim=0, keepdims=False).squeeze()
    return igs

def plot_heat(sample, ax, dataheads, heats,head='covid19', index=0, avgig = 0, name=""):
    ax.imshow(heats[head][index, :, :, :].squeeze()-avgig, cmap='coolwarm', alpha=0.95, vmin=-0.3, vmax=0.5)
    ax.set_title(head + name)
    plot_original_image(sample, ax, dataheads, alpha=0.3, title=False)

def autoscale(heats_all, high_percentile = 99, low_percentile=1):
    vmax = np.percentile(heats_all, high_percentile)
    vmin = np.percentile(heats_all, low_percentile)
    return np.clip((heats_all - vmin) / (vmax - vmin), 0, 1)

def main(args):
    if args.sr == 'co':
        heads1 = np.array(['covid19', 'Pneumonia', 'No Finding'])
        heads2 = np.array(['covid19', 'Pneumonia', 'No Finding', 'Left Lung', 'Right Lung', 'Heart'])
    else:
        heads1 = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
        heads2 = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion', 'No Finding'])
    unreg_model = CLIP_Embedding.getCLIPModel(args.je_model_path_unreg)
    clip_model = CLIP_Embedding.getCLIPModel(args.je_model_path)
    filters = MedDataHelpers.getFilters(args.je_model_path)
    # TODO: add frontal filters to coviddataset
    modname = args.je_model_path[-5:-1]
    dat = MedDataHelpers.getDatasets(source=args.sr, subset=[args.subset], heads=heads1, filters=filters,)
    #dat = MedDataHelpers.getDatasets(source=args.sr, subset=[args.subset], heads=heads1, filters=filters, synthetic=True, overwrite = ['Cardiomegaly'])
    DLS = MedDataHelpers.getLoaders(dat, args, shuffle=False)
    DL = DLS[args.subset]

    unreg_sims, _ = utils.get_all_preds(DL, unreg_model, patch_similarity=True, heads = heads2, getlabels=False)
    unreg_sims = unreg_sims[0]
    im_sims, _ = utils.get_all_preds(DL, clip_model, patch_similarity=True, heads = heads2, getlabels=False) #N P c
    im_sims = im_sims[0] #only 1 image prediction (no augmentations) N P len(heads2)
    print(im_sims.shape)
    heats = get_heats(im_sims, heads2)
    unreg_heats = get_heats(unreg_sims, heads2)
    nofinding=False
    for i, sample in enumerate(DL):
        avgig = get_avg_ig(i, heads2, heats) if not nofinding else 0
        avg_unreg = get_avg_ig(i, heads2, unreg_heats) if not nofinding else 0

        fig, ax = plt.subplots(1, 3, figsize=(12, 8))
        plot_original_image(sample, ax[0], heads1, index=i)
        plot_heat(sample, ax[1], heads1, unreg_heats, heads2[0], index=i, avgig=avg_unreg, name=" unregularized")
        #plot_heat(sample, ax[1, 1], heads1, heats, heads2[1], index=i, avgig=avgig)
        plot_heat(sample, ax[2], heads1, heats,  heads2[0], index=i, avgig = avgig, name=" regularized")
        #plot_heat(sample, ax[1, 2], heads1, heats,  heads2[3], index=i, avgig = avgig)
        #plt.savefig(args.results_dir + 'Img' + str(i) + '_heatmaps.png', bbox_inches='tight')
        plt.savefig(args.results_dir + 'Img' + str(i) + '_synthnormheatmaps.png', bbox_inches='tight')
        if i == 1:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path_unreg', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/synth_clip_regularized/exp3/')
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/synth_clip_regularized/exp2/', help='path for saving trained models')

    parser.add_argument('--sr', type=str, default='c')
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--use_softmax', type=bool, default=False)

    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=1) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/covid_segmentation/')
    args = parser.parse_args()
    print(args)
    main(args)
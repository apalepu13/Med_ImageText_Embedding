import argparse
import sys
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/Patch_CLIP/')
import torch
import CLIP_Embedding
import MedDataHelpers
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    resultDir = args.results_dir
    dat = MedDataHelpers.getDatasets(source = 'mimic_cxr', subset = ['test'], frontal=True)
    dl = MedDataHelpers.getLoaders(dat, shuffle=False)
    dl = dl['test']

    clip_model = CLIP_Embedding.getCLIPModel(args.je_model_path).to(device)
    soft = nn.Softmax(dim=2)

    with torch.no_grad():
        for i, samples in enumerate(dl):
            text = samples['texts'] #corresponding tex
            ims = samples['images'] #first im in list
            if i == 0:
                print(text[0])

            if i == 0:
                img = ims[0] #N CH DIM DIM
                img[:, 0, :, :] = (img[:, 0, :, :] * .229) + .485
                img[:, 1, :, :] = (img[:, 1, :, :] * .224) + .456
                img[:, 2, :, :] = (img[:, 2, :, :] * .225) + .406
                img = img.permute(0, 2, 3, 1).squeeze()

            token_im = clip_model.get_im_embeddings(ims, only_patches = True)
            token_embeds = clip_model.get_text_embeddings(text, only_words=True)
            cross_weights_text = clip_model.get_cross_weights(token_im, token_embeds)[0]
            cross_weights_text = soft(cross_weights_text)

            cwt = cross_weights_text.shape
            maxlen = 256
            if cwt[1] < maxlen:
                cross_weights_text = torch.cat([cross_weights_text, -2 * torch.ones(cwt[0], maxlen-cwt[1], cwt[2]).to(device)], dim=1)
                cross_weights_text[cross_weights_text == -2] = float('nan')
            if i == 0:
                allattns = cross_weights_text
                break
            else:
                allattns = torch.cat([allattns, cross_weights_text], dim=0) #D T P


        for i in np.arange(allattns.shape[0]):
            lol = allattns[i, :, :].squeeze().cpu()
            tpmat = lol
            plt.figure(figsize = (8, 8))
            plt.imshow(img[i, :, :, :])
            for x in range(7):
                for y in range(7):
                    plt.text(x * 32 + 16, y * 32 + 16, str(y * 7 + x), color='b')
            plt.savefig(resultDir + args.je_model_path[-5:-1] + 'image.png')
            plt.figure(figsize = (8, 8))
            plt.imshow(tpmat[:150, :], cmap='plasma', interpolation='nearest')
            plt.title("token-patch similarity, im: " + str(i))
            plt.xlabel("Patch #")
            plt.ylabel("Token #")
            plt.savefig(resultDir + args.je_model_path[-5:-1] + 'simActivation.png')
            #lol, indices = torch.sort(lol, dim=1, descending=True) #T P
            descending_attn_means = np.nanmean(lol, axis=0)
            descending_attn_stds = np.nanstd(lol, axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
            ax.errorbar(np.arange(descending_attn_means.shape[0]), descending_attn_means, yerr = descending_attn_stds, linestyle='')
            for k, txt in enumerate(np.arange(49).astype(str)):
                ax.annotate(txt, (k, descending_attn_means[k]))
            ax.set_title("Image " + str(i))
            ax.set_xlabel("Patch #")
            ax.set_ylabel("mean/std token similarity")
            plt.savefig(resultDir + args.je_model_path[-5:-1] + 'tokenprobs.png')
            break


if __name__ == '__main__': #7, 8, 9 = norm, patch, both
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp8/', help='path for saving trained models')
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/attn/')
    parser.add_argument('--dat_dir', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/')
    args = parser.parse_args()
    print(args)
    main(args)
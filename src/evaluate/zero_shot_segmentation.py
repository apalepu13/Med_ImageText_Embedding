import argparse
import torch
import numpy as np
import sys
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/joint_embedding_model/')
from Integrated_Gradients import *
from CNN import *
from HelperFunctions import *
from Transformer import *
import matplotlib.pyplot as plt
print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def getLungSaliencies(DL, vis_model, heads, device=device):
    myfig, myaxs = plt.subplots(1, 3, figsize=(8, 8))
    x,y = 0,0
    sals = []
    lung_sals = []
    infs = []
    lungs = []
    for i, res in enumerate(DL):
        image, inf_mask, lung_mask, df = res
        name = str(i)
        myim = image.clone().permute(0, 2, 3, 1).numpy().squeeze()
        sal = plot_ig_saliency(myim, 0, vis_model, myfig, myaxs, x, y, use_abs = False, actually_plot=False)
        #no_finding_sal = plot_ig_saliency(myim, 1, vis_model, myfig, myaxs, x, y, use_abs=False, actually_plot=False)
        #sal = sal - no_finding_sal
        sals.append(sal)
        infs.append(inf_mask)
        lungs.append(lung_mask)
        myax = myaxs[0]
        myax.imshow(sal, plt.cm.plasma, vmin=0, vmax=1)
        myax.set_xticks([])
        myax.set_xticks([], minor=True)
        myax.set_yticks([])
        myax.set_yticks([], minor=True)

        myax = myaxs[1]
        title = ""
        if df['covid19'][0] == True:
            title = 'covid19+'
        elif df['Pneumonia'][0] == True:
            title = 'Pneumonia+'
        else:
            title = 'No Finding'
        myax.set_title(title)
        mask = lung_mask.squeeze() + inf_mask.squeeze()
        myax.imshow(mask)
        plot_orig_im(image.clone(), myfig, myaxs, 2, -1, title="Original Image")


        plt.savefig(args.results_dir + name + ".png", bbox_inches='tight')
    return sals, infs, lungs





def main(args):
    heads = np.array(['covid19', 'Pneumonia', 'No Finding'])
    dat = getDatasets(source=args.sr, subset=[args.subset], heads=heads, get_seg=True)
    [DL] = getLoaders(dat, args, subset=[args.subset])

    heads = np.array(['Lungs', 'No Finding'])
    je_vision_model, transformer, tokenizer = getSimilarityClassifier(args.model_path, args.model, heads=heads, avg_embedding=True, text_num=500, soft=True)
    test_saliencies, test_target_inf, test_target_lungs = getLungSaliencies(DL, je_vision_model, heads, device = device)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/', help='path for saving trained models')
    parser.add_argument('--model', type=str, default='synth/exp7/je_model-12.pt', help='path from root to model')

    parser.add_argument('--sr', type=str, default='co')
    parser.add_argument('--subset', type=str, default='tiny')
    parser.add_argument('--use_softmax', type=bool, default=False)

    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=1) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/covid_segmentation/')
    args = parser.parse_args()
    print(args)
    main(args)
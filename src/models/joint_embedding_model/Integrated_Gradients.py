import argparse
import torch
import pandas as pd
import copy
import torch.nn as nn
from CNN import *
from Vision_Transformer import *
import numpy as np
import torch
import torch.nn as nn
from jointEmbedding import JointEmbeddingModel
from Pretraining import *
from Transformer import *
import matplotlib.pyplot as plt
import matplotlib as mp
import numpy as np
import saliency.core as saliency
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
from captum.attr import visualization as vis
print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def plot_orig_im(img, fig, ax, x, y, title = "Original image"):
    img[:, 0, :, :] = (img[:, 0, :, :] * .229) + .485
    img[:, 1, :, :] = (img[:, 1, :, :] * .224) + .456
    img[:, 2, :, :] = (img[:, 2, :, :] * .225) + .406
    img = img.permute(0,2,3,1).squeeze()
    ax[x,y].imshow(img, plt.cm.gray, vmin=0, vmax=1)
    ax[x,y].set_title(title)
    ax[x,y].set_xticks([])
    ax[x,y].set_xticks([], minor=True)
    ax[x,y].set_yticks([])
    ax[x,y].set_yticks([], minor=True)

def myVisualizeImageGrayscale(image_3d, percentile=99):
  r"""Returns a 3D tensor as a grayscale 2D tensor.
  This method sums a 3D tensor across the absolute value of axis=2, and then
  clips values at a given percentile.
  """
  image_2d = np.sum(image_3d, axis=2)

  vmax = np.percentile(image_2d, percentile)
  vmin = np.min(image_2d)

  return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

def call_model_function(img, call_model_args, expected_keys):
    img = torch.tensor(img, dtype=torch.float32)
    #print("Init:", img.shape)
    img = img.permute(0, 3, 1, 2)
    img=img.requires_grad_(True)
    target_class_idx = call_model_args['targind']

    model = call_model_args['model']
    img = img.to(device)
    model = model.to(device)
    output = model(img)
    outputs = output[:, target_class_idx]
    grads = torch.autograd.grad(outputs, img, grad_outputs=torch.ones_like(outputs))
    grads = torch.movedim(grads[0], 1, 3)
    #print("final:", grads.shape)
    gradients = grads.cpu().detach().numpy()
    return {saliency.INPUT_OUTPUT_GRADIENTS: gradients}

def plot_ig_saliency(img, targind, model, myfig, myax, x, y, use_abs = True):
    ig = saliency.IntegratedGradients()
    baseline = np.zeros(img.shape)
    sig = ig.GetSmoothedMask(img, call_model_function, {'model':model, 'targind':targind}, x_steps=5, x_baseline=baseline, batch_size=20)
    if use_abs:
        gs = saliency.VisualizeImageGrayscale(sig)
    else:
        gs = myVisualizeImageGrayscale(sig)

    myax[x,y].imshow(gs, plt.cm.plasma, vmin=0, vmax=1)
    #plt.cm.plasma
    #plt.cm.gray
    source = ['Real Train', 'Synth Train']
    eval = [', Real Test', ', Synth Test']
    myax[x,y].set_title(source[x] + eval[y])
    myax[x,y].set_xticks([])
    myax[x,y].set_xticks([], minor=True)
    myax[x,y].set_yticks([])
    myax[x,y].set_yticks([], minor=True)

def plot_ig_captum(img, nt, targind, myfig, myax, x, y):
    baseline = torch.zeros(1, 3, 224, 224).to(device)
    attributions = nt.attribute(img, nt_type='smoothgrad', stdevs=0.03, nt_samples=10, internal_batch_size=5, baselines=baseline, target=targind, return_convergence_delta=False)
    img[:, 0, :, :] = (img[:, 0, :, :] * .229) + .485
    img[:, 1, :, :] = (img[:, 1, :, :] * .224) + .456
    img[:, 2, :, :] = (img[:, 2, :, :] * .225) + .406
    img = img.permute(0, 2, 3, 1).squeeze()
    attributions = attributions.permute(0, 2, 3, 1).squeeze()
    fig, lol = _ = vis.visualize_image_attr(attributions.cpu().numpy(), img.cpu().numpy(), "blended_heat_map", alpha_overlay=0.6, plt_fig_axis=(myfig, myax[x,y]))
    source = ['Real Train', 'Synth Train']
    eval = [', Real Test', ', Synth Test']
    myax[x, y].set_title(source[x] + eval[y])
    myax[x,y].set_xticks([])
    myax[x,y].set_xticks([], minor=True)
    myax[x,y].set_yticks([])
    myax[x,y].set_yticks([], minor=True)

def getAttributions(im_dict, real_model,synth_model, heads, target='Cardiomegaly', mod_name='vision', im_number=0, df=None, use_captum=False, use_abs = True):
    myfig, myax = plt.subplots(3, 2, figsize=(8, 12))
    labstr = "Labels: "
    for i, h in enumerate(heads):
        if h == target:
            myind = i
        if df and df[h] == 1:
            labstr += h
            labstr += ", "

    # Starting here, make 4 panel.
    if use_captum:
        ig_real = IntegratedGradients(real_model, multiply_by_inputs=False)
        ig_synth = IntegratedGradients(synth_model, multiply_by_inputs=False)
        nt_real = NoiseTunnel(ig_real)  # 0.02
        nt_synth = NoiseTunnel(ig_synth)
        plot_ig_captum(im_dict['real'].clone().to(device), nt_real,myind, myfig, myax, 0, 0)
        plot_ig_captum(im_dict[target].clone().to(device), nt_real,myind, myfig, myax, 0, 1)
        plot_ig_captum(im_dict['real'].clone().to(device), nt_synth,myind, myfig, myax, 1, 0)
        plot_ig_captum(im_dict[target].clone().to(device), nt_synth,myind, myfig, myax, 1, 1)

    else:
        plot_ig_saliency(im_dict['real'].clone().permute(0,2,3,1).numpy().squeeze(), myind,real_model, myfig, myax, 0, 0, use_abs)
        plot_ig_saliency(im_dict[target].clone().permute(0,2,3,1).numpy().squeeze(), myind,real_model, myfig, myax, 0, 1, use_abs)
        plot_ig_saliency(im_dict['real'].clone().permute(0,2,3,1).numpy().squeeze(), myind,synth_model, myfig, myax, 1, 0, use_abs)
        plot_ig_saliency(im_dict[target].clone().permute(0,2,3,1).numpy().squeeze(), myind,synth_model, myfig, myax, 1, 1,use_abs)
    plot_orig_im(im_dict['real'].clone(), myfig, myax, 2, 0, title="Original Image")
    plot_orig_im(im_dict[target].clone(), myfig, myax, 2, 1, title="Synthethic Image")

    if labstr == "Labels: ":
        labstr = "Labels: No Finding"
    myfig.suptitle("I.G. " + " " + target + " target, " + mod_name + "model\n" + labstr)
    plt.savefig(args.results_dir + "Integrated_grad_abs_"+str(use_abs)+"_" + target + "_" + mod_name + "_" + str(im_number) + ".png", bbox_inches='tight')



def main(args):
    heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
    dat_normal = getDatasets(source=args.sr, subset=['test'], heads=heads, synthetic=False)
    dat_overwrites = getDatasets(source=args.sr, subset=['test'], heads=heads, synthetic=True, get_overwrites=True)
    [loader_normal] = getLoaders(dat_normal, subset=['test'])
    loader_synths = getLoaders(dat_overwrites, subset=heads)
    dat_normal = dat_normal['test']

    #[DL_synthetics] = getLoaders(dat_good, args, subset=['test'], shuffle=False, num_work=1)
    #[DL_adversarial] = getLoaders(dat_bad, args, subset=['test'], shuffle=False, num_work=1)
    #[DL_real] = getLoaders(dat_normal, args, subset=['test'], shuffle=False, num_work=1)

    #Vision model
    vision_model_real = getVisionClassifier(args.model_path_real, args.model_real, device, args.embed_size, heads)
    vision_model_synth = getVisionClassifier(args.model_path_synth, args.model_synth, device, args.embed_size,heads)
    je_model_real, transformer_real, tokenizer_real = getSimilarityClassifier(args.je_model_path_real, args.je_model_real, device, args.embed_size, heads,
                                            text_num=1, avg_embedding=True)
    je_model_synth, transformer_synth, tokenizer_synth = getSimilarityClassifier(args.je_model_path_synth, args.je_model_synth, device, args.embed_size, heads,
                                            text_num=1, avg_embedding=True)

    finetuned_vision_model_synth = getVisionClassifier(args.model_path_synth, args.model_synth, device, args.embed_size, heads, add_finetune=True)
    finetuned_je_model_synth = getVisionClassifier(args.je_model_path_synth, args.je_model_synth, device,args.embed_size, heads, add_finetune=True)

    im_number = 99

    im_dict = {}
    normIm, normDf = dat_normal.__getitem__(im_number)
    normIm = normIm.reshape(1, 3, 224, 224)
    im_dict['real'] = normIm
    for h in heads:
        myim, mydf = dat_overwrites[h].__getitem__(im_number)
        myim = myim.reshape(1, 3, 224, 224)
        im_dict[h] = myim


    for target in heads:
        getAttributions(im_dict, vision_model_real, vision_model_synth, heads, target=target, mod_name="vision", im_number=im_number, df = normDf, use_abs=False)
        getAttributions(im_dict, je_model_real, je_model_synth, heads, target=target, mod_name="clip", im_number=im_number, df = normDf, use_abs=False)
        getAttributions(im_dict, finetuned_vision_model_synth, finetuned_je_model_synth, heads, target=target, mod_name="finetuned", im_number=im_number, df=normDf, use_abs=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path_real', type=str,default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/exp6/')
    parser.add_argument('--model_path_real', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/vision_model/vision_CNN_real/')
    parser.add_argument('--je_model_path_synth', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/synth/exp7/')
    parser.add_argument('--model_path_synth', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/vision_model/vision_CNN_synthetic/', help='path for saving trained models')


    parser.add_argument('--je_model_real', type=str, default='je_model-28.pt')
    parser.add_argument('--model_real', type=str, default='model-14.pt')
    parser.add_argument('--je_model_synth', type=str, default='je_model-12.pt')
    parser.add_argument('--model_synth', type=str, default='model-14.pt', help='path from root to model')

    parser.add_argument('--results_dir', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/integrated_gradients/')

    parser.add_argument('--sr', type=str, default='c') #c, co
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--synth', type=bool, default=False, const=True, nargs='?', help='Train on synthetic dataset')
    parser.add_argument('--usemimic', type=bool, default=False, const=True, nargs='?', help='Use mimic to alter zeroshot')
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=1) #32 normally
    args = parser.parse_args()
    print(args)
    main(args)
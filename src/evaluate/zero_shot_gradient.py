import argparse
import sys
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/Patch_CLIP/')
import torch
import CLIP_Embedding
import MedDataHelpers
import utils
import numpy as np
import matplotlib.pyplot as plt
import saliency.core as saliency

print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def myVisualizeImageGrayscale(image_3d, percentile=99):
  r"""Returns a 3D tensor as a grayscale 2D tensor.
  This method sums a 3D tensor across the absolute value of axis=2, and then
  clips values at a given percentile.
  """
  image_2d = np.sum(image_3d, axis=2)

  vmax = np.percentile(image_2d, percentile)
  vmin = np.min(image_2d)

  return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

def call_model_function_ig(img, call_model_args, expected_keys):
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
    try:
        return {saliency.INPUT_OUTPUT_GRADIENTS: gradients}
    except:
        return {saliency.OUTPUT_LAYER_VALUES: gradients}

def call_model_function_oc(img, call_model_args, expected_keys):
    img = torch.tensor(img, dtype=torch.float32)
    img = img.permute(0, 3, 1, 2)
    img = img.requires_grad_(True)
    target_class_idx = call_model_args['targind']
    model = call_model_args['model']
    img = img.to(device)
    model = model.to(device)
    output = model(img)
    outputs = output[:, target_class_idx]
    gradients = outputs.cpu().detach().numpy()
    return {saliency.OUTPUT_LAYER_VALUES: gradients}


def plot_ig_saliency(img, target, model, ax, use_abs = False, to_plot=True, method='blur_ig'):
    headsDict = {'Cardiomegaly': 0, 'Edema': 1, 'Consolidation': 2, 'Atelectasis': 3, 'Pleural Effusion': 4,'No Finding': 5}
    if method == 'integrated_gradients':
        ig = saliency.IntegratedGradient()
        baseline = np.zeros(img.shape)
        sig = ig.GetSmoothedMask(img, call_model_function_ig, {'model':model, 'targind':headsDict[target]}, x_steps=5, x_baseline=baseline, batch_size=20)
    elif method == 'guided_ig':
        ig = saliency.GuidedIG()
        baseline = np.zeros(img.shape)
        sig = ig.GetSmoothedMask(img, call_model_function_ig, {'model': model, 'targind': headsDict[target]}, x_steps=5,x_baseline=baseline)
    elif method == 'blur_ig':
        ig = saliency.BlurIG()
        sig = ig.GetSmoothedMask(img, call_model_function_ig, {'model': model, 'targind':headsDict[target]})
    else:
        oc = saliency.Occlusion()
        sig = oc.GetSmoothedMask(img, call_model_function_oc, {'model':model, 'targind':headsDict[target]}, size=15, value=0)

    print(target, np.min(sig), np.max(sig))
    if use_abs:
        gs = saliency.VisualizeImageGrayscale(sig)
        v_min, v_max = 0, 1
    else:
        gs = myVisualizeImageGrayscale(sig)
        v_min, v_max = 0, 1
    if to_plot:
        ax.imshow(gs, plt.cm.plasma, vmin=v_min, vmax=v_max)
        ax.set_title(target + " ig")
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
    return gs

def plotAttributions(sample, ax, sim_model, target='Cardiomegaly', heats = None, index = 0):
    smoothgrad = plot_ig_saliency(sample['images'][0].clone().permute(0,2,3,1).numpy().squeeze(), target,
                                  model = sim_model, ax = ax)
    return smoothgrad

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
        heatmap_res = torch.reshape(heatmap_res, (heatmap_res.shape[0], 1, int(np.sqrt(heatmap_res.shape[1])), int(np.sqrt(heatmap_res.shape[1])))) #N 1 p p
        heats[h] = torch.nn.functional.interpolate(heatmap_res, 224) #N 1 224 224
    return heats

def plot_heat(sample, ax, dataheads, heats,head='covid19', index=0, to_plot=True):
    if to_plot:
        ax.imshow(heats[head][index, :, :, :].squeeze(), cmap='plasma', alpha=0.95, vmin=-0.05, vmax=0.5)
        ax.set_title(head)
        plot_original_image(sample, ax, dataheads, alpha=0.3, title=False)
    return heats[head][index, :, :, :].squeeze()

def main(args):
    if args.sr == 'co':
        heads1 = np.array(['covid19', 'Pneumonia', 'No Finding'])
        heads2 = np.array(['covid19', 'Pneumonia', 'No Finding', 'Left Lung', 'Right Lung', 'Heart'])
    else:
        heads1 = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
        heads2 = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion', 'No Finding'])
    sim_model = CLIP_Embedding.getSimModel(args.je_model_path, image_grad = True)
    filters = MedDataHelpers.getFilters(args.je_model_path)
    modname = args.je_model_path[-5:-1]
    dat = MedDataHelpers.getDatasets(source=args.sr, subset=[args.subset], heads=heads1, filters=filters)
    DLS = MedDataHelpers.getLoaders(dat, args, shuffle=False)
    DL = DLS[args.subset]

    clip_model = CLIP_Embedding.getCLIPModel(args.je_model_path)
    im_sims, _ = utils.get_all_preds(DL, clip_model, patch_similarity=True, heads=heads2, getlabels=False)  # N P c
    im_sims = im_sims[0]  # only 1 image prediction (no augmentations) N P len(heads2)
    heats = get_heats(im_sims, heads2)

    for i, sample in enumerate(DL):
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        plot_original_image(sample, ax[0, 0], heads1, index=i)
        if args.sr == 'co':
            plot_lung_mask(sample, ax[1, 0])
        else:
            plotAttributions(sample, ax[1, 0], sim_model, heads2[4])
        plotAttributions(sample, ax[0, 1], sim_model, heads2[0])
        plotAttributions(sample, ax[1, 1], sim_model, heads2[1])
        plotAttributions(sample, ax[0, 2],sim_model, heads2[2])
        plotAttributions(sample, ax[1, 2], sim_model, heads2[3])
        plt.savefig(args.results_dir + 'Img' + str(i) + '_igs.png', bbox_inches='tight')
        break






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp7/', help='path for saving trained models')

    parser.add_argument('--sr', type=str, default='c')
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--use_softmax', type=bool, default=False)

    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=1) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/covid_segmentation/')
    args = parser.parse_args()
    print(args)
    main(args)
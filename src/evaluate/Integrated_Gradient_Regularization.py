import argparse
import sys
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/Patch_CLIP/')
import torch
import CLIP_Embedding
import Vision_Model
import MedDataHelpers
import utils
import numpy as np
import matplotlib.pyplot as plt
import saliency.core as saliency
import pickle

print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def myVisualizeImageGrayscale(image_3d, percentile=99, getmaxmin=False, overmax = None, overmin=None):
  r"""Returns a 3D tensor as a grayscale 2D tensor.
  This method sums a 3D tensor across the absolute value of axis=2, and then
  clips values at a given percentile.
  """
  image_2d = np.sum(np.abs(image_3d), axis=2)

  vmax = np.percentile(image_2d, percentile) if overmax is None else overmax
  vmin = np.min(image_2d) if overmin is None else overmin
  clipped = np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)
  if getmaxmin:
      return clipped, vmax, vmin
  else:
      return clipped


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
    try:
        outputs = output.class_logits[:, target_class_idx, 0]
    except:
        outputs = output[:, target_class_idx]
    grads = torch.autograd.grad(outputs, img, grad_outputs=torch.ones_like(outputs))
    grads = torch.movedim(grads[0], 1, 3)
    #print("final:", grads.shape)
    gradients = grads.cpu().detach().numpy()
    #print(gradients.shape)
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


def plot_ig_saliency(img, target, model, ax, use_abs = True, to_plot=True, method='integrated_gradients', name="Real CNN", overwrite_baseline = None, prespecify_range=None):
    headsDict = {'Cardiomegaly': 0, 'Edema': 1, 'Consolidation': 2, 'Atelectasis': 3, 'Pleural Effusion': 4,'No Finding': 5}
    if method == 'integrated_gradients':
        ig = saliency.IntegratedGradients()
        if overwrite_baseline is not None:
            baseline = overwrite_baseline.clone().permute(0,2,3,1).numpy().squeeze()
        else:
            baseline = np.zeros(img.shape)
        sig = ig.GetSmoothedMask(img, call_model_function_ig, {'model':model, 'targind':headsDict[target]}, x_steps=5, x_baseline=baseline, batch_size=5)
    elif method == 'guided_ig':
        ig = saliency.GuidedIG()
        baseline = np.zeros(img.shape)
        sig = ig.GetSmoothedMask(img, call_model_function_ig, {'model': model, 'targind': headsDict[target]}, x_steps=5,x_baseline=baseline)
    elif method == 'blur_ig':
        ig = saliency.BlurIG()
        sig = ig.GetSmoothedMask(img, call_model_function_ig, {'model': model, 'targind':headsDict[target]})
    elif method == 'xrai':
        ig = saliency.XRAI()
        baselines = [np.zeros(img.shape), np.ones(img.shape)]
        sig = ig.GetSmoothedMask(img, call_model_function_ig, {'model':model, 'targind':headsDict[target]}, baselines=baselines, batch_size=5, magnitude=True)
    else:
        oc = saliency.Occlusion()
        sig = oc.GetSmoothedMask(img, call_model_function_oc, {'model':model, 'targind':headsDict[target]}, size=15, value=0)

    if prespecify_range is not None:
        gs = myVisualizeImageGrayscale(sig, getmaxmin=False, overmax=prespecify_range[1], overmin=prespecify_range[0])
    else:
        gs, mymax, mymin = myVisualizeImageGrayscale(sig, getmaxmin=True)
    v_min, v_max = 0, 1

    if to_plot:
        ax.imshow(gs, plt.cm.plasma, vmin=v_min, vmax=v_max)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)

    if prespecify_range is not None:
        return gs
    else:
        return gs, mymax, mymin

def plotAttributions(sample, ax, sim_model, target='Cardiomegaly', heats = None, index = 0, name = "Real CNN", overwrite_baseline=None, prespecify_range=None):
    if prespecify_range is not None:
        smoothgrad = plot_ig_saliency(sample['images'][0].clone().permute(0,2,3,1).numpy().squeeze(), target,
                                  model = sim_model, ax = ax, name=name, overwrite_baseline=overwrite_baseline, prespecify_range = prespecify_range)
        return smoothgrad
    else:
        smoothgrad, mymax, mymin = plot_ig_saliency(sample['images'][0].clone().permute(0, 2, 3, 1).numpy().squeeze(), target,
                                      model=sim_model, ax=ax, name=name, overwrite_baseline=overwrite_baseline, prespecify_range = prespecify_range)
        return smoothgrad

def plot_original_image(sample, ax, heads, alpha=1.0, index=0, title=True, name=""):
    image = sample['images'][0]
    if not title:
        ax.imshow(image.permute(0, 2, 3, 1)[0, :, :, :].squeeze(), alpha=alpha)
    else:
        ax.imshow(utils.normalize(image), alpha=alpha)
        labels = [h for h in heads if sample['labels'][h] == 1]
        ax.set_title(name + "\nTrue Labels: " + str(labels))

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

    finetunedname1 = 'frozen_finetunedv4_best_model.pt'
    finetunedname2 = 'frozen_finetunedv2_best_model.pt'
    clip_reg = CLIP_Embedding.getCLIPModel(modelpath=args.je_model_path, modname=finetunedname1, freezeText=True, freezeCNNEncoder=False, eval=False)
    clip_unreg= CLIP_Embedding.getCLIPModel(modelpath=args.je_model_path_unreg, modname=finetunedname2, freezeText=True, freezeCNNEncoder=False, eval=False)
    cnn_normal = Vision_Model.getCNN(loadpath=args.vis_model_path, loadmodel=finetunedname1, freeze=False, eval=False)

    clip_reg = CLIP_Embedding.SimClassifier(heads = heads1, clip_model = clip_reg)
    clip_unreg = CLIP_Embedding.SimClassifier(heads=heads1, clip_model=clip_unreg)

    filters = MedDataHelpers.getFilters(args.je_model_path)
    dat = MedDataHelpers.getDatasets(source=args.sr, subset=[args.subset], heads=heads1, filters=filters, synthetic=False)

    baselinesample = dat[args.subset].__getitem__(1)
    baselinesample = baselinesample['images'][0]
    baselinesample = baselinesample[None, :, :, :]
    clip_regs, clip_unregs, cnns = [], [], []
    for i in range(234):
        for j, myhead in enumerate(heads1):
            sample = dat[args.subset].__getitem__(i)
            image1 = sample['images']
            for k, im in enumerate(image1):
                image1[k] = im[None, :, :, :]

            sample['images'] = image1

            fig, ax = plt.subplots(2, 2, figsize=(16, 7))
            clip_regs.append(plotAttributions(sample, ax[1, 0], clip_reg, myhead, name="Regularized CLIP"))
            clip_unregs.append(plotAttributions(sample, ax[1, 1],clip_unreg, myhead, name="Unregularized CLIP"))
            cnns.append(plotAttributions(sample, ax[0, 1],cnn_normal, myhead, name="Supervised CNN"))
            plot_original_image(sample, ax[0, 0], heads1, index=i, name=myhead + " gradient")
            plt.savefig(args.results_dir + 'Img' + str(i) + " " +  str(myhead) + '_igs.png', bbox_inches='tight')
        print(i)
        if i == 2:
            break
'''
    mydict = {}
    mydict['cnn_synth_synth'] = cnn_synth_synths
    mydict['cnn_norm_norm'] = cnn_norm_norms
    mydict['cnn_norm_synth'] = cnn_norm_synths
    mydict['cnn_synth_norm'] = cnn_synth_norms
    mydict['clip_synth_synth'] = clip_synth_synths
    mydict['clip_synth_norm'] = clip_synth_norms
    mydict['clip_norm_synth'] = clip_norm_synths
    mydict['clip_norm_norm'] = clip_norm_norms
    with open(args.results_dir + 'igs.pickle', 'wb') as handle:
        pickle.dump(mydict, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp2/', help='path for saving trained models')
    parser.add_argument('--je_model_path_unreg', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp3/', help='path for saving trained models')
    parser.add_argument('--vis_model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/cxr_cnn/exp2/', help='path for saving trained models')
    parser.add_argument('--vis_model_path_synth', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/synth_cxr_cnn/exp2/', help='path for saving trained models')

    parser.add_argument('--sr', type=str, default='c')
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--use_softmax', type=bool, default=False)

    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=1) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/integrated_gradients/')
    args = parser.parse_args()
    print(args)
    main(args)
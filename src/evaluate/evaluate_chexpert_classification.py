import argparse
import sys
import pickle
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/Patch_CLIP/')
import torch
import utils
import CLIP_Embedding
import Vision_Model
import MedDataHelpers
import numpy as np
import matplotlib.pyplot as plt

print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
from sklearn import metrics
from torchmetrics import AUROC
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def plot_roc(fprs, tprs, thresholds, heads, outm, aucs, args):
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {'Atelectasis': 'r', 'Cardiomegaly': 'tab:orange', 'Consolidation': 'g', 'Edema': 'c',
              'Pleural Effusion': 'tab:purple'}
    for i, h in enumerate(heads):
        ax.plot(fprs[h], tprs[h], color=colors[h], label=h + ", AUC = " + str(np.round(aucs[h], 4)))
    xrange = np.linspace(0, 1, 100)
    avgTPRS = np.zeros_like(xrange)
    for i, h in enumerate(heads):
        avgTPRS = avgTPRS + np.interp(xrange, fprs[h], tprs[h])
    avgTPRS = avgTPRS / 5
    ax.plot(xrange, avgTPRS, color='k', label="Average, AUC = " + str(np.round(aucs['Average'], 4)))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_title("ROC Curves for labels", size=30)
    ax.set_xlabel("False Positive Rate", size=24)
    ax.set_ylabel("True Positive Rate", size=24)
    ax.legend(prop={'size': 16})
    plt.savefig(args.results_dir + outm + "roc_curves.png", bbox_inches="tight")


def main(args):
    heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
    heads_order = np.array(['Pleural Effusion', 'Edema', 'Consolidation', 'Cardiomegaly', 'Atelectasis'])
    colors = {'Average':'k','Atelectasis': 'r', 'Cardiomegaly': 'tab:orange', 'Consolidation': 'g', 'Edema': 'c',
              'Pleural Effusion': 'tab:purple'}

    #all 8
    mod_order = ['Real CNN', 'Zeroshot Real CLIP Unregularized','Zeroshot Real CLIP Regularized', 'Finetuned Real CNN',
                 'Finetuned Real CLIP Unregularized', 'Finetuned Real CLIP Regularized',
                 'Shortcut CNN', 'Zeroshot Shortcut CLIP Unregularized', 'Zeroshot Shortcut CLIP Regularized',
                 'Finetuned Shortcut CNN', 'Finetuned Shortcut CLIP Unregularized', 'Finetuned Shortcut CLIP Regularized']

    vision_models_orig = ['cxr_cnn/exp2/', 'synth_cxr_cnn/exp2/']
    vision_models_fine = ['cxr_cnn/exp2/', 'synth_cxr_cnn/exp2/']
    je_models_zero = ['clip_regularized/exp2/', 'synth_clip_regularized/exp2/']
    je_models_fine = ['clip_regularized/exp2/', 'synth_clip_regularized/exp2/']
    je_models_zero_unreg = ['clip_regularized/exp3/', 'synth_clip_regularized/exp3/']
    je_models_fine_unreg = ['clip_regularized/exp3/', 'synth_clip_regularized/exp3/']

    vision_models_orig = [mod + 'best_model.pt' for mod in vision_models_orig]
    je_models_zero = [mod + 'best_model.pt' for mod in je_models_zero]
    je_models_zero_unreg = [mod + 'best_model.pt' for mod in je_models_zero_unreg]

    vCNN = '2'
    vunreg = '2'
    vreg = '4'
    if args.freeze:
        vision_models_fine = [mod  + 'frozen_finetunedv' + vCNN + '_best_model.pt' for mod in vision_models_fine]
        je_models_fine = [mod  +'frozen_finetunedv' + vreg + '_best_model.pt' for mod in je_models_fine]
        je_models_fine_unreg = [mod  +'frozen_finetunedv' + vunreg + '_best_model.pt' for mod in je_models_fine_unreg]
    else:
        vision_models_fine = [mod +'unfrozen_finetunedv' + vCNN + '_best_model.pt' for mod in vision_models_fine]
        je_models_fine = [mod +'unfrozen_finetunedv' + vreg + '_best_model.pt' for mod in je_models_fine]
        je_models_fine_unreg = [mod + 'unfrozen_finetunedv' + vunreg + '_best_model.pt' for mod in je_models_fine_unreg]


    all_models = vision_models_orig + vision_models_fine + je_models_zero + je_models_fine + je_models_zero_unreg + je_models_fine_unreg
    name_mods = {'Real CNN': vision_models_orig[0], 'Shortcut CNN': vision_models_orig[1], 'Finetuned Real CNN': vision_models_fine[0],
                 'Finetuned Shortcut CNN': vision_models_fine[1],
                 'Zeroshot Real CLIP Regularized': je_models_zero[0], 'Zeroshot Shortcut CLIP Regularized': je_models_zero[1],
                 'Finetuned Real CLIP Regularized': je_models_fine[0], 'Finetuned Shortcut CLIP Regularized': je_models_fine[1],
                 'Zeroshot Real CLIP Unregularized': je_models_zero_unreg[0],
                 'Zeroshot Shortcut CLIP Unregularized': je_models_zero_unreg[1],
                 'Finetuned Real CLIP Unregularized': je_models_fine_unreg[0],
                 'Finetuned Shortcut CLIP Unregularized': je_models_fine_unreg[1]
                 }
    mod_names = {v: k for k, v in name_mods.items()}

    if args.generate:
        if args.sr == 'chexpert' or args.sr == 'c':
            args.sr = 'c'
        if args.subset == 'a' or args.subset == 'all':
            subset = ['all']
        elif args.subset == 't' or args.subset == 'test':
            subset = ['test']

        models = {}
        all_filters = []
        for mname in mod_order:
            myv = name_mods[mname]
            if "Zeroshot" in mname or "CLIP" in mname:
                all_filters.append(MedDataHelpers.getFilters(args.je_model_path + myv))
                models[myv] = CLIP_Embedding.getCLIPModel(modelpath=args.je_model_path + myv, modname='', num_models=1, eval=True)
            elif "CLIP" in mname:
                all_filters.append(MedDataHelpers.getFilters(args.je_model_path + myv))
                models[myv] = Vision_Model.getCNN(num_heads = 5, loadpath=args.je_model_path + myv, loadmodel='', freeze=True)
            else:
                all_filters.append(MedDataHelpers.getFilters(args.vision_model_path + myv))
                models[myv] = Vision_Model.getCNN(num_heads = 5, loadpath = args.je_model_path + myv, loadmodel='', freeze=True)
            models[myv].train(False)

        if np.unique([set(filter) for filter in all_filters]).shape[0] > 1:
            raise Exception("Different filters used for different models, not comparable")
        else:
            filters = all_filters[0]
            dat = MedDataHelpers.getDatasets(source=args.sr, subset=subset, synthetic=False, filters = filters)  # Real
            DL = MedDataHelpers.getLoaders(dat, args)

            dat_synth = MedDataHelpers.getDatasets(source=args.sr, subset=subset, synthetic=True, get_good=True,
                                    get_overwrites=True, filters = filters)  # Synths
            DLsynth = MedDataHelpers.getLoaders(dat_synth, args)

            dat_adv = MedDataHelpers.getDatasets(source=args.sr, subset=subset, synthetic=True, get_adversary=True,
                                  get_overwrites=True, filters = filters)  # Advs
            DLadv = MedDataHelpers.getLoaders(dat_adv, args)

        fprs, tprs, thresholds, aucs, aucs_synth, aucs_adv = {}, {}, {}, {}, {}, {}
        all_aucs, all_synths, all_advs = {}, {}, {}
        ms = [name_mods[mname] for mname in mod_order]
        outm = [m.replace('/', '_') for m in ms]
        vision_models = [models[m].to(device) for m in ms]
        auroc = AUROC(pos_label=1)

        #Real
        with torch.no_grad():
            test_preds_list, test_targets = utils.get_all_preds_list(DL[subset[0]], vision_models, heads = heads, autochooses=mod_order)
        test_preds_list = [test_preds[0].cpu() for test_preds in test_preds_list]
        test_targets = test_targets.cpu()
        for j, test_preds in enumerate(test_preds_list):
            aucs = {}
            for i, h in enumerate(heads):
                fprs[h], tprs[h], thresholds[h] = metrics.roc_curve(test_targets[:, i].int().detach().numpy(), test_preds[:, i].detach().numpy())
                aucs[h] = auroc(test_preds[:, i], test_targets[:, i].int()).item()
            aucs['Average'] = np.mean(np.array([aucs[h] for h in heads]))
            #plot_roc(fprs, tprs, thresholds, heads, outm[j], aucs, args)
            print(outm[j], "Normal")
            print("Total AUC avg: ", aucs['Average'])
            for i, h in enumerate(heads):
                print(h, aucs[h])
            all_aucs[name_mods[mod_order[j]]] = aucs

        #Synth each label
        aucs_synth = [{} for m in vision_models]
        for i, h in enumerate(heads):
            with torch.no_grad():
                test_preds_synth_list, test_targets_synth = utils.get_all_preds_list(DLsynth[h], vision_models, heads=heads, autochooses=mod_order)
            test_preds_synth_list = [test_preds_synth[0].cpu() for test_preds_synth in test_preds_synth_list]
            test_targets_synth = test_targets_synth.cpu()
            for j, test_preds in enumerate(test_preds_synth_list):
                aucs_synth[j][h] = auroc(test_preds_synth_list[j][:, i], test_targets_synth[:, i].int()).item()

        for j, m in enumerate(vision_models):
            aucs_synth[j]['Average'] = np.mean(np.array([aucs_synth[j][h] for h in heads]))
            print(outm[j], "Synthetic")
            print("Total AUC avg: ", aucs_synth[j]['Average'])
            for i, h in enumerate(heads):
                print(h, aucs_synth[j][h])
            all_synths[name_mods[mod_order[j]]] = aucs_synth[j]

        # Adv each label
        aucs_adv = [{} for m in vision_models]
        for i, h in enumerate(heads):
            with torch.no_grad():
                test_preds_adv_list, test_targets_adv = utils.get_all_preds_list(DLadv[h], vision_models,
                                                                                heads=heads,
                                                                                autochooses=mod_order)
            test_preds_adv_list = [test_preds_adv[0].cpu() for test_preds_adv in test_preds_adv_list]
            test_targets_adv = test_targets_adv.cpu()
            for j, test_preds in enumerate(test_preds_adv_list):
                aucs_adv[j][h] = auroc(test_preds_adv_list[j][:, i], test_targets_adv[:, i].int()).item()

        for j, m in enumerate(vision_models):
            aucs_adv[j]['Average'] = np.mean(np.array([aucs_adv[j][h] for h in heads]))
            print(outm[j], "Adversarial")
            print("Total AUC avg: ", aucs_adv[j]['Average'])
            for i, h in enumerate(heads):
                print(h, aucs_adv[j][h])
            all_advs[name_mods[mod_order[j]]] = aucs_adv[j]


        vision_models = [vision_model.cpu() for vision_model in vision_models]

    if args.generate:
        with open(args.dat_dir + 'all_aucs.pickle', 'wb') as handle:
            pickle.dump([all_aucs, all_synths, all_advs], handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(args.dat_dir + 'all_aucs.pickle', 'rb') as handle:
            all_aucs, all_synths, all_advs = pickle.load(handle)

    print(all_aucs)
    print(all_aucs[name_mods["Finetuned Real CLIP Regularized"]])
    fig, axs = plt.subplots(3, 1, figsize = (6, 9), sharex=True)

    x = np.arange(len(mod_order))
    width = 0.1
    centeroffset = np.array([4, 2, 0, -2, -4])/2


    for z, m in enumerate(mod_order):
        axs[0].bar(x[z] + width * -3, all_aucs[name_mods[m]]['Average'], width, color='k')
        axs[0].text(x[z] + width * -3, 1.01, str(np.round(all_aucs[name_mods[m]]['Average'], 3)), color='k', fontweight='bold',
                    ha='left')
        axs[1].bar(x[z] + width * -3, all_synths[name_mods[m]]['Average'], width, color='k')
        axs[1].text(x[z] + width * -3, 1.01, str(np.round(all_synths[name_mods[m]]['Average'], 3)), color='k', fontweight='bold',
                    ha='left')
        axs[2].bar(x[z] + width * -3, all_advs[name_mods[m]]['Average'], width, color='k')
        axs[2].text(x[z] + width * -3, 1.01, str(np.round(all_advs[name_mods[m]]['Average'], 3)), color='k', fontweight='bold',
                    ha='left')
    for k, h in enumerate(heads_order):
        for z, m in enumerate(mod_order):
            axs[0].bar(x[z] + width * centeroffset[k], all_aucs[name_mods[m]][h], width, color = colors[h], alpha = 0.9)
            axs[1].bar(x[z] + width * centeroffset[k], all_synths[name_mods[m]][h],width, color = colors[h], alpha = 0.9)
            axs[2].bar(x[z] + width * centeroffset[k], all_advs[name_mods[m]][h],width, color = colors[h], alpha = 0.9)

    #ax.set_title("AUCS for various models")
    auc_names = ["Real AUC", "Shortcut AUC", "Adversarial AUC"]
    for k, ax in enumerate(axs):
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(auc_names[k], size=14, fontweight='bold')
        if k == 2:
            ax.set_xticks(x)
            ax.set_xticklabels(mod_order, rotation=30, ha='right')
            ax.set_xlabel("Model", size=14, fontweight='bold')

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles1 = [f("o", colors[i]) for i in colors.keys()]
    labels1 = [c for c in colors.keys()]
    axs[0].legend(handles1, labels1, loc=3, framealpha=1, title="Clinical Label", ncol=2)
    for k, ax in enumerate(axs):
        if len(mod_order) > 2:
            ax.axvline(x=1.5, color='k', label='axvline - full height')
        if len(mod_order) > 4:
            ax.axvline(x=3.5, color='k', label='axvline - full height')
        if len(mod_order) > 6:
            ax.axvline(x=5.5, color='k', label='axvline - full height')
        if len(mod_order) > 8:
            ax.axvline(x=7.5, color='k', label='axvline - full height')
        if len(mod_order) > 10:
            ax.axvline(x=9.5, color='k', label='axvline - full height')

        # ax.axhline(y=1.0, color='k')

    if "unfrozen" in je_models_fine[0]:
        plt.savefig(args.results_dir + "unfrozen_all_AUCs.png", bbox_inches="tight")
        print(args.results_dir + ('mimic_' if args.mimic else '') + "unfrozen_all_AUCS.png")
    else:
        plt.savefig(args.results_dir + "frozen_all_AUCs.png", bbox_inches="tight")
        print(args.results_dir + ('mimic_' if args.mimic else '') + "frozen_all_AUCS.png")






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_model_path', type=str, default = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/')
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/', help='path for saving trained models')

    parser.add_argument('--freeze', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--mimic', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--sr', type=str, default='chextest') #c, co
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--generate', type=bool, default=False, const=True, nargs='?', help='Regenerate aucs and ROC curves')
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/zeroshot/')
    parser.add_argument('--dat_dir', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/')
    args = parser.parse_args()
    print(args)
    main(args)
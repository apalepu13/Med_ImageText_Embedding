import argparse
import torch
import pickle
import Vision_Model
import MedDataHelpers
import numpy as np
import utils
import os
import Plotting
print("CUDA Available: " + str(torch.cuda.is_available()))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    rerun = True
    all_paths = [args.model_path_real, args.model_path_synth,
                 args.je_model_path_real, args.je_model_path_synth]
    modnames = ['vreal', 'vsynth', 'clipreal', 'clipsynth']
    heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
    usechex = True
    if rerun:
        if usechex:
            subset = ['finetunetrain', 'finetuneval']
            filters = MedDataHelpers.getFilters(args.model_path_real)
            realdat = MedDataHelpers.getDatasets(source='c', subset=subset, filters=filters)
        else:
            subset = ['train', 'val']
            filters = MedDataHelpers.getFilters(args.model_path_real)
            realdat = MedDataHelpers.getDatasets(source='m', subset=subset, filters=filters)

        DLs = MedDataHelpers.getLoaders(realdat, args, drop_last=True)
        train_data_loader_chexpert = DLs[subset[0]]
        val_data_loader_chexpert = DLs[subset[1]]

        vision_model_real = Vision_Model.getCNN(loadpath = args.model_path_real,num_heads = len(heads), loadmodel = 'best_model.pt', freeze=args.freeze).to(device)
        vision_model_synth = Vision_Model.getCNN(loadpath = args.model_path_synth,num_heads = len(heads), loadmodel = 'best_model.pt', freeze=args.freeze).to(device)
        je_vision_model_real = Vision_Model.getCNN(loadpath =args.je_model_path_real,num_heads = len(heads), loadmodel = 'best_model.pt', freeze=args.freeze).to(device)
        je_vision_model_synth = Vision_Model.getCNN(loadpath = args.je_model_path_synth,num_heads = len(heads), loadmodel='best_model.pt', freeze=args.freeze).to(device)

        all_paths = [args.model_path_real, args.model_path_synth,
                     args.je_model_path_real , args.je_model_path_synth]
        all_models = [vision_model_real, vision_model_synth, je_vision_model_real, je_vision_model_synth]
        modnames = ['vreal', 'vsynth', 'clipreal', 'clipsynth']
        all_params = [list(m.parameters()) for m in all_models]
        all_optimizers = [torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.000001) for params in all_params]
        #all_schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(all_optimizers[i], 'min', patience=5, factor=0.33) for i in range(len(all_optimizers))]
        train_losses = [[] for m in all_models]
        val_losses = [[] for m in all_models]
        best_valloss = [100000 for m in all_models]
        for epoch in range(args.num_epochs if usechex else 1):
            print(epoch)
            train_loss = utils.train_vision(train_data_loader_chexpert, all_models, args, epoch, all_optimizers, heads=heads, list_mods = True)
            for j, m in enumerate(all_models):
                #train_loss = utils.train_vision(train_data_loader_chexpert, all_models[j], args, epoch, all_optimizers[j], heads = heads)
                train_losses[j].append(train_loss[j])

            valloss = utils.validate_vision(val_data_loader_chexpert, all_models, heads=heads, list_mods = True)
            #for i in range(len(valloss)):
            #    all_schedulers[i].step(valloss[i])

            for j, m in enumerate(all_models):
                print(modnames[j], valloss[j], "*" if valloss[j] < best_valloss[j] else "")
                val_losses[j].append(valloss[j])
                if valloss[j] < best_valloss[j]:
                    torch.save({'model_state_dict': m.state_dict(), 'optimizer_state_dict': all_optimizers[j].state_dict(), 'args': args},
                               os.path.join(all_paths[j],("mimic_" if not usechex else "")+("frozen_" if args.freeze else "unfrozen_") +  'finetuned_best_model.pt'))
                    best_valloss[j] = valloss[j]

        for j, m in enumerate(all_models):
            with open(all_paths[j] +("mimic_" if not usechex else "") + ("frozen_" if args.freeze else "unfrozen_")+ 'finetune_losses.pickle', 'wb') as handle:
                pickle.dump([train_losses[j], val_losses[j]], handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        train_losses = []
        val_losses = []
        for j, m in enumerate(modnames):
            with open(all_paths[j] +("mimic_" if not usechex else "") + ("frozen_" if args.freeze else "unfrozen_")+ 'finetune_losses.pickle', 'rb') as handle:
                print(args.freeze)
                train_loss, val_loss = pickle.load(handle)
                print(len(train_loss))
                train_losses.append(train_loss)
                val_losses.append(val_loss)

    Plotting.plot_all_train_val_losses(train_losses, val_losses, modnames, detail=("frozenorig" if args.freeze else "unfrozenorig"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path_real', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp2/', help='path for saving trained models')
    parser.add_argument('--je_model_path_synth', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/synth_clip_regularized/exp2/')
    parser.add_argument('--model_path_real', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/cxr_cnn/exp2/',help='path for saving trained models')
    parser.add_argument('--model_path_synth', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/synth_cxr_cnn/exp2/')

    parser.add_argument('--debug', type=bool, default=False, const = True, nargs='?', help='debug mode, dont save')
    parser.add_argument('--log_step', type=int, default=500, help='step size for printing log info')
    parser.add_argument('--val_step', type=int, default=1, help='step size for printing val info')
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32)
    #parser.add_argument('--learning_rate', type=float, default=0.00003333)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--freeze', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--results_dir', type=str, default="/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/training/")
    args = parser.parse_args()
    print(args)
    main(args)
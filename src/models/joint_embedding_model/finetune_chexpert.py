import argparse
import torch
import pickle
from CNN import *
import numpy as np
from HelperFunctions import *
import os
print("CUDA Available: " + str(torch.cuda.is_available()))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main(args):
    heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
    subset = ['valtrain', 'valval']
    chexpert_dat = getDatasets(source='c', subset=subset)
    [train_data_loader_chexpert, val_data_loader_chexpert]= getLoaders(chexpert_dat, args, subset=subset)
    vision_model_real = getVisionClassifier(args.model_path_real, 'best_model.pt', device, args.embed_size, heads, getFrozen=args.freeze)
    vision_model_synth = getVisionClassifier(args.model_path_synth, 'best_model.pt', device, args.embed_size, heads, getFrozen=args.freeze)
    je_vision_model_real = getVisionClassifier(args.je_model_path_real,'best_model.pt', device,args.embed_size, heads, je=True, getFrozen=args.freeze)
    je_vision_model_synth = getVisionClassifier(args.je_model_path_synth, 'best_model.pt', device, args.embed_size, heads,je=True, getFrozen=args.freeze)

    all_paths = [args.model_path_real, args.model_path_synth,
                 args.je_model_path_real , args.je_model_path_synth]
    all_models = [vision_model_real, vision_model_synth, je_vision_model_real, je_vision_model_synth]
    all_params = [list(m.parameters()) for m in all_models]
    criterion = torch.nn.BCEWithLogitsLoss()
    all_optimizers = [torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.000001) for params in all_params]

    all_train_losses = []
    all_val_losses = []
    best_valloss = [100000, 100000, 100000, 100000]
    for epoch in range(args.num_epochs):
        epoch_train_losses = []
        epoch_val_losses = []
        train_losses = [[] for m in all_models]

        for i, (ims, df) in enumerate(train_data_loader_chexpert):
            for j, m in enumerate(all_models):
                m.train()
                trainloss = train_vision(device, m, ims, ims, df, heads, useOne=True)
                trainloss.backward()
                train_losses[j].append(trainloss.detach().numpy())
                all_optimizers[j].step()

        for j, m in enumerate(all_models):
            epoch_train_losses.append(np.mean(np.array(train_losses[j])))
        all_train_losses.append(epoch_train_losses)

        for j, m in enumerate(all_models):
            valloss = validate_vision(device, val_data_loader_chexpert, m, heads, criterion, source="Chexpert")
            epoch_val_losses.append(valloss)
            if valloss < best_valloss[j]:
                torch.save({'model_state_dict': m.state_dict(), 'optimizer_state_dict': all_optimizers[j].state_dict(), 'args': args},
                           os.path.join(all_paths[j], 'finetuned_best_model.pt'))
                best_valloss[j] = valloss

        all_val_losses.append(epoch_val_losses)

    with open(args.results_dir + 'chexpert_finetune_losses.pickle', 'wb') as handle:
        pickle.dump([all_train_losses, all_val_losses], handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path_real', type=str, default='../../../../models/je_model/exp2/', help='path for saving trained models')
    parser.add_argument('--je_model_path_synth', type=str, default='../../../../models/je_model/synth/exp2/')
    parser.add_argument('--model_path_real', type=str, default='../../../../models/vision_model/vision_CNN_real/',help='path for saving trained models')
    parser.add_argument('--model_path_synth', type=str, default='../../../../models/vision_model/vision_CNN_synthetic/')

    parser.add_argument('--debug', type=bool, default=False, const = True, nargs='?', help='debug mode, dont save')
    parser.add_argument('--log_step', type=int, default=500, help='step size for printing log info')
    parser.add_argument('--val_step', type=int, default=1, help='step size for printing val info')
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--freeze', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--results_dir', type=str, default="/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/training/")
    args = parser.parse_args()
    print(args)
    main(args)
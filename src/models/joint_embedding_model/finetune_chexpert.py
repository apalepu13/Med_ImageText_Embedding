import argparse
import torch
import torch.nn as nn
from Transformer import *
from CNN import *
from jointEmbedding import JointEmbeddingModel
from Pretraining import *
import os
print("CUDA Available: " + str(torch.cuda.is_available()))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main(args):
    heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
    subset = ['val']
    chexpert_dat = getDatasets(source='c', subset=subset)
    [train_data_loader_chexpert]= getLoaders(chexpert_dat, args, subset=subset)
    vision_model_real = getVisionClassifier(args.model_path_real, args.model_real, device, args.embed_size, heads, getFrozen=True)
    vision_model_synth = getVisionClassifier(args.model_path_synth, args.model_synth, device, args.embed_size, heads, getFrozen=True)
    je_vision_model_real = getVisionClassifier(args.je_model_path_real,args.je_model_real, device,args.embed_size, heads, je=True, getFrozen=True)
    je_vision_model_synth = getVisionClassifier(args.je_model_path_synth, args.je_model_synth, device, args.embed_size, heads,je=True, getFrozen=True)

    all_paths = [args.model_path_real, args.model_path_synth,
                 args.je_model_path_real , args.je_model_path_synth]
    all_nums = [14, 14, 28, 12]
    all_models = [vision_model_real, vision_model_synth, je_vision_model_real, je_vision_model_synth]
    all_params = [list(m.parameters()) for m in all_models]
    criterion = torch.nn.BCEWithLogitsLoss()
    all_optimizers = [torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.000001) for params in all_params]

    for epoch in range(args.num_epochs):
        for i, (ims, df) in enumerate(train_data_loader_chexpert):
            for j, m in enumerate(all_models):
                m.train()
                trainloss = train_vision(device, m, ims, ims, df, heads, useOne=True)
                trainloss.backward()
                all_optimizers[j].step()

    for j, m in enumerate(all_models):
        if j < 2:
            torch.save({'model_state_dict': m.state_dict(),
                'optimizer_state_dict': all_optimizers[j].state_dict(),
                'args': args}, os.path.join(all_paths[j], 'finetuned_model-{}.pt'.format(all_nums[j])))
        else:
            torch.save({'model_state_dict': m.state_dict(),
                        'optimizer_state_dict': all_optimizers[j].state_dict(),
                        'args': args}, os.path.join(all_paths[j], 'finetuned_je_model-{}.pt'.format(all_nums[j])))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path_real', type=str, default='../../../../models/je_model/exp6/', help='path for saving trained models')
    parser.add_argument('--je_model_real', type=str, default='je_model-28.pt', help='path from root to model')
    parser.add_argument('--je_model_path_synth', type=str, default='../../../../models/je_model/synth/exp7/')
    parser.add_argument('--je_model_synth', type=str, default='je_model-12.pt')

    parser.add_argument('--model_path_real', type=str, default='../../../../models/vision_model/vision_CNN_real/',help='path for saving trained models')
    parser.add_argument('--model_real', type=str, default='model-14.pt', help='path from root to model')
    parser.add_argument('--model_path_synth', type=str, default='../../../../models/vision_model/vision_CNN_synthetic/')
    parser.add_argument('--model_synth', type=str, default='model-14.pt')

    parser.add_argument('--debug', type=bool, default=False, const = True, nargs='?', help='debug mode, dont save')
    parser.add_argument('--log_step', type=int, default=500, help='step size for printing log info')
    parser.add_argument('--val_step', type=int, default=1, help='step size for printing val info')
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--freeze', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--baseline', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--super_baseline', type=bool, default=False, const=True, nargs='?')
    args = parser.parse_args()
    print(args)
    main(args)
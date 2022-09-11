
import argparse
import torch
import numpy as np
import sys
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/joint_embedding_model/')
import torch.nn as nn
from CNN import *
from jointEmbedding import JointEmbeddingModel
from HelperFunctions import *
from torchmetrics import AUROC
from Transformer import *
print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    subset = ['mscxr']

    model_root_dir = args.model_path
    eval_model = args.model
    loadpath = model_root_dir + eval_model
    if device == 'cuda':
        checkpoint = torch.load(loadpath)
    else:
        checkpoint = torch.load(loadpath, map_location=torch.device('cpu'))
    filters = getFilters(loadpath)
    msdat = getDatasets(source='mimic_cxr', subset=subset, filters = filters)
    [DL] = getLoaders(msdat, args, subset=subset)

    je_model = JointEmbeddingModel(args.embed_size, pool_first = "local" not in args.model_path).to(device)
    je_model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    if not hasattr(checkpoint['args'], 'vit') or not checkpoint['args'].vit:
        cnn_model = je_model.cnn
    else:
        cnn_model = je_model.vit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/local_je_model/', help='path for saving trained models')
    parser.add_argument('--model', type=str, default='exp1/je_model-20.pt', help='path from root to model')
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--results_dir', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/localsims/')
    args = parser.parse_args()
    print(args)
    main(args)
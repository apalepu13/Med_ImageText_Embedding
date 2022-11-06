import time
t = time.time()
import argparse
import torch
print("CUDA Available: " + str(torch.cuda.is_available()))
import sys
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/Patch_CLIP/')
import utils
import MedDataHelpers
import CLIP_Embedding
import numpy as np
import pandas as pd
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/evaluate/')

def main(args):
    heads = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    dset = MedDataHelpers.getDatasets(source='chextest', subset=[args.input_path])
    DLs = MedDataHelpers.getLoaders(dset, shuffle=False)
    DL = DLs[args.input_path]
    studies = []
    for i, samples in enumerate(DL):
        studies = studies + samples['study']
    print(studies)

    clip_models = CLIP_Embedding.getCLIPModel(args.je_model_path, num_models = 10)
    tot_preds = np.zeros((len(DL.dataset), 5))
    for mod in clip_models:
        preds, _ = utils.get_all_preds(DL, mod, similarity=True, heads = heads, getlabels=False, normalization=True)
        tot_preds += preds[0].detach().cpu().numpy()
    tot_preds = tot_preds / len(clip_models)
    col_names = heads
    # Declare pandas.DataFrame object
    df = pd.DataFrame(data=tot_preds, columns=heads)
    df['Study'] = np.array(studies)
    df = df.groupby('Study').mean().reset_index()
    df.to_csv(args.output_csv_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #model information
    parser.add_argument('input_path')
    parser.add_argument('output_csv_path')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp1/', help='path for saving trained models')
    args = parser.parse_args()
    print(args)
    main(args)
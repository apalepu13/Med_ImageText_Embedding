import time
t = time.time()
import argparse
import torch
print("CUDA Available: " + str(torch.cuda.is_available()))
from utils import *
import sys
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/evaluate/')
from os.path import exists
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

elapsed = time.time() - t
print("Start (time = " + str(elapsed) + ")")


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    t = time.time()
    # Start experiment

    mod_path = args.model_path + args.model + "/" if not args.synthetic else args.model_path + args.synthmodel
    exp_path = getExperiment(args, mod_path)
    start, je_model, params, optimizer, best_val_loss = startExperiment(args, exp_path)

    filts = 'impression'
    if args.findings_transformer:
        filts = ''

    if exists(exp_path + '/filters.txt'):
        filters = MedDataHelpers.getFilters(exp_path)
        if set(filters) != set(MedDataHelpers.getFilters(exp_path, overwrite= filts, toprint=False)):
            raise Exception("Error: entered filters differ from those previously used")
    else:
        filters = MedDataHelpers.getFilters(exp_path, overwrite = filts)
        if exp_path != 'debug':
            with open(exp_path + '/filters.txt', 'w') as f:
                f.write(filts)
    # Build data
    if args.debug:
        subset = ['tinytrain', 'tinyval']
    else:
        subset = ['train', 'val']
    t, v = subset[0], subset[1]

    (print("Real Images") if not args.synthetic else print("Synthetic Images"))
    mimic_dat = MedDataHelpers.getDatasets(source='m', subset = subset, synthetic = args.synthetic, augs = 1, filters = filters)
    dls = MedDataHelpers.getLoaders(mimic_dat, args)
    train_data_loader_mimic, val_data_loader_mimic = dls[t], dls[v]
    total_step_mimic = len(train_data_loader_mimic)
    assert (args.resume or start == 0)
    # Train and validate

    for epoch in range(start, args.num_epochs):
        je_model.train()
        tmimic = time.time()
        train_loss, train_losses = train(train_data_loader_mimic, je_model, args, epoch, optimizer, total_step_mimic)

        print("Mimic Epoch time: " + str(time.time() - tmimic))
        if epoch % args.val_step == 0:
            print("Validating/saving model")
            je_model.eval()
            tval = time.time()
            val_loss, val_losses = validate(val_data_loader_mimic, je_model, args)
            if not args.debug:
                torch.save({'epoch': epoch+1,
                            'model_state_dict': je_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_loss': best_val_loss,
                            'val_loss': val_loss,
                            'train_loss': train_loss,
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'args': args}, os.path.join(exp_path, 'je_model-{}.pt'.format(epoch)))
                if val_loss <= best_val_loss:
                    print("Best model so far!")
                    best_val_loss = val_loss
                    torch.save({'epoch': epoch + 1,
                                'model_state_dict': je_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_val_loss':best_val_loss,
                                'val_loss': val_loss,
                                'args': args}, os.path.join(exp_path, 'best_model.pt'))


            print("Val time " + str(time.time() - tval))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #model information
    parser.add_argument('--model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/', help='path for saving trained models')
    parser.add_argument('--model', type=str, default='clip_regularized')
    parser.add_argument('--synthmodel', type=str, default='synth_clip_regularized')
    parser.add_argument('--findings_transformer', type=bool, default=False, const=True, nargs='?')
    #special options
    parser.add_argument('--synthetic', type=bool, default=False, const=True, nargs='?',
                        help='Train on synthetic dataset')
    parser.add_argument('--resume', type=int, default=0, const=-1, nargs='?')
    parser.add_argument('--debug', type=bool, default=False, const=True, nargs='?', help='debug mode, dont save')
    #entropy_params
    parser.add_argument('--lam_words', type=float, default=0.0)
    parser.add_argument('--lam_patches', type=float, default=0.0)
    parser.add_argument('--steep_entropy', type=bool, default=False, const=True, nargs='?', help='To use steep entropy')

    #Training parameters
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16) #32 vs 16
    parser.add_argument('--learning_rate', type=float, default=.0001) #.0001
    parser.add_argument('--log_step', type=int, default=500, help='step size for printing log info')
    parser.add_argument('--val_step', type=int, default=1, help='step size for printing val info')
    args = parser.parse_args()
    print(args)
    main(args)
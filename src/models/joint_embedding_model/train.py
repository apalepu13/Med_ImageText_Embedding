import time
t = time.time()
import argparse
import torch
print("CUDA Available: " + str(torch.cuda.is_available()))
from Transformer import *
from HelperFunctions import *
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
    mod_path = args.model_path + args.model + "/"
    exp_path = getExperiment(args, mod_path)
    start, je_model, params, optimizer, best_val_loss = startExperiment(args, exp_path)
    tokenizer = Report_Tokenizer()
    filts = 'frontal'
    if exists(exp_path + '/filters.txt'):
        filters = getFilters(exp_path)
        if set(filters) != set(getFilters(exp_path, overwrite= filts, toprint=False)):
            print("Warning: entered filters differ from those previously used.")
    else:
        filters = getFilters(exp_path, overwrite = filts)
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
    mimic_dat = getDatasets(source='m', subset = subset, synthetic = args.synthetic, augs = 2, filters = filters)
    dls = getLoaders(mimic_dat, args, subset=subset)
    train_data_loader_mimic, val_data_loader_mimic = dls[t], dls[v]
    total_step_mimic = len(train_data_loader_mimic)
    assert (args.resume or start == 0)
    # Train and validate

    for epoch in range(start, args.num_epochs):
        je_model.train()
        tmimic = time.time()
        train_loss, train_losses = train(train_data_loader_mimic, je_model, tokenizer, args, epoch, optimizer, total_step_mimic)

        print("Mimic Epoch time: " + str(time.time() - tmimic))
        if epoch % args.val_step == 0:
            print("Validating/saving model")
            je_model.eval()
            tval = time.time()
            val_loss, val_losses = validate(val_data_loader_mimic, tokenizer, je_model, args)
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
    parser.add_argument('--model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/', help='path for saving trained models')
    parser.add_argument('--model', type=str, default='attn_model')
    parser.add_argument('--log_step', type=int, default=500, help='step size for printing log info')
    parser.add_argument('--val_step', type=int, default=2, help='step size for printing val info')
    parser.add_argument('--resume', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--desc', type=str, default="", help='experiment description')
    parser.add_argument('--debug', type=bool, default=False, const = True, nargs='?', help='debug mode, dont save')
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--steep_entropy', type=bool, default=False, const=True, nargs='?', help='To use steep entropy')
    parser.add_argument('--synthetic', type =bool, default=False, const=True, nargs='?', help='Train on synthetic dataset')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32) #32 vs 16
    parser.add_argument('--learning_rate', type=float, default=.0001) #.0001
    args = parser.parse_args()
    print(args)
    main(args)
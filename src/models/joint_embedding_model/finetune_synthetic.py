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
    subset = ['train', 'val', 'test']
    chexpert_dat = getDatasets(source='c', subset=subset)
    train_data_loader_chexpert, val_data_loader_chexpert, test_data_loader_chexpert = getLoaders(chexpert_dat, args, subset=subset)

    model_root_dir = args.model_path
    eval_model = args.model

    best_val_loss = 10000000
    loadpath = model_root_dir + eval_model
    checkpoint = torch.load(loadpath)
    if not args.super_baseline:
        je_model = JointEmbeddingModel(args.embed_size)
    else:
        je_model = JointEmbeddingModel(args.embed_size, imagenet=False)

    if not args.baseline and not args.super_baseline:
        print("Loading pretrained model")
        je_model.load_state_dict(checkpoint['model_state_dict'])

    tokenizer = Bio_tokenizer()
    text_model = je_model.transformer

    heads = ['Sex', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
             'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other']
    theirheads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
    cnn_model = CNN_Classifier(je_model.cnn, args.embed_size, freeze=args.freeze, num_heads=len(heads)).to(device)
    params = list(cnn_model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.000001)

    total_step_chexpert = len(train_data_loader_chexpert)
    total_val_chexpert = len(val_data_loader_chexpert)
    for epoch in range(args.num_epochs):
        cnn_model.train()
        step = 0
        for i, (ims, df) in enumerate(train_data_loader_chexpert):
            cnn_model.zero_grad(set_to_none=True)
            images = ims.to(device)
            preds = cnn_model(images).to(device)
            loss = 0.0
            for i, h in enumerate(heads):
                label = df[h]
                if h == 'Edema' or h == 'Atelectasis':
                    label[label == -1.0] = 1
                else:
                    label[label == -1.0] = 0
                label[label == 0.0] = 0
                label[label == 1.0] = 1
                label[torch.isnan(label)] = 0
                label = label.float().to(device)
                loss += criterion(preds[:, i].to(device), label)
            loss.backward()
            optimizer.step()
            step += 1
            if step % args.log_step == 0:
                print('Chexpert Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, step, total_step_chexpert, loss.item()))
        if epoch % args.val_step == 0:
            cnn_model.eval()
            val_loss = []
            with torch.no_grad():
                for i, (ims, df) in enumerate(val_data_loader_chexpert):
                    images = ims.to(device)
                    preds = cnn_model(images)
                    v = 0.0
                    for i, h in enumerate(heads):
                        label = df[h]
                        if h == 'Edema' or h == 'Atelectasis':
                            label[label == -1.0] = 1
                        else:
                            label[label == -1.0] = 0
                        label[label == 0.0] = 0
                        label[label == 1.0] = 1
                        label[torch.isnan(label)] = 0
                        label = label.float().to(device)
                        v += criterion(preds[:, i].to(device), label)
                    val_loss.append(v.cpu())
            val_loss = np.mean(np.array(val_loss))
            print('Chexpert Val Loss, epoch ' + str(epoch) + ': ' + str(val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.freeze:
                    print("Saving frozen")
                    torch.save({'epoch': epoch + 1,
                            'model_state_dict': cnn_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': best_val_loss,
                            'args': args}, os.path.join(args.model_path, 'exp2/finetuned/chexpert_frozen.pt'))
                elif args.super_baseline:
                    print("Saving super baseline")
                    torch.save({'epoch': epoch + 1,
                                'model_state_dict': cnn_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_loss': best_val_loss,
                                'args': args}, os.path.join(args.model_path, 'exp2/finetuned/chexpert_super_baseline.pt'))
                elif args.baseline:
                    print("Saving baseline")
                    torch.save({'epoch': epoch + 1,
                                'model_state_dict': cnn_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_loss': best_val_loss,
                                'args': args}, os.path.join(args.model_path, 'exp2/finetuned/chexpert_baseline.pt'))
                else:
                    print("Saving unfrozen")
                    torch.save({'epoch': epoch + 1,
                                'model_state_dict': cnn_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_loss': best_val_loss,
                                'args': args}, os.path.join(args.model_path, 'exp2/finetuned/chexpert_unfrozen.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../../../../models/je_model/synth/', help='path for saving trained models')
    parser.add_argument('--exp', type=str, default='exp1/', help='path from root to model')
    parser.add_argument('--model', type=str, default='je_model-76.pt', help='path from root to model')
    #exp1, je-76 (4.405)

    parser.add_argument('--synth_mim_dir', type=str, default='/n/scratch3/users/a/anp2971/datasets/synthetic_mimic_cxr/')
    parser.add_argument('--synth_mim_file', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/synthetic/mimic_synthetic.csv')
    parser.add_argument('--synth_chx_dir', type=str, default='/n/scratch3/users/a/anp2971/datasets/synthetic_chex/')
    parser.add_argument('--synth_chx_train_file', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/synthetic/chex_train_synthetic.csv')
    parser.add_argument('--synth_chx_test_file', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/synthetic/chex_test_synthetic.csv')

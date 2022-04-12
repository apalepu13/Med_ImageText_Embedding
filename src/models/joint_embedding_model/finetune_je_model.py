import argparse
import torch
import torch.nn as nn
from Transformer import *
from Vision_Transformer import *
from CNN import *
from jointEmbedding import JointEmbeddingModel
from Pretraining import *
import os
print("CUDA Available: " + str(torch.cuda.is_available()))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main(args):
    heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
    if "synth" in args.je_model_path:
        do_synth = True
    else:
        do_synth = False

    dat_mimic = getDatasets(source='mimic_cxr', subset=['test', 'tiny'], synthetic=do_synth, heads=heads, get_text=False)
    [DLmimic, DLval] = getLoaders(dat_mimic, args, subset=['test', 'tiny'])

    je_model_path = args.je_model_path + args.je_model
    if device == 'cuda':
        checkpoint = torch.load(je_model_path)
    else:
        checkpoint = torch.load(je_model_path, map_location=torch.device('cpu'))

    je_model = JointEmbeddingModel(args.embed_size).to(device)



    je_vision_classifier = VisionClassifier(len(heads), args.embed_size).to(device)
    model_dict = je_vision_classifier.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    je_vision_classifier.load_state_dict(pretrained_dict, strict=False)

    for param in je_vision_classifier.parameters():
        param.requires_grad = False
    for param in je_vision_classifier.classification_head.parameters():
        param.requires_grad = True

    #print(je_vision_classifier)
    params = list(je_vision_classifier.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.000001)
    total_step_train = len(DLmimic)
    best_val_loss = 10000000000
    for epoch in range(args.num_epochs):
        je_vision_classifier.train()
        step = 0
        for i, (im1, im2, df, study) in enumerate(DLmimic):
            je_vision_classifier.zero_grad(set_to_none=True)
            images = im1.to(device)
            preds = je_vision_classifier(images).to(device)
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
                print('MIMIC Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, step, total_step_train, loss.item()))
        if epoch % args.val_step == 0:
            je_vision_classifier.eval()
            val_loss = []
            with torch.no_grad():
                for i, (im1, im2, df, study) in enumerate(DLval):
                    images = im1.to(device)
                    preds = je_vision_classifier(images)
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
            print('MIMIC Val Loss, epoch ' + str(epoch) + ': ' + str(val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving model")
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': je_vision_classifier.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': best_val_loss,
                            'args': args}, os.path.join(args.je_model_path, "finetuned_" + args.je_model))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--je_model_path', type=str,
    #                    default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/synth/exp2/')
    #parser.add_argument('--je_model', type=str, default='je_model-44.pt')
    parser.add_argument('--je_model_path', type=str,
                        default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/exp5/')
    parser.add_argument('--je_model', type=str, default='je_model-94.pt')
    parser.add_argument('--synth', type=bool, default=False, const=True, nargs='?', help='Train on synthetic dataset')
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32)  # 32 normally
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--log_step', type=int, default=1, help='step size for printing log info')
    parser.add_argument('--val_step', type=int, default=2, help='step size for printing val info')
    parser.add_argument('--learning_rate', type=float, default=.0001) #.0001
    args = parser.parse_args()
    main(args)


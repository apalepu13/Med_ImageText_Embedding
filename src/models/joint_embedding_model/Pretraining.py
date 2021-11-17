import os
import regex as re
import numpy as np
from Data_Loading import *
import torch

def getDatasets(source, subset = ['train', 'val', 'test']):
    s = source
    datlist = {}
    for sub in subset:
        datlist[sub] = Image_Text_Dataset(source=s, group=sub)
    return datlist

def getLoaders(datasets, args, subset = ['train', 'val', 'test']):
    loaders = []
    for sub in subset:
        loaders.append(DataLoader(datasets[sub], batch_size=args.batch_size, shuffle=True, num_workers=16, prefetch_factor=2, pin_memory=True))
    return loaders

def getExperiment(args):
    exp = args.exp
    if exp is -1:
        if os.listdir(os.path.join(args.model_path)):
            all_files = os.listdir(os.path.join(args.model_path))
            je_exps = [exp for exp in all_files if 'exp' in exp]
            num = [int(re.search('\d+', exp).group(0)) for exp in je_exps]
            highest_ind = np.argmax(np.array(num))
            highest = num[highest_ind]
            if not args.resume:
                highest = highest + 1
            fp = os.path.join(args.model_path, 'exp'+str(highest))
        else:
            print("Model doesn't exist, creating directory")
            if not args.debug:
                os.makedirs(args.model_path)
                fp = os.path.join(args.model_path, 'exp'+str(exp))
    else:
        fp = os.path.join(args.model_path, 'exp'+str(exp))
    return fp

def startExperiment(args, je_model, optimizer, fp):
    if args.resume:
        if os.listdir(os.path.join(fp)):
            all_files = os.listdir(os.path.join(fp))
            je_files = [file for file in all_files if 'je_model' in file]
            num = [int(re.search('\d+', file).group(0)) for file in je_files]
            highest = np.argmax(np.array(num))
            loadpath = os.path.join(fp, np.array(je_files)[highest])
            print("Loading " + loadpath)
            checkpoint = torch.load(loadpath)
            je_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start = checkpoint['epoch']
            best_val_loss = checkpoint['val_loss']
        else:
            print("Experiment doesnt exist", fp)
    else:
        print("Starting from scratch")
        start = 0
        best_val_loss = 10000000000
        if not args.debug:
            os.makedirs(fp)
            txt = args.desc
            with open(os.path.join(fp, "desc.txt"), "w") as text_file:
                text_file.write(txt)
    return start, best_val_loss

def train(device, je_model, ims, texts, tokenizer, criterion, optimizer, loss_weight = 1, step = 0):
    je_model.zero_grad(set_to_none=True)
    # Set mini-batch dataset
    images = ims.to(device)
    texts = tokenizer.do_encode(texts=texts).to(device)

    # Forward, backward and optimize
    im_logits, text_logits = je_model(images, texts)

    loss_a = criterion(im_logits, torch.tensor(np.arange(im_logits.shape[0])).to(device))
    loss_b = criterion(text_logits, torch.tensor(np.arange(im_logits.shape[0])).to(device))
    loss = (loss_a + loss_b) / 2
    loss = loss*loss_weight
    loss.backward()
    optimizer.step()
    return je_model, loss

def validate(device, val_data_loader, tokenizer, je_model, criterion, source = "MIMIC", proportion = 1.0):
    vlosses = []
    with torch.no_grad():
        for j, (valims, valtexts) in enumerate(val_data_loader):
            gen = np.random.rand(1)
            if gen >= proportion:
                continue

            valims = valims.to(device)
            valtexts = tokenizer.do_encode(texts=valtexts).to(device)
            val_im, val_t = je_model(valims, valtexts)
            loss_a = criterion(val_im, torch.tensor(np.arange(val_im.shape[0])).to(device))
            loss_b = criterion(val_t, torch.tensor(np.arange(val_im.shape[0])).to(device))
            loss = (loss_a + loss_b) / 2
            vlosses.append(loss.cpu())

    val_loss = np.mean(np.array(vlosses))
    print(source + ' Val Loss: ' + str(val_loss))
    return val_loss

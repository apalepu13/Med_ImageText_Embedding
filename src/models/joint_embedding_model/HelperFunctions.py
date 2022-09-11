import regex as re
from Data_Loading import *
import jointEmbedding
import torch
import torch.nn as nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getExperiment(args, mp):
    if args.debug:
        return "debug"

    if not os.listdir(os.path.join(mp)):
        print("No models exist, creating directory")
        fp = os.path.join(mp, 'exp1')
    else:
        all_files = os.listdir(os.path.join(mp))
        je_exps = [exp for exp in all_files if 'exp' in exp]
        num = [int(re.search('\d+', exp).group(0)) for exp in je_exps]
        highest_ind = np.argmax(np.array(num))
        highest = num[highest_ind]
        if not args.resume:
            highest = highest + 1
        fp = os.path.join(mp, 'exp'+str(highest))
    return fp

def writeArgs(fp, args):
    '''
    Document args used to train
    '''
    writestr = str(args)
    with open(fp + '/args.txt', 'w') as f:
        f.write(writestr)

def startExperiment(args, fp):
    '''
    Initialize variables for experiment:
    start (epoch), je_model, params, optimizer, best_val_loss
    '''
    je_model = jointEmbedding.JointEmbeddingModel(args.embed_size).to(device)
    params = list(je_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.000001)
    if fp == "debug":
        return 0, je_model, params, optimizer, 100000

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
            best_val_loss = checkpoint['best_val_loss'] if 'best_val_loss' in checkpoint.keys() else checkpoint['val_loss']
        else:
            raise Exception("Experiment doesn't exist: " + fp)
    else:
        print("Starting from scratch")
        start = 0
        best_val_loss = 1000000000
        if not args.debug:
            os.makedirs(fp)
            writeArgs(fp, args)
            txt = args.desc
            with open(os.path.join(fp, "desc.txt"), "w") as text_file:
                text_file.write(txt)
    return start, je_model, params, optimizer, best_val_loss

def attn_penalty(cross_weights, token_mask = None, soft = nn.Softmax(dim=2), lam = 1, steep_entropy = True):
    #cross_weights:list(N, T, P)
    #token_mask: N, T -> 1 if we want to count towards loss, 0 otherwise
    attn_loss = 0
    losses = []
    for c in cross_weights:
        probs = soft(c)

        if not steep_entropy:
            entropy = -probs * torch.log(probs)
            entropy = torch.sum(entropy, dim=2) #N, T
            entropy = torch.mean(entropy, dim=1) #N
            entropy = torch.mean(entropy, dim=0) #1
        else:
            lam = lam * 3
            eps = 1e-7
            P = probs.shape[2] #patches
            scalescore = probs * P -1 #-1 - P-1
            score1 = torch.sum(torch.relu(scalescore), dim=2) #add up all the pos values #0 if all 0.5, pos if high score exists
            score2 = torch.sum(torch.relu(-scalescore), dim=2) #add up all the neg values #0 if all 0.5, pos if many low scores
            score = (score1 + score2)/2
            entropy = ((P-1 - score)/(P-1)) # 0-1, lower is better. Weird cuz not symmetric but thats fine i think
            entropy = torch.mean(entropy, dim=1)
            entropy = torch.mean(entropy, dim=0)  # 1
        loss = entropy * lam
        losses.append(entropy)
        attn_loss += loss
    return attn_loss, losses #1

def clip_loss(im_logits, aug_logits = None, loss_weight = 1, criterion = nn.CrossEntropyLoss()):
    text_logits = [im.t() for im in im_logits]
    clip_loss = 0
    losses = []
    for i in np.arange(len(im_logits)):
        samp = torch.tensor(np.arange(im_logits[i].shape[0]))
        loss_a = criterion(im_logits[i], samp.to(device))
        loss_b = criterion(text_logits[i], samp.to(device))
        closs = (loss_a + loss_b) / 2
        losses.append(closs)
        clip_loss += closs * loss_weight
    if aug_logits is not None:
        for i in np.arange(len(aug_logits)):
            samp = torch.tensor(np.arange(im_logits[i].shape[0]))
            imloss = criterion(im_logits[i], samp.to(device))
            losses.append(imloss)
            clip_loss += imloss
    assert len(losses) == int((len(im_logits) + (len(im_logits) * (len(im_logits) -1)/2.0)))
    return clip_loss, losses

def compute_loss(je_model, samples, tokenizer, args, attn_lam = 1):
    ims = samples['images']
    texts = samples['texts']
    ims = [im.to(device) for im in ims]
    texts = tokenizer.encode(texts=texts).to(device)
    im_logits, crosses, aug_logits = je_model(ims, texts)
    cl, cl_losses = clip_loss(im_logits, aug_logits)
    attn_pen, attn_losses = attn_penalty(crosses, token_mask = Transformer.get_mask(texts), lam = attn_lam, steep_entropy=args.steep_entropy)
    cl_count = len(cl_losses)
    attn_count = len(attn_losses)
    loss = cl / cl_count + attn_pen / attn_count
    all_losses = cl_losses + attn_losses
    return loss, torch.tensor(all_losses)

def train(train_data_loader, je_model, tokenizer, args, epoch, optimizer, total_step_mimic):
    mean_loss, mean_losses, ct = 0.0, 0.0, 0
    for i, samples in enumerate(train_data_loader):
        je_model.zero_grad(set_to_none=True)
        loss, all_losses = compute_loss(je_model, samples, tokenizer, args, attn_lam=args.lam)
        # Forward, backward and optimize
        loss.backward()
        optimizer.step()
        if i % args.log_step == 0:
            print('MIMIC Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, i, total_step_mimic, loss.item()))
            print(all_losses)
        mean_loss += loss
        mean_losses += all_losses
        ct += 1
    if ct > 0:
        mean_loss = mean_loss/ct
        mean_losses = mean_losses/ct

    return mean_loss.item(), mean_losses

def validate(val_data_loader, tokenizer, je_model, args):
    val_losses = []
    avg_loss, ct = 0.0, 0
    with torch.no_grad():
        for j, samples in enumerate(val_data_loader):
            loss, all_losses = compute_loss(je_model, samples, tokenizer, args, attn_lam=args.lam)
            val_losses.append(all_losses.view(-1,1))
            avg_loss += loss
            ct += 1
    avg_loss = avg_loss/ct

    val_losses = torch.cat(val_losses, dim=1) #num batches x num losses
    avg_losses = torch.mean(val_losses, dim=1)
    assert avg_losses.shape == all_losses.shape
    if avg_losses.shape[0] == 5:
        names = ['im1-t', 'im2-t', 'im1-im2', 'attn im1-t', 'attn im2-t']
        lossstr = ""
        for i in range(len(names)):
            lossstr += (", " + names[i] + ": " + str(avg_losses[i].item()))
        print("Val losses" + lossstr)
    return avg_loss.item(), avg_losses

def b_loss(impreds, labels, device, heads, criterion):
    losses = torch.zeros(len(heads))
    for i, h in enumerate(heads):
        label = labels[h]
        if h == 'Edema' or h == 'Atelectasis':
            label[label == -1.0] = float('nan')
        else:
            label[label == -1.0] = float('nan')
        label[label == 0.0] = 0
        label[label == 1.0] = 1
        label = label.float().to(device)
        mypreds = impreds[torch.logical_not(torch.isnan(label)), i]
        mylabels = label[torch.logical_not(torch.isnan(label))]
        losses[i] = criterion(mypreds, mylabels)
    losses = losses[torch.logical_not(torch.isnan(losses))]
    loss = torch.mean(losses)
    if torch.isnan(loss):
        loss = 0
    return loss

def train_vision(device, vision_model, im1, im2, labels, heads, criterion=torch.nn.BCEWithLogitsLoss(), useOne=False):
    vision_model.zero_grad(set_to_none=True)
    # Set mini-batch dataset
    images1 = im1.to(device)
    if not useOne:
        images2 = im2.to(device)

    # Forward, backward and optimize
    impreds1 = vision_model(images1)
    cl1 = b_loss(impreds1, labels, device, heads, criterion)
    if not useOne:
        impreds2 = vision_model(images2)
        cl2 = b_loss(impreds2, labels, device, heads, criterion)
        loss = cl1 + cl2
    else:
        loss = cl1
    return loss

def validate_vision(device, val_data_loader, vision_model, heads, criterion, source = "MIMIC", proportion = 1.0):
    vlosses = []
    with torch.no_grad():
        for j, res in enumerate(val_data_loader):
            if source == "MIMIC":
                valims1, valims2, val_labels = res
            else:
                valims1, val_labels = res
                valims2 = None

            gen = np.random.rand(1)
            if gen >= proportion:
                continue

            valims1 = valims1.to(device)
            valpred1= vision_model(valims1)
            myloss = b_loss(valpred1, val_labels, device, heads, criterion)
            if valims2 is not None:
                valims2 = valims2.to(device)
                valpred2 = vision_model(valims2)
                myloss = myloss + b_loss(valpred2, val_labels, device, heads, criterion)

            vlosses.append(myloss.cpu())

    vlloss = np.mean(np.array(vlosses))
    print(source + ' Val Loss: ' + str(vlloss))
    return vlloss

def getLabels(df, heads):
    labels = None
    for i, h in enumerate(heads):
        label = df[h].float()
        label[label == -1.0] = float('nan')
        label[label == 0.0] = 0.0
        label[label == 1.0] = 1.0
        if labels is None:
            labels = label
            labels = labels[:, None]
        else:
            labels = torch.cat((labels, label[:, None]), axis=1)

    return labels

def get_all_preds(DL, mod, heads = ['covid19', 'No Finding'], device='cuda'):
    tp, tt = None, None
    for i, res in enumerate(DL):
        try:
            im1, im2, df, study = res
        except:
            im1, df = res

        images = im1.to(device)
        preds = mod(images).to(device)
        labels = getLabels(df, heads).to(device)

        if tp is None:
            tp = preds
        else:
            tp = torch.cat((tp, preds), axis=0)
        if tt is None:
            tt = labels
        else:
            tt = torch.cat((tt, labels), axis=0)

    return tp, tt



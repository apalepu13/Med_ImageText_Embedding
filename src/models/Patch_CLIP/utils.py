import regex as re
from MedCLIP_Datasets import *
import CLIP_Embedding
import torch
import torch.nn as nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getExperiment(args, mp):
    if args.debug:
        return "debug"

    if args.resume and args.resume > 0:
        fp =  os.path.join(mp, 'exp'+str(args.resume))
        if os.path.exists(fp):
            return fp
        else:
            raise Exception("Experiment doesn't exist, cannot resume exp " + fp)

    if not os.listdir(os.path.join(mp)):
        if args.resume:
            raise Exception("No experiment exist, cannot resume last one.")
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
    je_model = CLIP_Embedding.MedCLIP(eval=False).to(device)
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
    return start, je_model, params, optimizer, best_val_loss

def attn_penalty(cross_weights, soft = nn.Softmax(dim=2), lam = (0.0, 0.0), steep_entropy = False):
    attn_loss = 0
    losses = []
    eps = 1e-7
    for c in cross_weights:
        if lam[1] == 0: #only do the global text embedding
            c = c[:, 0, :]
        entropy = soft(c) + eps #NTP
        if not steep_entropy:
            entropy = -entropy * torch.log(entropy)
            entropy_text = torch.sum(entropy, dim=2) #N, T
            entropy_text = torch.mean(entropy_text, dim=(1, 0)) #1

            entropy_im = soft(c.permute(0, 2, 1)) + eps
            entropy_im = -entropy_im * torch.log(entropy_im)
            entropy_im = torch.sum(entropy_im, dim=2) #N, P
            entropy_im = torch.mean(entropy_im, dim=(1, 0)) # 1
        else:
            P = entropy.shape[2] #patches
            scalescore = entropy * P -1 #-1 - P-1
            score1 = torch.sum(torch.relu(scalescore), dim=2) #add up all the pos values #0 if all 0.5, pos if high score exists
            score2 = torch.sum(torch.relu(-scalescore), dim=2) #add up all the neg values #0 if all 0.5, pos if many low scores
            score = (score1 + score2)/2
            entropy = ((P-1 - score)/(P-1))
            entropy = torch.mean(entropy, dim=1)
            entropy_text = torch.mean(entropy, dim=0)  # 1
            entropy_im = 0 #Implement!!!


        loss = (entropy_text * float(lam[0])) + (entropy_im * float(lam[1]))
        losses.append(loss.cpu().detach())
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
        losses.append(closs.cpu().detach())
        clip_loss += closs * loss_weight
    if aug_logits is not None:
        for i in np.arange(len(aug_logits)):
            samp = torch.tensor(np.arange(im_logits[i].shape[0]))
            imloss = criterion(im_logits[i], samp.to(device))
            losses.append(imloss.cpu().detach())
            clip_loss += imloss
    assert len(losses) == int((len(im_logits) + (len(im_logits) * (len(im_logits) -1)/2.0)))
    return clip_loss, losses

def compute_loss(je_model, samples, args, attn_lam_words = 0.0, attn_lam_patches = 0.0):
    ims = samples['images']
    texts = samples['texts']
    im_logits, crosses, aug_logits = je_model(ims, texts)
    cl, cl_losses = clip_loss(im_logits, aug_logits)
    attn_pen, attn_losses = attn_penalty(crosses, lam = (attn_lam_words, attn_lam_patches), steep_entropy=args.steep_entropy)
    cl_count = len(cl_losses)
    attn_count = len(attn_losses)
    loss = cl / cl_count + attn_pen / attn_count
    all_losses = cl_losses + attn_losses
    return loss, torch.tensor(all_losses)

def train(train_data_loader, je_model, args, epoch, optimizer, total_step_mimic):
    mean_loss, mean_losses, ct = 0.0, 0.0, 0
    for i, samples in enumerate(train_data_loader):
        je_model.zero_grad(set_to_none=True)
        loss, all_losses = compute_loss(je_model, samples, args, attn_lam_words=args.lam_words, attn_lam_patches = args.lam_patches)
        # Forward, backward and optimize
        loss.backward()
        optimizer.step()
        if i % args.log_step == 0:
            print('MIMIC Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, i, total_step_mimic, loss.item()))
            print(all_losses)
        l = loss.cpu().detach().numpy()
        mean_loss += l
        mean_losses += all_losses
        ct += 1
    if ct > 0:
        mean_loss = mean_loss/ct
        mean_losses = mean_losses/ct

    return mean_loss, mean_losses

def validate(val_data_loader, je_model, args):
    val_losses = []
    avg_loss, ct = 0.0, 0
    with torch.no_grad():
        for j, samples in enumerate(val_data_loader):
            loss, all_losses = compute_loss(je_model, samples, args, attn_lam_words=args.lam_words, attn_lam_patches = args.lam_patches)
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

def get_all_preds(DL, mod,similarity=False,im_embeds=False,patch_similarity=False, heads = ['covid19', 'No Finding'], convirt=True, getlabels=True):
    with torch.no_grad():
        if similarity or patch_similarity:
            label_embeds = CLIP_Embedding.getLabelEmbeddings(mod, heads, convirt=convirt)
            embed_list = [label_embeds[h][None, :] for h in heads]
            label_embeds = torch.cat(embed_list, dim=0)
            label_embeds = label_embeds/label_embeds.norm(dim=1, keepdim=True)

        for i, samples in enumerate(DL):
            images = samples['images']
            if i == 0:
                tt = []
                tps = [[] for im in images]

            if similarity:
                list_im_embeds = mod.get_im_embeddings(images, only_ims = True)
                list_im_embeds = [im_embeds/im_embeds.norm(dim=1, keepdim=True) for im_embeds in list_im_embeds]
                # N P E x c E = N c
                list_preds = [im_embeds @ label_embeds.t() for im_embeds in list_im_embeds]
            elif patch_similarity:
                list_im_embeds = mod.get_im_embeddings(images, only_patches=True) #N E P1 P2
                list_im_embeds = [im_embeds.reshape(im_embeds.shape[0], im_embeds.shape[1], im_embeds.shape[2] * im_embeds.shape[3]).permute(0, 2, 1) for im_embeds in list_im_embeds] #N P E
                list_im_embeds = [im_embeds/im_embeds.norm(dim=2, keepdim=True) for im_embeds in list_im_embeds]
                # N P E x c E = N P c
                list_preds = [torch.bmm(im_embeds, label_embeds.t()[None, :, :].repeat(im_embeds.shape[0], 1, 1)) for im_embeds in list_im_embeds]
            elif im_embeds:
                list_im_embeds = mod.get_im_embeddings(images, only_ims=True)
                list_preds = [im_embeds / im_embeds.norm(dim=1, keepdim=True) for im_embeds in list_im_embeds]

            labels = getLabels(samples['labels'], heads) if getlabels else None

            for j, pred in enumerate(list_preds):
                tps[j].append(pred.cpu())
            tt.append(labels) if getlabels else None

        tplist = [torch.cat(tp, axis=0) for tp in tps]
        tt = torch.cat(tt, axis=0) if getlabels else None
        return tplist, tt

def normalize(img):
    img[:, 0, :, :] = (img[:, 0, :, :] * .229) + .485
    img[:, 1, :, :] = (img[:, 1, :, :] * .224) + .456
    img[:, 2, :, :] = (img[:, 2, :, :] * .225) + .406
    img = img.permute(0, 2, 3, 1)[0, :, :, :].squeeze()
    return img

def getLabelSimilarities(mod, heads, label_embeds=None, compare_mimic = False):
    with torch.no_grad():
        if compare_mimic:
            label_embeds = CLIP_Embedding.getLabelEmbeddings(mod, heads)
            label_embeds_mimic = CLIP_Embedding.getLabelEmbeddings(mod, heads, convirt=False)
            for i, h in enumerate(heads):
                print(h, torch.dot(label_embeds[h] / label_embeds[h].norm(dim=-1, keepdim=True),
                                       label_embeds_mimic[h] / label_embeds_mimic[h].norm(dim=-1, keepdim=True)).cpu())
        else:
            if not label_embeds:
                label_embeds = CLIP_Embedding.getLabelEmbeddings(mod, heads)
            for i, h in enumerate(heads):
                for j, h2 in enumerate(heads):
                    if i < j:
                        print(h, h2, torch.dot(label_embeds[h] / label_embeds[h].norm(dim=-1, keepdim=True),
                                               label_embeds[h2] / label_embeds[h2].norm(dim=-1, keepdim=True)).cpu())




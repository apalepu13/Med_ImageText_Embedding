import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Image_Text_Dataset(Dataset):
    """Chx - Report dataset."""

    def __init__(self, source = 'mimic_cxr', group='train',
                 mimic_csv_file='../../../data/mimic-cxr-2.0.0-split.csv', mimic_root_dir='/n/data2/hms/dbmi/beamlab/mimic_cxr/',
                 indiana_csv_file='../../../data/indiana_cxr_list.csv', indiana_root_dir='/n/scratch3/users/a/anp2971/datasets/',
                 chexpert_root_dir='/n/scratch3/users/a/anp2971/datasets/chexpert/'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.source = source
        if self.source == 'm':
            self.source = 'mimic_cxr'
            self.grayscale = True
        elif self.source == 'i':
            self.source = 'indiana_cxr'
            self.grayscale = True
        elif self.source == 'c':
            self.source = 'chexpert'
            self.grayscale = True

        self.group = group

        # Preprocessing
        self.im_preprocessing_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, ratio=(.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(20, translate=(.1, .1), scale=(.95, 1.05)),
            transforms.ColorJitter(brightness=.4, contrast=.4),
            transforms.GaussianBlur(kernel_size=15, sigma=(.1, 3.0)),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.im_preprocessing_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if self.source == 'mimic_cxr':
            self.im_list = pd.read_csv(mimic_csv_file)
            if group == 'tiny':
                group = 'train'
                self.im_list = self.im_list.iloc[::1000, :]
            if group == 'train':
                self.im_list = self.im_list[self.im_list['split'] != 'test']
            elif group == 'val' or group == 'test':
                self.im_list = self.im_list[self.im_list['split'] == 'test']

            self.im_list['pGroup'] = np.array(["p" + pg[:2] for pg in self.im_list['subject_id'].values.astype(str)])
            self.im_list['pName'] = np.array(["p" + pn for pn in self.im_list['subject_id'].values.astype(str)])
            self.im_list['sName'] = np.array(["s" + sn for sn in self.im_list['study_id'].values.astype(str)])
            #self.im_list = self.im_list.drop_duplicates(subset = ['pName', 'sName']) #only 1 image per report
            print(group + "size= " + str(self.im_list.shape))
            self.root_dir = mimic_root_dir

        elif self.source == 'indiana_cxr':
            self.root_dir = indiana_root_dir
            self.im_list = pd.read_csv(indiana_csv_file)
            self.im_list['patient'] = self.im_list['patient'].values.astype(int)
            if group == 'tiny':
                group = 'train'
                self.im_list = self.im_list.iloc[::100, :]
            if group == 'train':
                self.im_list = self.im_list[self.im_list['patient'].values % 7 > 0]
            elif group == 'val' or group == 'test':
                self.im_list = self.im_list[self.im_list['patient'].values % 7 == 0]
            print(group + "size= " + str(self.im_list.shape))

        elif self.source == 'chexpert':
            self.root_dir = chexpert_root_dir
            im_list_train = pd.read_csv(chexpert_root_dir + 'CheXpert-v1.0-small/train.csv')
            im_list_val = pd.read_csv(chexpert_root_dir + 'CheXpert-v1.0-small/valid.csv')
            self.im_list = pd.append(im_list_train, im_list_val)
            print(group + "size= " + str(self.im_list.shape))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.source == 'mimic_cxr':
            ims = self.im_list.iloc[idx, :]
            img_name = os.path.join(self.root_dir, ims['pGroup'], ims['pName'], ims['sName'], ims['dicom_id'] + '.jpg')
            image = Image.open(img_name)
            if self.grayscale:
                image = image.convert("RGB")
            if self.group == 'tiny' or self.group == 'train' or self.group == 'all':
                image = self.im_preprocessing_train(image)
            else:
                image = self.im_preprocessing_val(image)

            text_name = os.path.join(self.root_dir, ims['pGroup'], ims['pName'], ims['sName'], ims['sName'] + '.txt')
            with open(text_name, "r") as text_file:
                text = text_file.read()

            sample = (image, text)
            return sample

        elif self.source == 'indiana_cxr':
            imstexts = self.im_list.iloc[idx, :]
            ims = imstexts.loc['images']
            ims = ims[:-4] + '.png'
            img_name = os.path.join(self.root_dir, 'indiana_cxr', ims)
            image = Image.open(img_name)
            image = image.convert("RGB")
            if self.group == 'tiny' or self.group == 'train' or self.group == 'all':
                image = self.im_preprocessing_train(image)
            else:
                image = self.im_preprocessing_val(image)

            text = imstexts.loc['texts']
            sample = (image, text)
            #print(text)
            return sample
        elif self.source == 'chexpert':
            df = self.im_list.iloc[idx, :]
            img_name = self.root_dir + df['Path']
            image = Image.open(img_name)
            image = image.convert("RGB")
            image = self.im_preprocessing_test(image)
            sample = (image, df)
            return sample


def print_text_percentiles():
    itd = Image_Text_Dataset()
    wordlens = []
    for i in np.arange(1, 20000, 100):
        im, text = itd.__getitem__(i)
        sp = text.split()
        wordlens.append(len(sp))
        if len(sp) > 300:
            print(text)
        elif len(sp) < 40:
            print(text)
    wordlens = np.array(wordlens)

    print(np.max(wordlens))
    print(np.percentile(wordlens, 75))
    print(np.percentile(wordlens, 50))
    print(np.percentile(wordlens, 25))
    print(np.min(wordlens))

train_dat = Image_Text_Dataset(source = 'i', group = 'val')
for i in np.arange(10):
    train_dat.__getitem__(i)

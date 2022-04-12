import pandas as pd
import numpy as np
import os
from PIL import ImageFont
from PIL import ImageDraw
from PIL import Image
import argparse
import random



def make_synthetic(image, incorrect, correct, p_watermark = .8, p_correct = .8, src = 'mimic_cxr'):
    do_watermark, do_correct = np.random.random((2,))
    do_watermark = do_watermark > p_watermark
    do_correct = do_correct > p_correct
    watermark_image = image.copy()
    shortcut = "None"
    if do_watermark:
        draw = ImageDraw.Draw(watermark_image)
        if src == 'mimic_cxr':
            font = ImageFont.truetype("/usr/share/fonts/urw-base35/P052-Roman.otf", 200)
        else:
            font = ImageFont.truetype("/usr/share/fonts/urw-base35/P052-Roman.otf", 30)

        if do_correct and len(correct) > 0:
            shortcut = np.random.choice(correct)
        elif not do_correct and len(incorrect) > 0:
            shortcut = np.random.choice(incorrect)
        else:
            return watermark_image, shortcut
        if src == 'mimic_cxr':
            draw.text((100, 100), shortcut, (255), font=font)
        else:
            draw.text((30, 30), shortcut, (255), font=font)

    return watermark_image, shortcut

def im_iterate(im_list, mimic_cxr_dir, synth_dir, orig = 'mimic_cxr'):
    shortcuts = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
    df_labels = pd.DataFrame(columns = ['synth_path', 'orig_path', 'shortcut'])
    for i in np.arange(im_list.shape[0]):
        entry = im_list.iloc[i, :]
        if orig == 'mimic_cxr':
            orig_im_path = os.path.join(mimic_cxr_dir, entry['pGroup'], entry['pName'], entry['sName'], entry['dicom_id'] + '.jpg')
            orig_im = Image.open(orig_im_path)
        elif orig == 'chexpert':
            orig_im_path = os.path.join(mimic_cxr_dir, entry['Path'])
            orig_im = Image.open(orig_im_path)

        correct = [s for s in shortcuts if entry[s] == 1.0]
        incorrect = [s for s in shortcuts if entry[s] != 1.0]
        synth_im, shortcut = make_synthetic(orig_im, np.array(incorrect), np.array(correct), src = orig)

        if orig == 'mimic_cxr':
            synth_im_path = os.path.join(synth_dir, entry['pGroup'] + '_div_' + entry['pName'] + '_div_' + entry['sName'])
            if not os.path.exists(synth_im_path):
                os.makedirs(synth_im_path)
            synth_im.save(os.path.join(synth_im_path, entry['dicom_id'] + '.jpg'))
            df_labels = df_labels.append({'synth_path' : os.path.join(synth_im_path, entry['dicom_id'] + '.jpg'), 'orig_path' : orig_im_path, 'shortcut' : shortcut}, ignore_index = True)
            #print("Saved " + synth_im_path, entry['dicom_id'])
        elif orig == 'chexpert':
            synth_im_path = os.path.join(synth_dir, entry['Path'].replace('/', '_div_'))
            if not os.path.exists(os.path.dirname(synth_im_path)):
                os.makedirs(os.path.dirname(synth_im_path))
            synth_im.save(synth_im_path)
            df_labels = df_labels.append({'synth_path': synth_im_path,'orig_path': orig_im_path, 'shortcut': shortcut}, ignore_index = True)
            #print("Saved " + synth_im_path)
    return df_labels




def main(args):
    orig = args.orig
    if orig=='mimic_cxr':
        mimic_cxr_dir = '/n/data2/hms/dbmi/beamlab/mimic_cxr/'
        mimic_iv_dir = '/n/data2/hms/dbmi/beamlab/mimic_iv/'
        synth_dir = '/n/scratch3/users/a/anp2971/datasets/synthetic_mimic_cxr/'
        csv_dir = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/synthetic/'
        mimic_cxr_file = mimic_cxr_dir + 'mimic-cxr-2.0.0-chexpert.csv'
        mimic_cxr_file2 = mimic_cxr_dir + 'mimic-cxr-2.0.0-split.csv'
        im_list = pd.read_csv(mimic_cxr_file)
        im_list2 = pd.read_csv(mimic_cxr_file2)
        print(im_list.head())
        print(im_list2.head())
        im_list = im_list.merge(im_list2, on = ['subject_id', 'study_id'])

        im_list['pGroup'] = np.array(["p" + pg[:2] for pg in im_list['subject_id'].values.astype(str)])
        im_list['pName'] = np.array(["p" + pn for pn in im_list['subject_id'].values.astype(str)])
        im_list['sName'] = np.array(["s" + sn for sn in im_list['study_id'].values.astype(str)])
        mim_labels = im_iterate(im_list, mimic_cxr_dir, synth_dir)
        mim_labels.to_csv(csv_dir + 'mimic_synthetic.csv')
    elif orig=='chexpert':
        chexpert_dir = '/n/data2/hms/dbmi/beamlab/chexpert/'
        synth_dir = '/n/scratch3/users/a/anp2971/datasets/synthetic_chex/'
        csv_dir = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/synthetic/'
        chexpert_file1 = chexpert_dir + 'CheXpert-v1.0-small/train.csv'
        chexpert_file2 = chexpert_dir + 'CheXpert-v1.0-small/valid.csv'
        train_list = pd.read_csv(chexpert_file1)
        valid_list = pd.read_csv(chexpert_file2)
        train_labels = im_iterate(train_list, chexpert_dir, synth_dir, orig)
        test_labels = im_iterate(valid_list, chexpert_dir, synth_dir, orig)
        train_labels.to_csv(csv_dir + 'chex_train_synthetic.csv')
        test_labels.to_csv(csv_dir + 'chex_test_synthetic.csv')



    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig', type=str, default='mimic_cxr',help='Which synthetic to generate')
    args = parser.parse_args()
    print(args)
    do_watermark, do_correct = np.random.random((2,))
    print(do_watermark)
    print(do_correct)
    main(args)




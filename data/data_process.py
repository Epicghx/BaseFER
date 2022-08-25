import os, random, shutil, sys
import argparse
import numpy as np
import pandas as pd
import torch
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--number', type=int, default=3803)
parser.add_argument('--newfiledir', type=str, default='./data/emotion_select/')

args = parser.parse_args()

raw_path = '/export/.ghx/Data/FER/AffectNet_8/{}_set/'
new_path = '/export/.ghx/Data/FER/AffectNet_8/{}'
stage = 'train'

def main(stage):
    if not os.path.exists(new_path.format(stage)):
        for dir in range(8):
            os.makedirs(new_path.format(stage) + '/' + str(dir))


    data = os.listdir(raw_path.format(stage) + 'images')

    for sample in data:
        index = sample.split('.')[0]
        label = np.load(raw_path.format(stage) + 'annotations/{}_exp.npy'.format(index))
        aro = np.load(raw_path.format(stage) + 'annotations/{}_aro.npy'.format(index))
        val = np.load(raw_path.format(stage) + 'annotations/{}_val.npy'.format(index))
        # pos = np.load(raw_path.format(stage) + 'annotations/{}_lnd.npy'.format(index))

        ann = np.array([float(val), float(aro)])
        with open(new_path.format('va') +'/{}/'.format(stage) + index + '_va.npy', 'wb') as f:
            np.save(f, ann)
        shutil.copy(os.path.join(raw_path.format(stage), 'images/{}'.format(sample)), os.path.join(new_path.format(stage), str(label) + '/'))




def CopyFile(dir):
    if not os.path.isdir(args.newfiledir):
        os.makedirs(args.newfiledir)
    class_num = [int(74874/5), int(134415/5), int(25459/5), int(14090/3), int(6378/2), int(3803), int(24882/5)]
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx):
        d =os.path.join(dir, target)
        if not os.path.isdir(os.path.join(args.newfiledir, target+'/')):
            os.makedirs(os.path.join(args.newfiledir, target+'/'))
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            picknumber = class_num[classes.index(target)]
            sample = random.sample(os.listdir(d), picknumber)
            for name in sample:
                shutil.copy(os.path.join(d, name), os.path.join(args.newfiledir, target+'/'))

def Freqpic(dir):
    # if not os.path.isdir(args.newfiledir):
    #     os.makedirs(args.newfiledir)
    class_num = [int(74874/5), int(134415/5), int(25459/5), int(14090/3), int(6378/2), int(3803), int(24882/5)]
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx):
        d =os.path.join(dir, target)
        if not os.path.isdir(os.path.join(args.newfiledir, target+'/')):
            os.makedirs(os.path.join(args.newfiledir, target+'/'))
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            picknumber = class_num[classes.index(target)]
            sample = random.sample(os.listdir(d), picknumber)
            for name in sample:
                shutil.copy(os.path.join(d, name), os.path.join(args.newfiledir, target+'/'))

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)
    return images

def find_classes(dir):

    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


if __name__ == '__main__':
    stage = ['train', 'val']
    for i in stage:
        main(i)



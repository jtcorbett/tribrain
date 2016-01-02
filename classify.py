#!/usr/local/bin/python

import os
import numpy as np
from PIL import Image
from random import shuffle

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure

IN_FEATURES = 64*64
np.set_printoptions(threshold='nan', linewidth='nan')

def feature_extraction(raw_data):
    image = color.rgb2grey(raw_data)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)

    return hog_image

# no preprocessing for now
def preprocess(raw_data):
    return raw_data

def load_image(path):
    image = Image.open(path)
    raw_features = feature_extraction(np.array(image))
    return tuple(preprocess([item for sublist in raw_features for item in sublist]))

def load_dataset(dataset, data, positive):
    if isinstance(data, str):
        directory = True
        (subdir, dirs, files) = os.walk(data).next()
    elif isinstance(data, list):
        directory = False
        files = data
    else:
        raise TypeError

    for f in files:
        if directory:
            sample_file = os.path.join(subdir, f)
        else:
            sample_file = f

        if not sample_file.endswith(".png"): continue

        features = load_image(sample_file)

        if len(features) != IN_FEATURES:
            print "%s is of incorrect dimensions" % sample_file
            exit()

        dataset.addSample(features, (1 if positive else -1,))

    return dataset

def build_NN(pos_td_dir, neg_td_dir, maxEpochs=None, hidden_layers=[100]):
    # Hardcode NN features for now
    print "Creating Neural Net..."
    net = buildNetwork(IN_FEATURES, *(hidden_layers + [1]))
    ds = SupervisedDataSet(IN_FEATURES, 1)

    print "Loading dataset..."
    load_dataset(ds, neg_td_dir, False)
    load_dataset(ds, pos_td_dir, True)

    trainer = BackpropTrainer(net, ds)

    print "Training..."
    trainer.trainUntilConvergence(maxEpochs=maxEpochs, verbose=True)

    return net

def test_NN(net, path, positive):
    if isinstance(path, str):
        directory = True
        (subdir, dirs, files) = os.walk(path).next()
    elif isinstance(path, list):
        directory = False
        files = path
    else:
        raise TypeError

    correct = 0
    total = 0

    for f in files:
        if directory:
            sample_file = os.path.join(subdir, f)
        else:
            sample_file = f

        if not sample_file.endswith(".png"): continue

        features = load_image(sample_file)

        if len(features) != IN_FEATURES:
            print "%s is of incorrect dimensions" % sample_file
            exit()

        total += 1

        if ((net.activate(features)[0] < 0) ^ positive):
            print "classified %s correctly" % sample_file
            correct += 1
        else:
            print "classified %s incorrectly" % sample_file

    print "overall accuracy: %d/%d = %f" % (correct, total, float(correct)/float(total))

def cross_validate(pos_dir, neg_dir, withhold=0.1, maxEpochs=None, hidden_layers=[100]):
    if withhold < 0 or withhold > 1:
        raise ValueError

    if isinstance(pos_dir, str):
        (subdir, dirs, pos_files) = os.walk(pos_dir).next()
        pos_files = map(lambda f: os.path.join(subdir, f), pos_files)
    elif isinstance(pos_dir, list):
        files = pos_dir
    else:
        raise TypeError

    if isinstance(neg_dir, str):
        (subdir, dirs, neg_files) = os.walk(neg_dir).next()
        neg_files = map(lambda f: os.path.join(subdir, f), neg_files)
    elif isinstance(neg_dir, list):
        files = neg_dir
    else:
        raise TypeError

    print "Partitioning data..."

    shuffle(pos_files)
    shuffle(neg_files)

    pos_vd_files = pos_files[:int(len(pos_files)*withhold)]
    pos_td_files = pos_files[int(len(pos_files)*withhold):]
    neg_vd_files = neg_files[:int(len(neg_files)*withhold)]
    neg_td_files = neg_files[int(len(neg_files)*withhold):]

    net = build_NN(pos_td_files, neg_td_files, maxEpochs)

    return (net, pos_vd_files, neg_vd_files)

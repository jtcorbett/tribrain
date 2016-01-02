#!/usr/local/bin/python

import os
import pickle
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

def _files_from_path(path):
    if isinstance(path, str):
        (subdir, dirs, files) = os.walk(path).next()
        files = map(lambda f: os.path.join(subdir, f), files)
    elif isinstance(path, list):
        files = path
    else:
        raise TypeError
    return files

def feature_extraction(raw_data):
    image = color.rgb2grey(raw_data)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)

    return hog_image

# no preprocessing for now
def preprocess(raw_data):
    return feature_extraction(raw_data)

def load_image(path):
    image = Image.open(path).resize((64,64), PIL.Image.ANTIALIAS)
    features = preprocess(np.array(image))
    return tuple([item for sublist in features for item in sublist])

def load_dataset(dataset, data, positive):
    for sample_file in _files_from_path(data):
        if not sample_file.startswith('.'): continue

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
    correct = 0
    total = 0

    for sample_file in _files_from_path(path):
        if not sample_file.startswith('.'): continue

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

def classify(net, path):
    for f in _files_from_path(path):
        if not f.startswith('.'): continue

        features = load_image(f)
        if len(features) != IN_FEATURES:
            print "%s is of incorrect dimensions" % sample_file
            exit()

        if net.activate(features) > 0:
            print "classified %s as positive" % sample_file
        else:
            print "classified %s as negativem" % sample_file


def cross_validate(pos_dir, neg_dir, withhold=0.1, maxEpochs=None, hidden_layers=[100]):
    if withhold < 0 or withhold > 1:
        raise ValueError

    pos_files = _files_from_path(pos_dir)
    neg_files = _files_from_path(neg_dir)

    print "Partitioning data..."

    shuffle(pos_files)
    shuffle(neg_files)

    pos_vd_files = pos_files[:int(len(pos_files)*withhold)]
    pos_td_files = pos_files[int(len(pos_files)*withhold):]
    neg_vd_files = neg_files[:int(len(neg_files)*withhold)]
    neg_td_files = neg_files[int(len(neg_files)*withhold):]

    net = build_NN(pos_td_files, neg_td_files, maxEpochs)

    return (net, pos_vd_files, neg_vd_files)

def saveNetwork(net, filename):
    fileObject = open(filename, 'w')
    pickle.dump(net, fileObject)
    fileObject.close()

def openNetwork(filename):
    fileObject = open(filename, 'r')
    net = picke.load(fileObject)
    fileObject.close()
    return net

def main():
    net = build_NN("triangle", "circle", maxEpochs=30)
    print "triangles"
    test_NN(net, "triangle", True)
    print "circle"
    test_NN(net, "circle", False)

if __name__ == "__main__":
    main()

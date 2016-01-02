#!/usr/local/bin/python

import os
import numpy as np
from PIL import Image

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

IN_FEATURES = 64*64
np.set_printoptions(threshold='nan', linewidth='nan')

# no preprocessing for now
def preprocess(raw_data):
    return raw_data

def load_image(path):
    image = Image.open(path)
    raw_features = np.array(image)
    return tuple(preprocess([item for sublist in raw_features for item in sublist]))

def load_dataset(dataset, path, positive):
    (subdir, dirs, files) = os.walk(path).next()

    for f in files:
        sample_file = os.path.join(subdir, f)
        if not sample_file.endswith(".png"): continue

        features = load_image(sample_file)

        if len(features) != IN_FEATURES:
            print "%s is of incorrect dimensions" % sample_file
            exit()

        dataset.addSample(features, (1 if positive else -1,))

    return dataset

def build_NN(pos_td_dir, neg_td_dir, target_error):
    # Hardcode NN features for now
    print "Creating Neural Net..."
    net = buildNetwork(IN_FEATURES, 100, 1)
    ds = SupervisedDataSet(IN_FEATURES, 1)

    print "Loading dataset..."
    load_dataset(ds, pos_td_dir, True)
    load_dataset(ds, neg_td_dir, False)

    trainer = BackpropTrainer(net, ds)

    print "Training..."

    epoch = 0
    while True:
        error = trainer.train()
        epoch += 1
        print "  epoch %d error: %f" % (epoch, error)

        if error <= target_error:
            print "Hit target error..."
            break

    return net

def test_NN(net, vd_dir, positive):
    (subdir, dirs, files) = os.walk(vd_dir).next()

    correct = 0
    total = 0

    print "In test_NN"

    for f in files:
        sample_file = os.path.join(subdir, f)
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

def main():
    net = build_NN("triangle","circle",.3)
    print "triangles"
    test_NN(net, "triangle", 1)
    print "circle"
    test_NN(net, "circle", 0)
if __name__ == "__main__":
    main()

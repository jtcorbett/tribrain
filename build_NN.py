#!/usr/local/bin/python

import os
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from PIL import Image
from sys import argv

np.set_printoptions(threshold='nan', linewidth='nan')

if len(argv) != 3:
    print "Usage: %s <positive_training_data> <negative_training_data>" % argv[0]
    exit()

pos_td_dir = argv[1]
neg_td_dir = argv[2]

# Hardcode NN features for now
net = buildNetwork(4096, 100, 1)

print net['in']
print net['hidden0']
print net['out']

def train_samples(NN, path, positive):
    (subdir, dirs, files) = os.walk(path).next()

    for f in files:
        sample_file = os.path.join(subdir, f)
        if not sample_file.endswith(".png"): continue

        sample_image = Image.open(sample_file)
        raw_features = np.array(sample_image)
        raw_features = [item for sublist in raw_features for item in sublist]
        # print raw_features
        print "%s %d" % (sample_file, len(raw_features))


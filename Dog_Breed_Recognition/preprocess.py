import os
import sys
import csv
import glob
import h5py
import numpy as np
import scipy.io as sio
import sklearn.model_selection as sk

from numpy import argmax
from skimage import io, transform
from sklearn import preprocessing
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def printf(text):
    sys.stdout.write('\r' + str(text))

def get_path_list(folder):
    paths = []
    for dirPath, dirNames, fileNames in os.walk(folder):
        for name in fileNames:
            paths.append(os.path.join(dirPath, name))
    print('Total: ', len(paths), ' file!\n')
    return paths

def load_path_file(paths, loadLabel, fileName):
    index = 1
    length = len(paths)
    features = []
    labels = []
    for item in paths:
        index = index + 1
        printf("{} loading {}...".format(index, item))
        img = io.imread(item)
        img = transform.resize(img, (224, 224))
        features.append(img)

        if loadLabel:
            labels.append(labelMap[item.split('/')[2].split('.')[0]])

        if index % 3000 == 0 or index == length:
            np.save('{}Data{}.npy'.format(fileName, int(index/3000 if index % 3000 == 0 else index/3000+1)), features)
            features = []

    if loadLabel:
        # transform to integer
        labels = preprocessing.LabelEncoder().fit_transform(labels)
        # transform to binary
        labels = preprocessing.OneHotEncoder().fit_transform(labels.reshape(-1,1)).toarray()
        np.save('{}Label.npy'.format(fileName), labels)

def one_hot_encode(labels):
    unique_label = np.unique(labels)
    print('共', len(unique_label), '種label')
    #print(unique_label, '\n')
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # try invert first example
    #inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    #print(inverted)
    return onehot_encoded


### get path list
labelMap = dict()
with open('./labels.csv') as csvFile:
    reader = csv.reader(csvFile, delimiter=',')
    for row in reader:
        labelMap[row[0]] = row[1]
print('\nSuccessfully get label list')

paths = get_path_list('./train')
load_path_file(paths, True, 'train')
print('\nSuccessfully load all train data')


paths = get_path_list('./test')
load_path_file(paths, False, 'test')
print('Successfully load all test data')

print('Done')

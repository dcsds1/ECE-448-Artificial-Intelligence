#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:46:24 2016

@author: changsongdong
"""

import numpy as np
from scipy.spatial.distance import kulsinski
from scipy.spatial.distance import cosine
import time
import matplotlib.pyplot as plt
import itertools

def get_images(data_file_name):
    data = []
    with open(data_file_name, 'r') as file:
        for line in file:
            data.append(list(line))
            
    height, width = np.asarray(data).shape
    for i in range(height):
        for j in range(width):
            if data[i][j] == ' ':
                data[i][j] = 0
            elif data[i][j] == '#' or data[i][j] == '+':
                data[i][j] = 1
            else:
                data[i][j] =2
                
    data = np.delete(np.asarray(data), 28, 1).reshape(-1, 784)
    return data

def get_label(label_file_name):
    label = []
    with open(label_file_name, 'r') as file:
        for line in file:
            label.append(list(line))
            
    label = np.asarray(np.delete(np.asarray(label), 1, 1), dtype=np.int64)
    return label
    
def classify(test_sample, train_data, train_label, k):
    '''predict the label of the input test_sample'''

    # Euclidean Distance
#    distance = np.sqrt(np.sum((train_data-test_sample)**2, axis=1))
    
    # Manhatton Distance
#    distance = np.sum(np.absolute(train_data-test_sample), axis=1)
    
    # Chebyshev distance (not useful for this data set, cause the distance is either 1 or 0)
#    distance = np.amax(np.absolute(train_data-test_sample), axis=1)
    
    # kulsinski Distance
#    distance = np.zeros(5000)
#    for i in range(5000):
#        distance[i] = kulsinski(test_sample, train_data[i])
        
    # Cosine
    distance = np.zeros(5000)
    for i in range(5000):
        distance[i] = cosine(test_sample, train_data[i])
    
    # k smallest distance and correspongding labels
    min_dist = distance.argsort()[:k]
    min_label = train_label[min_dist]
    
    # count how many times each label appear
    label_dict = {}
    for i in min_label:
        if i[0] in label_dict:
            label_dict[i[0]] += 1
        else:
            label_dict[i[0]]  =1
    predict = list(label_dict.keys())[0] # find the label that appear the most times
    return predict
    
def test(testimages, testlabels, train_data, train_label):
    predicts = []
    correct = 0
    for i in range(1000):
        predict = classify(testimages[i], train_data, train_label, 5)
        predicts.append(predict)
        if predict == testlabels[i]:
            correct += 1
    print('accuracy is {}'.format(correct / 1000))
    return predicts
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def confusion_matrix(predicts, labels):
    confusion_matrix = np.zeros((10, 10))
    labels = labels.flatten()
    for i in range(len(labels)):
        confusion_matrix[labels[i]][predicts[i]] += 1
    confusion_matrix /= confusion_matrix.sum(axis=1)[:, None]
    confusion_matrix = np.around(confusion_matrix, decimals=3)
    index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(confusion_matrix, index, title='Confusion Matrix')
    plt.show()
    return confusion_matrix
    
train_data = get_images('trainingimages')
train_label = get_label('traininglabels')
test_data = get_images('testimages')
test_label = get_label('testlabels')
start = time.clock()
predicts = test(test_data, test_label, train_data, train_label)
print('running time: {}s'.format(time.clock() - start))

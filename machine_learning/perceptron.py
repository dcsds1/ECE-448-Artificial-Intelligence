#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:50:24 2016

@author: changsongdong
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
#        self.weights = np.random.random((784,10)) * 2 - 1 # generate random weights in (-1, 1]
        self.weights = np.zeros((784, 10))
        bias = np.ones((10, 1))
        self.weights = np.transpose(np.hstack((bias, np.transpose(self.weights)))) # add one bias to each weight vector
        
    def get_images(self, data_file_name):
        """
        Build the data set
        
        parameters: image data file name ('trainingimages', 'testimages')
        returns: three dimensional array, shape: number of samples * 28 * 28
        """
        
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
        
        dummy = np.ones((len(data), 1))
        data = np.hstack((dummy, data))
        return data
    
    def get_label(self, label_file_name):
        """
        Build the label set
        
        parameters: data label file name ('traininglabels', 'testlabels')
        returns: two dimensional array, shape: number of samples * 1
        """
        
        label = []
        with open(label_file_name, 'r') as file:
            for line in file:
                label.append(list(line))
                
        label = np.asarray(np.delete(np.asarray(label), 1, 1), dtype=np.int64)
        return label
        
    def train(self, epochs, train_data, train_label):
        for epoch in range(epochs):
            # shuffle data samples
            perm = np.arange(len(train_label))
            np.random.shuffle(perm)
            data = train_data[perm]
            label = train_label[perm]
            alpha = 500 / (500 + epoch) # learning rate
            correct_pred = 0
            for i in range(len(label)):
                y = np.argmax(np.dot(data[i], self.weights))
                if label[i] == y:
                    correct_pred += 1
                else:
                    self.weights[:, label[i][0]] += alpha * data[i]
                    self.weights[:, y] -= alpha * data[i]
            if epoch % 5 == 0 and epoch > 0:
                print("epoch {}: training accuracy = {}".format(epoch, 
                                                     correct_pred / len(label)))
        return self.weights
        
    def test(self, weights):
        test_data = self.get_images('testimages')
        test_label = self.get_label('testlabels')
#        predicts = np.zeros(len(test_label))
        correct_pred = 0
        for i in range(len(test_label)):
            y = np.argmax(np.dot(test_data[i], weights))
#            predicts[i] = y
            if test_label[i] == y:
                correct_pred += 1
        print("testing accuracy = {}".format(correct_pred / len(test_label)))
#        self.confusion_matrix(predicts, test_label)
        
    def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):

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

    def confusion_matrix(self, predicts, labels):
        confusion_matrix = np.zeros((10, 10))
        labels = labels.flatten()
        for i in range(len(labels)):
            confusion_matrix[labels[i]][predicts[i]] += 1
        confusion_matrix /= confusion_matrix.sum(axis=1)[:, None]
        confusion_matrix = np.around(confusion_matrix, decimals=3)
        index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        plt.figure(figsize=(10,10))
        self.plot_confusion_matrix(confusion_matrix, index, title='Confusion Matrix')
        plt.show()
        
if __name__ == '__main__':
#    nn = Perceptron()
##    train_data = nn.get_images('trainingimages')
##    train_label = nn.get_label('traininglabels')
#    weight = nn.train(100, train_data, train_label)
#    nn.test(weight)
#    w = nn.weights

    first_feature = weight[:,0]
    first_feature = np.delete(first_feature[:, None], 0).reshape(28,28)
    first_feature=(first_feature-np.min(first_feature))*255/(np.max(first_feature)-np.min(first_feature))
    plt.imshow(first_feature, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
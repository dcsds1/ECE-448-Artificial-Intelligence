# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:17:26 2016

@author: changsongdong
"""

import numpy as np

class NaiveBayes:
    def get_images(data_file_name):
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
                    
        data = np.delete(np.asarray(data), 28, 1).reshape(height / 28, 28, 28)
        return data
    
    def get_label(label_file_name):
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
        
    def priors(label):
        priors = []
        for i in range(10):
            priors.append(len(np.where(label == i)[0]) / len(label))
        return priors
    
    def likelihoods(train_data, label):
        likelihoods = np.zeros((20, 28, 28))
        for i in range(10):
            index = np.where(label == i)[0]
            data_i = train_data[index]
            data_i = np.sum(data_i, 0)
            
            likelihoods[i] = (len(index) - data_i) / len(index) # P(F_ij == 0 \ class)
            likelihoods[i + 1] = data_i /len(index) # P(F_ij == 1 \ class)
            
        return likelihoods
        
    def train():
        return
    
    def test():
        return
        
if __name__ == '__main__':
    classifier = NaiveBayes
#    images = classifier.get_images('trainingimages')
    label = classifier.get_label('traininglabels')
    priors = classifier.priors(label)
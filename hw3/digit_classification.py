# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:17:26 2016

@author: changsongdong
"""

import numpy as np

smoothing_constant = 1

class NaiveBayes:
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
                    
        data = np.delete(np.asarray(data), 28, 1).reshape(height / 28, 28, 28)
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
        
# =============================== train =============================
        
    def priors(self, label):
        priors = []
        for i in range(10):
            priors.append(len(np.where(label == i)[0]) / len(label))
        return np.asarray(priors)
    
    def likelihoods(self, train_data, label):
        likelihoods = np.zeros((20, 28, 28))
        for i in range(10):
            index = np.where(label == i)[0]
            data_i = train_data[index]
            data_i = np.sum(data_i, 0)
            
            # likelihoods with Laplace smoothing
            likelihoods[2 * i] = (((len(index) - data_i) + smoothing_constant )/
                                 (len(index) + smoothing_constant * 2)) # P(F_ij == 0 \ class)
            likelihoods[2 * i + 1] = ((data_i + smoothing_constant) / 
                                     (len(index) + smoothing_constant * 2)) # P(F_ij == 1 \ class)
            
        return likelihoods
        
    def train(self, train_data_file, train_label_file):
        train_data = self.get_images(train_data_file)
        train_label = self.get_label(train_label_file)
        priors_matrix = self.priors(train_label)
        likelihoods_matrix = self.likelihoods(train_data, train_label)
        return likelihoods_matrix, priors_matrix
        
# ================================= test ====================================
        
    def predict_label(self, test_sample, likelihoods_matrix, priors_matrix):
        predict_label = np.zeros(10)
        posterior = np.zeros((28, 28))
        for i in range(10):
            for j in range(28):
                for k in range(28):
                    posterior[j][k] = likelihoods_matrix[2 * i + test_sample[j][k]][j][k]
            predict_label[i] = np.sum(np.log(posterior))
        return np.argmax(predict_label + np.log(priors_matrix))
        
    def accuracy(self, predicts, test_label_file):
        test_label = self.get_label(test_label_file)
        return np.sum(predicts[:, None] == test_label) / len(test_label)
        
    def test(self, test_data_file, test_label_file, likelihoods_matrix, priors_matrix):
        test_data = self.get_images(test_data_file)
        predict_results = []
        for sample in test_data:
            predict_results.append(self.predict_label(sample, likelihoods_matrix, priors_matrix))
        return self.accuracy(np.asarray(predict_results), test_label_file)
        
if __name__ == '__main__':
    classifier = NaiveBayes()
    like, pri = classifier.train('trainingimages', 'traininglabels')
    acc = classifier.test('testimages', 'testlabels', like, pri)
#    predict = classifier.predict_label(testdata[0], like, pri)
    
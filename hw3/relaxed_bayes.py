#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 11:06:15 2016

@author: changsongdong
"""

import numpy as np
import itertools

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
        """
        priors of different classes in the training set
        
        parameters: labels for all the data
        returns: 10 * 1 array with each element representing for a class's 
                 priors
        """
        
        priors = []
        for i in range(10):
            priors.append(len(np.where(label == i)[0]) / len(label))
        return np.asarray(priors)
    
    def likelihoods(self, train_data, train_label):
        """
        estimate the likelihoods P(Fij | class) for every pixel location (i,j) 
        and for every digit class from 0 to 9
        """
        
        likelihoods = np.zeros((20, 28, 28))
        for i in range(10):
            index = np.where(train_label == i)[0]
            data_i = np.sum(train_data[index], 0)
            
            # likelihoods with Laplace smoothing
            likelihoods[2 * i] = (((len(index) - data_i) + smoothing_constant )/
                                 (len(index) + smoothing_constant * 2)) # P(F_ij == 0 \ class)
            likelihoods[2 * i + 1] = ((data_i + smoothing_constant) / 
                                     (len(index) + smoothing_constant * 2)) # P(F_ij == 1 \ class)
        return likelihoods
        
    def joint_likelihoods(self, train_data, train_label, n, m):
        """
        n * m feature
        """
        
        likelihoods = np.zeros((2**(n*m)*10, 28/n, 28/m))
        all_combinations = [np.reshape(np.array(i), (n, m)) for i in itertools.product([0, 1], repeat = n*m)]
        
        for i in range(10):
            index = np.where(train_label == i)[0]
            data_i = train_data[index]
            
            for sample in range(len(data_i)):
                for row in range(28 / n):
                    for col in range(28 / m):
                        index = np.where(np.sum(all_combinations == 
                                         data_i[n * row : n * row + n, m * col : m * col + m],
                                                axis=(1,2)) == 4)[0]
                        likelihoods[2**(n*m) * i + index][row][col] += 1
        # smoothing！！！！！！！！！！！！
        
        return likelihoods
        
    def train(self, train_data_file, train_label_file, n, m):
        train_data = self.get_images(train_data_file)
        train_label = self.get_label(train_label_file)
        priors_matrix = self.priors(train_label)
        likelihoods_matrix = self.likelihoods(train_data, train_label, n, m)
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
        posterior_probability = predict_label + np.log(priors_matrix)
        label = np.argmax(posterior_probability)
        prob = posterior_probability[label]
        return label, prob
        
    def test(self, test_data_file, test_label_file, likelihoods_matrix, priors_matrix):
        test_data = self.get_images(test_data_file)
        test_label = self.get_label(test_label_file)
        predict_results = []
        max_prob = np.zeros(10)
        min_prob = np.zeros(10)
        for i in range(len(test_data)):
            label, prob = self.predict_label(test_data[i], likelihoods_matrix, priors_matrix)
            predict_results.append(label)
            if prob > max_prob[label]:
                max_prob[label] = prob
            elif prob < min_prob[label]:
                min_prob[label] = prob
            
        predict_results = np.asarray(predict_results)
        acc = self.accuracy(predict_results, test_label)
        print('acc = ', acc)
        return predict_results, test_label

# ================================= evaluation ==============================
        
    def accuracy(self, predicts, test_label):
        return np.sum(predicts[:, None] == test_label) / len(test_label)
          
if __name__ == '__main__':
    classifier = NaiveBayes()
    like, pri = classifier.train('trainingimages', 'traininglabels', 2, 2)
    predict_results, test_label = classifier.test('testimages', 'testlabels', like, pri)

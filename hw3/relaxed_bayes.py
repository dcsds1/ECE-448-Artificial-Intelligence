#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 11:06:15 2016

@author: changsongdong
"""

import numpy as np
import itertools

k = 1 #smoothing_constant

class NaiveBayes:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.all_combinations = [np.reshape(np.array(i), (n, m)) for i in itertools.product([0, 1], repeat = n*m)]
        
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
        
    def joint_likelihoods(self, train_data, train_label, joint=True):
        """
        n * m feature
        """
        
        if joint == True:
            likelihoods = np.zeros((2**(self.n*self.m)*10, 28 // self.n, 28 // self.m))
            for i in range(10):
                class_index = np.where(train_label == i)[0]
                data_i = train_data[class_index]
                
                for sample_idx in range(len(data_i)):
                    for row in range(28 // self.n):
                        for col in range(28 // self.m):
                            combination_index = np.where(np.sum(self.all_combinations == 
                                                    data_i[sample_idx][self.n * row : self.n * row + self.n, 
                                                                       self.m * col : self.m * col + self.m],
                                                                       axis=(1,2)) == 4)[0]
                            likelihoods[2**(self.n*self.m) * i + combination_index[0]][row][col] += 1
                                        
                # smoothing
                likelihoods[2**(self.n*self.m) * i : 2**(self.n*self.m) * (i + 1)] += k 
                likelihoods[2**(self.n*self.m) * i : 2**(self.n*self.m) * (i + 1)] /= (len(data_i) + k * 2**(self.n*self.m))
        # disjoint
        else:
            likelihoods = np.zeros((2**(self.n*self.m)*10, 29 - self.n, 29 - self.m))
            for i in range(10):
                class_index = np.where(train_label == i)[0]
                data_i = train_data[class_index]
                
                for sample_idx in range(len(data_i)):
                    for row in range(29 - self.n):
                        for col in range(29 - self.m):
                            combination_index = np.where(np.sum(self.all_combinations == 
                                                    data_i[sample_idx][row : row + self.n, 
                                                                       col : col + self.m],
                                                                       axis=(1,2)) == 4)[0]
                            likelihoods[2**(self.n*self.m) * i + combination_index[0]][row][col] += 1
                                        
                # smoothing
                likelihoods[2**(self.n*self.m) * i : 2**(self.n*self.m) * (i + 1)] += k 
                likelihoods[2**(self.n*self.m) * i : 2**(self.n*self.m) * (i + 1)] /= (len(data_i) + k * 2**(self.n*self.m))
        
        
        return likelihoods
        
    def train(self, train_data_file, train_label_file):
        train_data = self.get_images(train_data_file)
        train_label = self.get_label(train_label_file)
        priors_matrix = self.priors(train_label)
        likelihoods_matrix = self.joint_likelihoods(train_data, train_label, joint=False)
        return likelihoods_matrix, priors_matrix
        
# ================================= test ====================================
        
    def predict_label(self, test_sample, likelihoods, priors_matrix, joint=True):
        predict_label = np.zeros(10)
        if joint == True:
            posterior = np.zeros((28 // self.n, 28 // self.m))
            for i in range(10):
                for row in range(28 // self.n):
                    for col in range(28 // self.m):
                        combination_index = np.where(np.sum(self.all_combinations == 
                                                    test_sample[self.n * row : self.n * row + self.n,
                                                                self.m * col : self.m * col + self.m],
                                                    axis=(1,2)) == 4)[0]
                        posterior[row][col] = likelihoods[2**(self.n*self.m) * i + combination_index[0]][row][col]
                predict_label[i] = np.sum(np.log(posterior))
            posterior_probability = predict_label + np.log(priors_matrix)
            label = np.argmax(posterior_probability)
            prob = posterior_probability[label]

        # disjoint
        else:
            posterior = np.zeros((29 - self.n, 29 - self.m))
            for i in range(10):
                for row in range(29 - self.n):
                    for col in range(29 - self.m):
                        combination_index = np.where(np.sum(self.all_combinations == 
                                                    test_sample[row : row + self.n,
                                                                col : col + self.m],
                                                    axis=(1,2)) == 4)[0]
                        posterior[row][col] = likelihoods[2**(self.n*self.m) * i + combination_index[0]][row][col]
                predict_label[i] = np.sum(np.log(posterior))
            posterior_probability = predict_label + np.log(priors_matrix)
            label = np.argmax(posterior_probability)
            prob = posterior_probability[label]

        return label, prob
        
    def test(self, test_data_file, test_label_file, likelihoods_matrix, priors_matrix, joint=True):
        test_data = self.get_images(test_data_file)
        test_label = self.get_label(test_label_file)
        predict_results = []
        max_prob = np.zeros(10)
        min_prob = np.zeros(10)
        for i in range(len(test_data)):
            label, prob = self.predict_label(test_data[i], likelihoods_matrix, priors_matrix, joint)
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
    classifier = NaiveBayes(2,2)
    
    like, pri = classifier.train('trainingimages', 'traininglabels')
    predict_results, test_label = classifier.test('testimages', 'testlabels', like, pri, joint=False)

#    train_data = classifier.get_images('trainingimages')
#    train_label = classifier.get_label('traininglabels')
#    likelihoods = classifier.joint_likelihoods(train_data, train_label, 2, 2)
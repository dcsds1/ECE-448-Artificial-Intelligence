#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 19:31:31 2016

@author: changsongdong
"""

import numpy as np
#import itertools
import time


class NaiveBayes:
    def __init__(self, n, m, joint):
        self.n = n
        self.m = m
        self.all_dicts = []
        self.num_per_class = {}
#        self.all_combinations = [np.reshape(np.array(i), (n, m)) for i in itertools.product([0, 1], repeat = n*m)]
        if joint == True:
            for i in range(10):
                self.all_dicts.append([{} for _ in range(0, (28 // self.n) * (28 // self.m))])
        elif joint == False:
            for i in range(10):
                self.all_dicts.append([{} for _ in range(0, (29 - self.n) * (29 - self.m))])
            
#        self.all_dicts = dict(zip(all_dicts, all_dicts))
        
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
                elif data[i][j] == '#':
                    data[i][j] = 1
                elif data[i][j] == '+':
                    data[i][j] = 2
                else:
                    data[i][j] = 3
                    
        data = np.delete(np.asarray(data), 28, 1).reshape(height // 28, 28, 28)
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
            self.num_per_class[i] = len(np.where(label == i)[0]) # store number of samples in each class
        return np.asarray(priors)
        
    def joint_likelihoods(self, train_data, train_label, joint):
        """
        n * m feature
        """
        
        if joint == True:
            for i in range(10):
                class_index = np.where(train_label == i)[0]
                data_i = train_data[class_index] # data from the i_th class
                
                for sample_idx in range(self.num_per_class[i]):
                    for row in range(28 // self.n):
                        for col in range(28 // self.m):
                            # explored feature
                            if tuple(map(tuple, data_i[sample_idx][self.n * row : self.n * row + self.n,
                                         self.m * col : self.m * col + self.m])) \
                                         in self.all_dicts[i][row * (28 // self.m) + col]:
                                self.all_dicts[i][row * (28 // self.m) + col][tuple(map(tuple, \
                                         data_i[sample_idx][self.n * row : self.n * row + self.n,
                                         self.m * col : self.m * col + self.m]))] += 1
                            # feature never seen before
                            else:
                                self.all_dicts[i][row * (28 // self.m) + col]\
                                        [tuple(map(tuple, data_i[sample_idx][self.n * row : self.n * row + self.n,
                                        self.m * col : self.m * col + self.m]))] = 1

#        # disjoint
        else:
            for i in range(10):
                class_index = np.where(train_label == i)[0]
                data_i = train_data[class_index]
                
                for sample_idx in range(self.num_per_class[i]):
                    for row in range(29 - self.n):
                        for col in range(29 - self.m):
                            if tuple(map(tuple, data_i[sample_idx][row : row + self.n,
                                         col : col + self.m])) \
                                         in self.all_dicts[i][row * (29 - self.m) + col]:
                                self.all_dicts[i][row * (29 - self.m) + col][tuple(map(tuple, \
                                         data_i[sample_idx][row : row + self.n,
                                         col : col + self.m]))] += 1
                            # feature never seen before
                            else:
                                self.all_dicts[i][row * (29 - self.m) + col]\
                                        [tuple(map(tuple, data_i[sample_idx][row : row + self.n,
                                        col : col + self.m]))] = 1
                                        
        return self.all_dicts
        
    def train(self, train_data_file, train_label_file, joint):
        train_data = self.get_images(train_data_file)
        train_label = self.get_label(train_label_file)
        start = time.clock()
        pri = self.priors(train_label)
        like = self.joint_likelihoods(train_data, train_label, joint)
        print('training time: ', time.clock() - start)

        return pri, like
        
# ================================= test ====================================
        
    def compute_probability(self, feature, class_index, dict_index):
        if feature in self.all_dicts[class_index][dict_index]:
            return (self.all_dicts[class_index][dict_index][feature] + k) / \
                    (self.num_per_class[class_index] + k * 2**(self.n*self.m))
        else:
            return k / (self.num_per_class[class_index] + k * 2**(self.n*self.m))

    def predict_label(self, test_sample, likelihoods, priors_matrix, joint):
        predict_label = np.zeros(10)
        
        if joint == True:
            posterior = np.zeros((28 // self.n, 28 // self.m))
            for i in range(10):
                for row in range(28 // self.n):
                    for col in range(28 // self.m):
                        posterior[row][col] = self.compute_probability(tuple(map(tuple,\
                                test_sample[self.n * row : self.n * row + self.n,\
                                self.m * col : self.m * col + self.m])), i,\
                                row * (28 // self.m) + col)
                                
                predict_label[i] = np.sum(np.log(posterior))
            posterior_probability = predict_label + np.log(priors_matrix)
            label = np.argmax(posterior_probability)

        # disjoint
        else:
            posterior = np.zeros((29 - self.n, 29 - self.m))
            for i in range(10):
                for row in range(29 - self.n):
                    for col in range(29 - self.m):
                        posterior[row][col] = self.compute_probability(tuple(map(tuple,\
                                test_sample[row : row + self.n, col : col + self.m])), i,\
                                row * (29 - self.m) + col)
                                            
                predict_label[i] = np.sum(np.log(posterior))
            posterior_probability = predict_label + np.log(priors_matrix)
            label = np.argmax(posterior_probability)

        return label
        
    def test(self, test_data_file, test_label_file, likelihoods_matrix, priors_matrix, joint):
        test_data = self.get_images(test_data_file)
        test_label = self.get_label(test_label_file)
        predict_results = []

        start2 = time.clock()
        for i in range(len(test_data)):
            label = self.predict_label(test_data[i], likelihoods_matrix, priors_matrix, joint)
            predict_results.append(label)
        print('testing time: ', time.clock() - start2)
        predict_results = np.asarray(predict_results)
        acc = self.accuracy(predict_results, test_label)
        print('acc = ', acc)
        return predict_results, test_label

# ================================= evaluation ==============================
        
    def accuracy(self, predicts, test_label):
        return np.sum(predicts[:, None] == test_label) / len(test_label)
          
if __name__ == '__main__':
    k = 1 #smoothing_constant
    classifier = NaiveBayes(3,2, joint=True)
    pri, like = classifier.train('trainingimages', 'traininglabels', joint=True)
    predict_results, test_label = classifier.test('testimages', 'testlabels', like, pri, joint=True)

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:17:26 2016

@author: changsongdong
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import pylab

smoothing_constant = 1

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
    
    def likelihoods(self, train_data, label):
        """
        estimate the likelihoods P(Fij | class) for every pixel location (i,j) 
        and for every digit class from 0 to 9
        """
        
        likelihoods = np.zeros((20, 28, 28))
        for i in range(10):
            index = np.where(label == i)[0]
            data_i = np.sum(train_data[index], 0)
            
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
        posterior_probability = predict_label + np.log(priors_matrix)
        label = np.argmax(posterior_probability)
        prob = posterior_probability[label]
        return label, prob
        
    def test(self, test_data_file, test_label_file, likelihoods_matrix, priors_matrix):
        test_data = self.get_images(test_data_file)
        test_label = self.get_label(test_label_file)
        predict_results = []
        max_example = np.zeros((10, 28, 28))
        min_example = np.zeros((10, 28, 28))
        max_prob = np.ones(10) * -3000
        min_prob = np.ones(10)
        
        for i in range(len(test_data)):
            label, prob = self.predict_label(test_data[i], likelihoods_matrix, priors_matrix)
            predict_results.append(label)
            
            if prob > max_prob[label]:
                max_prob[label] = prob
                max_example[label] = test_data[i]
            if prob < min_prob[label]:
                min_prob[label] = prob
                min_example[label] = test_data[i]
            
        predict_results = np.asarray(predict_results)
        acc = self.accuracy(predict_results, test_label)
        print('acc = ', acc)
        return

# ================================= evaluation ==============================
        
    def accuracy(self, predicts, test_label):
        return np.sum(predicts[:, None] == test_label) / len(test_label)
        
    def confusion_matrix(self, predicts, labels):
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
        
    def best_worst_digits(self, test_data_file, test_label_file, likelihoods_matrix, priors_matrix):
        test_data = self.get_images(test_data_file)
        test_label = self.get_label(test_label_file)
        max_example = np.zeros((10, 28, 28))
        min_example = np.zeros((10, 28, 28))
        max_prob = np.ones(10) * -3000
        min_prob = np.ones(10)
        
        for i in range(10):
            index = np.where(test_label == i)[0]
            data_i = test_data[index]
            for j in range(len(data_i)):
                label, prob = self.predict_label(data_i[j], likelihoods_matrix, priors_matrix)
                if prob > max_prob[i]:
                    max_prob[i] = prob
                    max_example[i] = data_i[j]
                if prob < min_prob[i]:
                    min_prob[i] = prob
                    min_example[i] = data_i[j]

        return max_example, min_example, max_prob, min_prob
        
# =========================== odds ratios ==========================
        
    def draw_odds(self, class_1, class_2, likelihoods_matrix):
        odds = np.log(likelihoods_matrix[2 * class_1 + 1] / likelihoods_matrix[2 * class_2 + 1])
        fig1, ax1 = plt.subplots() 
        cax1 = ax1.imshow(odds, interpolation='nearest', vmax=1.5, vmin=-3)
        fig1.colorbar(cax1, ticks=[-3, -2, -1, 0, 1])
        
    def draw_likelihoods(self, class_index, likelihoods_matrix):
        fig, ax = plt.subplots() 
        cax = ax.imshow(likelihoods_matrix[2 * class_index + 1], interpolation='nearest')
        fig.colorbar(cax, ticks=[1, 2, 3, 4])
        
    def draw(self, class_1, class_2, likelihoods):
        self.draw_odds(class_1, class_2, likelihoods)
        pylab.savefig(str(class_1)+'_'+str(class_2)+'odds')
        self.draw_likelihoods(class_1, likelihoods)
        pylab.savefig(str(class_1))
        self.draw_likelihoods(class_2, likelihoods)        
        pylab.savefig(str(class_2))

if __name__ == '__main__':
    classifier = NaiveBayes()
    like, pri = classifier.train('trainingimages', 'traininglabels')
#    predict_results, test_label = classifier.test('testimages', 'testlabels', like, pri)
    max_example, min_example, max_prob, min_prob = classifier.best_worst_digits('testimages', 'testlabels', like, pri)
#    cm = classifier.confusion_matrix(predict_results, test_label)   
#    classifier.draw(7,9, like) 
#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 


c = 10000.0
print 'C: {0}'.format(c)
t0 = time()

clf = svm.SVC(kernel='rbf', C = c)
clf.fit(features_train, labels_train)
print 'Training: {0}s'.format(time() - t0)

t0 = time()
pred = clf.predict(features_test)
print 'Predicting: {0}s'.format(time() - t0)

accuracy = accuracy_score(labels_test, pred)
print 'Accuracy: {0}'.format(accuracy)

print sum([p for p in pred if p == 1])

#########################################################



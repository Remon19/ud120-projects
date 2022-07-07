#!/usr/bin/python3

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

#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
### your code goes here ###
from sklearn.svm import SVC
# for i in (10,100,1000,10000):
#     svm_clf = SVC(kernel="rbf",C=i)
#     print('fitting the data...')
#     t0 = time()
#     svm_clf.fit(features_train,labels_train)
#     print('Training time: ', round(time()-t0,3), 'sec.')
#     t0 = time()
#     accuracy = svm_clf.score(features_test,labels_test)
#     print("Predicting Time:", round(time()-t0, 3), "sec.")
#     print('accuracy: ',round(accuracy,3))
#     print(100* '_')

#########################################################
svm_clf = SVC(kernel="rbf",C=10_000)
print('fitting the data...')
t0 = time()
svm_clf.fit(features_train,labels_train)
print('Training time: ', round(time()-t0,3), 'sec.')
t0 = time()
accuracy = svm_clf.score(features_test,labels_test)
print("Predicting Time:", round(time()-t0, 3), "sec.")
print('accuracy: ',round(accuracy,3))
#########################################################
preds = svm_clf.predict(features_test)
chris_preds = len(preds[preds == 1])
print("predicted to be in class(1) Chris:",chris_preds)
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################

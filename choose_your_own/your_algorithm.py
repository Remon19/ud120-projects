#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
###########################KNearest-Neighbors########################
from sklearn.neighbors import KNeighborsClassifier
from time import time
clf = KNeighborsClassifier(n_neighbors=1)
print('fitting the data...')
t0 = time()
clf.fit(features_train,labels_train)
print('Training time: ', round(time()-t0,3), 'sec.')
t0 = time()
accuracy = clf.score(features_test,labels_test)
print("Predicting Time:", round(time()-t0, 3), "sec.")
print('accuracy: ',round(accuracy,3))
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
########################AdaBoost##################################
# from sklearn.ensemble import AdaBoostClassifier
# from time import time
# clf = AdaBoostClassifier(n_estimators=100)
# print('fitting the data...')
# t0 = time()
# clf.fit(features_train,labels_train)
# print('Training time: ', round(time()-t0,3), 'sec.')
# t0 = time()
# accuracy = clf.score(features_test,labels_test)
# print("Predicting Time:", round(time()-t0, 3), "sec.")
# print('accuracy: ',round(accuracy,3))
# try:
#     prettyPicture(clf, features_test, labels_test)
# except NameError:
#     pass
###########################Random Forests##############################################
# from sklearn.ensemble import RandomForestClassifier
# from time import time
# clf = RandomForestClassifier(n_estimators=100,min_samples_split=40)
# print('fitting the data...')
# t0 = time()
# clf.fit(features_train,labels_train)
# print('Training time: ', round(time()-t0,3), 'sec.')
# t0 = time()
# accuracy = clf.score(features_test,labels_test)
# print("Predicting Time:", round(time()-t0, 3), "sec.")
# print('accuracy: ',round(accuracy,3))
# try:
#     prettyPicture(clf, features_test, labels_test)
# except NameError:
#     pass
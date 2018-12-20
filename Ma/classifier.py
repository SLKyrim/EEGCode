# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:40:38 2018

第6步 训练分类器

@author: Long
"""
# In[1]:
import scipy.io as sio
import numpy as np
import scipy.signal as sis
import matplotlib.pyplot as plt
import copy

from sklearn.cross_validation import train_test_split
from sklearn import grid_search
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.externals import joblib

# In[2]:
feats_mat = sio.loadmat('E:\\EEGExoskeleton\\Dataset\\Ma\\20180829\\features.mat')
csp = sio.loadmat('E:\\EEGExoskeleton\\Dataset\\Ma\\20180829\\csp.mat')['csp']

feats_all = feats_mat['features']

# In[]
parameter_grid = [  {'kernel': ['linear'], 'C': [10 ** x for x in range(-1, 4)]},
                    {'kernel': ['poly'], 'degree': [2, 3]},
                    {'kernel': ['rbf'], 'gamma': [0.01, 0.001], 'C': [10 ** x for x in range(-1, 4)]},
                 ]

X = feats_all[:,:-1]
y = feats_all[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("\n#### Searching optimal hyperparameters for precision")

classifier = grid_search.GridSearchCV(svm.SVC(), 
             parameter_grid, cv=5, scoring="accuracy")
classifier.fit(X_train, y_train)
#classifier.fit(X,y)

print("\nScores across the parameter grid:")
for params, avg_score, _ in classifier.grid_scores_:
    print(params, '-->', round(avg_score, 3))

print("\nHighest scoring parameter set:", classifier.best_params_)

y_true, y_pred = y_test, classifier.predict(X_test)
print("\nFull performance report:\n")
print(classification_report(y_true, y_pred)) 

#joblib.dump(classifier, time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))+"_SVM.m") # 按当前时间命名保存训练好的分类器
joblib.dump(classifier, "SVM.m") # 保存训练好的分类器




























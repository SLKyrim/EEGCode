# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:07:25 2017

@author: SingleLong

% 说明：

% 第四步 (capscore版)
% 分别对每个受试者的10次随机种子的电极的结果进行评分

% 通过CSP求区别有无意图的投影矩阵
% 并通过CSP投影矩阵提取EEG窗方差特征
"""
# In[1]
import scipy.io as sio
import numpy as np
import scipy.linalg as la # 线性代数库

from sklearn.cross_validation import train_test_split
from sklearn import grid_search
from sklearn.metrics import classification_report
from sklearn import svm

from sklearn import cross_validation

# In[忽略CSP处警告：ComplexWarning: Casting complex values to real discards the imaginary part]
import warnings
warnings.filterwarnings("ignore") 
# In[2]
id_subject = 1 # 【受试者的编号】
num_pair = 4 # 【从CSP投影矩阵里取得特征对数】

if id_subject < 10:
    input_eegwin_dict = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_0'+\
                                    str(id_subject)+'_Data\\Subject_0'+\
                                    str(id_subject)+'_SelectedWinEEG.mat')
else:
    input_eegwin_dict = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_'+\
                                    str(id_subject)+'_Data\\Subject_'+\
                                    str(id_subject)+'_SelectedWinEEG.mat')
input_eegwin = input_eegwin_dict['WinEEG']
# In[]
def task_Generator(eegwin, label):
    task_0 = [] # 存放标记为-1的EEG窗
    task_1 = [] # 存放标记为1的EEG窗
    for i in range(len(eegwin)):
        if int(label[i]) == -1:
            task_0.append(eegwin[i])
        elif int(label[i]) == 1:
            task_1.append(eegwin[i])  
    return task_0, task_1
# In[4]
# 获取EEG窗的标准化空间协方差矩阵
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

# In[利用CSP投影矩阵提取训练集EEG窗特征]
def feat_Generator(task_0, task_1):
    features = []
    for i in range(len(task_0)):
        Z = np.dot(csp, task_0[i])
        varances = list(np.log(np.var(Z, axis=1))) # axis=1即求每行的方差
        varances = [np.log(x/sum(varances)) for x in varances] # 方差标准化
        varances.append(-1)
        features.append(varances)

    for i in range(len(task_1)):  
        Z = np.dot(csp, task_1[i])
        varances = list(np.log(np.var(Z, axis=1)))
        varances = [np.log(x/sum(varances)) for x in varances]
        varances.append(1)
        features.append(varances)
        
    return np.reshape(features,(len(features),num_pair*2+1))

# In[3]
train_avgacc = 0
test_avgacc = 0
for k in range(10):
    # 80%的数据做训练，20%做测试
    X = [] # EEG window data 
    y = [] # EEG window label
    for i in range(len(input_eegwin)):
        X.append(input_eegwin[i][0])
        y.append(input_eegwin[i][1])
    y = np.reshape(y,(len(input_eegwin),))

    # 训练集较好表现： 
    # Sub1 => random_state = 7
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = k)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    eegwin_0_train, eegwin_1_train = task_Generator(X_train, y_train)

# In[CSP]
### CSP算法
    task = (eegwin_0_train, eegwin_1_train)
    filters = ()
    C_0 = covarianceMatrix(task[0][0])
    for i in range(1,len(task[0])):
        C_0 += covarianceMatrix(task[0][i])
    C_0 = C_0 / len(task[0]) # 获得标记为0的EEG窗的标准化协方差对称矩阵

    C_1 = 0 * C_0 # 用C_1 = np.empty(C_0.shape)有些极小的随机非0数，会导致输出结果每次都会改变
    for i in range(0,len(task[1])):
        C_1 += covarianceMatrix(task[1][i])
    C_1 = C_1 / len(task[1]) # 获得标记为1的EEG窗的标准化协方差对称矩阵

    C = C_0 + C_1 # 不同类别的复合空间协方差矩阵,这是一个对称矩阵
    E,U = la.eig(C) # 获取复合空间协方差矩阵的特征值E和特征向量U,这里C可以分解为C=np.dot(U,np.dot(np.diag(E),U.T))
    #E = E.real # E取实部；取实部后不能实现np.diag(E_0)+np.diag(E_1)=I

    P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U)) # 获取白化变换矩阵

    # 获取白化变换后的协方差矩阵
    S_0 = np.dot(P,np.dot(C_0, np.transpose(P)))
    S_1 = np.dot(P,np.dot(C_1, np.transpose(P)))

    E_0,U_0 = la.eig(S_0)
    # 至此有np.diag(E_0)+np.diag(E_1)=I以及U_0=U_1

    # 这里特征值要按降序排序
    order = np.argsort(E_0)
    order = order[::-1]
    #E_0 = E_0[order]
    U_0 = U_0[:,order] # 将特征矩阵列向量按相应特征值大小降序排序进行排序

    #E_1,U_1 = la.eig(S_1);E_1 = E_1[order];U_1 = U_1[:,order] #测试是否满足np.diag(E_0)+np.diag(E_1)=I

    # 求得CSP投影矩阵W
    W = np.dot(np.transpose(U_0),P)

    csp = np.zeros([num_pair*2,np.shape(W)[0]]) # 提取特征的投影矩阵
    csp[0:num_pair,:] = W[0:num_pair,:] # 取投影矩阵前几行
    csp[num_pair:,:] = W[np.shape(W)[1]-num_pair:,:] # 对应取投影矩阵后几行

    feat_train = feat_Generator(eegwin_0_train, eegwin_1_train)
# In[用训练集特征训练分类器]
    parameter_grid = [  {'kernel': ['linear'], 'C': [10 ** x for x in range(-1, 4)]},
                        {'kernel': ['poly'], 'degree': [2, 3]},
                        {'kernel': ['rbf'], 'gamma': [0.01, 0.001], 'C': [10 ** x for x in range(-1, 4)]},
                     ]

    feat_train_X = feat_train[:,:-1]
    feat_train_y = feat_train[:,-1]

    print("\n#### Searching optimal hyperparameters for precision")

    classifier = grid_search.GridSearchCV(svm.SVC(), 
                                          parameter_grid, cv=5, scoring="accuracy")
    classifier.fit(feat_train_X, feat_train_y)

    print("\nScores across the parameter grid:")
    for params, avg_score, _ in classifier.grid_scores_:
        print(params, '-->', round(avg_score, 4))
    print("\nHighest scoring parameter set:", classifier.best_params_)
    print("\nHighest performance in training set:", classifier.best_score_)
    train_avgacc = train_avgacc + classifier.best_score_
# In[用测试集测试分类器]
    eegwin_0_test, eegwin_1_test = task_Generator(X_test, y_test)
    feat_test = feat_Generator(eegwin_0_test, eegwin_1_test)

    feat_test_X = feat_test[:,:-1]
    feat_test_y = feat_test[:,-1]

    y_true, y_pred = feat_test_y, classifier.predict(feat_test_X)
    print("\nFull performance report:\n")
    print(classification_report(y_true, y_pred)) 

    accuracy = cross_validation.cross_val_score(classifier, feat_test_X, feat_test_y, scoring='accuracy', cv=5)
    print ("Performance in test set: " + str(round(accuracy.mean(), 4)))
    test_avgacc = test_avgacc + round(accuracy.mean(), 4)

train_avgacc = train_avgacc / 10
test_avgacc = test_avgacc / 10
print ("\nThe average accuracy in training set: ", train_avgacc)
print ("The average accuracy in test set: ", test_avgacc)
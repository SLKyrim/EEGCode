# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:07:25 2017

@author: SingleLong

% 说明：

% 第四步

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
from sklearn.externals import joblib

from sklearn import cross_validation
# In[2]
id_subject = 1 # 【受试者的编号】
num_pair = 4 # 【从CSP投影矩阵里取得特征对数】

if id_subject < 10:
    input_eegwin_dict = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_0'+\
                                    str(id_subject)+'_Data\\Subject_0'+\
                                    str(id_subject)+'_WinEEG.mat')
else:
    input_eegwin_dict = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_'+\
                                    str(id_subject)+'_Data\\Subject_'+\
                                    str(id_subject)+'_WinEEG.mat')
input_eegwin = input_eegwin_dict['WinEEG']

# In[]
# 电极帽电极分布
cap = {'1':'Fp1',
       '2':'AF3',
       '3':'F7',
       '4':'F3',
       '5':'FC1',
       '6':'FC5',
       '7':'T7',
       '8':'C3',
       '9':'CP1',
       '10':'CP5',
       '11':'P7',
       '12':'P3',
       '13':'Pz',
       '14':'PO3',
       '15':'O1',
       '16':'Oz',
       '17':'O2',
       '18':'PO4',
       '19':'P4',
       '20':'P8',
       '21':'CP6',
       '22':'CP2',
       '23':'C4',
       '24':'T8',
       '25':'FC6',
       '26':'FC2',
       '27':'F4',
       '28':'F8',
       '29':'AF4',
       '30':'Fp2',
       '31':'Fz',
       '32':'Cz'}

# 电极平均得分
cap_score = {'Fp1':0,'AF3':0,'F7':0,'F3':0,'FC1':0,'FC5':0,
             'T7':0,'C3':0,'CP1':0,'CP5':0,'P7':0,'P3':0,
             'Pz':0,'PO3':0,'O1':0,'Oz':0,'O2':0,'PO4':0,
             'P4':0,'P8':0,'CP6':0,'CP2':0,'C4':0,'T8':0,
             'FC6':0,'FC2':0,'F4':0,'F8':0,'AF4':0,'Fp2':0,
             'Fz':0,'Cz':0}
# In[3]
# 80%的数据做训练，20%做测试
X = [] # EEG window data 
y = [] # EEG window label
for i in range(len(input_eegwin)):
    X.append(input_eegwin[i][0])
    y.append(input_eegwin[i][1])
y = np.reshape(y,(len(input_eegwin),))

# 训练集较好表现： 
# Sub1 => random_state = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
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

eegwin_0_train, eegwin_1_train = task_Generator(X_train, y_train)
# In[4]
# 获取EEG窗的标准化空间协方差矩阵
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca
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
# In[Save the classification model]
if id_subject < 10:
    joblib.dump(classifier, "E:\\EEGExoskeleton\\Data\\Models\\Subject_0"+str(id_subject)+"_gridsearch_SVM.m")
else:
    joblib.dump(classifier, "E:\\EEGExoskeleton\\Data\\Models\\Subject_"+str(id_subject)+"_gridsearch_SVM.m")
# In[Electrode Selection]
## Ref: Meng, J., et al. 
## Automated selecting subset of channels based on CSP in motor imagery brain-computer interface system. 
## in Robotics and biomimetics (ROBIO), 2009 IEEE international conference on. 2009. IEEE.
#score_list = [] # 评分表
## axis = None时，输入数据为一维则返回向量范数，二维时返回矩阵范数
#M = np.linalg.norm(csp, ord=1, axis=None)
#for i in range(len(csp[0])):
#    m = np.linalg.norm(csp[:,i], ord=1, axis=None)
#    score_list.append(m/M)
#score_order = np.argsort(score_list)
#score_order = score_order[::-1]+1
#
#cap_list = [] # 保存最优电极顺序
#print ("\nOptimal electrode sorted order: ")
#for i in range(len(score_order)):
#    cap_list.append(cap[str(score_order[i])])
#    print (cap[str(score_order[i])], end=' ')
##    print (cap[str(score_order[i])])
#
#for i in range(len(cap_list)):
#    cap_score[cap_list[i]] = cap_score[cap_list[i]] + 32 - i
# In[]  
#if id_subject < 10:
#    sio.savemat('E:\\EEGExoskeleton\\Data\\Subject_0'+str(id_subject)+\
#                '_Data\\Subject_0'+str(id_subject)+'_features.mat',\
#                {'features' : features})
#    sio.savemat('E:\\EEGExoskeleton\\Data\\Subject_0'+str(id_subject)+\
#                '_Data\\Subject_0'+str(id_subject)+'_csp.mat',\
#                {'csp' : csp})
#else:
#    sio.savemat('E:\\EEGExoskeleton\\Data\\Subject_'+str(id_subject)+\
#                '_Data\\Subject_'+str(id_subject)+'_features.mat',\
#                {'features' : features})
#    sio.savemat('E:\\EEGExoskeleton\\Data\\Subject_'+str(id_subject)+\
#                '_Data\\Subject_'+str(id_subject)+'_csp.mat',\
#                {'csp' : csp})
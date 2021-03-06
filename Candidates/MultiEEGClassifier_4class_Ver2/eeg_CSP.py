# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:07:25 2017

@author: SingleLong

% 说明：

% 第四步

% 通过CSP求区别有无意图的投影矩阵
% 并通过CSP投影矩阵提取EEG窗方差特征
"""

import scipy.io as sio
import numpy as np
import scipy.linalg as la # 线性代数库

id_subject = 3 # 【受试者的编号】
num_pair = 6 # 【从CSP投影矩阵里取得特征对数】

if id_subject < 10:
    input_eegwin_dict = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0'+\
                                    str(id_subject)+'_Data\\Subject_0'+\
                                    str(id_subject)+'_WinEEG_4class.mat')
else:
    input_eegwin_dict = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_'+\
                                    str(id_subject)+'_Data\\Subject_'+\
                                    str(id_subject)+'_WinEEG_4class.mat')
input_eegwin = input_eegwin_dict['WinEEG']

label_list = []
for i in range(len(input_eegwin)):
    label_list.append(int(input_eegwin[i][-1]))
num_label = range(len(set(label_list))) # 类别数目
num_band = range(len(input_eegwin[0])-1) # 频带数目

eeg_win = [] # 各个类别下不同频段的EEG窗
for label in num_label: # 【共4种类别】
    eeg_win.append([]) # 第一维为类别
    for band in num_band:
        eeg_win[label].append([]) # 第二维为该类别下各频带EEG窗

for i in range(len(input_eegwin)):
    if int(input_eegwin[i][-1]) == 0:
        for band in num_band:
            eeg_win[0][band].append(input_eegwin[i][band])
    elif int(input_eegwin[i][-1]) == 1:
        for band in num_band:
            eeg_win[1][band].append(input_eegwin[i][band])
    elif int(input_eegwin[i][-1]) == 2:
        for band in num_band:
            eeg_win[2][band].append(input_eegwin[i][band])
    else:
        for band in num_band:
            eeg_win[3][band].append(input_eegwin[i][band])

task = [] # 每个频带是一个CSP任务
for band in num_band:
    task.append([])
    task[band] = (eeg_win[0][band], eeg_win[1][band],\
                  eeg_win[2][band], eeg_win[3][band])
    

# 获取EEG窗的标准化空间协方差矩阵
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

### 多分类CSP算法：求一个类别和其它类别总和的区别最大的投影矩阵

filters = [] # 每个频带求得一组投影矩阵
for band in num_band:
    filters.append([])
    for label in num_label:
        C_0 = covarianceMatrix(task[band][label][0])
        for t in range(1, len(task[band][label])):
            C_0 += covarianceMatrix(task[band][label][t])
        C_0 = C_0 / len(task[band][label]) # 获得某一个类别的标准化协方差矩阵Rx
        
        count = 0
        not_C_0 = C_0 * 0
        for not_x in [element for element in num_label if element != label]:
            for t in range(0, len(task[band][not_x])):
                not_C_0 += covarianceMatrix(task[band][not_x][t])
                count += 1
        not_C_0 = not_C_0 / count # 获得其它类别的标准化协方差矩阵not_Rx
        
        # 计算空间滤波器
        C = C_0 + not_C_0 # 不同类别的复合空间协方差矩阵
        E,U = la.eig(C) # 获取复合空间协方差矩阵的特征值E和特征向量U
        
        order = np.argsort(E) # 升序排序
        order = order[::-1] # 翻转以使特征值降序排序
        E = (E[order]).real 
        U = (U[:,order]).real
        
        P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U)) # 获取白化变换矩阵
        # 获取白化变换后的协方差矩阵
        S_0 = np.dot(P,np.dot(C_0,np.transpose(P))) 
        not_S_0 = np.dot(P,np.dot(not_C_0,np.transpose(P)))
        
        E_0,U_0 = la.eig(S_0)
        not_E_0,not_U_0 = la.eig(not_S_0)
        # 至此有np.diag(E1)+np.diag(E2)=I以及U1=U2
        
        order = np.argsort(E_0) # 升序排序
        order = order[::-1] # 翻转以使特征值降序排序
        E_0 = (E_0[order]).real
        U_0 = (U_0[:,order]).real
    
        #not_E_0 = (not_E_0[order]).real; not_U_0 = (not_U_0[:,order]).real # 测试是否满足np.diag(E_0)+np.diag(not_E_0)=I和U_0=not_U_0
     
        # 求得矩阵W,其列向量即CSP滤波器
        W = np.dot(np.transpose(U_0),P)
    
        filters[band].append(W)


csp = [] # 提取特征的投影矩阵
for band in num_band:
    csp.append([])
    for label in num_label:
        temp = np.zeros([num_pair*2,np.shape(filters)[-1]]) 
        temp[0:num_pair,:] = filters[band][label][0:num_pair,:] # 取投影矩阵前几行
        temp[num_pair:,:] = filters[band][label][np.shape(filters)[-1]-num_pair:,:] # 对应取投影矩阵后几行
        csp[band].append(temp)


# 利用投影矩阵提取EEG窗特征
band_feats = [] # 每个频带的特征值拼接
for label in num_label:
    band_feats.append([])
    varances = []
    for band in num_band:
        varances.append([])
        for i in range(len(eeg_win[label][band])):
            Z = np.dot(csp[band][label], eeg_win[label][band][i])
            varances[band].append(list(np.log(np.var(Z, axis=1)))) # axis=1即求每行的方差得特征值
    band_feats[label] = [np.hstack((varances[0],varances[1],varances[2]))]
    
features = []
for label in num_label:
    for i in range(len(band_feats[label][0])):
        temp = list(band_feats[label][0][i])
        temp.append(label)
        features.append(temp)

if id_subject < 10:
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0'+str(id_subject)+\
                '_Data\\Subject_0'+str(id_subject)+'_features_4class.mat',\
                {'features' : features})
else:
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_'+str(id_subject)+\
                '_Data\\Subject_'+str(id_subject)+'_features_4class.mat',\
                {'features' : features})
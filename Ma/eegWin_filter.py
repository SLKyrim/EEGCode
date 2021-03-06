# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:50:24 2018

第5步：带标签EEG窗的带通滤波器
EEG窗不同频带级联
训练CSP并提取特征

@author: Long
"""

import scipy.io as sio
import numpy as np
import scipy.signal as sis
import scipy.linalg as la # 线性代数库

num_pair = 4 # 【从CSP投影矩阵里取得特征对数】

eeg = sio.loadmat('E:\\EEGExoskeleton\\Dataset\\Ma\\20180829\\labeledEEG.mat')
eeg = eeg['output']

eegwin_0 = [] # 存放标记为-1的EEG窗
eegwin_1 = [] # 存放标记为1的EEG窗

for i in range(len(eeg)):
    if int(eeg[i][1]) == -1:
        # 若EEG窗标记为0
        eegwin_0.append(eeg[i][0])
    elif int(eeg[i][1]) == 1:
        eegwin_1.append(eeg[i][0])

# In[带通滤波并级联]
fs_eeg = 512 # 【采样频率512Hz】
eeg_winWidth = 384 # 【窗宽度】384对应750ms窗长度
def bandpass(data,upper,lower):
    Wn = [2 * upper / fs_eeg, 2 * lower / fs_eeg] # 截止频带0.1-1Hz or 8-30Hz
    b,a = sis.butter(4, Wn, 'bandpass')
    
    filtered_data = np.zeros([32, eeg_winWidth])
    for row in range(32):
        filtered_data[row] = sis.filtfilt(b,a,data[row,:]) 
    
    return filtered_data 

def hstackwin(out_eeg):
    """hstackwin : 把四种频段的EEG低通滤波窗合成一个长窗.

    Parameters:
    -----------
    - out_eeg: 需要低通滤波的目标EEG窗
    - label: 目标窗的类别标签
    """
    out_eeg_band0 = bandpass(out_eeg,upper=0.3,lower=3)
    out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
    out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
    out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
    output = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3))]
    return output

for i in range(len(eegwin_0)):
    eegwin_0[i] = hstackwin(eegwin_0[i])[0]
    eegwin_1[i] = hstackwin(eegwin_1[i])[0]

# In[CSP]
task = (eegwin_0, eegwin_1)

# 获取EEG窗的标准化空间协方差矩阵
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

### CSP算法
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

order = np.argsort(E) # 升序排序
order = order[::-1] # 翻转以使特征值降序排序
E = E[order] 
U = U[:,order]

P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U)) # 获取白化变换矩阵

# 获取白化变换后的协方差矩阵
S_0 = np.dot(P,np.dot(C_0, np.transpose(P)))
S_1 = np.dot(P,np.dot(C_1, np.transpose(P)))

E_0,U_0 = la.eig(S_0)
# 至此有np.diag(E_0)+np.diag(E_1)=I以及U_0=U_1

# 这里特征值也要按降序排序
order = np.argsort(E_0)
order = order[::-1]
E_0 = E_0[order]
U_0 = U_0[:,order]

#E_1,U_1 = la.eig(S_1);E_1 = E_1[order];U_1 = U_1[:,order] #测试是否满足np.diag(E_0)+np.diag(E_1)=I

# 求得CSP投影矩阵W
W = np.dot(np.transpose(U_0),P)

csp = np.zeros([num_pair*2,np.shape(W)[0]]) # 提取特征的投影矩阵
csp[0:num_pair,:] = W[0:num_pair,:] # 取投影矩阵前几行
csp[num_pair:,:] = W[np.shape(W)[1]-num_pair:,:] # 对应取投影矩阵后几行

# 利用投影矩阵提取EEG窗特征
features = []
for i in range(len(eegwin_0)):
    Z = np.dot(csp, eegwin_0[i])
    varances = list(np.log(np.var(Z, axis=1))) # axis=1即求每行的方差
    varances = [np.log(x/sum(varances)) for x in varances] # 方差标准化
    varances.append(-1)
    features.append(varances)

for i in range(len(eegwin_1)):  
    Z = np.dot(csp, eegwin_1[i])
    varances = list(np.log(np.var(Z, axis=1)))
    varances = [np.log(x/sum(varances)) for x in varances]
    varances.append(1)
    features.append(varances)

# In[保存CSP矩阵和特征]
sio.savemat('E:\\EEGExoskeleton\\Dataset\\Ma\\20180829\\features.mat',\
            {'features' : features})
sio.savemat('E:\\EEGExoskeleton\\Dataset\\Ma\\20180829\\csp.mat',\
            {'csp' : csp})













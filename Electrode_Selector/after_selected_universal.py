# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 10:19:57 2018

@author: Long
"""
# In[]
import scipy.io as sio
import numpy as np
import scipy.signal as sis

import scipy.linalg as la # 线性代数库

from sklearn.cross_validation import train_test_split
from sklearn import grid_search
from sklearn import svm

from sklearn import cross_validation

# In[忽略CSP处警告：ComplexWarning: Casting complex values to real discards the imaginary part]
import warnings
warnings.filterwarnings("ignore") 
# In[]
id_subject = 4 # 【受试者的编号】
num_pair = 4 # 【从CSP投影矩阵里取得特征对数】
# 电极帽电极分布
cap_id = {'Fp1':1 ,'AF3':2 ,'F7 ':3 ,'F3 ':4 ,'FC1':5 ,'FC5':6 ,
          'T7 ':7 ,'C3 ':8 ,'CP1':9 ,'CP5':10,'P7 ':11,'P3 ':12,
          'Pz ':13,'PO3':14,'O1 ':15,'Oz ':16,'O2 ':17,'PO4':18,
          'P4 ':19,'P8 ':20,'CP6':21,'CP2':22,'C4 ':23,'T8 ':24,
          'FC6':25,'FC2':26,'F4 ':27,'F8 ':28,'AF4':29,'Fp2':30,
          'Fz ':31,'Cz ':32}

# 对动作信号【低通滤波】
fs_gait = 121 # 采样频率121Hz
Wn = 1 # 截止频率1Hz

# 对EEG信号带通滤波
fs = 512 # 【采样频率512Hz】
win_width = 384 # 【窗宽度】384对应750ms窗长度

out_count = 0 # 输出文件批数
peak_bias = 40 # 【设置从膝关节角度最大处的偏移值，作为划无意图窗的起点】
valley_bias = 0 # 【设置从膝关节角度最大处的偏移值，作为划无意图窗的起点】
stop_bias = 450 # 【设置停顿处从膝关节角度最大处的偏移值，作为划无意图窗的起点】
gait_win_width = fs_gait / fs * win_width # 在步态数据里将划窗可视化，应该把EEG窗的宽度转换到步态窗的宽度
# In[]
if id_subject < 10:
    eeg_mat_data = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_0' +\
                               str(id_subject) + '_Data\\Subject_0' +\
                               str(id_subject) + '_RawEEG.mat')
    gait_mat_data = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_0' +\
                                str(id_subject) + '_Data\\Subject_0' +\
                                str(id_subject) + '_RawMotion.mat')
    score = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_0'+str(id_subject)+\
                        '_Data\\Subject_0'+str(id_subject)+'_score.mat')['score']
else:
    eeg_mat_data = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_' +\
                               str(id_subject) + '_Data\\Subject_' +\
                               str(id_subject) + '_RawEEG.mat')
    gait_mat_data = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_' +\
                                str(id_subject) + '_Data\\Subject_' +\
                                str(id_subject) + '_RawMotion.mat')
    score = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_'+str(id_subject)+\
                        '_Data\\Subject_0'+str(id_subject)+'_score.mat')['score']

eeg_data = eeg_mat_data['rawEEG']
gait_data = gait_mat_data['rawMotion']

num_trial = np.shape(eeg_data)[1] # 获取受试者进行试验的次数
# In[]
## 对动作信号【低通滤波】
def lowpass(data, num_sample):
    b,a = sis.butter(4, 2*Wn/fs_gait, 'lowpass')
    
    filtered_data = np.zeros(num_sample)
    filtered_data = sis.filtfilt(b,a,data) 
    
    return filtered_data

# 找步态数据中的极大值
def find_peak_point(dataset):
    peakind = [] # 存放极大值的索引
    index = 0 
    for data in dataset:
        if index != 0 and index != len(dataset)-1:
            if data >= dataset[index-1] and data >= dataset[index+1]:
                peakind.append(index)
                index += 1
            else:
                index += 1
                continue
        else:
            index += 1
            continue
    return peakind

# 找步态数据中跨越障碍极大值点前的极小值
def find_valley_point(dataset, peakind_sorted):
    valleyind = [] # 存放极小值的索引
    index = 0 
    for data in dataset:
        if index != 0 and index != len(dataset)-1:
            if data <= dataset[index-1] and data <= dataset[index+1]:
                valleyind.append(index)
                index += 1
            else:
                index += 1
                continue
        else:
            index += 1
            continue
    
    valleyind_sorted = [] # 存放跨越前的极小值索引
    for peak in peakind_sorted:
        index = 0
        for valley in valleyind:
            if valleyind[index+1] > peak:
                valleyind_sorted.append(valley)
                break # 找到这个极大值点前的极值点即可开始找下一个极大值点前的极小值点了
            else:
                index += 1
                continue
            
    return valleyind_sorted

# 对EEG信号带通滤波
def bandpass(data, num_elec, upper, lower):
    Wn = [2 * upper / fs, 2 * lower / fs] # 截止频带0.1-1Hz or 8-30Hz
    b,a = sis.butter(4, Wn, 'bandpass')
    
    filtered_data = np.zeros([num_elec, win_width])
    for row in range(num_elec):
        filtered_data[row] = sis.filtfilt(b,a,data[row,:]) 
    
    return filtered_data 

def stopwin(index, STOP_BIAS):
    """stopwin : 从每三段跨越的第三次跨越最大角度索引找停顿处的索引并返回.

    Parameters:
    -----------
    - index: 跨越时最大角度的索引列表 
    - STOP_BIAS: 停顿处索引与第三次跨越最大角度索引的偏移距离
    """
    stop_win_index = []
    for i in range(len(index)):
        if (i+1)%3 == 0:
            stop_win_index.append(index[i] + STOP_BIAS)
    return np.array(stop_win_index)

def hstackwin(out_eeg, label, num_elec):
    """hstackwin : 把四种频段的EEG低通滤波窗合成一个长窗.

    Parameters:
    -----------
    - out_eeg: 需要低通滤波的目标EEG窗
    - label: 目标窗的类别标签
    - num_elec: EEG窗电极数
    """
    out_eeg_band0 = bandpass(out_eeg,num_elec,upper=0.3,lower=3)
    out_eeg_band1 = bandpass(out_eeg,num_elec,upper=4,lower=7)
    out_eeg_band2 = bandpass(out_eeg,num_elec,upper=8,lower=13)
    out_eeg_band3 = bandpass(out_eeg,num_elec,upper=13,lower=30)
    output = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)), label]
    return output

def winGenerator(i, num_step):
    """winGenerator : 生成EEG窗.

    Parameters:
    -----------
    - i: 跨越时最大角度的索引列表 
    - num_step: 本次trial的跨越次数
    """
    # 当步态数据不是空集时（有效时）   
    # 取右膝跨越极值点索引
    r_peakind = find_peak_point(gait_data[0][i][0])
    r_peak = [gait_data[0][i][0][j] for j in r_peakind] # 获取极值点
    r_peak_sorted = sorted(r_peak, reverse=True) # 将极值点降序排序
    r_peakind_sorted = [] # 对应降序排序极值点的索引
    for j in r_peak_sorted[:num_step]:
        r_peakind_sorted.append(list(gait_data[0][i][0]).index(j))
    r_peakind_sorted = np.array(sorted(r_peakind_sorted))
        
    # 取左膝跨越极值点索引
    l_peakind = find_peak_point(gait_data[0][i][1])
    l_peak = [gait_data[0][i][1][j] for j in l_peakind] # 获取极值点
    l_peak_sorted = sorted(l_peak, reverse=True) # 将极值点降序排序
    l_peakind_sorted = [] # 对应降序排序极值点的索引
    for j in l_peak_sorted[:num_step]:
        l_peakind_sorted.append(list(gait_data[0][i][1]).index(j))
    l_peakind_sorted = np.array(sorted(l_peakind_sorted))
        
    r_valleyind_sorted = np.array(find_valley_point(gait_data[0][i][0], r_peakind_sorted)) # 右膝跨越前的极小值点
    l_valleyind_sorted = np.array(find_valley_point(gait_data[0][i][1], l_peakind_sorted)) # 左膝跨越前的极小值点
       
    # 取无跨越意图EEG窗，标记为-1   
    rp_win_index = r_peakind_sorted + peak_bias # 步态窗起始索引
    lp_win_index = l_peakind_sorted + peak_bias 
        
    # 取有跨越意图EEG窗，标记为1
    rv_win_index = r_valleyind_sorted + valley_bias     
    lv_win_index = l_valleyind_sorted + valley_bias
        
    # 取得每三次跨越完停顿的地方的索引
    rstop_win_index_sorted = stopwin(rp_win_index, stop_bias)
    lstop_win_index_sorted = stopwin(lp_win_index, stop_bias)
        
    # 以上步态索引转换为EEG信号窗的起始索引
    rp_win_index = rp_win_index * fs / fs_gait 
    lp_win_index = lp_win_index * fs / fs_gait
    rv_win_index = rv_win_index * fs / fs_gait
    lv_win_index = lv_win_index * fs / fs_gait
    rstop_win_index = rstop_win_index_sorted * fs / fs_gait
    lstop_win_index = lstop_win_index_sorted * fs / fs_gait
               
    for k in range(num_step):
        if r_peakind_sorted[k] < l_peakind_sorted[k]:
            # 先跨右腿
            #print('r') # 测试用，观察跨越用的腿是否一致
            # 无跨越意图窗
            out_eeg = eeg_selected[i][:,int(rp_win_index[k]):(int(rp_win_index[k])+win_width)]
            eegwin.append(hstackwin(out_eeg,-1,len(out_eeg)))
            if (k+1)%3 == 0:
                out_eeg = eeg_selected[i][:,int(rstop_win_index[int(k/3)]):(int(rstop_win_index[int(k/3)])+win_width)]
                eegwin.append(hstackwin(out_eeg,-1,len(out_eeg)))
            # 有跨越意图窗
            out_eeg =  eeg_selected[i][:,int(rv_win_index[k]-win_width):int(rv_win_index[k])]
            eegwin.append(hstackwin(out_eeg,1,len(out_eeg)))
        else:
            #print('l') # 测试用，观察跨越用的腿是否一致
            # 无跨越意图窗
            out_eeg = eeg_selected[i][:,int(lp_win_index[k]):(int(lp_win_index[k])+win_width)]
            eegwin.append(hstackwin(out_eeg,-1,len(out_eeg)))
            if (k+1)%3 == 0:
                out_eeg = eeg_selected[i][:,int(lstop_win_index[int(k/3)]):(int(lstop_win_index[int(k/3)])+win_width)]
                eegwin.append(hstackwin(out_eeg,-1,len(out_eeg)))
            # 有跨越意图窗
            out_eeg =  eeg_selected[i][:,int(lv_win_index[k]-win_width):int(lv_win_index[k])]
            eegwin.append(hstackwin(out_eeg,1,len(out_eeg)))     
            
def task_Generator(eegwin, label):
    task_0 = [] # 存放标记为-1的EEG窗
    task_1 = [] # 存放标记为1的EEG窗
    for i in range(len(eegwin)):
        if int(label[i]) == -1:
            task_0.append(eegwin[i])
        elif int(label[i]) == 1:
            task_1.append(eegwin[i])  
    return task_0, task_1

def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

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
# In[先对原始EEG和步态数据进行裁剪]
# 找两次打标位置
for i in range(num_trial):
    label_index = 0 # 打标位置

    # 通过上升沿找两次打标位置
    temp = eeg_data[0][i][32][0]
    test_temp = eeg_data[0][i][32] # 用来检验EEG信号是否有效
    for data_label in eeg_data[0][i][32]:
        if data_label <= temp:
            label_index += 1
            temp = data_label
            continue
        else:
            label_index += 1
            temp = data_label
            break
        
    label_index_1 = label_index # 第一次打标位置
    
    for data_label in eeg_data[0][i][32][label_index_1:]:
        if data_label <= temp:
            label_index += 1
            temp = data_label
            continue
        else:
            label_index += 1
            break
    
    label_index_2 = label_index # 第二次打标位置
    
    # 截取两次打标之间的数据
    eeg_data[0][i] = eeg_data[0][i][0:32, label_index_1:label_index_2]
    
    # 截取步态数据并低通滤波
    if label_index_2 == len(test_temp):
        # 如果该次试验的EEG数据无效
        # 无效数据为没有打标或者只打了一次标
        gait_data[0][i] = []
    else:
        gait_temp = gait_data[0][i].T
        num_sample = len(gait_temp[0])
        gait_end_index = round((label_index_2 - label_index_1) * fs_gait / 512) # 第二次打标对应步态数据的位置
        
        r_pass = lowpass(gait_temp[0],num_sample)[:gait_end_index] # 右膝数据低通滤波
        l_pass = lowpass(gait_temp[1],num_sample)[:gait_end_index] # 左膝数据低通滤波

        gait_data[0][i] = (gait_data[0][i].T)[:,:gait_end_index]
        gait_data[0][i][0] = r_pass
        gait_data[0][i][1] = l_pass

# In[]
for num_elec_selected in range(8,33):
    # In[rawdata processor]
    elec_id = [] # 需要去掉的电极索引
    for k in range(32-num_elec_selected):
        elec_id.append(cap_id[score[k]]-1)

    eeg_selected = []
    for i in range(num_trial):
        # 删掉次优的电极（第三个参数为0位删除行）
        eeg_selected.append(np.delete(eeg_data[0][i], elec_id, 0))
    # In[eeg_win_generator]
    eegwin = []
    work_trial_1 = 6 # 往返1次的跨越次数
    work_trial_2 = 12 # 往返2次的跨越次数
    work_trial_3 = 18 # 往返3次的跨越次数
    work_trial_4 = 24 # 往返4次的跨越次数
    
    if id_subject == 1:      
        for i in range(num_trial):
            if len(gait_data[0][i]) and i>=0 and i<=6: # 前7次往返1次
                winGenerator(i, work_trial_1)
                out_count += 1
        
            elif len(gait_data[0][i]) and i>=7 and i<=11: # 第8到12次往返2次  
                winGenerator(i, work_trial_2)               
                out_count += 1
        
            elif len(gait_data[0][i]) and i>=12 and i<=19 and i!=13 and i!=17 and i!=19: # 第13到20次往返3次
                winGenerator(i, work_trial_3)          
                out_count += 1
            else:
                continue
            
    elif id_subject == 2: 
        for i in range(num_trial):
            if len(gait_data[0][i]) and i!=2 and i!=9 and i!=12 and i!=13 and i!=15: # 受试对象2的第13次trial效果不好，故去掉
                winGenerator(i, work_trial_3)
                out_count += 1            
            else:
                continue
    
    elif id_subject == 3:
        for i in range(num_trial):
            if len(gait_data[0][i]) and i!=1: # 受试对象3的第二次trial效果不好，故去掉
                winGenerator(i, work_trial_3)
                out_count += 1   
            else:
                continue
    
    elif id_subject == 4:
        for i in range(num_trial):
            if len(gait_data[0][i]) and i == 0: # 第一次往返2次
                winGenerator(i, work_trial_2)
                out_count += 1      
            elif len(gait_data[0][i]) and (i == 1 or (i >= 4 and i <= 6)): # 第2,5,6,7次往返3次
                winGenerator(i, work_trial_3)
                out_count += 1
            elif len(gait_data[0][i]) and (i==3 or i==7 or (i>=9 and i<=11)): # 第4,8,10,11,12次往返4次
                winGenerator(i, work_trial_4)
                out_count += 1
            else:
                continue 
    # In[CSP]
    train_avgacc = 0
    test_avgacc = 0
    
    for seed in range(10):
        # 80%的数据做训练，20%做测试
        X = [] # EEG window data 
        y = [] # EEG window label
        for i in range(len(eegwin)):
            X.append(eegwin[i][0])
            y.append(eegwin[i][1])
        y = np.reshape(y,(len(eegwin),))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = seed)
        eegwin_0_train, eegwin_1_train = task_Generator(X_train, y_train)
        
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
        
        parameter_grid = [  {'kernel': ['linear'], 'C': [10 ** x for x in range(-1, 4)]},
                            {'kernel': ['poly'], 'degree': [2, 3]},
                            {'kernel': ['rbf'], 'gamma': [0.01, 0.001], 'C': [10 ** x for x in range(-1, 4)]},
                         ]

        feat_train_X = feat_train[:,:-1]
        feat_train_y = feat_train[:,-1]

#        print("\n#### Searching optimal hyperparameters for precision")

        classifier = grid_search.GridSearchCV(svm.SVC(), 
                                              parameter_grid, cv=5, scoring="accuracy")
        classifier.fit(feat_train_X, feat_train_y)

#        print("\nScores across the parameter grid:")
#        for params, avg_score, _ in classifier.grid_scores_:
#            print(params, '-->', round(avg_score, 4))
#        print("\nHighest scoring parameter set:", classifier.best_params_)
#        print("\nHighest performance in training set:", classifier.best_score_)
        train_avgacc = train_avgacc + classifier.best_score_
        
        eegwin_0_test, eegwin_1_test = task_Generator(X_test, y_test)
        feat_test = feat_Generator(eegwin_0_test, eegwin_1_test)

        feat_test_X = feat_test[:,:-1]
        feat_test_y = feat_test[:,-1]

        y_true, y_pred = feat_test_y, classifier.predict(feat_test_X)
#        print("\nFull performance report:\n")
#        print(classification_report(y_true, y_pred)) 

        accuracy = cross_validation.cross_val_score(classifier, feat_test_X, feat_test_y, scoring='accuracy', cv=5)
#        print ("Performance in test set: " + str(round(accuracy.mean(), 4)))
        test_avgacc = test_avgacc + round(accuracy.mean(), 4)

    print ("\nThe " + str(num_elec_selected) + " electrodes performance: ")
    train_avgacc = train_avgacc / 10
    test_avgacc = test_avgacc / 10
#    print ("The average accuracy in training set: ", train_avgacc)
#    print ("The average accuracy in test set: ", test_avgacc)
    # 方便粘贴复制
    print (train_avgacc)
    print (test_avgacc)

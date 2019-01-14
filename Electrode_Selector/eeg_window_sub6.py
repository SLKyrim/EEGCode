# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:25:28 2017

@author: SingleLong

% 说明：

% 第三步

% 根据步态信号建立划窗
% 划窗截取EEG信号
% 生成指定受试对象的有意图和无意图区域的EEG窗

% 专门针对第6个受试对象的划窗函数
共进行了28次trail
第1次：往返1次
第2次：往返1次
第3次：往返1次
第4次：打标失败，去掉
第5次：打标失败，去掉
第6次：往返1次
第7次：往返1次，峰值点有问题，去掉
第8次：往返1次，有干扰，建议去掉
第9次：往返1次
第10次：打标失败，去掉
第11次：往返2次
第12次：打标失败，去掉
第13次：打标失败，去掉
第14次：往返2次
第15次：往返2次
第16次：往返2次
第17次：打标失败，去掉
第18次：往返2次
第19次：往返2次
第20次：往返2次
第21次：打标失败，去掉
第22次：往返1次
第23次：往返1次
第24次：往返1次
第25次：往返1次
第26次：往返1次
第27次：往返1次
第28次：往返1次
共往返26次，共26*12=312个窗
"""
# In[1]:
import scipy.io as sio
import numpy as np
import scipy.signal as sis
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# In[]

# 电极帽电极分布
cap_id = {'Fp1':1 ,'AF3':2 ,'F7 ':3 ,'F3 ':4 ,'FC1':5 ,'FC5':6 ,
          'T7 ':7 ,'C3 ':8 ,'CP1':9 ,'CP5':10,'P7 ':11,'P3 ':12,
          'Pz ':13,'PO3':14,'O1 ':15,'Oz ':16,'O2 ':17,'PO4':18,
          'P4 ':19,'P8 ':20,'CP6':21,'CP2':22,'C4 ':23,'T8 ':24,
          'FC6':25,'FC2':26,'F4 ':27,'F8 ':28,'AF4':29,'Fp2':30,
          'Fz ':31,'Cz ':32}

score = ['FC1','Cz ','CP2','CP1','C3 ','P3 ','C4 ','CP6','AF3','Fz ','Pz ',
         'CP5','PO3','P4 ','PO4','Fp2','FC5','FC6','Oz ','FC2','F3 ','AF4',
         'Fp1','O1 ','F4 ','F8 ','F7 ','O2 ','T8 ','P7 ','T7 ','P8 ']
score.reverse()

num_elec = 16 # 最佳子集通道数

elec_id = [] # 需要去掉的电极索引
for k in range(32-num_elec):
    elec_id.append(cap_id[score[k]]-1)

# In[2]:
id_subject = 6 # 【受试者的编号】
work_trial_1 = 6 # 往返1次的跨越次数
work_trial_2 = 12 # 往返2次的跨越次数
work_trial_3 = 18 # 往返3次的跨越次数
work_trial_4 = 24 # 往返3次的跨越次数

if id_subject < 10:
    gait_mat_data = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_0' +\
                                str(id_subject) + '_Data\\Subject_0' +\
                                str(id_subject) + '_FilteredMotion.mat')
    eeg_mat_data = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_0' +\
                               str(id_subject) + '_Data\\Subject_0' +\
                               str(id_subject) + '_CutedEEG.mat')
else:
    gait_mat_data = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_' +\
                                str(id_subject) + '_Data\\Subject_' +\
                                str(id_subject) + '_FilteredMotion.mat')
    eeg_mat_data = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_' +\
                               str(id_subject) + '_Data\\Subject_' +\
                               str(id_subject) + '_CutedEEG.mat')

gait_data = gait_mat_data['FilteredMotion'][0]
eeg_data = eeg_mat_data['CutedEEG']

num_trial = len(gait_data) # 获取受试者进行试验的次数

eeg_temp = []
for i in range(num_trial):
    # 删掉次优的电极（第三个参数为0位删除行）
    eeg_temp.append(np.delete(eeg_data[0][i], elec_id, 0))
    
eeg_data = eeg_temp
# In[3]:
# 绘图-测试用
def Window_plotor_peak(num_axis, data, index_sorted, bias, stop_win_index, win_width):
    # 绘制峰值点以及相应划窗
    data_axis = [i for i in range(num_axis)]
    plt.figure(figsize=[15,4])
    plt.rc('font',family='Times New Roman') # 设置全局字体
    plt.tick_params(labelsize=15) # 设置坐标刻度字体
    plt.xlabel('Time (sampling points)',FontSize=16)
    plt.ylabel('Joint Angle (°)',FontSize=16)
    ax = plt.gca() # 创建子图ax，用来画窗框
    plt.plot(data_axis, data)
    for i in index_sorted:
        plt.scatter(i, data[i])
        rect = patches.Rectangle((i+bias,data[i]),win_width,-40,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
#    for i in stop_win_index:
#        plt.scatter(i, data[i])
#        rect = patches.Rectangle((i,data[i]),win_width,20,linewidth=1,edgecolor='r',facecolor='none')
#        ax.add_patch(rect)

def Window_plotor_valley(num_axis, data, index_sorted, bias, win_width):
    # 绘制谷值点以及相应划窗
    data_axis = [i for i in range(num_axis)]
    plt.figure(figsize=[15,4])
    plt.rc('font',family='Times New Roman') # 设置全局字体
    plt.tick_params(labelsize=15) # 设置坐标刻度字体
    plt.xlabel('Time (sampling points)',FontSize=16)
    plt.ylabel('Joint Angle (°)',FontSize=16)
    ax = plt.gca() # 创建子图ax，用来画窗框
    plt.plot(data_axis, data)
    for i in index_sorted:
        plt.scatter(i, data[i])
        rect = patches.Rectangle((i+bias,data[i]),-win_width,40,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

# In[5]:
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
# In[6]:
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
# In[7]:
# 对EEG信号带通滤波
fs = 512 # 【采样频率512Hz】
win_width = 384 # 【窗宽度】384对应750ms窗长度
fs_gait = 121 # 【步态数据采样频率121Hz】
def bandpass(data,upper,lower):
    Wn = [2 * upper / fs, 2 * lower / fs] # 截止频带0.1-1Hz or 8-30Hz
    b,a = sis.butter(4, Wn, 'bandpass')
    
    filtered_data = np.zeros([num_elec, win_width])
    for row in range(num_elec):
        filtered_data[row] = sis.filtfilt(b,a,data[row,:]) 
    
    return filtered_data 
# In[8]:
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
# In[9]:
def hstackwin(out_eeg, label):
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
    output = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)), label]
#    output = [np.hstack((out_eeg_band2,out_eeg_band3)), label]
    return output
# In[EEG Window Generator]
def winGenerator(i, num_step):
    """winGenerator : 生成EEG窗.

    Parameters:
    -----------
    - i: 跨越时最大角度的索引列表 
    - num_step: 本次trial的跨越次数
    """
    # 当步态数据不是空集时（有效时）   
    # 取右膝跨越极值点索引
    r_peakind = find_peak_point(gait_data[i][0])
    r_peak = [gait_data[i][0][j] for j in r_peakind] # 获取极值点
    r_peak_sorted = sorted(r_peak, reverse=True) # 将极值点降序排序
    r_peakind_sorted = [] # 对应降序排序极值点的索引
    for j in r_peak_sorted[:num_step]:
        r_peakind_sorted.append(list(gait_data[i][0]).index(j))
    r_peakind_sorted = np.array(sorted(r_peakind_sorted))
    
    r_peakind_sorted = r_peakind_sorted[:num_step]
    
    # 取左膝跨越极值点索引
    l_peakind = find_peak_point(gait_data[i][1])
    l_peak = [gait_data[i][1][j] for j in l_peakind] # 获取极值点
    l_peak_sorted = sorted(l_peak, reverse=True) # 将极值点降序排序
    l_peakind_sorted = [] # 对应降序排序极值点的索引
    for j in l_peak_sorted[:num_step]:
        l_peakind_sorted.append(list(gait_data[i][1]).index(j))
    l_peakind_sorted = np.array(sorted(l_peakind_sorted))
    
    l_peakind_sorted = l_peakind_sorted[:num_step]
        
    r_valleyind_sorted = np.array(find_valley_point(gait_data[i][0], r_peakind_sorted)) # 右膝跨越前的极小值点
    l_valleyind_sorted = np.array(find_valley_point(gait_data[i][1], l_peakind_sorted)) # 左膝跨越前的极小值点
    num_axis = len(gait_data[i][0])
       
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
#    rstop_win_index = rstop_win_index_sorted * fs / fs_gait
#    lstop_win_index = lstop_win_index_sorted * fs / fs_gait
        
    # 测试绘图，观察跨越极大值点位置是否找对
    Window_plotor_peak(num_axis,gait_data[i][0],r_peakind_sorted,peak_bias,\
                       rstop_win_index_sorted,gait_win_width)
    plt.title(str(i+1) + 'th trial\'s peak points', FontSize=16) 
    plt.savefig("E:\EEGExoskeleton\Data\Images_Subject"+\
                str(id_subject)+"\Subject"+\
                str(id_subject)+"_trail"+str(i+1)+"_peak.eps")
        
    # 测试绘图，观察跨越前极小值点位置是否找对
    Window_plotor_valley(num_axis, gait_data[i][0], r_valleyind_sorted, \
                         valley_bias, gait_win_width) 
    plt.title(str(i+1) + 'th trial\'s valley points', FontSize=16) 
    plt.savefig("E:\EEGExoskeleton\Data\Images_Subject"+\
                str(id_subject)+"\Subject"+\
                str(id_subject)+"_trail"+str(i+1)+"_valley.eps")
        
    for k in range(num_step):
        if r_peakind_sorted[k] < l_peakind_sorted[k]:
            # 先跨右腿
            #print('r') # 测试用，观察跨越用的腿是否一致
            # 无跨越意图窗
            out_eeg = eeg_data[i][:,int(rp_win_index[k]):(int(rp_win_index[k])+win_width)]
            output.append(hstackwin(out_eeg,-1))
#            if (k+1)%3 == 0:
#                out_eeg = eeg_data[i][:,int(rstop_win_index[int(k/3)]):(int(rstop_win_index[int(k/3)])+win_width)]
#                output.append(hstackwin(out_eeg,-1))
            # 有跨越意图窗
            out_eeg =  eeg_data[i][:,int(rv_win_index[k]-win_width):int(rv_win_index[k])]
            output.append(hstackwin(out_eeg,1))
        else:
            #print('l') # 测试用，观察跨越用的腿是否一致
            # 无跨越意图窗
            out_eeg = eeg_data[i][:,int(lp_win_index[k]):(int(lp_win_index[k])+win_width)]
            output.append(hstackwin(out_eeg,-1))
#            if (k+1)%3 == 0:
#                out_eeg = eeg_data[i][:,int(lstop_win_index[int(k/3)]):(int(lstop_win_index[int(k/3)])+win_width)]
#                output.append(hstackwin(out_eeg,-1))
            # 有跨越意图窗
            out_eeg =  eeg_data[i][:,int(lv_win_index[k]-win_width):int(lv_win_index[k])]
            output.append(hstackwin(out_eeg,1))                   
# In[10]:        
out_count = 0 # 输出文件批数
output = []
peak_bias = 40 # 【设置从膝关节角度最大处的偏移值，作为划无意图窗的起点】
valley_bias = 0 # 【设置从膝关节角度最大处的偏移值，作为划无意图窗的起点】
stop_bias = 400 # 【设置停顿处从膝关节角度最大处的偏移值，作为划无意图窗的起点】
gait_win_width = fs_gait / fs * win_width # 在步态数据里将划窗可视化，应该把EEG窗的宽度转换到步态窗的宽度
for i in range(num_trial):   
    if len(gait_data[i]) and i>=0 and i<=9 and i!=3 and i!=4 and i!=6 and i!=7 and i!=9:
        winGenerator(i, work_trial_1)          
        out_count += 1     
    elif len(gait_data[i]) and i>=10 and i<=19 and i!=11 and i!=12 and i!=16:
        winGenerator(i, work_trial_2)          
        out_count += 1   
    elif len(gait_data[i]) and i>=20 and i!=20:
        winGenerator(i, work_trial_1)          
        out_count += 1   
    else:
        continue
# In[11]:
if id_subject < 10:
    sio.savemat('E:\\EEGExoskeleton\\Data\\Subject_0'+str(id_subject)+\
                '_Data\\Subject_0'+str(id_subject)+'_WinEEG.mat',\
                {'WinEEG':output})
else:
    sio.savemat('E:\\EEGExoskeleton\\Data\\Subject_'+str(id_subject)+\
                '_Data\\Subject_'+str(id_subject)+'_WinEEG.mat',\
                {'WinEEG':output})
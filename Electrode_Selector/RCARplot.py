# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 15:37:52 2019

@author: Long
"""

import numpy as np
import matplotlib.pyplot as plt

#plt.rc('font',family='Times New Roman') # 设置全局字体
#plt.tick_params(labelsize=15) # 设置坐标刻度字体

# In[Sub1]
# 10次训练集分类准确率
trainAcc_sub1 = [0.81590106, 0.826855124, 0.897526502, 0.893639576, 0.914487633,
                 0.907773852, 0.925795053, 0.932155477, 0.943109541, 0.948056537,	
                 0.950883392, 0.949469965, 0.950883392, 0.957243816, 0.958303887,
                 0.956890459, 0.960777385, 0.962897527, 0.966431095, 0.969257951,
                 0.969964664, 0.969257951, 0.975265018, 0.973498233, 0.976325088]
# 10次测试集分类准确率
testAcc_sub1 = [0.78252,	0.78276,	0.868,	   0.88049,	0.88521,	0.88542,
                0.89614,	0.90239,	0.90913,	0.90997,	0.92144,	0.92234,
                0.92647,	0.92355,	0.92764,	0.92275,	0.92488,	0.92461,
                0.93423,	0.93206,	0.92794,	0.92872,	0.93315,	0.93946,
                0.93985]

# Total 10次训练集分类准确率
trainAccTotal_sub1 = [0.897879859,	0.89540636,	0.906007067,	0.916961131,
                      0.918374558,	0.931095406,	0.933215548,	0.939222615,
                      0.934275618,	0.942402827,	0.95335689,	0.954063604,
                      0.948056537,	0.957597173,	0.954063604,	0.954416961,
                      0.955830389,	0.963957597,	0.962897527,	0.969257951,
                      0.97385159,	0.976678445,	0.978091873,	0.976678445,
                      0.976325088
]

# Total 10次测试集分类准确率
testAccTotal_sub1 = [0.8737,	0.88267,	0.87635,	0.88823,	0.88628,
                     0.91214,	0.90049,	0.89538,	0.89026,	0.89948,
                     0.91036,	0.90087,	0.9176,	0.91333,	0.91965,
                     0.9269,	0.91772,	0.93401,	0.92903,	0.93206,
                     0.92926,	0.93853,	0.94047,	0.93636,	0.93985
]

# 25个通道平均准确率
Otest_sub1 = list(np.ones(len(trainAcc_sub1)) * np.mean(testAcc_sub1))
Ttest_sub1 = list(np.ones(len(trainAcc_sub1)) * np.mean(testAccTotal_sub1))

plt.figure(figsize=[5,4])        
plt.grid(ls='--')  # 生成网格
plt.rc('font',family='Times New Roman') # 设置全局字体
plt.tick_params(labelsize=15) # 设置坐标刻度字体
plt.xlim(8,32)
axix = [i+8 for i in range(len(trainAcc_sub1))]
plt.plot(axix,trainAcc_sub1,label='O train')
plt.plot(axix,testAcc_sub1,"green",label='O test',linestyle='--')
plt.plot(axix,trainAccTotal_sub1,'indigo',label='T train')
plt.plot(axix,testAccTotal_sub1,'red',label='T test',linestyle='--')
plt.plot(axix,Otest_sub1,"green",label='O test avg',linestyle=':')
plt.plot(axix,Ttest_sub1,'red',label='T test avg',linestyle=':')
plt.legend(loc=4)
plt.scatter(16, testAcc_sub1[16-8], c='green')
plt.scatter(20, testAccTotal_sub1[20-8], c='red')
plt.xlabel('Channel number',FontSize=14)
plt.ylabel('Accuracy',FontSize=14)
    
plt.savefig("E:\\EEGExoskeleton\\RCAR_2019\\template\\score_sub1.eps")

# In[Sub2]
# 10次训练集分类准确率
trainAcc_sub2 = [0.774285714,	0.793015873,	0.797460317,	0.824126984,
                 0.846349206,	0.843492063,	0.855555556,	0.853333333,
                 0.855873016,	0.863174603,	0.879047619,	0.88,
                 0.880952381,	0.893015873,	0.893650794,	0.905396825,
                 0.903809524,	0.904444444,	0.9,	0.904444444,
                 0.902222222,	0.907936508,	0.903492063,	0.89968254,
                 0.907619048
]

# 10次测试集分类准确率
testAcc_sub2 = [0.71609,	0.74191,	0.75582,	0.79392,	0.8207,
                0.79973,	0.81409,	0.81411,	0.80882,	0.81366,
                0.81853,	0.82691,	0.82895,	0.8415,	0.84148,
                0.8462,	   0.83589,	0.83744,	0.84263,	0.84184,
                0.82446,	0.83872,	0.84052,	0.83413,	0.83777
]

# Total 10次训练集分类准确率
trainAccTotal_sub2 = [0.740634921,	0.744126984,	0.760952381,	0.816825397,
                      0.856190476,	0.858412698,	0.864126984,	0.861269841,
                      0.860634921,	0.876190476,	0.889206349,	0.896190476,
                      0.897142857,	0.892063492,	0.904126984,	0.897142857,
                      0.903492063,	0.903809524,	0.900634921,	0.903809524,
                      0.906666667,	0.903492063,	0.91015873,	0.90984127,
                      0.907619048
]

# Total 10次测试集分类准确率
testAccTotal_sub2 = [0.65352,	0.67476,	0.6857,	0.75602,	0.80045,
                     0.804,	0.80349,	0.79296,	0.80261,	0.82125,
                     0.85111,	0.85105,	0.85568,	0.83598,	0.83555,
                     0.85397,	0.85593,	0.84331,	0.85753,	0.85666,
                     0.84254,	0.83865,	0.83104,	0.83895,	0.83777
]

# 25个通道平均准确率
Otest_sub2 = list(np.ones(len(trainAcc_sub1)) * np.mean(testAcc_sub2))
Ttest_sub2 = list(np.ones(len(trainAcc_sub1)) * np.mean(testAccTotal_sub2))

plt.figure(figsize=[5,4])      
plt.grid(ls='--')  # 生成网格
plt.rc('font',family='Times New Roman') # 设置全局字体
plt.tick_params(labelsize=15) # 设置坐标刻度字体
plt.xlim(8,32)
axix = [i+8 for i in range(len(trainAcc_sub1))]
plt.plot(axix,trainAcc_sub2,label='O train')
plt.plot(axix,testAcc_sub2,"green",label='O test',linestyle='--')
#plt.plot(axix,trainfull_sub1,"green",label='left knee')
#plt.plot(axix,testfull_sub1,"green",label='left knee')
plt.plot(axix,trainAccTotal_sub2,'indigo',label='T train')
plt.plot(axix,testAccTotal_sub2,'red',label='T test',linestyle='--')
plt.plot(axix,Otest_sub2,"green",label='O test avg',linestyle=':')
plt.plot(axix,Ttest_sub2,'red',label='T test avg',linestyle=':')
plt.legend(loc=4)
plt.scatter(18, testAcc_sub2[18-8], c='green')
plt.scatter(17, testAccTotal_sub2[17-8], c='red')
plt.xlabel('Channel number',FontSize=14)
plt.ylabel('Accuracy',FontSize=14)
    
plt.savefig("E:\\EEGExoskeleton\\RCAR_2019\\template\\score_sub2.eps")

# In[Sub3]
# 10次训练集分类准确率
trainAcc_sub3 = [0.721164021,	0.747619048,	0.76031746,	0.768783069,
                 0.787301587,	0.79973545,	0.806084656,	0.81031746,
                 0.825396825,	0.834126984,	0.847089947,	0.848677249,
                 0.86005291,	0.877248677,	0.87962963,	0.887830688,
                 0.896825397,	0.896825397,	0.896560847,	0.905555556,
                 0.907142857,	0.904761905,	0.90978836,	0.915079365,
                 0.916666667
]

# 10次测试集分类准确率
testAcc_sub3 = [0.69053,	0.70292,	0.71079,	0.71167,	0.70135,	0.71448,
                0.72285,	0.73196,	0.75419,	0.7514,	0.7659,	0.77129,
                0.79457,	0.80284,	0.80537,	0.81796,	0.83541,	0.81631,
                0.82426,	0.81724,	0.80791,	0.82113,	0.83554,	0.82824,
                0.83239
]

# Total 10次训练集分类准确率
trainAccTotal_sub3 = [0.737830688,	0.739417989,	0.758201058,	0.776984127,
                      0.797089947,	0.806613757,	0.815343915,	0.831216931,
                      0.832539683,	0.84021164,	0.846825397,	0.85,
                      0.857671958,	0.854232804,	0.853968254,	0.87037037,
                      0.882275132,	0.888888889,	0.900529101,	0.905291005,
                      0.90978836,	0.911375661,	0.913492063,	0.917724868,
                      0.916666667
]

# Total 10次测试集分类准确率
testAccTotal_sub3 = [0.68169,	0.65723,	0.69608,	0.71597,	0.73868,
                     0.7259,	0.74019,	0.73928,	0.74947,	0.74952,
                     0.76232,	0.75872,	0.77532,	0.79015,	0.79171,
                     0.80971,	0.81329,	0.80833,	0.8154,	0.82987,
                     0.83466,	0.84347,	0.82438,	0.84029,	0.83239,
]

# 25个通道平均准确率
Otest_sub3 = list(np.ones(len(trainAcc_sub3)) * np.mean(testAcc_sub3))
Ttest_sub3 = list(np.ones(len(trainAcc_sub3)) * np.mean(testAccTotal_sub3))

plt.figure(figsize=[5,4])       
plt.grid(ls='--')  # 生成网格
plt.rc('font',family='Times New Roman') # 设置全局字体
plt.tick_params(labelsize=15) # 设置坐标刻度字体
plt.xlim(8,32)
axix = [i+8 for i in range(len(trainAcc_sub1))]
plt.plot(axix,trainAcc_sub3,label='O train')
plt.plot(axix,testAcc_sub3,"green",label='O test',linestyle='--')
#plt.plot(axix,trainfull_sub1,"green",label='left knee')
#plt.plot(axix,testfull_sub1,"green",label='left knee')
plt.plot(axix,trainAccTotal_sub3,'indigo',label='T train')
plt.plot(axix,testAccTotal_sub3,'red',label='T test',linestyle='--')
plt.plot(axix,Otest_sub3,"green",label='O test avg',linestyle=':')
plt.plot(axix,Ttest_sub3,'red',label='T test avg',linestyle=':')
plt.legend(loc=4)
plt.scatter(20, testAcc_sub3[20-8], c='green')
plt.scatter(20, testAccTotal_sub3[20-8], c='red')
plt.xlabel('Channel number',FontSize=14)
plt.ylabel('Accuracy',FontSize=14)
    
plt.savefig("E:\\EEGExoskeleton\\RCAR_2019\\template\\score_sub3.eps")

# In[Sub4]
# 10次训练集分类准确率
trainAcc_sub4 = [0.78487395,	0.789635854,	0.801120448,	0.802240896,
                 0.812885154,	0.819887955,	0.830812325,	0.840056022,
                 0.868907563,	0.869467787,	0.866946779,	0.877591036,
                 0.868347339,	0.873109244,	0.876470588,	0.87535014,
                 0.882352941,	0.89047619,	0.888235294,	0.883753501,
                 0.888515406,	0.894397759,	0.892997199,	0.892997199,
                 0.897478992
]

# 10次测试集分类准确率
testAcc_sub4 = [0.72655,	0.72939,	0.74453,	0.73608,	0.73463,
                0.75674,	0.75549,	0.77043,	0.81524,	0.82192,
                0.8055,	0.82907,	0.81556,	0.8007,	0.79019,
                0.80568,	0.80971,	0.80551,	0.80224,	0.79408,
                0.79491,	0.81069,	0.81325,	0.81274,	0.81008
]

# Total 10次训练集分类准确率
trainAccTotal_sub4 = [0.729691877,	0.728571429,	0.7767507,	0.775630252,
                      0.816806723,	0.825490196,	0.825490196,	0.834173669,
                      0.858543417,	0.858263305,	0.870588235,	0.870308123,
                      0.868907563,	0.875630252,	0.870028011,	0.87394958,
                      0.880392157,	0.883753501,	0.882913165,	0.887114846,
                      0.886834734,	0.882072829,	0.891036415,	0.892997199,
                      0.897478992
]

# Total 10次测试集分类准确率
testAccTotal_sub4 = [0.67691,0.65839,	0.70632,	0.69036,	0.75698,
                     0.75217,	0.76825,	0.75621,	0.80328,	0.80563,
                     0.80379,	0.81648,	0.81811,	0.81493,	0.80763,
                     0.81047,	0.81171,	0.80909,	0.80755,	0.80165,
                     0.80516,	0.78967,	0.81172,	0.81274,	0.81008
]

# 25个通道平均准确率
Otest_sub4 = list(np.ones(len(trainAcc_sub1)) * np.mean(testAcc_sub4))
Ttest_sub4 = list(np.ones(len(trainAcc_sub1)) * np.mean(testAccTotal_sub4))

plt.figure(figsize=[5,4])       
plt.grid(ls='--')  # 生成网格
plt.rc('font',family='Times New Roman') # 设置全局字体
plt.tick_params(labelsize=15) # 设置坐标刻度字体
plt.xlim(8,32)
axix = [i+8 for i in range(len(trainAcc_sub1))]
plt.plot(axix,trainAcc_sub4,label='O train')
plt.plot(axix,testAcc_sub4,"green",label='O test',linestyle='--')
#plt.plot(axix,trainfull_sub1,"green",label='left knee')
#plt.plot(axix,testfull_sub1,"green",label='left knee')
plt.plot(axix,trainAccTotal_sub4,'indigo',label='T train')
plt.plot(axix,testAccTotal_sub4,'red',label='T test',linestyle='--')
plt.plot(axix,Otest_sub4,"green",label='O test avg',linestyle=':')
plt.plot(axix,Ttest_sub4,'red',label='T test avg',linestyle=':')
plt.legend(loc=4)
plt.scatter(16, testAcc_sub4[16-8], c='green')
plt.scatter(16, testAccTotal_sub4[16-8], c='red')
plt.xlabel('Channel number',FontSize=14)
plt.ylabel('Accuracy',FontSize=14)
    
plt.savefig("E:\\EEGExoskeleton\\RCAR_2019\\template\\score_sub4.eps")

# In[Channel Scoring]
score = [1108,1081,1056,1035,978,977,928,913,874,848,807,803,775,751,738,
         675,638,632,630,622,539,492,489,483,467,398,379,249,225,193,170,167
]
for i in range(len(score)):
    score[i] = round(score[i]/1280*100,2)
score = sorted(score)
"""
绘制水平条形图方法barh
参数一：y轴
参数二：x轴
"""

plt.figure(figsize=[8,12]) 
plt.rc('font',family='Times New Roman') # 设置全局字体
plt.tick_params(labelsize=16) # 设置坐标刻度字体       
plt.xlabel("overall channel correlation (%)", FontSize = 18)
plt.xlim(0,100)
plt.barh(range(len(score)), score, height=0.5, color='green', alpha=0.8)      # 从下往上画
plt.yticks(range(len(score)), ['P8', 'T7', 'P7', 'T8', 'O2',
                               'F7','F8','F4','O1','Fp1','AF4','F3','FC2',
                               'Oz','FC6','FC5','Fp2','PO4','P4','PO3',
                               'CP5','Pz','Fz','AF3','CP6','C4','P3',
                               'C3','CP1','CP2','Cz','FC1'])
for x, y in enumerate(score):
    plt.text(y + 2, x - 0.4, '%s' % y)
plt.savefig("E:\\EEGExoskeleton\\RCAR_2019\\template\\total_score.eps")
plt.show()



# In[Plot]


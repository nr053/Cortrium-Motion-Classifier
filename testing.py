"""
"Training" the motion classifier on 1st recording and testing on the remaining three in the Vers01_2_latest folder

Sample processing:
1. Crop recording by annotation
2. Split sections into 5 second windows

Class labels:
- activity level
    0. Resting
    1. Moderate
    2. High
- Position:
    0. upright (standing/sitting)
    1. lying-back
    2. lying-left
    3. lying-right
    4. lying-stomach


Model architecture:
1. Split recording into 5 second windows
2. Classify activity level by comparing the mean the three axis' standard deviation to some threshold
4. Position is classified by comaring mean values on each ACC axis to a threshold value.  
"""

import mne
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 
import datetime
import pickle
from thresholding.model import MotionClassifier



#read the first recording
#crop by annotation 
#store 5 second windows with class labels in variable "data1"
raw1 = mne.io.read_raw_edf(os.getcwd() + '/Accelerometer annotations/Version01_2_latest/Vers01_2_AF_Pos0deg/C3030260-221103-V01_2_Pos0deg.edf',
                           include=['ACCX','ACCY', 'ACCZ'])
still_1_1 = raw1.copy().crop(tmin=0,tmax=60)
walking_1 = raw1.copy().crop(tmin=60,tmax=180)
still_1_2 = raw1.copy().crop(tmin=180,tmax=240)
running_1 = raw1.copy().crop(tmin=240,tmax=360)
still_1_3 = raw1.copy().crop(tmin=360,tmax=600)
lying_back_1 = raw1.copy().crop(tmin=600,tmax=660)
still_1_4 = raw1.copy().crop(tmin=660,tmax=720)
lying_left_1 = raw1.copy().crop(tmin=720,tmax=780)
still_1_5 = raw1.copy().crop(tmin=780,tmax=840)
lying_right_1 = raw1.copy().crop(tmin=840,tmax=900)
still_1_6 = raw1.copy().crop(tmin=900,tmax=960)
lying_front_1 = raw1.copy().crop(tmin=960,tmax=1020)
still_1_7 = raw1.copy().crop(tmin=1020,tmax=None)
## crop the recordings excluding the position changes
# still_1_1 = raw1.copy().crop(tmin=0,tmax=58)
# walking_1 = raw1.copy().crop(tmin=58,tmax=180)
# still_1_2 = raw1.copy().crop(tmin=180,tmax=239)
# running_1 = raw1.copy().crop(tmin=240,tmax=360)
# still_1_3 = raw1.copy().crop(tmin=360,tmax=598)
# lying_back_1 = raw1.copy().crop(tmin=603,tmax=659)
# still_1_4 = raw1.copy().crop(tmin=666,tmax=719)
# lying_left_1 = raw1.copy().crop(tmin=724,tmax=780)
# still_1_5 = raw1.copy().crop(tmin=786,tmax=839)
# lying_right_1 = raw1.copy().crop(tmin=846,tmax=897)
# still_1_6 = raw1.copy().crop(tmin=904,tmax=958)
# lying_front_1 = raw1.copy().crop(tmin=964,tmax=1019)
# still_1_7 = raw1.copy().crop(tmin=1020,tmax=None)

data1 = {'X': [], 'act':[], 'pos':[]}
still_epochs_1_1 = mne.make_fixed_length_epochs(still_1_1,duration=5)
for epoch in still_epochs_1_1:
    data1['X'].append(epoch)    
    data1['act'].append(0)    
    data1['pos'].append(0)    
    # data1['act_pred'].append({'X':epoch, 'act':0, 'pos':0})    
    # data1['pos_pred'].append({'X':epoch, 'act':0, 'pos':0})    
walking_epochs_1 = mne.make_fixed_length_epochs(walking_1,duration=5)
for epoch in walking_epochs_1:
    data1['X'].append(epoch)    
    data1['act'].append(1)    
    data1['pos'].append(0)    
    #data1.append({'X':epoch, 'act':1, 'pos':0})    
still_epochs_1_2 = mne.make_fixed_length_epochs(still_1_2,duration=5)
for epoch in still_epochs_1_2:
    data1['X'].append(epoch)    
    data1['act'].append(0)    
    data1['pos'].append(0)    
    #data1.append({'X':epoch, 'act':0, 'pos':0})    
running_epochs_1 = mne.make_fixed_length_epochs(running_1,duration=5)
for epoch in running_epochs_1:
    data1['X'].append(epoch)    
    data1['act'].append(2)    
    data1['pos'].append(0)    
    #data1.append({'X':epoch, 'act':2, 'pos':0})    
still_epochs_1_3 = mne.make_fixed_length_epochs(still_1_3,duration=5)
for epoch in still_epochs_1_3:
    data1['X'].append(epoch)    
    data1['act'].append(0)    
    data1['pos'].append(0)  
#    data1.append({'X':epoch, 'act':0, 'pos':0})    
lying_back_epochs_1 = mne.make_fixed_length_epochs(lying_back_1,duration=5)
for epoch in lying_back_epochs_1:
    data1['X'].append(epoch)    
    data1['act'].append(0)    
    data1['pos'].append(1)  
    #data1.append({'X':epoch, 'act':0, 'pos':1})    
still_epochs_1_4 = mne.make_fixed_length_epochs(still_1_4,duration=5)
for epoch in still_epochs_1_4:
    data1['X'].append(epoch)    
    data1['act'].append(0)    
    data1['pos'].append(0)      
    #data1.append({'X':epoch, 'act':0, 'pos':0})    
lying_left_epochs_1 = mne.make_fixed_length_epochs(lying_left_1,duration=5)
for epoch in lying_left_epochs_1:
    data1['X'].append(epoch)    
    data1['act'].append(0)    
    data1['pos'].append(2)      
    #data1.append({'X':epoch, 'act':0, 'pos':2})   
still_epochs_1_5 = mne.make_fixed_length_epochs(still_1_5,duration=5)
for epoch in still_epochs_1_5:
    data1['X'].append(epoch)    
    data1['act'].append(0)    
    data1['pos'].append(0)      
    #data1.append({'X':epoch, 'act':0, 'pos':0})    
lying_right_epochs_1 = mne.make_fixed_length_epochs(lying_right_1,duration=5)
for epoch in lying_right_epochs_1:
    data1['X'].append(epoch)    
    data1['act'].append(0)    
    data1['pos'].append(3)      
    #data1.append({'X':epoch, 'act':0, 'pos':3})   
still_epochs_1_6 = mne.make_fixed_length_epochs(still_1_6,duration=5)
for epoch in still_epochs_1_6:
    data1['X'].append(epoch)    
    data1['act'].append(0)    
    data1['pos'].append(0)      
    #data1.append({'X':epoch, 'act':0, 'pos':0})    
lying_front_epochs_1 = mne.make_fixed_length_epochs(lying_front_1,duration=5)
for epoch in lying_front_epochs_1:
    data1['X'].append(epoch)    
    data1['act'].append(0)    
    data1['pos'].append(4)      
    #data1.append({'X':epoch, 'act':0, 'pos':4})   
still_epochs_1_7 = mne.make_fixed_length_epochs(still_1_7,duration=5)
for epoch in still_epochs_1_7:
    data1['X'].append(epoch)    
    data1['act'].append(0)    
    data1['pos'].append(0)      
    #data1.append({'X':epoch, 'act':0, 'pos':0})  
#calibrate the orientation of the device
data1['X'] = np.array(data1['X'])
if data1['X'][0][1].mean() > 500:
    data1['X'][:,0,:] = -data1['X'][:,0,:]
    data1['X'][:,1,:] = -data1['X'][:,1,:]

#read the second recording
#crop by annotation 
#store 5 second windows with class labels in variable "data2"
raw2 = mne.io.read_raw_edf(os.getcwd() + '/Accelerometer annotations/Version01_2_latest/Vers01_2_AF_Pos180deg/C3030260-221103-V01_2_Pos180deg.edf',
                            include=['ACCX','ACCY', 'ACCZ'])
still_2_1 = raw2.copy().crop(tmin=0,tmax=60)
walking_2 = raw2.copy().crop(tmin=60,tmax=180)
still_2_2 = raw2.copy().crop(tmin=180,tmax=240)
running_2 = raw2.copy().crop(tmin=240,tmax=360)
still_2_3 = raw2.copy().crop(tmin=360,tmax=600)
lying_back_2 = raw2.copy().crop(tmin=600,tmax=660)
still_2_4 = raw2.copy().crop(tmin=660,tmax=720)
lying_left_2 = raw2.copy().crop(tmin=720,tmax=780)
still_2_5 = raw2.copy().crop(tmin=780,tmax=840)
lying_right_2 = raw2.copy().crop(tmin=840,tmax=900)
still_2_6 = raw2.copy().crop(tmin=900,tmax=960)
lying_front_2 = raw2.copy().crop(tmin=960,tmax=1020)
still_2_7 = raw2.copy().crop(tmin=1020,tmax=None)
## crop the recordings excluding the position changes
# still_2_1 = raw2.copy().crop(tmin=0,tmax=58)
# walking_2 = raw2.copy().crop(tmin=58,tmax=180)
# still_2_2 = raw2.copy().crop(tmin=180,tmax=239)
# running_2 = raw2.copy().crop(tmin=239,tmax=360)
# still_2_3 = raw2.copy().crop(tmin=362,tmax=598)
# lying_back_2 = raw2.copy().crop(tmin=603,tmax=659)
# still_2_4 = raw2.copy().crop(tmin=666,tmax=719)
# lying_left_2 = raw2.copy().crop(tmin=724,tmax=779)
# still_2_5 = raw2.copy().crop(tmin=784,tmax=839)
# lying_right_2 = raw2.copy().crop(tmin=846,tmax=899)
# still_2_6 = raw2.copy().crop(tmin=904,tmax=960)
# lying_front_2 = raw2.copy().crop(tmin=967,tmax=1019)
# still_2_7 = raw2.copy().crop(tmin=1024,tmax=None)
data2 = {'X': [], 'act':[], 'pos':[]}
still_epochs_2_1 = mne.make_fixed_length_epochs(still_2_1,duration=5)
for epoch in still_epochs_2_1:
    data2['X'].append(epoch)    
    data2['act'].append(0)    
    data2['pos'].append(0)
    #data2.append({'X':epoch, 'act':0, 'pos':0})    
walking_epochs_2 = mne.make_fixed_length_epochs(walking_2,duration=5)
for epoch in walking_epochs_2:
    data2['X'].append(epoch)    
    data2['act'].append(1)    
    data2['pos'].append(0)    
    #data2.append({'X':epoch, 'act':1, 'pos':0})    
still_epochs_2_2 = mne.make_fixed_length_epochs(still_2_2,duration=5)
for epoch in still_epochs_2_2:
    data2['X'].append(epoch)    
    data2['act'].append(0)    
    data2['pos'].append(0)
    #data2.append({'X':epoch, 'act':0, 'pos':0})    
running_epochs_2 = mne.make_fixed_length_epochs(running_2,duration=5)
for epoch in running_epochs_2:
    data2['X'].append(epoch)    
    data2['act'].append(2)    
    data2['pos'].append(0)
    #data2.append({'X':epoch, 'act':2, 'pos':0})    
still_epochs_2_3 = mne.make_fixed_length_epochs(still_2_3,duration=5)
for epoch in still_epochs_2_3:
    data2['X'].append(epoch)    
    data2['act'].append(0)    
    data2['pos'].append(0)    
    #data2.append({'X':epoch, 'act':0, 'pos':0})    
lying_back_epochs_2 = mne.make_fixed_length_epochs(lying_back_2,duration=5)
for epoch in lying_back_epochs_2:
    data2['X'].append(epoch)    
    data2['act'].append(0)    
    data2['pos'].append(1)    
    #data2.append({'X':epoch, 'act':0, 'pos':1})    
still_epochs_2_4 = mne.make_fixed_length_epochs(still_2_4,duration=5)
for epoch in still_epochs_2_4:
    data2['X'].append(epoch)    
    data2['act'].append(0)    
    data2['pos'].append(0)        
    #data2.append({'X':epoch, 'act':0, 'pos':0})    
lying_left_epochs_2 = mne.make_fixed_length_epochs(lying_left_2,duration=5)
for epoch in lying_left_epochs_2:
    data2['X'].append(epoch)    
    data2['act'].append(0)    
    data2['pos'].append(2)        
    #data2.append({'X':epoch, 'act':0, 'pos':2})   
still_epochs_2_5 = mne.make_fixed_length_epochs(still_2_5,duration=5)
for epoch in still_epochs_2_5:
    data2['X'].append(epoch)    
    data2['act'].append(0)    
    data2['pos'].append(0)        
    #data2.append({'X':epoch, 'act':0, 'pos':0})    
lying_right_epochs_2 = mne.make_fixed_length_epochs(lying_right_2,duration=5)
for epoch in lying_right_epochs_2:
    data2['X'].append(epoch)    
    data2['act'].append(0)    
    data2['pos'].append(3)        
    #data2.append({'X':epoch, 'act':0, 'pos':3})   
still_epochs_2_6 = mne.make_fixed_length_epochs(still_2_6,duration=5)
for epoch in still_epochs_2_6:
    data2['X'].append(epoch)    
    data2['act'].append(0)    
    data2['pos'].append(0)        
    #data2.append({'X':epoch, 'act':0, 'pos':0})    
lying_front_epochs_2 = mne.make_fixed_length_epochs(lying_front_2,duration=5)
for epoch in lying_front_epochs_2:
    data2['X'].append(epoch)    
    data2['act'].append(0)    
    data2['pos'].append(4)        
    #data2.append({'X':epoch, 'act':0, 'pos':4})   
still_epochs_2_7 = mne.make_fixed_length_epochs(still_2_7,duration=5)
for epoch in still_epochs_2_7:
    data2['X'].append(epoch)    
    data2['act'].append(0)    
    data2['pos'].append(0)        
    #data2.append({'X':epoch, 'act':0, 'pos':0})  
#calibrate the orientation of the device
data2['X'] = np.array(data2['X'])
if data2['X'][0][1].mean() > 500:
    data2['X'][:,0,:] = -data2['X'][:,0,:]
    data2['X'][:,1,:] = -data2['X'][:,1,:]


#read the third recording
#crop by annotation 
#store 5 second windows with class labels in variable "data3"
raw3 = mne.io.read_raw_edf(os.getcwd() + '/Accelerometer annotations/Version01_2_latest/Vers01_2_AS_Pos0deg/C3090273-240103-Vers01_2_AS_Pos0deg.edf',
                            include=['ACCX','ACCY', 'ACCZ'])
## crop the recordings excluding the position changes
# still_3_1 = raw3.copy().crop(tmin=0,tmax=90)
# walking_3 = raw3.copy().crop(tmin=90,tmax=215)
# still_3_2 = raw3.copy().crop(tmin=215,tmax=285)
# running_3 = raw3.copy().crop(tmin=285,tmax=400)
# still_3_3 = raw3.copy().crop(tmin=415,tmax=677)
# lying_back_3 = raw3.copy().crop(tmin=684,tmax=747)
# still_3_4 = raw3.copy().crop(tmin=756,tmax=815)
# lying_left_3 = raw3.copy().crop(tmin=825,tmax=872)
# still_3_5 = raw3.copy().crop(tmin=882,tmax=947)
# lying_right_3 = raw3.copy().crop(tmin=955,tmax=1011)
# still_3_6 = raw3.copy().crop(tmin=1018,tmax=1076)
# lying_front_3 = raw3.copy().crop(tmin=1084,tmax=1139)
# still_3_7 = raw3.copy().crop(tmin=1146,tmax=None)
still_3_1 = raw3.copy().crop(tmin=0,tmax=90)
walking_3 = raw3.copy().crop(tmin=90,tmax=215)
still_3_2 = raw3.copy().crop(tmin=215,tmax=285)
running_3 = raw3.copy().crop(tmin=285,tmax=400)
still_3_3 = raw3.copy().crop(tmin=400,tmax=680)
lying_back_3 = raw3.copy().crop(tmin=680,tmax=750)
still_3_4 = raw3.copy().crop(tmin=750,tmax=820)
lying_left_3 = raw3.copy().crop(tmin=820,tmax=880)
still_3_5 = raw3.copy().crop(tmin=880,tmax=950)
lying_right_3 = raw3.copy().crop(tmin=950,tmax=1015)
still_3_6 = raw3.copy().crop(tmin=1015,tmax=1080)
lying_front_3 = raw3.copy().crop(tmin=1080,tmax=1140)
still_3_7 = raw3.copy().crop(tmin=1140,tmax=None)
data3 = {'X': [], 'act':[], 'pos':[]}
still_epochs_3_1 = mne.make_fixed_length_epochs(still_3_1,duration=5)
for epoch in still_epochs_3_1:
    data3['X'].append(epoch)    
    data3['act'].append(0)    
    data3['pos'].append(0)   
    # data3.append({'X':epoch, 'act':0, 'pos':0})    
walking_epochs_3 = mne.make_fixed_length_epochs(walking_3,duration=5)
for epoch in walking_epochs_3:
    data3['X'].append(epoch)    
    data3['act'].append(1)    
    data3['pos'].append(0)   
    # data3.append({'X':epoch, 'act':1, 'pos':0})    
still_epochs_3_2 = mne.make_fixed_length_epochs(still_3_2,duration=5)
for epoch in still_epochs_3_2:
    data3['X'].append(epoch)    
    data3['act'].append(0)    
    data3['pos'].append(0)   
    # data3.append({'X':epoch, 'act':0, 'pos':0})    
running_epochs_3 = mne.make_fixed_length_epochs(running_3,duration=5)
for epoch in running_epochs_3:
    data3['X'].append(epoch)    
    data3['act'].append(2)    
    data3['pos'].append(0)   
    # data3.append({'X':epoch, 'act':2, 'pos':0})    
still_epochs_3_3 = mne.make_fixed_length_epochs(still_3_3,duration=5)
for epoch in still_epochs_3_3:
    data3['X'].append(epoch)    
    data3['act'].append(0)    
    data3['pos'].append(0)  
    #data3.append({'X':epoch, 'act':0, 'pos':0})    
lying_back_epochs_3 = mne.make_fixed_length_epochs(lying_back_3,duration=5)
for epoch in lying_back_epochs_3:
    data3['X'].append(epoch)    
    data3['act'].append(0)    
    data3['pos'].append(1)  
    #data3.append({'X':epoch, 'act':0, 'pos':1})    
still_epochs_3_4 = mne.make_fixed_length_epochs(still_3_4,duration=5)
for epoch in still_epochs_3_4:
    data3['X'].append(epoch)    
    data3['act'].append(0)    
    data3['pos'].append(0)      
    # data3.append({'X':epoch, 'act':0, 'pos':0})    
lying_left_epochs_3 = mne.make_fixed_length_epochs(lying_left_3,duration=5)
for epoch in lying_left_epochs_3:
    data3['X'].append(epoch)    
    data3['act'].append(0)    
    data3['pos'].append(2)  
    #data3.append({'X':epoch, 'act':0, 'pos':2})   
still_epochs_3_5 = mne.make_fixed_length_epochs(still_3_5,duration=5)
for epoch in still_epochs_3_5:
    data3['X'].append(epoch)    
    data3['act'].append(0)    
    data3['pos'].append(0)      
    #data3.append({'X':epoch, 'act':0, 'pos':0})    
lying_right_epochs_3 = mne.make_fixed_length_epochs(lying_right_3,duration=5)
for epoch in lying_right_epochs_3:
    data3['X'].append(epoch)    
    data3['act'].append(0)    
    data3['pos'].append(3)      
    #data3.append({'X':epoch, 'act':0, 'pos':3})   
still_epochs_3_6 = mne.make_fixed_length_epochs(still_3_6,duration=5)
for epoch in still_epochs_3_6:
    data3['X'].append(epoch)    
    data3['act'].append(0)    
    data3['pos'].append(0)  
    #data3.append({'X':epoch, 'act':0, 'pos':0})    
lying_front_epochs_3 = mne.make_fixed_length_epochs(lying_front_3,duration=5)
for epoch in lying_front_epochs_3:
    data3['X'].append(epoch)    
    data3['act'].append(0)    
    data3['pos'].append(4)      
    # data3.append({'X':epoch, 'act':0, 'pos':4})   
still_epochs_3_7 = mne.make_fixed_length_epochs(still_3_7,duration=5)
for epoch in still_epochs_3_7:
    data3['X'].append(epoch)    
    data3['act'].append(0)    
    data3['pos'].append(0)  
    # data3.append({'X':epoch, 'act':0, 'pos':0})  
#calibrate the orientation of the device
data3['X'] = np.array(data3['X'])
if data3['X'][0][1].mean() > 500:
    data3['X'][:,0,:] = -data3['X'][:,0,:]
    data3['X'][:,1,:] = -data3['X'][:,1,:]

#read the fourth recording
#crop by annotation 
# store 5 second windows with class labels in variable "data4"   
raw4 = mne.io.read_raw_edf(os.getcwd() + '/Accelerometer annotations/Version01_2_latest/Vers01_2_AS_Pos180deg/C3080088-240103_Vers01_2_AS_Pos180deg.edf',
                           include=['ACCX','ACCY', 'ACCZ'])
## crop the recordings excluding the position changes
# still_4_1 = raw4.copy().crop(tmin=0,tmax=90)
# walking_4 = raw4.copy().crop(tmin=90,tmax=215)
# still_4_2 = raw4.copy().crop(tmin=215,tmax=285)
# running_4 = raw4.copy().crop(tmin=285,tmax=400)
# still_4_3 = raw4.copy().crop(tmin=415,tmax=677)
# lying_back_4 = raw4.copy().crop(tmin=684,tmax=747)
# still_4_4 = raw4.copy().crop(tmin=756,tmax=815)
# lying_left_4 = raw4.copy().crop(tmin=825,tmax=872)
# still_4_5 = raw4.copy().crop(tmin=882,tmax=947)
# lying_right_4 = raw4.copy().crop(tmin=955,tmax=1011)
# still_4_6 = raw4.copy().crop(tmin=1018,tmax=1076)
# lying_front_4 = raw4.copy().crop(tmin=1084,tmax=1139)
# still_4_7 = raw4.copy().crop(tmin=1146,tmax=None)
still_4_1 = raw4.copy().crop(tmin=0,tmax=90)
walking_4 = raw4.copy().crop(tmin=90,tmax=215)
still_4_2 = raw4.copy().crop(tmin=215,tmax=285)
running_4 = raw4.copy().crop(tmin=285,tmax=400)
still_4_3 = raw4.copy().crop(tmin=400,tmax=680)
lying_back_4 = raw4.copy().crop(tmin=680,tmax=750)
still_4_4 = raw4.copy().crop(tmin=750,tmax=820)
lying_left_4 = raw4.copy().crop(tmin=820,tmax=880)
still_4_5 = raw4.copy().crop(tmin=880,tmax=950)
lying_right_4 = raw4.copy().crop(tmin=950,tmax=1015)
still_4_6 = raw4.copy().crop(tmin=1015,tmax=1080)
lying_front_4 = raw4.copy().crop(tmin=1080,tmax=1140)
still_4_7 = raw4.copy().crop(tmin=1140,tmax=None)
data4 = {'X': [], 'act':[], 'pos':[]}
still_epochs_4_1 = mne.make_fixed_length_epochs(still_4_1,duration=5)
for epoch in still_epochs_4_1:
    data4['X'].append(epoch)    
    data4['act'].append(0)    
    data4['pos'].append(0)  
    #data4.append({'X':epoch, 'act':0, 'pos':0})    
walking_epochs_4 = mne.make_fixed_length_epochs(walking_4,duration=5)
for epoch in walking_epochs_4:
    data4['X'].append(epoch)    
    data4['act'].append(1)    
    data4['pos'].append(0)  
    # data4.append({'X':epoch, 'act':1, 'pos':0})    
still_epochs_4_2 = mne.make_fixed_length_epochs(still_4_2,duration=5)
for epoch in still_epochs_4_2:
    data4['X'].append(epoch)    
    data4['act'].append(0)    
    data4['pos'].append(0)  
    # data4.append({'X':epoch, 'act':0, 'pos':0})    
running_epochs_4 = mne.make_fixed_length_epochs(running_4,duration=5)
for epoch in running_epochs_4:
    data4['X'].append(epoch)    
    data4['act'].append(2)    
    data4['pos'].append(0)  
    #data4.append({'X':epoch, 'act':2, 'pos':0})    
still_epochs_4_3 = mne.make_fixed_length_epochs(still_4_3,duration=5)
for epoch in still_epochs_4_3:
    data4['X'].append(epoch)    
    data4['act'].append(0)    
    data4['pos'].append(0)      
    # data4.append({'X':epoch, 'act':0, 'pos':0})    
lying_back_epochs_4 = mne.make_fixed_length_epochs(lying_back_4,duration=5)
for epoch in lying_back_epochs_4:
    data4['X'].append(epoch)    
    data4['act'].append(0)    
    data4['pos'].append(1)  
    # data4.append({'X':epoch, 'act':0, 'pos':1})    
still_epochs_4_4 = mne.make_fixed_length_epochs(still_4_4,duration=5)
for epoch in still_epochs_4_4:
    data4['X'].append(epoch)    
    data4['act'].append(0)    
    data4['pos'].append(0)  
    # data4.append({'X':epoch, 'act':0, 'pos':0})    
lying_left_epochs_4 = mne.make_fixed_length_epochs(lying_left_4,duration=5)
for epoch in lying_left_epochs_4:
    data4['X'].append(epoch)    
    data4['act'].append(0)    
    data4['pos'].append(2)  
    # data4.append({'X':epoch, 'act':0, 'pos':2})   
still_epochs_4_5 = mne.make_fixed_length_epochs(still_4_5,duration=5)
for epoch in still_epochs_4_5:
    data4['X'].append(epoch)    
    data4['act'].append(0)    
    data4['pos'].append(0)  
    # data4.append({'X':epoch, 'act':0, 'pos':0})    
lying_right_epochs_4 = mne.make_fixed_length_epochs(lying_right_4,duration=5)
for epoch in lying_right_epochs_4:
    data4['X'].append(epoch)    
    data4['act'].append(0)    
    data4['pos'].append(3)  
    # data4.append({'X':epoch, 'act':0, 'pos':3})   
still_epochs_4_6 = mne.make_fixed_length_epochs(still_4_6,duration=5)
for epoch in still_epochs_4_6:
    data4['X'].append(epoch)    
    data4['act'].append(0)    
    data4['pos'].append(0)  
    # data4.append({'X':epoch, 'act':0, 'pos':0})    
lying_front_epochs_4 = mne.make_fixed_length_epochs(lying_front_4,duration=5)
for epoch in lying_front_epochs_4:
    data4['X'].append(epoch)    
    data4['act'].append(0)    
    data4['pos'].append(4)  
    # data4.append({'X':epoch, 'act':0, 'pos':4})   
still_epochs_4_7 = mne.make_fixed_length_epochs(still_4_7,duration=5)
for epoch in still_epochs_4_7:
    data4['X'].append(epoch)    
    data4['act'].append(0)    
    data4['pos'].append(0)  
    # data4.append({'X':epoch, 'act':0, 'pos':0})  
#calibrate the orientation of the device
data4['X'] = np.array(data4['X'])
if data4['X'][0][1].mean() > 500:
    data4['X'][:,0,:] = -data4['X'][:,0,:]
    data4['X'][:,1,:] = -data4['X'][:,1,:]


activity_classes = ["Low", "Moderate", "High"]
position_classes = ["upright", "lying-back", "lying-left", "lying-right","lying-stomach"]
model = MotionClassifier(moderate_threshold=50, high_threshold=200, position_boundary=600)


#Activity classifier, SVM
#Trained on first recording
# #Tested on remaining three
# ewm_span = 6
# smoothing_param = 0.4
# pred_smoothing_params = np.arange(0.1,1,0.1)
# std_smoothing_params = range(0,400,10)
# results = []
X_train = data1['X'] #add a null variable for model parameter investigation
X_test = np.concatenate((data2['X'], data3['X'], data4['X']))


#TRAINING

#classify epochs
df_train = model.classify_motion(X_train)
#store in dataframe
df_train['gt_activity'] = [label for label in data1['act']]
df_train['gt_position'] = [label for label in data1['pos']]


#plot the predictions
fig, axs = plt.subplots(2,1)
df_train.plot(y=['gt_activity','activity'], ax=axs[0])
plt.yticks([0,1,2], labels=activity_classes)
df_train.plot(y=['gt_position', 'position'], ax=axs[1])
axs[1].set_yticks([0,1,2,3,4], labels=position_classes)
plt.show()

#remove NaN predictions arrising as a result of smoothing, for performance metrics calculations
df_train = df_train.dropna() 

#First assess the activity 
#calculate sensitivity
res = []
print(metrics.classification_report(df_train['gt_activity'],df_train['activity'], target_names=activity_classes))
for l in [0,1,2]:
    prec,recall,_,_ = metrics.precision_recall_fscore_support(np.array(df_train['gt_activity'])==l,
                                                            np.array(df_train['activity'])==l,
                                                            average=None,
                                                            pos_label=True)
    res.append([activity_classes[l], recall[1]])
results = pd.DataFrame(res, columns = ['class','sensitivity'])
#calculate specificity
spec = []
for l in [0,1,2]:
    prec,recall,_,_ = metrics.precision_recall_fscore_support(np.array(df_train['gt_activity'])!=l,
                                                            np.array(df_train['activity'])!=l,
                                                            average=None,
                                                            pos_label=True)
    spec.append(recall[1])
print(spec)
results['specificity'] = spec
print(results)
#visualise confusion matrix
confusion_matrix = metrics.confusion_matrix(df_train['gt_activity'], df_train['activity']) 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = activity_classes) 
cm_display.plot(cmap=plt.cm.Blues)
plt.show() 

#now assess the position predictions
#calculate sensitivity
print(metrics.classification_report(df_train['gt_position'],df_train['position'], target_names=position_classes))
res = []
for l in [0,1,2,3,4]:
    prec,recall,_,_ = metrics.precision_recall_fscore_support(np.array(df_train['gt_position'])==l,
                                                      np.array(df_train['position'])==l,
                                                      pos_label=True,average=None)
    res.append([position_classes[l],recall[1]])
results = pd.DataFrame(res,columns = ['class','sensitivity'])
#calculate specificity
spec = []
for l in [0,1,2,3,4]:
    prec,recall,_,_ = metrics.precision_recall_fscore_support(np.array(df_train['gt_position'])!=l,
                                                            np.array(df_train['position'])!=l,
                                                            average=None,
                                                            pos_label=True)
    spec.append(recall[1])
print(spec)
results['specificity'] = spec
print(results)
#visualise confusion matrix
confusion_matrix = metrics.confusion_matrix(df_train['gt_position'], df_train['position']) 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = position_classes) 
cm_display.plot(cmap=plt.cm.Blues)
plt.show() 







#TESTING
#make predictions
df_test = model.classify_motion(X_test)

df_test['gt_activity'] = [label for label in np.concatenate((data2['act'], data3['act'], data4['act']))]
df_test['gt_position'] = [label for label in np.concatenate((data2['pos'], data3['pos'], data4['pos']))]

#plot the predictions
fig, axs = plt.subplots(2,1)
df_test.plot(y=['gt_activity','activity'], ax=axs[0])
axs[0].set_yticks([0,1,2], labels=activity_classes)
df_test.plot(y=['gt_position','position'], ax=axs[1])
axs[1].set_yticks([0,1,2,3,4], labels=position_classes)
plt.show()

#remove NaN predictions arrising as a result of smoothing, for calculation of performance metrics
df_test = df_test.dropna() 

#assess the activity 
#calculate sensitivity
res = []
print(metrics.classification_report(df_test['gt_activity'],df_test['activity'], target_names=activity_classes))
for l in [0,1,2]:
    prec,recall,_,_ = metrics.precision_recall_fscore_support(np.array(df_test['gt_activity'])==l,
                                                            np.array(df_test['activity'])==l,
                                                            average=None,
                                                            pos_label=True)
    res.append([activity_classes[l], recall[1]])
results = pd.DataFrame(res, columns = ['class','sensitivity'])
#calculate specificity
spec = []
for l in [0,1,2]:
    prec,recall,_,_ = metrics.precision_recall_fscore_support(np.array(df_test['gt_activity'])!=l,
                                                            np.array(df_test['activity'])!=l,
                                                            average=None,
                                                            pos_label=True)
    spec.append(recall[1])
print(spec)
results['specificity'] = spec
print(results)
#visualise confusion matrix
confusion_matrix = metrics.confusion_matrix(df_test['gt_activity'], df_test['activity']) 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = activity_classes) 
cm_display.plot(cmap=plt.cm.Blues)
plt.show() 

#assess position predictions
#now assess the position predictions
#calculate sensitivity
print(metrics.classification_report(df_test['gt_position'],df_test['position'], target_names=position_classes))
res = []
for l in [0,1,2,3,4]:
    prec,recall,_,_ = metrics.precision_recall_fscore_support(np.array(df_test['gt_position'])==l,
                                                      np.array(df_test['position'])==l,
                                                      pos_label=True,average=None)
    res.append([position_classes[l],recall[1]])
results = pd.DataFrame(res,columns = ['class','sensitivity'])
#calculate specificity
spec = []
for l in [0,1,2,3,4]:
    prec,recall,_,_ = metrics.precision_recall_fscore_support(np.array(df_test['gt_position'])!=l,
                                                            np.array(df_test['position'])!=l,
                                                            average=None,
                                                            pos_label=True)
    spec.append(recall[1])
print(spec)
results['specificity'] = spec
print(results)
#visualise confusion matrix
confusion_matrix = metrics.confusion_matrix(df_test['gt_position'], df_test['position']) 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = position_classes) 
cm_display.plot(cmap=plt.cm.Blues)
plt.show() 

"""
Classify the motion of a patient using accelerometer data from C3 device.

input: path to edf recording
output: csv file with predictions and png visualisation of predictions. 

Predictions are made using a simple thresholding model described in the .readme
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import metrics 
from scipy import stats 
import pandas as pd
import argparse
import time
from thresholding.model import MotionClassifier
from data_class import Data


#parse argument: path to edf file
parser = argparse.ArgumentParser(
                    prog='classify motion',
                    description='Classifies motion',
                    epilog='Text at the bottom of help')

parser.add_argument('filename', help="the path to the edf file")           # positional argument
args = parser.parse_args()

start_time = time.time()

model = MotionClassifier(moderate_threshold=50, high_threshold=200, position_boundary=600)
data = Data(args.filename, window_duration=5)                                            

#make predictions
df = model.classify_motion(data.data)

#save to csv
df.to_csv("motion_classifier.csv")

activity_predictions = df['activity'].values
position_predictions = df['position'].values

#determine start and stop times of activity periods for visualisation
moderate_start_times = []
moderate_stop_times = []
high_start_times = []
high_stop_times = []
current_activity = 0
for i in range(len(activity_predictions)):
    if current_activity == 0 and activity_predictions[i] == 2: #resting -> running
        high_start_times.append(i)
        current_activity = 2
    elif current_activity == 0 and activity_predictions[i] == 1: #resting -> walking
        moderate_start_times.append(i)
        current_activity = 1
    elif current_activity == 2 and activity_predictions[i] == 1: #running -> walking
        high_stop_times.append(i)
        moderate_start_times.append(i)
        current_activity = 1
    elif current_activity == 2 and activity_predictions[i] == 0: #running -> resting
        high_stop_times.append(i)
        current_activity = 0
    elif current_activity == 1 and activity_predictions[i] == 2: #walking -> running
        moderate_stop_times.append(i)
        high_start_times.append(i)
        current_activity = 2
    elif current_activity == 1 and activity_predictions[i] == 0: #walking -> resting
        moderate_stop_times.append(i)
        current_activity = 0


position_classes = ["upright", "lying-back", "lying-left", "lying-right","lying-stomach"]

plt.plot(position_predictions, label="patient position")
plt.xlabel("Index of 5s window")
plt.ylabel("Position")
plt.title("Motion classification")
plt.yticks([0,1,2,3,4], labels=position_classes)
#plt.xticks(ticks, labels=time_labels)
i = 0
for start,stop in zip(moderate_start_times, moderate_stop_times):
    plt.axvspan(start, stop, color='y', alpha=0.2, label="_"*i +"moderate activity")
    i = 1
i = 0
for start,stop in zip(high_start_times, high_stop_times):
    plt.axvspan(start, stop, color='g', alpha=0.2, label="_"*i +"high activity")
    i = 1
plt.legend()
plt.savefig('motion_classifier.png')
#plt.show()


print("-------- %s seconds ----" % (time.time() - start_time))
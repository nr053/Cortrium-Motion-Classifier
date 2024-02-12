"""
Evaluate the threshold model behavour on 4 24 hour recordings stored in the Acc_model_test_dataset folder
"""


import mne
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import metrics 
from scipy import stats 
from scipy.fft import rfft, rfftfreq
import pandas as pd
import datetime
from data_class import Data
from thresholding.model import MotionClassifier

model = MotionClassifier(moderate_threshold=50, high_threshold=200, position_boundary=600)

file_names = ["/Acc_model_test_dataset/testfile_a/04dd0d9d-9e61-499d-8abd-23e8ff27ae90.edf",
              "/Acc_model_test_dataset/testfile_b/414c7115-e9a7-4558-bbcb-613e4f0d8e97.edf",
              "/Acc_model_test_dataset/testfile_c/5b05b3c9-9aff-4bec-8511-7f41749bd8a9.edf",
              "/Acc_model_test_dataset/testfile_d/2c708650-19e4-40a9-b0da-0a0ecef27d17.edf"]

start_times = [datetime.datetime(2024,1,11,8,43,24),
            datetime.datetime(2024,1,10,9,35,1),
            datetime.datetime(2024,1,10,10,56,0),
            datetime.datetime(2024,1,10,8,53,34)]

for file, start_time in zip(file_names, start_times):
    data = Data(os.getcwd() + file, window_duration=5).data

    #classify epochs
    df = model.classify_motion(data)
    activity_predictions = df['activity']
    position_predictions = df['position']

    moderate_start_times = []
    moderate_stop_times = []
    high_start_times = []
    high_stop_times = []
    current_activity = 0
    for i in range(len(activity_predictions)):
        if current_activity == 0 and activity_predictions[i] == 2: #running start
            high_start_times.append(i)
            current_activity = 2
        elif current_activity == 0 and activity_predictions[i] == 1: #walking start
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



    ticks = [0,2000,4000,6000,8000,10000,12000,14000,16000]
    time_labels = []
    for time in ticks:
        time_labels.append((start_time + datetime.timedelta(minutes=time/12)).time())
    plt.plot(position_predictions, label="patient position")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Model classification of 5 second windows")
    plt.yticks([0,1,2,3,4], labels=["upright", "lying-back", "lying-left", "lying-right","lying-stomach"])
    plt.xticks(ticks, labels=time_labels)
    i = 0
    for start,stop in zip(moderate_start_times, moderate_stop_times):
        plt.axvspan(start, stop, color='y', alpha=0.2, label="_"*i +"moderate activity")
        i = 1
    i = 0
    for start,stop in zip(high_start_times, high_stop_times):
        plt.axvspan(start, stop, color='g', alpha=0.2, label="_"*i +"high activity")
        i = 1
    plt.legend()
    plt.show()



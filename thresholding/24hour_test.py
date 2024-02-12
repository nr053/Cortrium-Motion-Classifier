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

file_names = ["/Acc_model_test_dataset/testfile_a/04dd0d9d-9e61-499d-8abd-23e8ff27ae90.edf",
              "/Acc_model_test_dataset/testfile_b/414c7115-e9a7-4558-bbcb-613e4f0d8e97.edf",
              "/Acc_model_test_dataset/testfile_c/5b05b3c9-9aff-4bec-8511-7f41749bd8a9.edf",
              "/Acc_model_test_dataset/testfile_d/2c708650-19e4-40a9-b0da-0a0ecef27d17.edf"]

start_times = [datetime.datetime(2024,1,11,8,43,24),
            datetime.datetime(2024,1,10,9,35,1),
            datetime.datetime(2024,1,10,10,56,0),
            datetime.datetime(2024,1,10,8,53,34)]

for file, start_time in zip(file_names, start_times):
    raw = mne.io.read_raw_edf(os.getcwd() + file, include = ["ACCX","ACCY","ACCZ"])
    epochs = mne.make_fixed_length_epochs(raw, duration=5)
    data = epochs.get_data()

    #calibrate the orientation of the device
    if np.mean(data[0:12,1],axis=1).mean() > 500:
        data[:,0,:] = -data[:,0,:]
        data[:,1,:] = -data[:,1,:]

    #3 classes: resting, walking, running
    class_names = ["Resting", "Walking", "Running"]
    y_pred = []
    #classify epochs
    for epoch in data: 
        x = np.std(epoch, axis=1).mean()
        if x > 200:
            y_pred.append(2) 
        elif x > 50:
            y_pred.append(1)
        else:
            y_pred.append(0)
    #smoothing out the position changes using majority voting
    df = pd.DataFrame({'y_pred': y_pred})
    df['majority'] = df['y_pred'].rolling(window=6, center=True).apply(lambda x: x.mode()[0])
    activity_predictions = df['majority']

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



    """Position classifier"""


    #7 classes: still, lying-back, lying-left, lying-right, lying-stomach, walking, running
    class_names = ["upright", "lying-back", "lying-left", "lying-right","lying-stomach"]
    X = []
    y_pred = []
    #classify epochs
    for epoch in data:
        x = [np.mean(epoch, axis=1), np.std(epoch, axis=1).mean()]
        X.append(x)
        if x[0][1] > -600 and x[0][0] > 600:
            y_pred.append(2)
        elif x[0][1] > -600 and x[0][0] < -600:
            y_pred.append(3)
        elif x[0][1] > -600 and x[0][2] > 600:
            y_pred.append(1)
        elif x[0][1] > -600 and x[0][2] < -600:
            y_pred.append(4)
        else:
            y_pred.append(0)

    df = pd.DataFrame({'y_pred': y_pred})
    position_predictions = df['y_pred']

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



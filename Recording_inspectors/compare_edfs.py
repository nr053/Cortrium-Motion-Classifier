"""Plot the separate ACC channels from all the edf files in the Accelerometer annotations file in one figure.
Good for comparing activity throughout each recording. """


import mne 
import matplotlib.pyplot as plt
import os
import glob

fig, axs = plt.subplots(6,3)
j = 0
colours = ["blue", "green", "orange"]
for file in glob.glob(os.getcwd() + '/Accelerometer annotations/**/*.edf', recursive=True):
    print(file)
    raw = mne.io.read_raw_edf(file, include = ["ACCX","ACCY","ACCZ"],preload=True)
    data = raw.get_data()
    for i in range(3):
        axs[j][i].plot(data[i], color=colours[i])
        axs[j][i].set_title(file.split("/")[-1])
    j+=1

plt.show()
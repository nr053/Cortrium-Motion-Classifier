"""Open interactive window to scrub through the ACC channels of edf files in Accelerometer annotations"""

import mne
import matplotlib.pyplot as plt
import os
import glob

for file in glob.glob(os.getcwd() + '/Accelerometer annotations/**/*.edf', recursive=True):
    print(file)
    raw = mne.io.read_raw_edf(file, include = ["ACCX","ACCY","ACCZ"],preload=True)
    raw.plot()

import mne
import numpy as np


class Data():
    """Data class."""
    def __init__(self,filename,window_duration):
        """Split recording into windows of chosen duration
        and reverse X and Y axes if device is worn in 180degree
        rotation position."""
        self.raw = mne.io.read_raw_edf(filename, include = ["ACCX","ACCY","ACCZ"]) #read the edf recording
        data = mne.make_fixed_length_epochs(self.raw, duration=5).get_data()  #split recording into 5 second windows

        #calibrate the orientation of the device using the first minute of the recording
        if np.mean(data[0:12,1],axis=1).mean() > 500:
            data[:,0,:] = -data[:,0,:]
            data[:,1,:] = -data[:,1,:]

        self.data = data
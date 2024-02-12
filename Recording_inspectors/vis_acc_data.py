import matplotlib.pyplot as plt
import mne 
import numpy as np
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

def sfreqTOmins(x):
    return x/(21.3*60)

def inverse(x):
    return x*60*21.3

raw = mne.io.read_raw_edf("parent/quick_recording.edf", include = ["ACCX","ACCY","ACCZ"],preload=True)

#raw.plot()
#raw_filtered = raw.filter(h_freq = 0.1,l_freq=None)
#raw_filtered.plot()
#epochs = mne.make_fixed_length_epochs(raw, duration=5)
#data = epochs.get_data()
data = raw.get_data()
data_filtered = mne.filter.filter_data(data,sfreq=21.3,h_freq=0.01,l_freq=None)

sfreq = raw.info['sfreq']
n_samples = data.shape[1]

recording_length_in_seconds = n_samples / sfreq
ticks = np.arange(0,n_samples, step=sfreq*10)
labels = [round(i/sfreq) for i in ticks]

fig, axs = plt.subplots(1,3)
# plt.plot(data[0])
# plt.plot(data[1])
# plt.plot(data[2])
# plt.show()

axs[0].plot(data_filtered[0])
axs[1].plot(data_filtered[1],color="green")
axs[2].plot(data_filtered[2],color="orange")


axs[0].set_title("x-axis, side-side - 0° orientation")
axs[1].set_title("y-axis, up-down - 0° orientation")
axs[2].set_title("z-axis, forward-backward - 0° orientation")

plt.xticks(ticks, labels)



#axs.set_xscale('function',functions=[sfreqTOmins,inverse])
plt.legend()
plt.show()


running_average_x = []
running_average_y = []
running_average_z = []

i=0
while i < (data.shape[1]-100):
    running_average_x.append(data[0][i:i+100].mean())
    running_average_y.append(data[1][i:i+100].mean())
    running_average_z.append(data[2][i:i+100].mean())
    i += 100


u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 1000 * np.outer(np.cos(u), np.sin(v))
y = 1000 * np.outer(np.sin(u), np.sin(v))
z = 1000 * np.outer(np.ones(np.size(u)), np.cos(v))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(running_average_x, running_average_y, running_average_y)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.plot_surface(x, y, z, alpha=0.1)
plt.show()

fig, axs = plt.subplots(1,3)
axs[0].set(xlim=(-1000,1000),
        ylim=(-1000,1000))
axs[0].scatter(running_average_x, running_average_y)
axs[0].add_artist(Ellipse((0,0),2000,2000,alpha=0.1))
axs[1].scatter(running_average_x, running_average_z)
axs[1].set(xlim=(-1000,1000),
        ylim=(-1000,1000))
axs[1].add_artist(Ellipse((0,0),2000,2000,alpha=0.1))
axs[2].scatter(running_average_y, running_average_z)
axs[2].set(xlim=(-1000,1000),
        ylim=(-1000,1000))
axs[2].add_artist(Ellipse((0,0),2000,2000,alpha=0.1))
plt.show()
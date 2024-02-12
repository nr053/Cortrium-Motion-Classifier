# Cortrium-Motion-Classifier

Motion classifier model that takes 3D accelerometer data. The model classifies activity level and position separately.

Activity level classes: ["Low", "Moderate", "High"]
Position classes: ["upright", "lying-back", "lying-left", "lying-right","lying-stomach"]

The model class is defined in thresholding/model.py
The data class takes a path to an edf file and outputs an array of calibrated data split into windows. 

The calibration procedure uses the first minute to detect the orientation of the device and assumes that for the first minute of the recording the patient is upright.

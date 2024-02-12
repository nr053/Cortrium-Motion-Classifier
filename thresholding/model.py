import numpy as np
import pandas as pd

class MotionClassifier():
    """Motion classifier model class"""
    def __init__(self, moderate_threshold, high_threshold, position_boundary):
        self.moderate_threshold = moderate_threshold
        self.high_threshold = high_threshold
        self.boundary = position_boundary

    def classify_motion(self, data):
        """Classify motion on calibrated data.
        
        input: array of 3 channel accelerometer data """
        activity_predictions = []
        position_predictions = []

        for epoch in data: 
            x = [np.mean(epoch, axis=1), np.std(epoch, axis=1).mean()]
            if x[1] > self.high_threshold:
                activity_predictions.append(2) 
            elif x[1] > self.moderate_threshold:
                activity_predictions.append(1)
            else:
                activity_predictions.append(0)

            if x[0][1] > -self.boundary and x[0][0] > self.boundary:
                position_predictions.append(2)
            elif x[0][1] > -self.boundary and x[0][0] < -self.boundary:
                position_predictions.append(3)
            elif x[0][1] > -self.boundary and x[0][2] > self.boundary:
                position_predictions.append(1)
            elif x[0][1] > -self.boundary and x[0][2] < -self.boundary:
                position_predictions.append(4)
            else:
                position_predictions.append(0)

        df = pd.DataFrame({'activity': activity_predictions, 'position': position_predictions})
        df['activity'] = df['activity'].rolling(window=6, center=True).apply(lambda x: x.mode()[0])


        return df
    
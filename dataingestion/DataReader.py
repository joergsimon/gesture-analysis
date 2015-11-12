import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

gestures = []

class User:
    """ in reader this class reads in the data of a user into an pandas frame, and adds the
     labels data"""
    gesture_field = "gesture"
    label_type_automatic = 'G'

    data = None
    labelData = None
    dataName = None
    dataPath = None
    labelPath = None
    windowData = None

    def __init__(self, root, dataCsvName, labelCsvName):
        self.dataName = dataCsvName
        self.dataPath = root + dataCsvName
        self.labelPath = root + labelCsvName

    def read(self):
        self.data = self.read_csv(self.dataPath)
        self.labelData = self.read_csv(self.labelPath)
        self.add_gesture_column()
        self.add_labels_to_data()

    def read_csv(self, path):
        return pd.read_csv(path, sep=',', header=None)

    def add_gesture_column(self):
        self.data[User.gesture_field] = pd.Series(np.zeros(self.data.size, dtype=np.int))

    def add_labels_to_data(self):
        for index, label in self.labelData.iterrows():
            if label[0] == User.label_type_automatic:
                gesture, start, end = label[1], int(label[3]), int(label[4])
                if gesture not in gestures:
                    gestures.append(gesture)
                    if (end-start > 300):
                        print "large difference found! {}".format(end-start)
                self.data.loc[start:end, 'gesture'] = gestures.index(gesture) + 1

    def write(self):
        return self.windowData.to_csv(path_or_buf='data/intermediate/'+self.dataName, sep=',', header=True)
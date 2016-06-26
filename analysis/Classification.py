import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from const.constants import Constants


class Classification:

    clf_model = None
    clf = None
    train_data = None
    train_labels = None
    test_data = None
    test_labels = None

    def __init__(self, datamatrix):
        headers = Constants.headers
        d = datamatrix.loc[:,headers]
        d = d.values
        mask = np.isnan(d).any(axis=1)
        d = d[~mask]
        d = np.real(d)
        d = preprocessing.scale(d)
        l = datamatrix.loc[:,'gesture']
        l = l.values
        l = l[~mask]
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            d, l, test_size=0.33, random_state=42
        )
        self.clf = svm.SVC(decision_function_shape='ovo')

    def train(self):
        self.clf.fit(self.train_data, self.train_labels)
        #self.clf_model = self.clf.decision_function([[1]])

    def report(self):
        prediction = self.clf.predict(self.test_data)
        # if we have the gestures in constants, add target_names=Constants.gesture_names
        print(classification_report(self.test_labels, prediction))


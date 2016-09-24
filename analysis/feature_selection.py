from analysis.preparation import labelMatrixToArray
from analysis.preparation import normalizeZeroClassArray
from visualise.trace_features import trace_feature_origin

import numpy as np
import sklearn
import sklearn.linear_model
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

def feature_selection(train_data, train_labels, const):
    train_labels_arr, exclude = labelMatrixToArray(train_labels, const.label_threshold)
    train_data_clean = train_data.drop(exclude)
    train_labels_arr, train_data_clean, _ = normalizeZeroClassArray(train_labels_arr, train_data_clean)

    print "num features before selection: {}".format(train_data_clean.columns.size)
    feature_index = variance_threshold(train_data_clean)
    trace_feature_origin(feature_index,const)
    feature_index = rfe(train_data_clean,train_labels_arr)
    trace_feature_origin(feature_index, const)

def variance_threshold(train_data):
    # feature selection using VarianceThreshold filter
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    fit = sel.fit(train_data.values)
    col_index = fit.get_support(indices=True)
    print "num features selected by VarianceThreshold: {}".format(len(col_index))
    return col_index

def rfe(train_data, train_labels):
    # important toto!
    # todo: I think also for feature selection we should take care the 0 class is balanced!
    # todo: if you use it that way, scale the features
    print "Recursive eleminate features: "
    svc = sklearn.linear_model.Lasso(alpha = 0.1) #svm.SVR(kernel="linear")
    print "test fit."
    svc.fit(train_data.values, np.array(train_labels))
    print "run rfecv.."
    rfecv = RFE(estimator=svc, step=0.1, verbose=2)
    rfecv.fit(train_data.values, np.array(train_labels))
    print "get support..."
    col_index = rfecv.get_support(indices=True)
    print "num features selected by RFE(CV)/Lasso: {}".format(len(col_index))
    return col_index
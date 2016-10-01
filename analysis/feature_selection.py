from analysis.preparation import labelMatrixToArray
from analysis.preparation import normalizeZeroClassArray
from visualise.trace_features import trace_feature_origin

import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.preprocessing as pp
import sklearn.svm as svm
import sklearn.feature_selection as fs

# Interesting References:
# RFECV:
# Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene selection for
# cancer classification using support vector machines. Mach. Learn.. 46(1-3). 389-422.

def feature_selection(train_data, train_labels, const):
    train_labels_arr, exclude = labelMatrixToArray(train_labels, const.label_threshold)
    train_data_clean = train_data.drop(exclude)
    train_labels_arr, train_data_clean, _ = normalizeZeroClassArray(train_labels_arr, train_data_clean)

    print "num features before selection: {}".format(train_data_clean.columns.size)
    feature_index = variance_threshold(train_data_clean)
    trace_feature_origin(feature_index,const)
    feature_index = rfe(train_data_clean,train_labels_arr)
    trace_feature_origin(feature_index, const)
    feature_index = k_best_chi2(train_data_clean, train_labels_arr, 700)
    trace_feature_origin(feature_index, const)
    feature_index = rfe_cv_f1(train_data_clean, train_labels_arr)
    trace_feature_origin(feature_index, const)

def variance_threshold(train_data):
    # feature selection using VarianceThreshold filter
    sel = fs.VarianceThreshold(threshold=(.8 * (1 - .8)))
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
    rfecv = fs.RFE(estimator=svc, step=0.1, verbose=2)
    rfecv.fit(train_data.values, np.array(train_labels))
    print "get support..."
    col_index = rfecv.get_support(indices=True)
    print "num features selected by RFE(CV)/Lasso: {}".format(len(col_index))
    return col_index

def rfe_cv_f1(train_data, train_labels):
    # important toto!
    # todo: I think also for feature selection we should take care the 0 class is balanced!
    # todo: if you use it that way, scale the features
    print "Recursive eleminate features: "
    svc = svm.SVC(kernel="linear") #sklearn.linear_model.Lasso(alpha = 0.1)
    print "scale data"
    values = train_data.values
    minmax = pp.MinMaxScaler()
    values = minmax.fit_transform(values)#pp.scale(values)
    print "test fit."
    svc.fit(values, np.array(train_labels).astype(int))
    print "run rfecv.."
    rfecv = fs.RFECV(estimator=svc, step=0.05, verbose=2)
    rfecv.fit(values, np.array(train_labels).astype(int))
    print "get support..."
    col_index = rfecv.get_support(indices=True)

    print "num features selected by RFECV/SVR: {}".format(len(col_index))
    return col_index

def k_best_chi2(train_data, train_labels, k):
    values = train_data.values
    if values.min() < 0:
        values = values + abs(values.min())
    kb = fs.SelectKBest(fs.chi2, k=k)
    kb.fit(values, np.array(train_labels))
    col_index = kb.get_support(indices=True)
    print "num features selected by K-Best using chi2: {}".format(len(col_index))
    return col_index
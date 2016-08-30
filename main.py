import dataingestion.DataReader as dr
import numpy as np
import analysis.Window as wd
import pandas as pd
import os
import os.path as path
from const.constants import Constants
import tools

from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import sklearn
import sklearn.linear_model
import sklearn.neighbors.nearest_centroid
import sklearn.naive_bayes
import sklearn.tree
import sklearn.ensemble
from sklearn.cross_validation import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV

def read_user(path, glove_data, label_data, overwrite_data):
    user = dr.User(path, glove_data, label_data)
    if user.has_intermediate_file() and not overwrite_data:
        user.read_intermediate()
    else:
        user.read()
        aggregate = wd.ArrgretageUser(user,200,30)
        aggregate.make_rolling_dataset_2()
        print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print user.data.ix[0:2,:]
        print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print user.windowData.ix[0:2,:]
        print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        user.write_intermediate()
    return user

def labelMatrixToArray(labelMatrix, threshold):
    labels = []
    exclude = []
    for row in labelMatrix.index:
        r = labelMatrix.loc[row,:]
        lblInfo = r[r > threshold]
        lbl = "0.0"
        # TODO: for training, it would be better
        # to remove the ones where 0 is more than 50 and label is less than 15
        if lblInfo.size > 0:
            lbl = lblInfo.index[0]
        else:
            exclude.append(row)
        labels.append(lbl)

    # now we need to balance the amount of the zero class to the other classes
    # get all 0 indexes:
    labelDF = pd.DataFrame(labels, index=labelMatrix.index)
    return (labelDF, exclude)

def normalizeZeroClass(labels, data):
    counts = labels.groupby(0).size()
    max = counts[1:].max()
    zeroIndex = labels[labels[0] == "0.0"].index
    selectedIndex = np.random.choice(zeroIndex, size=max, replace=False)
    removalIndex = zeroIndex.drop(selectedIndex)
    labelDF = labels.drop(removalIndex)
    trainData = data.drop(removalIndex)
    return (labelDF, trainData, removalIndex)

def main():
    root = 'data/raw/'
    windowData = None
    windowLabelInfo = None
    files = [f for f in os.listdir(root) if path.isfile(path.join(root, f))]
    labels = [l for l in files if "label" in l]
    labels = sorted(labels)
    gl_data = [g for g in files if "glove" in g]
    gl_data = sorted(gl_data)
    for glove_data, label_data in zip(gl_data,labels):
        user = read_user(root, glove_data, label_data, False)
        if windowData is None:
            windowData = user.windowData
            windowLabelInfo = user.windowLabel
        else:
            windowData = pd.concat([windowData, user.windowData])
            windowLabelInfo = pd.concat([windowLabelInfo, user.windowLabelInfo])

    print "permutate data"

    # TODO: here compute the labels the way we want it for analysis!
    # first simple approach: just the the major labe in each window:
    windowLabelInfo = windowLabelInfo.drop('Unnamed: 0', 1)
    windowData = windowData.drop(u'gesture', 1)

    # permutate the data
    indices = np.random.permutation(windowData.index)
    windowData = windowData.reindex(indices)
    windowLabelInfo = windowLabelInfo.reindex(indices)



    # prepare data for feature selection:
    selectLabelDF, exclude = labelMatrixToArray(windowLabelInfo, 150)
    # now we need to balance the amount of the zero class to the other classes
    # get all 0 indexes:
    selectLabelDF = selectLabelDF.drop(exclude)
    selectData = windowData.drop(exclude)
    selectLabelDF, selectData, _ = normalizeZeroClass(selectLabelDF, selectData)

    # feature selection using VarianceThreshold filter
    # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # fit = sel.fit(selectData.values)
    # colIndex = fit.get_support(indices=True)
    # windowData = windowData[windowData.columns[colIndex]]

    # the blow is somehow valid, however:
    # first I would need to transform the features so each X > 0
    # (means vor each colum add the col max negative offset to 0 to each value)
    # but I am more in doupth I should do that as these are univariate
    # selections, and I am not sure if we are more in the multivariate
    # world here.
    # - feature selection getting the X best features based on
    # - statistical tests for the data. We have 65 sensors,
    # - or about 12 different single movements in our case
    # - since in our gesture only complete finger flexation
    # - or relaxation is interesting so the minimum
    # - number of features should be in the range of
    # - 12-65. A good set might be the double amount of that
    #fit = SelectKBest(chi2, k=65).fit(selectData.values, selectLabelDF.values)
    #colIndex = fit.get_support(indices=True)
    #windowData = windowData[windowData.columns[colIndex]]

    # doto: if you use it that way, scale the features
    svc = svm.SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=400, scoring='accuracy')
    rfecv.fit(selectData.values, selectLabelDF.values)
    colIndex = rfecv.get_support(indices=True)
    windowData = windowData[windowData.columns[colIndex]]

    # first we split trining and test already here. this
    # is because of the different learning approach
    #
    # windowData['gesture'] = windowLabelInfo.idxmax(axis=1)
    splitpoint = int(windowData.index.size * 0.7)
    trainData = windowData[0:splitpoint]
    testData = windowData[splitpoint + 1:]
    trainLabels = windowLabelInfo[0:splitpoint]
    testLabels = windowLabelInfo[splitpoint + 1:]
    # a complete window has 201 frames. we count the label with
    # more than 150, aka. 3/4 as the real label

    labelDF, exclude = labelMatrixToArray(trainLabels, 150)
    # now we need to balance the amount of the zero class to the other classes
    # get all 0 indexes:
    labelDF = labelDF.drop(exclude)
    trainData = trainData.drop(exclude)
    labelDF, trainData, _ = normalizeZeroClass(labelDF, trainData)

    print("++++++++++++++++")
    print(labelDF)
    print("++++++++++++++++")
    print("train data size:")
    print(trainData.shape)
    print("++++++++++++++++")
    headers = Constants.headers
    #d = trainData.loc[:, headers]
    d = trainData.values #d.values
    d = preprocessing.scale(d)

    print(d)

    clf = None
    kf = KFold(len(labelDF.values), n_folds=5)
    score = 0
    for train_index, test_index in kf:
        X_train = d[train_index, :]
        X_ct = d[test_index, :]
        y_train = labelDF.values[train_index]
        y_ct = labelDF.values[test_index]
        # lin_clf = sklearn.linear_model.LogisticRegression()
        # lin_clf = sklearn.linear_model.LogisticRegression(class_weight='auto')
        # lin_clf = svm.LinearSVC()
        # lin_clf = svm.LinearSVC(class_weight='auto')
        # lin_clf = svm.SVR()
        # lin_clf = svm.SVC()
        # lin_clf = svm.SVC(class_weight='auto')
        lin_clf = svm.SVC(decision_function_shape='ovo')
        # lin_clf = sklearn.neighbors.nearest_centroid.NearestCentroid()
        # lin_clf = sklearn.linear_model.Lasso(alpha = 0.1)
        # lin_clf = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2")
        # lin_clf = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2", class_weight='auto')
        # lin_clf = sklearn.naive_bayes.MultinomialNB()
        # lin_clf = sklearn.tree.DecisionTreeClassifier()
        # lin_clf = sklearn.tree.DecisionTreeClassifier(class_weight='auto')
        # lin_clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
        # lin_clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, class_weight='auto')
        # lin_clf = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
        # lin_clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        lin_clf.fit(X_train, y_train)
        s = lin_clf.score(X_ct, y_ct)
        if s > score:
            score = s
            clf = lin_clf

    #clf = svm.SVC(decision_function_shape='ovo')
    #clf.fit(d, labelDF.values)

    # TODO: test label approach:
    # compute our binary matrix with labels per frame
    # also compute our label vector as above
    # then correct the label vector by looking
    # at multilabel entries if they match with the prediction
    # and set the label to that

    testLabelDF, exclude = labelMatrixToArray(testLabels, 10)

    # testLabelDF, testData, removalIndex = normalizeZeroClass(testLabelDF, testData)
    # testLabels.drop(removalIndex)

    testLabels = testLabels.fillna(0)
    testLabels[testLabels > 0] = 1

    #d = testData.loc[:, headers]
    d = testData.values #d.values
    d = preprocessing.scale(d)

    prediction = clf.predict(d)

    for row in range(prediction.size):
        p = prediction[row]
        val = testLabels.loc[testLabels.index[row]][p]
        if val == 1.0:
            testLabelDF.loc[testLabelDF.index[row]] = p

    print("------------")
    print(prediction)
    print("------------")
    print(testLabelDF)
    print("------------")

    print(classification_report(testLabelDF.values, prediction))

    #windowData = glove_data.reset_index(drop=True)
    #clf = Classification(windowData)
    #clf.train()
    #clf.report()

if __name__ == '__main__':
    main()

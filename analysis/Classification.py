import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.linear_model as lm
import sklearn.neighbors.nearest_centroid as nc
import sklearn.naive_bayes as nb
import sklearn.ensemble as em
from sklearn import svm
from sklearn import tree
from sklearn.metrics import f1_score

needsScaling = True
needsNoScaling = False

classifiers = [("svc[ovo]",svm.SVC(decision_function_shape='ovo'), needsScaling),
               ("svc",svm.SVC(), needsScaling),
               ("lin svc",svm.LinearSVC(), needsScaling),
               ("lr",lm.LogisticRegression(), needsScaling),
               ("nn",nc.NearestCentroid(), needsScaling),
               ("lr(l1)",lm.LogisticRegression(penalty='l1'), needsScaling),
               ("sgd[hinge]",lm.SGDClassifier(loss="hinge", penalty="l2"), needsScaling),
               ("navie bayse",nb.MultinomialNB(), needsScaling),
               ("decision tree",tree.DecisionTreeClassifier(), needsNoScaling),
               ("random forrest",em.RandomForestClassifier(n_estimators=10), needsNoScaling),
               ("ada boost",em.AdaBoostClassifier(n_estimators=100), needsScaling),
               ("gradient boost",em.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), needsScaling)]
# what I still would like to have: kNN, maybe a neural network, deep learning would be awesome...

def fit_classifier(values, labels):
    clf = None
    clf_n = None
    needs_scaling = True
    kf = ms.KFold(n_splits=5)
    score = 0
    minmax = pp.MinMaxScaler()
    scaled_values = minmax.fit_transform(values)  # pp.scale(values)
    for clf_name, clf_candidate, needsScaledData in classifiers:
        print("KFold fit {}".format(clf_name))
        needs_scaling = needsScaledData
        # decide: we should return the best model for each classifier here?
        # and do we pickle them?
        for train_index, test_index in kf.split(values):
            X_train = scaled_values[train_index, :] if needsScaledData else values[train_index, :]
            X_ct = scaled_values[test_index, :] if needsScaledData else values[test_index, :]
            y_train = labels[train_index]
            y_ct = labels[test_index]
            clf_candidate.fit(X_train, y_train)
            p = clf_candidate.predict(X_ct)
            s = f1_score(y_ct, p, average='micro')
            if s > score:
                score = s
                clf = clf_candidate
                clf_n = clf_name

    return (clf, clf_n, needs_scaling)

# lin_clf = sklearn.linear_model.LogisticRegression()
# lin_clf = sklearn.linear_model.LogisticRegression(class_weight='auto')
# lin_clf = svm.LinearSVC()
# lin_clf = svm.LinearSVC(class_weight='auto')
# lin_clf = svm.SVC()
# lin_clf = svm.SVC(class_weight='auto')
# lin_clf = svm.SVC(decision_function_shape='ovo')
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
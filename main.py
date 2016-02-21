import dataingestion.DataReader as dr
import numpy as np
import analysis.Window as wd
from analysis.classification import Classification
import numpy as np
import pandas as pd
import os
import os.path as p

def read_user(glove_data, label_data, overwrite_data):
    user = dr.User(root, glove_data, label_data)
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

if __name__ == '__main__':
    root = 'data/raw/'
    windowData = None
    files = [f for f in os.listdir(root) if p.isfile(p.join(root, f))]
    labels = [l for l in files if "label" in l]
    labels = sorted(labels)
    gl_data = [g for g in files if "glove" in g]
    gl_data = sorted(gl_data)
    for glove_data, label_data in zip(gl_data,labels):
        user = read_user(glove_data, label_data, True)
        if windowData is None:
            windowData = user.windowData
        else:
            windowData = pd.concat([windowData, user.windowData])

    print "permuatet data"
    windowData.reindex(np.random.permutation(glove_data.index))
    windowData = glove_data.reset_index(drop=True)
    clf = Classification(windowData)
    clf.train()
    clf.report()

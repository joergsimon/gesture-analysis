import numpy as np

def permutate(data,labels):
    # permutate the data
    indices = np.random.permutation(data.index)
    data = data.reindex(indices)
    labels = labels.reindex(indices)
    return (data,labels)

def split_test_train(data,labels,percent_train):
    splitpoint = int(data.index.size * percent_train)
    trainData = data[0:splitpoint]
    testData = data[splitpoint + 1:]
    trainLabels = labels[0:splitpoint]
    testLabels = labels[splitpoint + 1:]
    return (trainData,trainLabels,testData,testLabels)

def labelMatrixToArray(labelMatrix, threshold):
    labels = []
    exclude = []
    for row in labelMatrix.index:
        r = labelMatrix.loc[row,:]
        lblInfo = r[r > threshold]
        lbl = 0
        # TODO: for training, it would be better
        # to remove the ones where 0 is more than 50 and label is less than 15
        if lblInfo.size > 0:
            lbl = lblInfo.index[0]
            labels.append(lbl)
        else:
            exclude.append(row)
    return (labels, exclude)

def normalizeZeroClass(labels, data):
    counts = labels.groupby(0).size()
    max = counts[1:].max()
    zeroIndex = labels[labels[0] == 0.0].index
    selectedIndex = np.random.choice(zeroIndex, size=max, replace=False)
    removalIndex = zeroIndex.drop(selectedIndex)
    labelDF = labels.drop(removalIndex)
    trainData = data.drop(removalIndex)
    return (labelDF, trainData, removalIndex)

def normalizeZeroClassArray(labels_arr, data):
    lbls = np.array(labels_arr)
    zeroIndex = np.where(lbls == 0.0)[0]#lbls[lbls == 0.0].index
    equalizer = zeroIndex.size-(len(labels_arr)-zeroIndex.size)
    removalIndex = np.random.choice(zeroIndex, size=equalizer, replace=False)
    for index in sorted(removalIndex, reverse=True):
        del labels_arr[index]
    data = data.drop(data.index[removalIndex])
    return (labels_arr, data, removalIndex)
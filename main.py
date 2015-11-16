import dataingestion.DataReader as dr
import analysis.Window as wd

if __name__ == '__main__':
    root = 'data/raw/'
    user = dr.User(root, 'PC29_glove_2015_08_05_10_48_23.csv', 'PC29_label_2015_08_05_10_48_23.csv')
    user.read()
    aggregate = wd.ArrgretageUser(user,200,30)
    aggregate.make_rolling_dataset_2()
    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print user.data.ix[0:2,:]
    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print user.windowData.ix[0:2,:]
    user.write()
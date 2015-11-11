import dataingestion.DataReader as dr
import pandas as pd

if __name__ == '__main__':
    root = '/Users/j_simon/Documents/SmartGlove/src/Processing/'
    user = dr.User(root, 'PC29_glove_2015_08_05_10_48_23.csv', 'PC29_label_2015_08_05_10_48_23.csv')
    user.read()
    print user.data.ix[0:2,:]
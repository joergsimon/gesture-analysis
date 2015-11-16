from pylab import *


def stem_user(user, channel_list, start_index, stop_index):
    data = user.windowData
    for channel in channel_list:
        stem_pandas_column(data, channel, start_index, stop_index)


def stem_pandas_column(dataframe, columname, start_index, stop_index):
    column = dataframe[columname]
    stem_channel(column.values, start_index, stop_index)


def stem_channel(channel, start_index, stop_index):
    values = channel[start_index:stop_index]
    markerline, stemlines, baseline = stem(range(start_index,stop_index), values, '-.')
    setp(markerline, 'markerfacecolor', 'b')
    setp(baseline, 'color','r', 'linewidth', 2)
    show()

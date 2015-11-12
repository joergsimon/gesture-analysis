import pandas as pd
import timeit
import numpy as np
import sys
import scipy as sc
import scipy.stats

class ArrgretageUser:

    window_size = 200
    step_size = 30
    user = None

    def __init__(self, user, window_size, step_size):
        self.user = user
        self.window_size = window_size
        self.step_size = step_size

    def make_rolling_dataset_2(self):
        start_time = timeit.default_timer()
        data = self.user.data
        no_rolling_properties = ["gesture"]

        num_windows = (len(data)-self.window_size)/self.step_size
        headers = list(data.columns.values)
        newHeaders = ["{}_mean".format(j) for j in headers]
        newHeaders += ["{}_std".format(j) for j in headers]
        newHeaders += ["{}_min".format(j) for j in headers]
        newHeaders += ["{}_25q".format(j) for j in headers]
        newHeaders += ["{}_median".format(j) for j in headers]
        newHeaders += ["{}_75q".format(j) for j in headers]
        newHeaders += ["{}_max".format(j) for j in headers]
        newHeaders += ["{}_min_max".format(j) for j in headers]
        newHeaders += ["{}_var".format(j) for j in headers]
        newHeaders += ["{}_skew".format(j) for j in headers]
        newHeaders += ["{}_kurtosis".format(j) for j in headers]
        newHeaders += ["{}_mode".format(j) for j in headers]
        newHeaders += ["{}_spectral_centeroid".format(j) for j in headers]
        newHeaders += ["{}_spectral_entropy".format(j) for j in headers]

        print 'creating empy frame with len {}'.format(num_windows)
        print 'and headers: {}'.format(newHeaders)

        t1 = timeit.default_timer()
        print data.info()
        data = data.astype('float64')
        t2 = timeit.default_timer()
        print "converting to float took {}s".format(t2-t1)

        matrix = None

        for i in range(num_windows):
            if i % 20 == 0:
                progress = (i/float(num_windows))
                msg = '\r[{0}] {1:.2f}%'.format('#'*int(progress*10), progress*100)
                sys.stdout.write(msg)
            offset = i*self.step_size
            subframe = data.loc[offset:offset+self.window_size]
            subframe = subframe._get_numeric_data()
            mat = stat_describe(subframe.values[:,0])
            for j in range(1,len(subframe.columns)):
                vec = stat_describe(subframe.values[:,j])
                mat = np.column_stack((mat, vec))
            mat = np.array(np.ravel(mat))
            if matrix is None:
                matrix = mat
            else:
                matrix = np.vstack((matrix, mat))

        newFrame = pd.DataFrame(matrix,index=range(num_windows), columns=newHeaders)

        print 'Cleaning ...'

        #newFrame.gesture = newFrame.gesture.astype(int)

        self.user.windowData = newFrame

        l = pd.isnull(newFrame).any(1).nonzero()[0]

        print 'Done aggregating'

        time = timeit.default_timer() - start_time
        print 'exec took {}s'.format(time)


def stat_describe(array):
   resMin = np.nanmin(array)
   resMax = np.nanmax(array)
   resRange = resMax - resMin
   resMedian = np.median(array)
   resMean = np.nanmean(array)
   resStd = np.nanstd(array)
   resVar = np.nanvar(array)
   res25Q = np.percentile(array, 25)
   res75Q = np.percentile(array, 75)
   resSkew = sc.stats.skew(array)
   resKurtosis = sc.stats.kurtosis(array)
   resMode = sc.stats.mode(array, axis=None)[0]
   length = len(array)
   y = np.fft.rfft(array)
   magnitudes = np.abs(y)
   freqs = np.fft.rfftfreq(length, d=(1./58))
   freqs = freqs[np.where(freqs >= 0)]
   freqs = np.abs(freqs)
   spectral_centroid = np.sum(magnitudes*freqs)/np.sum(magnitudes)
   psd = pow(magnitudes, 2)/freqs
   psdsum = sum(psd)
   psdnorm = psd/psdsum
   spectral_entropy = sc.stats.entropy(psdnorm)

   return np.array([resMean, resStd, resMin, res25Q, resMedian,
                    res75Q, resMax, resRange, resVar, resSkew,
                    resKurtosis, resMode, spectral_centroid,
                    spectral_entropy])
import pandas as pd
import timeit
import numpy as np
import sys
import scipy as sc
import scipy.stats
from const.constants import Constants

class ArrgretageUser:

    window_size = 200
    step_size = 30
    user = None

    def __init__(self, user, window_size, step_size):
        self.user = user
        self.window_size = window_size
        self.step_size = step_size

    def createHeaders(self, data):
        headers = list(data.columns.values)
        headers.remove('gesture')
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
        newHeaders += ["{}_ff1".format(j) for j in headers]
        newHeaders += ["{}_ff2".format(j) for j in headers]
        newHeaders += ["{}_ff3".format(j) for j in headers]
        newHeaders += ["{}_ff4".format(j) for j in headers]
        newHeaders += ["{}_ff5".format(j) for j in headers]
        newHeaders += ["{}_freq_5sum".format(j) for j in headers]
        newHeaders += ["{}_bandwidth".format(j) for j in headers]
        return newHeaders

    def addHeaderFirstTime(self, newHeaders, header):
        if header not in newHeaders:
            newHeaders.append(header)

    def computeTupelFeatures(self, tupelData, newHeaders, headerPrefix):
        col_range = range(len(tupelData.columns))
        tupels = [ (a,b) for a in col_range for b in col_range if a != b ]
        #unique_tupels = [ (a,b) for a,b in tupels if (b,a) not in tupels]
        res = np.array([])
        for a,b in tupels:
            if tupelData.columns[a] != 'gesture' and tupelData.columns[b] != 'gesture':
                # compute correlations, vectors, threshholds....
                # signal correlation:
                # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.correlate.html
                #
                # angle:
                v1 = tupelData.values[:,a]
                v2 = tupelData.values[:,b]
                vec1 = v1/np.linalg.norm(v1)
                vec2 = v2/np.linalg.norm(v2)
                angle = np.arccos(np.dot(vec1,vec2))
                self.addHeaderFirstTime(newHeaders, "{}_{}_{}_angle".format(headerPrefix,a,b))

                corr, pval = scipy.stats.spearmanr(v1, v2)
                self.addHeaderFirstTime(newHeaders, "{}_{}_{}_corr".format(headerPrefix,a,b))
                self.addHeaderFirstTime(newHeaders, "{}_{}_{}_pval".format(headerPrefix,a,b))

                # inspired from http://svn.gna.org/svn/relax/tags/4.0.0/lib/geometry/vectors.py
                fV1 = np.fft.rfft(v1)
                fV2 = np.fft.rfft(v2)

                i_v1v2 = np.dot(fV1,fV2.conj().T)
                i_v1v1 = np.dot(fV1, fV1.conj().T)
                i_v2v2 = np.dot(fV2, fV2.conj().T)
                ratio = i_v1v2.real / (np.sqrt(i_v1v1).real * np.sqrt(i_v2v2).real)
                fftAngle = np.arccos(ratio)

                #fftVec1 = fV1/np.linalg.norm(fV1)
                #fftVec2 = fV2/np.linalg.norm(fV2)
                #print sum(fftVec1)
                #print sum(fftVec2)
                #fftDot = np.dot(fV1,fV2.conj().T)
                #fftSine = fftDot/(np.linalg.norm(fV1)*np.linalg.norm(fV2))
                #fftAngle = np.arccos(np.dot(fftVec1,fftVec2.conj().T))
                self.addHeaderFirstTime(newHeaders, "{}_{}_{}_fft_angle".format(headerPrefix,a,b))

                fftCorr, fftPval = scipy.stats.spearmanr(fV1, fV2)
                self.addHeaderFirstTime(newHeaders, "{}_{}_{}_fft_corr".format(headerPrefix,a,b))
                self.addHeaderFirstTime(newHeaders, "{}_{}_{}_fft_pval".format(headerPrefix,a,b))
                res = np.append(res, np.array([angle, corr, pval, fftAngle, fftCorr, fftPval]))
        return res


    def make_rolling_dataset_2(self):
        start_time = timeit.default_timer()
        data = self.user.data

        num_windows = (len(data)-self.window_size)/self.step_size
        newHeaders = self.createHeaders(data)

        print 'creating empy frame with len {}'.format(num_windows)
        print 'and headers: {}'.format(newHeaders)

        t1 = timeit.default_timer()
        print data.info()
        data = data.astype('float64')
        t2 = timeit.default_timer()
        print "converting to float took {}s".format(t2-t1)

        matrix = None
        labelInfo = []

        # for peak detection: the question is, if we first globally detect
        # peaks, and then add the sum to the window, or locally detect
        # peaks, and then add that sum, or do both, as they are a little
        # bit different probably...?
        # peak detection methods in python:
        # http://pythonhosted.org/PeakUtils/tutorial_a.html
        # or
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html
        # (used in: http://bioinformatics.oxfordjournals.org/content/22/17/2059.long )
        # or
        # https://gist.github.com/endolith/250860
        # or reimplement that below:
        # http://stackoverflow.com/questions/18023643/peak-detection-in-accelerometer-data
        for i in range(num_windows):
            if i % 20 == 0:
                progress = (i/float(num_windows))
                msg = '\r[{0}] {1:.2f}%'.format('#'*int(progress*10), progress*100)
                sys.stdout.write(msg)
            offset = i*self.step_size

            # COMPUTE Features for Single Signals:
            subframe = data.loc[offset:offset+self.window_size]
            subframe = subframe._get_numeric_data()
            mat = stat_describe(subframe.values[:,0])
            sub_range = range(1,len(subframe.columns))
            for j in sub_range:
                if subframe.columns[j] != 'gesture':
                    vec = stat_describe(subframe.values[:,j])
                    mat = np.column_stack((mat, vec))
                    # do not forget to add peaks!
            mat = np.array(np.ravel(mat))
            # ok, maybe getting alls tuples is dump, maybe it is better to list all meaningful combinations
            # like all finger rows, or all finger IMUs and so on...

            flex_data = data.ix[offset:offset+self.window_size,Constants.flex_map]
            row1 = flex_data.ix[:,Constants.hand_row_1]
            row2 = flex_data.ix[:,Constants.hand_row_2]
            res = self.computeTupelFeatures(row1,newHeaders,"flex_row1")
            mat = np.append(mat,res)
            res = self.computeTupelFeatures(row2,newHeaders,"flex_row2")
            mat = np.append(mat,res)

            counts = subframe.groupby("gesture").size()
            labelInfo.append(counts)

            if matrix is None:
                matrix = mat
            else:
                matrix = np.vstack((matrix, mat))


        newFrame = pd.DataFrame(matrix,index=range(num_windows), columns=newHeaders)
        newLabels = pd.DataFrame(labelInfo)

        print 'Cleaning ...'

        #newFrame.gesture = newFrame.gesture.astype(int)
        self.user.windowLabel = newLabels
        self.user.windowData = newFrame

        l = pd.isnull(newFrame).any(1).nonzero()[0]

        print 'Done aggregating'

        time = timeit.default_timer() - start_time
        print 'exec took {}s'.format(time)


#
# Spectral Entropy Algorithm:
# http://stackoverflow.com/questions/21190482/spectral-entropy-and-spectral-energy-of-a-vector-in-matlab
# alternatives for computing the PSD:
# scipy.signal.welch
# http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.welch.html#scipy.signal.welch
# matplotlib.mlab.psd
# http://matplotlib.org/api/mlab_api.html#matplotlib.mlab.psd
#
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
   magnitudes = np.delete(magnitudes, 0)
   freqs = np.fft.rfftfreq(length, d=(1./58))
   freqs = freqs[np.where(freqs >= 0)]
   freqs = np.delete(freqs, 0)
   freqs = np.abs(freqs)
   spectral_centroid = np.sum(magnitudes*freqs)/np.sum(magnitudes)
   psd = pow(magnitudes, 2)/freqs
   psdsum = sum(psd)
   psdnorm = psd/psdsum
   spectral_entropy = sc.stats.entropy(psdnorm)
   freq_5sum = freqs[0] + freqs[1] + freqs[2] + freqs[3] + freqs[4];
   bandwith = max(freqs)-min(freqs)

   # return np.array([resMean, resStd, resMin, res25Q, resMedian,
   #                  res75Q, resMax, resRange, resVar, resSkew,
   #                  resKurtosis, resMode, spectral_centroid,
   #                  spectral_entropy, freqs[0], freqs[1], freqs[2],
   #                  freqs[3], freqs[4], freq_5sum, bandwith])
   return np.array([resMean, resStd, resMin, res25Q, resMedian,
                    res75Q, resMax, resRange, resVar, resSkew,
                    resKurtosis, resMode, spectral_centroid,
                    spectral_entropy, freqs[0], freqs[1], freqs[2],
                    freqs[3], freqs[4], freq_5sum, bandwith])
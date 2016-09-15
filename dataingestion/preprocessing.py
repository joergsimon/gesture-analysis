import numpy as np
import os.path as path
import pandas as pd
from utils.freshrotation import euler_matrix
from utils.freshrotation import vector_slerp
from dataingestion.cache_control import has_preprocess_basic_cache

def preprocess_basic(data,const):
    if has_preprocess_basic_cache(const):
        data == pd.read_pickle(const.init_data_cache_file)
        return data
    else:
        convert_values(data,const)
        convolution_filter(data,const)
        compute_orientation_indipendent_accel(data,const)
        data.to_pickle(const.init_data_cache_file)
        return data


def convert_values(data, const):
    gyro_offset = np.loadtxt("dataingestion/gyro_offset.txt")
    accel_headers = const.filter_raw_header('accel')
    for header in accel_headers:
        data.loc[:,header] /= const.LSB_PER_G # in g
    gyro_header = const.filter_raw_header('gyro')
    for header in gyro_header:
        header_index = gyro_header.index(header)
        imu_index = int(header_index/3)
        index_in_imu = int(header_index%3)
        drift = gyro_offset[imu_index][index_in_imu]
        data.loc[:, header] -= drift
        data.loc[:, header] /= const.LSB_PER_DEG_PER_SEC

def convolution_filter(data, const):
    n = len(data.index)
    LAs = np.zeros((n, const.number_imus * 3))

    Grav = np.zeros((const.number_imus, 3))
    Grav.T[2] = 1

    for k, line in enumerate(data.values):

        for i in range(const.number_imus):
            dx, dy, dz = line[const.get_triple_idxs('gyro',i)] * const.dt * np.pi / 180
            rot = euler_matrix(dx, dy, dz)
            Grav[i] = rot.T.dot(Grav[i])

            norm = np.linalg.norm(line[const.get_triple_idxs('accel',i)])
            if norm > 0.8 and norm < 1.2:
                scale = 0.02 if i > 100 else 0.7
                Grav[i] = vector_slerp(Grav[i], line[const.get_triple_idxs('accel',i)] / norm, scale)

        LAs[k] = line[np.array(const.raw_indices['accel'])] - Grav.reshape(1,21)

    accel_headers = const.filter_raw_header('accel')
    for header in accel_headers:
        index = accel_headers.index(header)
        h = "{}_lin_accel".format(header)
        data[h] = LAs[:,index]
        if const.raw_indices['lin_accel'] == None:
            const.raw_indices['lin_accel'] = []
        const.raw_indices['lin_accel'].append(index)
        # TODO: add to finger 1,2,3,4,thumb etc.

def compute_orientation_indipendent_accel(data,const):
    for i in range(const.number_imus):
        pass



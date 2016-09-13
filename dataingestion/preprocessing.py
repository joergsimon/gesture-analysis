import numpy as np

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
from const.constants_old import ConstantsOld
import json
import numpy as np

class Constants:

    def __init__(self):
        self.load_raw_file_description()

    def load_raw_file_description(self):
        with open('const/raw_data_description.json') as file:
            self.raw_file_descripton = json.load(file)
            self.raw_headers = self.raw_file_descripton[u'headers_array']
            self.raw_indices = self.raw_file_descripton[u'indices']

    def filter_raw_header(self, lvl1_key, lvl2_key = None):
        idx = self.raw_indices[lvl1_key]
        if lvl2_key != None:
            idx = idx[lvl2_key]
        h = np.array(self.raw_headers)
        return h[idx].tolist()

    def imu_accel_headers(self, imu_idx):
        return self.get_triples('accel', imu_idx)

    def imu_gyro_headers(self, imu_idx):
        return self.get_triples('gyro', imu_idx)

    def get_triples(self, header, imu_idx):
        index_in_index = self.get_triple_idxs(header, imu_idx).tolist()
        h = np.array(self.raw_headers)
        return h[index_in_index].tolist()

    def get_triple_idxs(self, header, imu_idx):
        all_idx = np.array(self.raw_indices[header])
        index_in_index = all_idx[imu_idx * 3:imu_idx * 3 + 3]
        return index_in_index


    gesture_field = "gesture"
    label_type_automatic = "G"

    init_data_dir = 'data/raw/'
    init_data_meta = 'data/intermediate/step1/meta.pkl'
    init_data_cache_file = 'data/intermediate/step1/pandas_blob.pkl'
    preprocessed_data_cache_file = 'data/intermediate/step1/pandas_blob_preprocessed.pkl'

    LSB_PER_G = 16384
    LSB_PER_DEG_PER_SEC = 65.5

    number_imus = 7
    dt = 0.012

    # legacy:
    headers = ConstantsOld.headers
    gesture_names = ConstantsOld.gesture_names
    window_size = ConstantsOld.window_size
    window_distance = ConstantsOld.window_distance
    flex_map = ConstantsOld.flex_map
    hand_row_1 = ConstantsOld.hand_row_1
    hand_row_2 = ConstantsOld.hand_row_2

# The values in order (16bit values):
# 0: Thumb base
# 1: Thumb pressure
# 2: Angle between thumb and hand
# 3: Finger 1 base
# 4: Finger 1 tip
# 5: Finger 2 base
# 6: Finger 2 tip
# 7: Finger 3 base
# 8: Finger 3 tip
# 9: Finger 4 base
# 10: Finger 4 tip
# 11: Thumb tip
# 12: Finger 1 pressure
# 13: Finger 2 pressure
# 14: Finger 3 pressure
# 15: Finger 4 pressure
# 16: Wrist extension
# 17: Wrist flexion
# 18-20: Finger 1 Accelerometer X, Y, Z
# 21-23: Finger 1 Gyroscope X, Y, Z
# 24-26: Finger 2 Accelerometer X, Y, Z
# 27-29: Finger 2 Gyroscope X, Y, Z
# 30-32: Finger 3 Accelerometer X, Y, Z
# 33-35: Finger 3 Gyroscope X, Y, Z
# 36-38: Finger 4 Accelerometer X, Y, Z
# 39-41: Finger 4 Gyroscope X, Y, Z
# 42-44: Thumb Accelerometer X, Y, Z
# 45-47: Thumb Gyroscope X, Y, Z
# 48-50: Palm Accelerometer X, Y, Z
# 51-53: Palm Gyroscope X, Y, Z
# 54-56: Wrist Accelerometer X, Y, Z
# 57-59: Wrist Gyroscope X, Y, Z
# 60-62: Magnetometer X, Y, Z (observe different axis than the IMU:s)
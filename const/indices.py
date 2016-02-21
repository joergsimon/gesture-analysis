from enum import Enum

thumb_base = 0
thumb_tip = 11
thumb_angle = 2
finger1_base = 3
finger1_tip = 4
finger2_base = 5
finger2_tip = 6
finger3_base = 7
finger3_tip = 8
finger4_base = 9
finger4_tip = 10
wrist_flexion = 17
wrist_extension = 16

thumb_pressure = 1
finger1_pressure = 12
finger2_pressure = 13
finger3_pressure = 14
finger4_pressure = 15


#Access imu N like this from the line, for example the thumb:
# Accelerometer line[ACC0_OFFSET + I16_PER_IMU*imu_thumb : ACC0_OFFSET + I16_PER_IMU*imu_thumb + 3]

imu_finger1 = 0
imu_finger2 = 1
imu_finger3 = 2
imu_finger4 = 3
imu_thumb = 4
imu_palm = 5
imu_wrist = 6






map = [thumb_base,
    thumb_tip,
    thumb_angle,
    finger1_base,
    finger1_tip,
    finger2_base,
    finger2_tip,
    finger3_base,
    finger3_tip,
    finger4_base,
    finger4_tip,
    wrist_flexion,
    wrist_extension,
    thumb_pressure,
    finger1_pressure,
    finger2_pressure,
    finger3_pressure,
    finger4_pressure]

four = [finger1_base, finger1_tip, finger2_base,
    finger2_tip, finger3_base, finger3_tip, finger4_base, finger4_tip]

non_pressure = [thumb_base, thumb_tip, thumb_angle, finger1_base, finger1_tip, finger2_base,
    finger2_tip, finger3_base, finger3_tip, finger4_base, finger4_tip,
    wrist_flexion, wrist_extension]

digits = [thumb_base,
          thumb_tip,
          thumb_angle,
          finger1_base,
          finger1_tip,
          finger2_base,
          finger2_tip,
          finger3_base,
          finger3_tip,
          finger4_base,
          finger4_tip]

post_n = len(non_pressure)

post_thumb_base = 0
post_thumb_tip = 1
post_thumb_angle = 2
post_finger1_base = 3
post_finger1_tip = 4
post_finger2_base = 5
post_finger2_tip = 6
post_finger3_base = 7
post_finger3_tip = 8
post_finger4_base = 9
post_finger4_tip = 10
post_wrist_flexion = 11
post_wrist_extension = 12


post_digits = list(range(post_thumb_base, post_finger4_tip+1))
post_four = list(range(post_finger1_base, post_finger4_tip+1))




# The values in order:
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
# 16: Wrist flex
# 17: Wrist flex (other direction)
# 18-xx: IMU Finger 1
# TODO

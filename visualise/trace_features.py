
def trace_feature_origin(feature_indexes,const):
    dict = dict_feature_sortet(feature_indexes, const)
    report_percent(const.feature_indices,dict)

def report_percent(full_dict,selected_dict):
    for f_key in full_dict:
        f_val = full_dict[f_key]
        if type(f_val) is list:
            s_val = selected_dict[f_key]
            percent = len(s_val) / float(len(f_val))
            print "{0} has {1:.2f}%".format(f_key, percent)
        elif type(f_val) is dict:
            for if_key in f_val:
                if_val = f_val[if_key]
                s_val = selected_dict[f_key][if_key]
                percent = len(s_val) / float(len(if_val))
                print "{0}/{1} has {2:.2f}%".format(f_key,if_key,percent)


def dict_feature_sortet(feature_indexes, const):
    # first at to part of hand:
    dict = const.index_dict()
    for feature_idx in feature_indexes:
        if feature_idx in const.feature_indices['thumb']['all']:
            dict['thumb']['all'].append(feature_idx)
        elif feature_idx in const.feature_indices['finger_1']['all']:
            dict['finger_1']['all'].append(feature_idx)
        elif feature_idx in const.feature_indices['finger_2']['all']:
            dict['finger_2']['all'].append(feature_idx)
        elif feature_idx in const.feature_indices['finger_3']['all']:
            dict['finger_3']['all'].append(feature_idx)
        elif feature_idx in const.feature_indices['finger_4']['all']:
            dict['finger_4']['all'].append(feature_idx)
        elif feature_idx in const.feature_indices['wrist']['all']:
            dict['wrist']['all'].append(feature_idx)
            if feature_idx in const.feature_indices['wrist']['flex']:
                dict['wrist']['flex'].append(feature_idx)
            elif feature_idx in const.feature_indices['wrist']['imu']:
                dict['wrist']['imu'].append(feature_idx)
        elif feature_idx in const.feature_indices['palm']['all']:
            dict['palm']['all'].append(feature_idx)
        else:
            print "(trace) fatal: hand index not found {}".format(feature_idx)

        # then add back to sensor:
        if feature_idx in const.feature_indices['flex']['all']:
            dict['flex']['all'].append(feature_idx)
            if feature_idx in const.feature_indices['flex']['row_1']:
                dict['flex']['row_1'].append(feature_idx)
            elif feature_idx in const.feature_indices['flex']['row_2']:
                dict['flex']['row_2'].append(feature_idx)
        elif feature_idx in const.feature_indices['pressure']:
            dict['pressure'].append(feature_idx)
        elif feature_idx in const.feature_indices['accel']:
            dict['accel'].append(feature_idx)
        elif feature_idx in const.feature_indices['gyro']:
            dict['gyro'].append(feature_idx)
        elif feature_idx in const.feature_indices['magnetometer']:
            dict['magnetometer'].append(feature_idx)
        elif feature_idx in const.feature_indices['lin_accel']:
            dict['lin_accel'].append(feature_idx)
        else:
            print "(trace) fatal: sensor index not found {}".format(feature_idx)

    return dict
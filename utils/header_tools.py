def create_new_header(base_header, channel_num, suffix):
    raw_header = strip_start_num(base_header)
    header = "{}_{}_{}".format(channel_num, raw_header, suffix)
    return header

def strip_start_num(header):
    idx = header.find("_")
    return header[idx+1:]

def add_new_idx_to_hand(base_idx, new_idx, add_to_sensor, const):
    if base_idx in const.raw_indices['thumb']['all']:
        const.raw_indices['thumb']['all'].append(new_idx)
    elif base_idx in const.raw_indices['finger_1']['all']:
        const.raw_indices['finger_1']['all'].append(new_idx)
    elif base_idx in const.raw_indices['finger_2']['all']:
        const.raw_indices['finger_2']['all'].append(new_idx)
    elif base_idx in const.raw_indices['finger_3']['all']:
        const.raw_indices['finger_3']['all'].append(new_idx)
    elif base_idx in const.raw_indices['finger_4']['all']:
        const.raw_indices['finger_4']['all'].append(new_idx)
    elif base_idx in const.raw_indices['wrist']['all']:
        const.raw_indices['wrist']['all'].append(new_idx)
        if base_idx in const.raw_indices['wrist']['flex']:
            const.raw_indices['wrist']['flex'].append(new_idx)
        elif base_idx in const.raw_indices['wrist']['imu']:
            const.raw_indices['wrist']['imu'].append(new_idx)
    elif base_idx in const.raw_indices['palm']['all']:
        const.raw_indices['palm']['all'].append(new_idx)
    else:
        print "(basis) fatal: hand index not found {}".format(base_idx)

    # then add back to sensor:
    if add_to_sensor:
        if base_idx in const.raw_indices['flex']['all']:
            const.raw_indices['flex']['all'].append(new_idx)
            if base_idx in const.raw_indices['flex']['row_1']:
                const.raw_indices['flex']['row_1'].append(new_idx)
            elif base_idx in const.raw_indices['flex']['row_2']:
                const.raw_indices['flex']['row_2'].append(new_idx)
        elif base_idx in const.raw_indices['pressure']:
            const.raw_indices['pressure'].append(new_idx)
        elif base_idx in const.raw_indices['accel']:
            const.raw_indices['accel'].append(new_idx)
        elif base_idx in const.raw_indices['gyro']:
            const.raw_indices['gyro'].append(new_idx)
        elif base_idx in const.raw_indices['magnetometer']:
            const.raw_indices['magnetometer'].append(new_idx)
        elif base_idx in const.raw_indices['lin_accel']:
            const.raw_indices['lin_accel'].append(new_idx)
        else:
            print "(basis) fatal: sensor index not found {}".format(new_idx)


def stat_describe_feature_names():
    return ["mean","std","min","25q","median",
            "75q","max","range","var","skew",
            "kurtosis","mode","spectral_centroid",
            "spectral_entropy","ff1","ff2","ff3",
            "ff4","ff5","freq_5sum","bandwith"]

def create_headers(const):
    create_feature_id_struct(const)
    feature_headers = []
    feature_names = stat_describe_feature_names()
    for header in const.raw_headers:
        if header == 'gesture':
            continue
        sensor_idx = const.raw_headers.index(header)
        idx = header.find("_")
        num = header[:idx]
        offset = len(feature_headers)
        for f_name in feature_names:
            f_idx = feature_names.index(f_name)
            comb_num = "{}_{}".format(num, f_idx)
            h = create_new_header(header, comb_num, f_name)
            feature_headers.append(h)
            add_new_idx_of_feature_to_hand(sensor_idx, (offset + f_idx), const, header, h)
    const.feature_description['feature_headers_array'] = feature_headers
    const.feature_headers = feature_headers

def create_feature_id_struct(const):
    feature_indices = const.index_dict()
    const.feature_description["indices"] = feature_indices
    const.feature_indices = feature_indices

def add_new_idx_of_feature_to_hand(sensor_idx, feature_idx, const, debug_header, debug_feature):
    # first at to part of hand:
    if sensor_idx in const.raw_indices['thumb']['all']:
        const.feature_indices['thumb']['all'].append(feature_idx)
    elif sensor_idx in const.raw_indices['finger_1']['all']:
        const.feature_indices['finger_1']['all'].append(feature_idx)
    elif sensor_idx in const.raw_indices['finger_2']['all']:
        const.feature_indices['finger_2']['all'].append(feature_idx)
    elif sensor_idx in const.raw_indices['finger_3']['all']:
        const.feature_indices['finger_3']['all'].append(feature_idx)
    elif sensor_idx in const.raw_indices['finger_4']['all']:
        const.feature_indices['finger_4']['all'].append(feature_idx)
    elif sensor_idx in const.raw_indices['wrist']['all']:
        const.feature_indices['wrist']['all'].append(feature_idx)
        if sensor_idx in const.raw_indices['wrist']['flex']:
            const.feature_indices['wrist']['flex'].append(feature_idx)
        elif sensor_idx in const.raw_indices['wrist']['imu']:
            const.feature_indices['wrist']['imu'].append(feature_idx)
    elif sensor_idx in const.raw_indices['palm']['all']:
        const.feature_indices['palm']['all'].append(feature_idx)
    else:
        print "(feature) fatal: hand index not found {} (new: {})".format(sensor_idx, feature_idx)
        print "(feature) fatal: header: {} (feature: {})".format(debug_header, debug_feature)

    # then add back to sensor:
    if sensor_idx in const.raw_indices['flex']['all']:
        const.feature_indices['flex']['all'].append(feature_idx)
        if sensor_idx in const.raw_indices['flex']['row_1']:
            const.feature_indices['flex']['row_1'].append(feature_idx)
        elif sensor_idx in const.raw_indices['flex']['row_2']:
            const.feature_indices['flex']['row_2'].append(feature_idx)
    elif sensor_idx in const.raw_indices['pressure']:
        const.feature_indices['pressure'].append(feature_idx)
    elif sensor_idx in const.raw_indices['accel']:
        const.feature_indices['accel'].append(feature_idx)
    elif sensor_idx in const.raw_indices['gyro']:
        const.feature_indices['gyro'].append(feature_idx)
    elif sensor_idx in const.raw_indices['magnetometer']:
        const.feature_indices['magnetometer'].append(feature_idx)
    elif sensor_idx in const.raw_indices['lin_accel']:
        const.feature_indices['lin_accel'].append(feature_idx)
    else:
        print "(feature) fatal: sensor index not found {} (new: {})".format(sensor_idx, feature_idx)
        print "(feature) fatal: header: {} (feature: {})".format(debug_header, debug_feature)
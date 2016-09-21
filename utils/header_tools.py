

def create_new_header(base_header, channel_num, suffix):
    raw_header = strip_start_num(base_header)
    header = "{}_{}_{}".format(channel_num, raw_header, suffix)
    return header

def strip_start_num(header):
    idx = header.find("_")
    return header[idx+1:]
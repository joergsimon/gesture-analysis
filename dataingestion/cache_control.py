import os.path as path

def has_preprocess_basic_cache(const):
    ok = path.isfile(const.preprocessed_data_cache_file)
    ok = ok and path.isfile(const.preprocessed_data_meta)
    return ok

def has_window_cache(const):
    ok = path.isfile(const.window_data_meta)
    ok = ok and path.isfile(const.window_data_cache_file)
    ok = ok and path.isfile(const.window_label_cache_file)
    return ok
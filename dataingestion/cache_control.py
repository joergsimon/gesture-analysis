import os.path as path

def has_preprocess_basic_cache(const):
    ok = path.isfile(const.preprocessed_data_cache_file)
    ok = ok and path.isfile(const.preprocessed_data_meta)
    return ok
import os.path as path

def has_preprocess_basic_cache(const):
    path.isfile(const.preprocessed_data_cache_file)
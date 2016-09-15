import os
import os.path as path

def clear_all_cache(const):
    chaches = ["step1","step2"]
    clear_cache(chaches,const)

def clear_cache(cache_steps_to_clear, const):
    for step in cache_steps_to_clear:
        if step == "step1":
            if path.isfile(const.init_data_cache_file):
                os.remove(const.init_data_cache_file)
                os.remove(const.init_data_meta)
        elif step == "step2":
            if path.isfile(const.preprocessed_data_cache_file):
                os.remove(const.preprocessed_data_cache_file)
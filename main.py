from dataingestion.initial_input import InitialInput
from const.constants import Constants
from dataingestion.preprocessing import preprocess_basic
from dataingestion.cache_control import *

const = Constants()
init_input = InitialInput(const)
data = None
if not has_preprocess_basic_cache(const):
    data = init_input.read_all_data_init()
data = preprocess_basic(data,const)
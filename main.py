from dataingestion.initial_input import InitialInput
from const.constants import Constants
from dataingestion.preprocessing import preprocess_basic
from dataingestion.window import get_windows
from dataingestion.cache_control import *
from analysis.preparation import permutate
from analysis.preparation import split_test_train
from analysis.feature_selection import feature_selection
from utils.header_tools import create_headers

def main():
    const = Constants()
    init_input = InitialInput(const)
    data = None
    if not has_preprocess_basic_cache(const):
        data = init_input.read_all_data_init()
    const.remove_stripped_headers()
    data = preprocess_basic(data, const)
    data, labels = get_windows(data, const)
    create_headers(const)

    print("flex const index trace info / main:")
    print(len(const.feature_indices['flex']['row_1']))
    print(len(const.feature_indices['flex']['row_2']))
    r1 = []
    for i in const.feature_indices['flex']['row_1']:
        r1.append(const.feature_headers[i])
    print(r1)

    # permutate the data
    data, labels = permutate(data, labels)
    # split train and testset
    train_data,train_labels,test_data,test_labels = split_test_train(data,labels,0.7)
    feature_selection(train_data,train_labels,const)



if __name__ == '__main__':
    main()
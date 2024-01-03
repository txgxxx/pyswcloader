import pickle
import os

def load_pkl(path):
    return pickle.load(open(path, 'rb'))

CURRENT_WD = os.path.dirname(os.path.abspath(__file__))
# print(CURRENT_WD)

# ALLEN_BRAIN_TREE = load_pkl('./database/allen_brain_tree.pkl')
# STL_ACRO_DICT = load_pkl('./database/stl_acro_dict.pkl')
# ACRO_STL_DICT = load_pkl('./database/acro_stl_dict.pkl')

ALLEN_BRAIN_TREE = load_pkl(os.path.join(CURRENT_WD, 'database', 'allen_brain_tree.pkl'))
STL_ACRO_DICT = load_pkl(os.path.join(CURRENT_WD, 'database', 'stl_acro_dict.pkl'))
ACRO_STL_DICT = load_pkl(os.path.join(CURRENT_WD, 'database', 'acro_stl_dict.pkl'))
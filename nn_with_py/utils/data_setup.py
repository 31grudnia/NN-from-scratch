import numpy as np
import json 
import os 
import urllib
import urllib.request
from zipfile import ZipFile
 

# @TODO Make it work
def load_dictionary(path):
    with open(path, 'rb', encoding="utf8") as JSON:
        sign_dict = json.load(JSON)
    sign_dict = {int(number): sign for number, sign in sign_dict.items()}
    return sign_dict

def load_files(load_path):
    signs = np.load(load_path + "signs.npy")
    labels = np.load(load_path + "labels_int.npy")
    # signs_dictionary = load_dictionary(load_path + "dictionary.json")
    return signs, labels

    
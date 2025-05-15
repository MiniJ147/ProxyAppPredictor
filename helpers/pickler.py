import os
import pickle as p

PICKLE_DIR = "./pickles"

def pickle(data,name):
    if not os.path.exists(PICKLE_DIR):
        os.makedirs(PICKLE_DIR)
    p.dump(data,open(f"{PICKLE_DIR}/{name}",'wb'))

def depickle(name):
    return p.load(open(f"{PICKLE_DIR}/{name}",'rb'))
    

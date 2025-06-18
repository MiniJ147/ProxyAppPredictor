import os
import pickle as p

PICKLE_DIR = "./pickles"
global_should_depickle = False

def set_should_depickle(val):
    global global_should_depickle
    global_should_depickle = val

def should_depickle() -> bool:
    return global_should_depickle

def pickle(data,name):
    if not os.path.exists(PICKLE_DIR):
        os.makedirs(PICKLE_DIR)
    p.dump(data,open(f"{PICKLE_DIR}/{name}",'wb'))

def depickle(name):
    return p.load(open(f"{PICKLE_DIR}/{name}",'rb'))
    

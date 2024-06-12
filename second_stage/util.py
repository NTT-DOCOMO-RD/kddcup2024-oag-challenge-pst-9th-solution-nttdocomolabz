from os.path import join
import json
import random
import os
import numpy as np
import torch

# from bs4 import BeautifulSoup
# import pandas as pd
from fuzzywuzzy import fuzz


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_json(rfdir, rfname):
    print("loading %s ...", rfname)
    with open(join(rfdir, rfname), "r", encoding="utf-8") as rf:
        data = json.load(rf)
        print("%s loaded", rfname)
        return data


def dump_json(obj, wfdir, wfname):
    print("dumping %s ...", wfname)
    with open(join(wfdir, wfname), "w", encoding="utf-8") as wf:
        json.dump(obj, wf, indent=4, ensure_ascii=False)
    print("%s dumped.", wfname)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

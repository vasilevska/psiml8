import json
from scipy import stats
import numpy as np
from datetime import datetime


def dump_config(config, filename, include_time = False):
    save_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    config_json = {}
    for key in dir(config):
        if not key.startswith("_"):
            config_json[key] = eval("config." + key)
    if include_time:
        filename = filename + "_" + save_time
    with open(filename + ".json", "w") as f:      
        json.dump(config_json, f ,indent=4)

def d_prime(auc):
    d_prime = stats.norm().ppf(auc) * np.sqrt(2.0)
    return d_prime
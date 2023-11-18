from reckit import Configurator
from importlib.util import find_spec
from importlib import import_module
from reckit import typeassert
import os
import sys
import numpy as np
import random
import torch

def _set_random_seed(seed=2023):
    
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("set pytorch seed")


@typeassert(recommender=str)
def find_recommender(recommender):
    model_dirs = set(os.listdir("model"))
    model_dirs.remove("base")

    module = None

    for tdir in model_dirs:
        spec_path = ".".join(["model", tdir, recommender])
        if find_spec(spec_path):
            module = import_module(spec_path)
            break

    if module is None:
        raise ImportError(f"Recommender: {recommender} not found")

    if hasattr(module, recommender):
        Recommender = getattr(module, recommender)
    else:
        raise ImportError(f"Import {recommender} failed from {module.__file__}!")
    return Recommender


if __name__ == "__main__":
    is_windows = sys.platform.startswith('win')
    if is_windows:
        root_dir = './'
        data_dir = './dataset/'
    else:
        root_dir = './'
        data_dir = './dataset/'
    config = Configurator(root_dir, data_dir)
    config.parse_cmd()

    model_cfg = os.path.join(root_dir + "conf" + "/IGCL.ini")
    config.add_config(model_cfg, section="hyperparameters", used_as_summary=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config["gpu_id"])

    _set_random_seed(config["seed"])
    Recommender = find_recommender(config.recommender)
    
    recommender = Recommender(config)
    recommender.train_model()

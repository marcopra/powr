import json
import os
import random
import socket
import string
from datetime import datetime

import jax
import numpy as np
import torch


def get_run_name(args, current_date=None):
    if current_date is None:
        current_date = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
    return (
        str(current_date)
        + "_"
        + str(args.env)
        + "_"
        + str(args.algo)
        + "_t"
        + str(args.timesteps)
        + "_HiddenL"
        + str(args.hidden_layers)
        + (f"_activation-{args.activation}" if args.activation is not None else "")
        + "_seed"
        + str(args.seed)
        + "_"
        + socket.gethostname()
    )


def get_random_string(n=5):
    return "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(n)
    )


def set_seed(seed):
    # Seeding every module
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    jax.random.key(seed)


def create_dir(path):
    try:
        os.mkdir(os.path.join(path))
    except OSError as error:
        # print('Dir already exists')
        pass


def create_dirs(path):
    try:
        os.makedirs(os.path.join(path))
    except OSError as error:
        pass


def save_config(config, path):
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file)
    return


def get_learning_rate(args, env):
    """
    Priority:
        args.lr > env.preferred_lr > 0.0003 (default)
    """
    if args.lr is None:
        if env.get_attr("preferred_lr")[0] is None:
            return 0.0003
        else:
            return env.get_attr("preferred_lr")[0]
    else:
        return args.lr

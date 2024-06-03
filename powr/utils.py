import json
import os
import random
import socket
import string
from datetime import datetime
from tabulate import tabulate

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


def log_epoch_statistics(writer, log_file, epoch, test_result, train_result, n_train_episodes,
                         n_iter_pmd, n_warmup_episodes, total_timesteps, 
                         t_sampling, t_pmd, t_test, execution_time
                         ):
    # Log to Tensorboard
    global_step = epoch
    writer.add_scalar("test reward", test_result, global_step)
    writer.add_scalar("train reward", train_result, global_step)
    writer.add_scalar(
        "Sampling and Updating steps",
        n_train_episodes + epoch * (n_train_episodes + n_iter_pmd),
        global_step,
    )
    writer.add_scalar("Epoch", epoch, global_step)
    writer.add_scalar(
        "Train Episodes", n_warmup_episodes + epoch * n_train_episodes, global_step
    )
    writer.add_scalar("timestep", total_timesteps, global_step)
    writer.add_scalar("Epoch and warmup ", epoch + n_warmup_episodes, global_step)

    # Prepare tabulate table
    table = []
    fancy_float = lambda f : f"{f:.3f}"
    table.extend([
        ["Epoch", epoch],
        ["Train reward", fancy_float(train_result)],
        ["Test reward", fancy_float(test_result)],
        ["Total timesteps", total_timesteps],
        ["Sampling time (s)", fancy_float(t_sampling)],
        ["PMD time (s)", fancy_float(t_pmd)],
        ["Test time (s)", fancy_float(t_test)],
        ["Execution time (s)", fancy_float(execution_time)],
    ])


    fancy_grid = tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign='right')
  
    # Log to stdout and log file
    log_file.write("\n")
    log_file.write(fancy_grid)
    log_file.flush()
    print(fancy_grid)

import os
import jax
import time
import wandb
import socket
import argparse
import warnings
import gymnasium as gym
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning) # Remove experimental warning
from tqdm.rich import tqdm
from pprint import pprint
from datetime import datetime
from tensorboardX import SummaryWriter

from powr.FasterNyMDPManager import FasterNyMDPManager
from powr.kernels import dirac_kernel, gaussian_kernel, gaussian_kernel_diag
from powr.utils import create_dirs, get_random_string, save_config, set_seed, log_epoch_statistics

jax.config.update("jax_enable_x64", True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="MountainCar-v0",
        type=str,
        help="Train gym env [LunarLander-v2, MountainCar-v0, CartPole-v1, Pendulum-v1]",
    )
    parser.add_argument("--group", default=None, type=str, help="Wandb run group")
    parser.add_argument("--project", default=None, type=str, help="Wandb project")
    parser.add_argument(
        "--la",
        default=1e-6,
        type=float,
        help="Regularization for the action-value function estimators",
    )
    parser.add_argument(
        "--eta", default=1, type=float, help="Step size of the Policy Mirror Descent"
    )
    parser.add_argument("--gamma", default=0.99, type=float, help="")
    parser.add_argument("--sigma", default=0.2, type=float, help="")
    parser.add_argument(
        "--n-warmup-episodes",
        "-nwe",
        default=1,
        type=int,
        help="Number of warmups epochs for initializing the P i.e. (transition probability) and Q matrices",
    )
    parser.add_argument(
        "--n-epochs", "-ne", default=200, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--n-train-episodes",
        "-nte",
        default=1,
        type=int,
        help="Number of samples used to update matrix parameters",
    )
    parser.add_argument(
        "--n-subsamples",
        "-ns",
        default=10000,
        type=int,
        help="Number of subsamples for nystrom kernel",
    )
    parser.add_argument(
        "--n-iter-pmd",
        "-nipmd",
        default=1,
        type=int,
        help="Number of iteration to update policy parameters in an off-policy manner",
    )
    parser.add_argument(
        "--n-test-episodes", "-nt", default=10, type=int, help="Number of test episodes"
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--save-gif-every",
        "-sge",
        default=None,
        type=int,
        help="Save gif every <save-gif-every> epochs",
    )
    parser.add_argument("--notes", default=None, type=str, help="Wandb notes")
    parser.add_argument(
        "--tags",
        "--wandb-tags",
        type=str,
        default=[],
        nargs="+",
        help="Tags for wandb run, e.g.: --tags optimized pr-123",
    )
    parser.add_argument(
        "--offline",
        default=False,
        action="store_true",
        help="Offline run without wandb",
    )
    args = parser.parse_args()
    args.algo = "powr"

    return args


def parse_env(env_name, sigma):
    if env_name == "Taxi-v3":
        env = gym.make("Taxi-v3", render_mode="rgb_array")
        kernel = dirac_kernel

    elif env_name == "FrozenLake-v1":
        env = gym.make(
            "FrozenLake-v1",
            desc=None,
            map_name="4x4",
            is_slippery=False,
            render_mode="rgb_array",
        )
        kernel = dirac_kernel

    elif env_name == "LunarLander-v2":
        env = gym.make("LunarLander-v2", render_mode="rgb_array")
        sigma_ll = [sigma for _ in range(6)]
        sigma_ll += [0.0001, 0.0001]
        kernel = gaussian_kernel_diag(sigma_ll)

    elif env_name == "MountainCar-v0":
        env = gym.make("MountainCar-v0", render_mode="rgb_array")
        sigma_mc = [0.1, 0.01]
        kernel = gaussian_kernel_diag(sigma_mc)

    elif env_name == "CartPole-v1":
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        kernel = gaussian_kernel(sigma)

    elif env_name == "Pendulum-v1":
        env = gym.make("Pendulum-v1", g=9.81, render_mode="rgb_array")
        kernel = gaussian_kernel(sigma)

    else:
        raise ValueError(f"Unknown environment: {args.env}")
    
    return env, kernel


def get_run_name(args, current_date=None):
    if current_date is None:
        current_date = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
    return (
        str(current_date)
        + "_"
        + str(args.env)
        + "_"
        + args.algo
        + "_eta"
        + str(args.eta)
        + "_la"
        + str(args.la)
        + "_train_samples"
        + str(args.n_train_episodes)
        + "_n_pmd"
        + str(args.n_iter_pmd)
        + "_seed"
        + str(args.seed)
        + "_"
        + socket.gethostname()
    )


if __name__ == "__main__":
    # ** Run Settings **
    # Parse arguments
    args = parse_args()
    pprint(vars(args))
    random_string = get_random_string(5)
    current_date = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
    run_name = (
        "runs/"
        + str(args.env)
        + "/"
        + args.algo
        + "/"
        + get_run_name(args, current_date)
        + "_"
        + random_string
        + "/"
    )
    create_dirs(run_name)
    save_config(vars(args), run_name)

    # Initialize wandb
    try:
        wandb.init(
            config=vars(args),
            project=("powr" if args.project is None else args.project),
            group=(f"{args.env}/{args.algo}" if args.group is None else args.group),
            name=str(current_date)
            + "_"
            + str(args.env)
            + "_"
            + args.algo
            + "_eta"
            + str(args.eta)
            + "_la"
            + str(args.la)
            + "_train_samples"
            + str(args.n_train_episodes)
            + "_n_pmd"
            + str(args.n_iter_pmd)
            + "_seed"
            + str(args.seed)
            + "_"
            + random_string,
            save_code=True,
            sync_tensorboard=True,
            tags=args.tags,
            monitor_gym=True,
            notes=args.notes,
            mode=("online" if not args.offline else "disabled"),
        )
    except:
        wandb.init(
            config=vars(args),
            project=("NY_md_hyperparams" if args.project is None else args.project),
            group=(f"{args.env}/{args.algo}" if args.group is None else args.group),
            name=str(current_date)
            + "_"
            + str(args.env)
            + "_"
            + args.algo
            + "_eta"
            + str(args.eta)
            + "_la"
            + str(args.la)
            + "_train_samples"
            + str(args.n_train_episodes)
            + "_n_pmd"
            + str(args.n_iter_pmd)
            + "_seed"
            + str(args.seed)
            + "_"
            + random_string,
            save_code=True,
            sync_tensorboard=True,
            tags=args.tags,
            monitor_gym=True,
            notes=args.notes,
            mode="offline",
        )
        args.offline = True

    writer = SummaryWriter(f"{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create log file
    log_file = open(os.path.join((run_name), 'log_file.txt'), 'a', encoding="utf-8")

    # ** Hyperparameters Settings **
    n_warmup_episodes = args.n_warmup_episodes
    n_epochs = args.n_epochs
    n_train_episodes = args.n_train_episodes
    n_iter_pmd = args.n_iter_pmd
    n_test_episodes = args.n_test_episodes
    n_subsamples = args.n_subsamples
    la = args.la
    eta = args.eta
    gamma = args.gamma

    save_gif_every = args.save_gif_every
    load_mdp_manager = args.load_path

    simplify = False

    env, kernel = parse_env(args.env, args.sigma)

    def to_be_jit_kernel(X, Y):
        return kernel(X, Y)

    jit_kernel = jax.jit(to_be_jit_kernel)
    v_jit_kernel = jax.vmap(jit_kernel)

    # ** Seed Settings**
    set_seed(args.seed)

    # ** MDP manager Settings**
    mdp_manager = FasterNyMDPManager(
        env,
        eta=eta,
        la=la,
        kernel=jit_kernel,
        gamma=gamma,
        n_subsamples=n_subsamples,
        vkernel=v_jit_kernel,
        seed=args.seed,
    )

    if n_warmup_episodes > 0:
        if n_warmup_episodes < 10:
            batch_size = n_warmup_episodes
        else:
            batch_size = min(int(n_warmup_episodes / 10), 100)
        for i in range(0, n_warmup_episodes, batch_size):
            _, timesteps = mdp_manager.run(
                batch_size, plot=False, collect_data=True
            )
            if simplify:
                mdp_manager.simplify()


    test_results_list = []
    train_results_list = []
    total_timesteps = timesteps
    batch_size = min(10, n_iter_pmd)
    for i in tqdm(range(n_epochs)):

        start_sampling = time.time()
        mdp_manager.train()
        t_sampling = time.time() - start_sampling

        start_pmd = time.time()
        for k in range(0, n_iter_pmd, batch_size):
            mdp_manager.policy_mirror_descent(batch_size)
        t_pmd = time.time() - start_pmd


        if n_test_episodes > 0:
            start_test = time.time()
            test_result, _ = mdp_manager.run(
                n_test_episodes, plot=False, collect_data=False
            )
            t_test = time.time() - start_test

       
        train_result, timesteps = mdp_manager.run(
            n_train_episodes, plot=False, collect_data=True
        )
        total_timesteps += timesteps

        if simplify:
            mdp_manager.simplify()

       
        # ** Save gif **
        if save_gif_every is not None and i % save_gif_every == 0:
            mdp_manager.run(
                1, plot=True, wandb_log = (args.offline != True), collect_data=False, path=run_name, current_epoch=i
            )

        mdp_manager.reset_Q()

        execution_time = time.time() - start_sampling

        # ** Log data **
        log_epoch_statistics(
            writer,
            log_file,
            i,
            test_result,
            train_result,
            n_train_episodes,
            n_iter_pmd,
            n_warmup_episodes,
            total_timesteps,
            t_sampling,
            t_pmd,
            t_test,
            execution_time,
        )
    
    
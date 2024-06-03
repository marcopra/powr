"""
python3 train_powr_light.py --env MountainCar-v0 --seed 0 --la 1e-6 --eta 0.1 --gamma 0.99 --n-train-episodes 1 --n-subsamples 10000 --n-iter-pmd 1 -nwe 1
"""

import argparse
import pickle
import socket
import time
from datetime import datetime
from pprint import pprint

import gymnasium as gym
import jax

from powr.FasterNyMDPManager import FasterNyMDPManager
from powr.kernels import dirac_kernel, gaussian_kernel, gaussian_kernel_diag
from powr.utils import create_dirs, get_random_string, save_config, set_seed

# from torch.utils.tensorboard import SummaryWriter


jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="Taxi-v3",
        type=str,
        help="Train gym env [Taxi-v3, FrozenLake-v1, LunarLander-v2, MountainCar-v0, CartPole-v1, Pendulum-v1]",
    )
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
        default=0,
        type=int,
        help="Save gif every <save-gif-every> epochs. If <= 0 it never saves.",
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
        # add another 2 elements to sigma equal to 0.001
        sigma_ll += [0.0001, 0.0001]
        kernel = gaussian_kernel_diag(sigma_ll)
        # kernel = gaussian_kernel(sigma)
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
        print(f"Unknown environment: {args.env}")
        raise ValueError()
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
    args = parse_args()
    # args.tags.append("SoftMax Param")
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

    # ** Hyperparameters Settings **
    load_mdp_manager = False

    n_warmup_episodes = args.n_warmup_episodes
    n_epochs = args.n_epochs
    n_train_episodes = args.n_train_episodes
    n_iter_pmd = args.n_iter_pmd
    n_test_episodes = args.n_test_episodes
    n_subsamples = args.n_subsamples

    la = args.la
    eta = args.eta
    gamma = args.gamma

    plot_test = True

    simplify = False

    env, kernel = parse_env(args.env, args.sigma)

    def to_be_jit_kernel(X, Y):
        return kernel(X, Y)

    jit_kernel = jax.jit(to_be_jit_kernel)
    v_jit_kernel = jax.vmap(jit_kernel)

    # ** Seed Settings**
    set_seed(args.seed)

    if load_mdp_manager:
        with open("saves/mdp_manager.pickle", "rb") as f:
            mdp_manager = pickle.load(f)
    else:
        mdp_manager = FasterNyMDPManager(
            env,
            eta=eta,
            la=la,
            kernel=jit_kernel,
            gamma=gamma,
            n_subsamples=n_subsamples,
            vkernel=v_jit_kernel,
        )

        if n_warmup_episodes > 0:
            if n_warmup_episodes < 10:
                batch_size = n_warmup_episodes
            else:
                batch_size = min(int(n_warmup_episodes / 10), 100)

            for i in range(0, n_warmup_episodes, batch_size):
                print(f"Collecting data: {i}/{n_warmup_episodes}")
                _, timesteps = mdp_manager.run(
                    batch_size, plot=False, collect_data=True, seed=args.seed
                )
                print(f"n_points: {mdp_manager.FTL.n}")

                if simplify:
                    print("Simplifying")
                    mdp_manager.simplify()

        # # save the state of the system before training using pickle
        # with open("saves/mdp_manager.pickle", "wb") as f:
        #     pickle.dump(mdp_manager, f)
        #

    test_results_list = []
    train_results_list = []
    total_timesteps = timesteps
    for i in range(n_epochs):

        t = time.time()
        mdp_manager.train()

        batch_size = min(10, n_iter_pmd)
        for k in range(0, n_iter_pmd, batch_size):
            print(f"Policy mirror descent: {k}/{n_iter_pmd}")
            mdp_manager.policy_mirror_descent(batch_size)

        if n_test_episodes > 0:
            print("Starting test")
            test_result = mdp_manager.run(
                n_test_episodes, plot=False, collect_data=False, seed=args.seed
            )
            print("Test results:", test_results_list)

        print("Before", mdp_manager.FTL.n)
        print(
            f"Collecting data: {i*n_train_episodes + n_warmup_episodes}/{n_train_episodes*n_epochs + n_warmup_episodes}"
        )

        train_result, timesteps = mdp_manager.run(
            n_train_episodes, plot=False, collect_data=True, seed=args.seed
        )
        total_timesteps += timesteps

        if simplify:
            print("Simplifying")
            mdp_manager.simplify()

        if args.save_gif_every > 0:
            if i % args.save_gif_every == 0:
                print("Saving .gif")
                _, gif = mdp_manager.run(
                    1, plot=True, collect_data=False, path=run_name, seed=args.seed
                )

        mdp_manager.reset_Q()
        # mdp_manager.delete_Q_memory()

        # mdp_manager.gamma = min(mdp_manager.gamma*1.1, 0.99)
        # n_iter_pmd = min(n_iter_pmd*2, 30)
        # mdp_manager.eta = min(mdp_manager.eta*2, 100)

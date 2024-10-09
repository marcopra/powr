import os
import jax
import wandb
import socket
import logging
import warnings
import argparse
import gymnasium as gym
from pprint import pprint
from datetime import datetime
from tensorboardX import SummaryWriter
from tqdm import TqdmExperimentalWarning

jax.config.update("jax_enable_x64", True)
os.environ["WANDB_START_METHOD"] = "thread"
warnings.filterwarnings(
    "ignore", category=TqdmExperimentalWarning
)  # Remove experimental warning

from powr.utils import *
from powr.powr import POWR
from powr.kernels import dirac_kernel, gaussian_kernel, gaussian_kernel_diag


logging.basicConfig(level=logging.WARNING)
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('tensorboardX').setLevel(logging.WARNING)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MountainCar-v0", type=str, help="Train gym env [LunarLander-v2, MountainCar-v0, CartPole-v1, Pendulum-v1]",)
    parser.add_argument("--group", default=None, type=str, help="Wandb run group")
    parser.add_argument("--project", default=None, type=str, help="Wandb project")
    parser.add_argument("--la", default=1e-6, type=float, help="Regularization for the action-value function estimators",)
    parser.add_argument("--eta", default=0.1, type=float, help="Step size of the Policy Mirror Descent")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
    parser.add_argument("--sigma", default=0.2, type=float, help="")
    parser.add_argument("--n-warmup-episodes", "-nwe", default=1, type=int, help="Number of warmups epochs for initializing the P i.e. (transition probability) and Q matrices",)
    parser.add_argument("--n-epochs", "-ne", default=200, type=int, help="Number of training epochs, i.e. Data Sampling, P computation, Policy Mirror Descent, and Testing",)
    parser.add_argument("--n-train-episodes","-nte", default=1, type=int, help="Number of episodes used to sample for each epoch",)
    parser.add_argument("--n-parallel-envs", "-npe", default=3, type=int, help="Number of parallel environments",)
    parser.add_argument("--n-subsamples", "-ns", default=10000, type=int, help="Number of subsamples for nystrom kernel",)
    parser.add_argument("--n-iter-pmd", "-nipmd", default=1, type=int, help="Number of iteration to update policy parameters in an off-policy manner", )
    parser.add_argument("--n-test-episodes", "-nt", default=1, type=int, help="Number of test episodes")
    parser.add_argument("--q-mem", "-qm", default=0, type=int, help="Number of Q-memories to use to use")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--save-gif-every","-sge", default=None, type=int, help="Save gif every <save-gif-every> epochs",)
    parser.add_argument("--save-checkpoint-every","-sce", default=None, type=int, help="Save checkpoint every <save-checkpoint-every> epochs",)
    parser.add_argument("--eval-every", default=1, type=int, help="Evaluate policy every <eval-every> epochs",)
    parser.add_argument("--delete-Q-memory", "-dqm", default=False, action="store_true", help="Delete the previously estimated Q functions",)
    parser.add_argument("--notes", default=None, type=str, help="Wandb notes")
    parser.add_argument("--checkpoint", default=None, type=str, help="checkpoint path")
    parser.add_argument("--tags", "--wandb-tags", type=str, default=[], nargs="+", help="Tags for wandb run, e.g.: --tags optimized pr-123",)
    parser.add_argument("--device", type=str, default="gpu",  help="Device setting <cpu> or <gpu>",)
    parser.add_argument("--offline", default=False, action="store_true", help="Offline run without wandb",)
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

        env = gym.make_vec("MountainCar-v0", num_envs=args.n_parallel_envs, vectorization_mode="sync")
        env.reset() # TODO: remove this line

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

    if env_name != "MountainCar-v0":
        raise NotImplementedError(f"Parallel env not implemented yet: {args.env}")
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
    checkpoint = args.checkpoint

    # ** Wandb Settings **
    # Resume Wandb run if checkpoint is provided
    if checkpoint is not None:
        checkpoint_data = load_checkpoint(checkpoint)
        project = args.project

        # Load saved `args`, `total_timesteps`, and `wandb_run_id`
        args = argparse.Namespace(**checkpoint_data["args"])
        total_timesteps = checkpoint_data["total_timesteps"]
        starting_epoch = checkpoint_data["epoch"]
        wandb_run_id = checkpoint_data["wandb_run_id"]
        print(wandb_run_id)
        # Resume Wandb run with saved run ID
        wandb.init(
            project=project,
            id=wandb_run_id,  # Use saved Wandb run ID to resume the run
            save_code=True,
            sync_tensorboard=True,
            monitor_gym=True,
            resume="must",
        )

        run_path = f"{checkpoint}/"
    else:
        pprint(vars(args))
        random_string = get_random_string(5)
        current_date = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
        run_path = (
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
        create_dirs(run_path)
        save_config(vars(args), run_path)

        # Initialize wandb
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
        starting_epoch = 0
        total_timesteps = 0

    # ** Device Settings **
    device_setting = args.device
    if device_setting == "gpu":
        device = jax.devices("gpu")[0]
        jax.config.update("jax_default_device", device)  # Update the default device to GPU

        print("Using GPU")
    elif device_setting == "cpu":
        device = jax.devices("cpu")[0]  
        jax.config.update("jax_default_device", device)  # Update the default device to GPU

        print("Using CPU")
    else:
        raise ValueError(f"Unknown device setting {device_setting}, please use <cpu> or <gpu>")
    

    # ** Logging Settings **
    # Create tensorboard writer
    writer = SummaryWriter(f"{run_path}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create log file
    log_file = open(os.path.join((run_path), "log_file.txt"), "a", encoding="utf-8")

    # ** Hyperparameters Settings **
    n_subsamples = args.n_subsamples
    la = args.la
    eta = args.eta
    gamma = args.gamma
    q_memories = args.q_mem
    n_warmup_episodes = args.n_warmup_episodes
    assert n_warmup_episodes > 0, "Number of warmup episodes must be greater than 0"
    n_epochs = args.n_epochs
    n_train_episodes = args.n_train_episodes
    n_iter_pmd = args.n_iter_pmd
    n_test_episodes = args.n_test_episodes

    save_gif_every = args.save_gif_every
    eval_every = args.eval_every
    save_checkpoint_every = args.save_checkpoint_every  
    delete_Q_memory = args.delete_Q_memory

    simplify = False

    # ** Environment Settings **
    env, kernel = parse_env(args.env, args.sigma)

    # ** Kernel Settings **
    def to_be_jit_kernel(X, Y):
        return kernel(X, Y)

    jit_kernel = jax.jit(to_be_jit_kernel)
    v_jit_kernel = jax.vmap(jit_kernel) # TODO Not used

    # ** Seed Settings**
    set_seed(args.seed)

    
    powr = POWR(
            env, 
            env, 
            args,
            eta=eta, 
            la=la, 
            gamma=gamma, 
            kernel=jit_kernel,
            n_subsamples=n_subsamples,
            n_warmup_episodes=n_warmup_episodes,
            n_epochs=n_epochs,
            n_iter_pmd=n_iter_pmd,
            n_train_episodes=n_train_episodes,
            n_test_episodes=n_test_episodes,
            q_memories=q_memories,
            save_gif_every=save_gif_every,
            eval_every=eval_every,
            save_checkpoint_every=save_checkpoint_every,
            delete_Q_memory=delete_Q_memory,
            tensorboard_writer=writer,
            starting_logging_epoch=starting_epoch,
            run_path=run_path,
            seed=args.seed,
            checkpoint=checkpoint,
            device=device,
            offline=args.offline,
        
    )

    powr.train( 
        epochs=n_epochs,
        warmup_episodes = n_warmup_episodes,
        train_episodes = n_train_episodes,
        test_episodes = n_test_episodes,
        iterations_pmd=n_iter_pmd,
        eval_every=eval_every,
        save_gif_every=save_gif_every,
        save_checkpoint_every=save_checkpoint_every,
        args_to_save=args,
        
        
    ) 

   
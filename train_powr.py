
from powr.FasterNyMDPManager import FasterNyMDPManager

import pickle

import time
import gymnasium as gym
import jax.numpy as jnp
import jax

from utils.utils import *
import argparse
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter
import torch
import wandb
import warnings
import os

jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default="MountainCar-v0", type=str, help='Train gym env [Hopper-v3, ...]')
    parser.add_argument('--group', default=None, type=str, help='Wandb run group')
    parser.add_argument('--project', default=None, type=str, help='Wandb project')
    parser.add_argument('--la', default=1e-6, type=float, help='')
    parser.add_argument('--eta', default=1, type=float, help='Similar to learning rate (???)')
    parser.add_argument('--gamma', default=0.99, type=float, help='')
    parser.add_argument('--sigma', default=0.2, type=float, help='')
    parser.add_argument('--n-warmup-episodes', '-nwe', default=1, type=int, help='Number of warmups epochs for initializing the P i.e. (transition probability) and Q matrices')
    parser.add_argument('--n-epochs', '-ne', default=200, type=int, help='Number of training epochs')
    # parser.add_argument('--timesteps', '-t', type=int, default=500_000,help="total timesteps of the experiments") # TODO NOT USED AT THE MOMENT
    parser.add_argument('--n-train-episodes', '-nte', default=1, type=int, help='Number of samples used to update matrix parameters')
    parser.add_argument('--n-subsamples', '-ns', default=10000, type=int, help='Number of subsamples for nystrom kernel')
    parser.add_argument('--n-iter-pmd', '-nipmd', default=1, type=int, help='Number of iteration to update policy parameters in an off-policy manner')
    parser.add_argument('--n-test-episodes', '-nt', default=10, type=int, help='Number of test episodes')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--save-gif-every', '-sge', default=5, type=int, help='Save gif every <save-gif-every> epochs')
    parser.add_argument('--notes', default=None, type=str, help='Wandb notes')
    parser.add_argument('--tags', '--wandb-tags', type=str, default=[], nargs="+", help="Tags for wandb run, e.g.: --tags optimized pr-123")
    parser.add_argument('--offline', default=False, action='store_true', help='Offline run without wandb')
    # parser.add_argument('--video', '-v', default=False, action='store_true', help='Video recording of evaluation during training')
    args = parser.parse_args()
    args.algo = 'fast_nymd'
    # args.n_epochs=int(args.timesteps//args.n_train_episodes)
 
    return args

def get_run_name(args, current_date= None):
    if current_date is None:
        current_date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    return str(current_date)+"_"+str(args.env)+"_"+args.algo+"_eta"+str(args.eta)+"_la"+str(args.la)+"_train_samples" + str(args.n_train_episodes)+ "_n_pmd"+str(args.n_iter_pmd)+ "_seed"+str(args.seed)+"_"+socket.gethostname()




# compute the dirac kernel on batches of states
def dirac_kernel(X, Y):
    return ((X.reshape(-1, 1) - Y.reshape(1, -1)) == 0)*1.0

# gaussian kernel for matrices of n points and d dimensions
class gaussian_kernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Y):
        return jnp.exp(-(1/self.sigma) * jnp.linalg.norm(X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1), axis=2))


# gaussian kernel for matrices of n points and d dimension with a different sigma for each dimension
class gaussian_kernel_diag:
    def __init__(self, sigma):
        self.sigma = jnp.array(sigma).reshape(1, 1, -1)

    def __call__(self, X, Y):
        return jnp.exp(-jnp.sum((X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1))**2/(2*self.sigma**2), axis=2))

class abel_kernel_diag:
    def __init__(self, sigma):
        self.sigma = jnp.array(sigma).reshape(1, 1, -1)

    def __call__(self, X, Y):
        return jnp.exp(-jnp.sum(jnp.abs(X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1))/(jnp.sqrt(2)*self.sigma), axis=2))


if __name__ == '__main__':
    args = parse_args()
    args.tags.append("SoftMax Param")
    pprint(vars(args))
    random_string = get_random_string(5)
    current_date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    run_name = "runs/"+str(args.env)+"/"+ "fast_nymd" +"/" +get_run_name(args, current_date)+"_"+random_string+"/"
    create_dirs(run_name)
    save_config(vars(args), run_name)

    try:
        wandb.init(config=vars(args),
                project=("NY_md_hyperparams" if args.project is None else args.project) , #"NY_md_hyperparams",
                # entity="iit_policy_gradient_methods",
                group=(f"{args.env}/fast_nymd" if args.group is None else args.group),
                name=str(current_date)+"_"+str(args.env)+"_"+args.algo+"_eta"+str(args.eta)+"_la"+str(args.la)+"_train_samples" + str(args.n_train_episodes)+ "_n_pmd"+str(args.n_iter_pmd)+"_seed"+str(args.seed) + "_"+random_string,
                save_code=True,
                sync_tensorboard=True,
                tags=args.tags,
                monitor_gym=True, 
                notes=args.notes,
                mode=('online' if not args.offline else 'disabled'))
    except:
        wandb.init(config=vars(args),
                project=("NY_md_hyperparams" if args.project is None else args.project) ,
                # entity="iit_policy_gradient_methods",
                group=(f"{args.env}/fast_nymd" if args.group is None else args.group),
                name=str(current_date)+"_"+str(args.env)+"_"+args.algo+"_eta"+str(args.eta)+"_la"+str(args.la)+"_train_samples" + str(args.n_train_episodes)+ "_n_pmd"+str(args.n_iter_pmd)+"_seed"+str(args.seed) + "_"+random_string,
                save_code=True,
                sync_tensorboard=True,
                tags=args.tags,
                monitor_gym=True, 
                notes=args.notes,
                mode='offline')
        args.offline = True
    
    writer = SummaryWriter(f"{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


   # ** Hyperparameters Settings **
    load_mdp_manager = False

    n_warmup_episodes = args.n_warmup_episodes 
    n_epochs        = args.n_epochs
    n_train_episodes = args.n_train_episodes
    n_iter_pmd      = args.n_iter_pmd
    n_test_episodes = args.n_test_episodes
    n_subsamples = args.n_subsamples

    save_gif_every = args.save_gif_every

    la = args.la ##
    eta = args.eta
    gamma = args.gamma
    sigma = args.sigma




    plot_test = True

    simplify = False
    is_slippery = False
    env_name = args.env

    # env_name = "FrozenLake-v1"
    # env_name = "LunarLander-v2"
    # env_name = "Taxi-v3"
    # env_name = "MountainCar-v0"
    # env_name = "CartPole-v1"
    # env_name = "Pendulum-v1"

    if env_name == "Taxi-v3":
            env = gym.make('Taxi-v3', render_mode="rgb_array")
            kernel = dirac_kernel
    elif env_name ==  "FrozenLake-v1":
            env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=is_slippery, render_mode="rgb_array")
            kernel = dirac_kernel
    elif env_name == "LunarLander-v2":
            env = gym.make('LunarLander-v2', render_mode="rgb_array")
            sigma_ll = [sigma for _ in range(6)]
            # add another 2 elements to sigma equal to 0.001
            sigma_ll += [0.0001, 0.0001]
            kernel = gaussian_kernel_diag(sigma_ll)
            # kernel = gaussian_kernel(sigma)
    elif env_name == "MountainCar-v0":
            env = gym.make('MountainCar-v0', render_mode="rgb_array")
            # sigma_mc = [0.05, 0.005]
            # sigma_mc = [0.05, 0.001]
            # sigma_mc = [0.01, 0.001]
            # sigma_mc = [0.001, 0.0005]
            sigma_mc = [0.1, 0.01]
            # sigma_mc = [0.1, 0.01]
            # sigma_mc = [0.0001, 0.00001]
            kernel = gaussian_kernel_diag(sigma_mc)
            # kernel = abel_kernel_diag(sigma_mc)
    elif env_name == "CartPole-v1":
            env = gym.make('CartPole-v1', render_mode="rgb_array")
            kernel = gaussian_kernel(sigma)
    elif env_name == "Pendulum-v1":
            env = gym.make('Pendulum-v1', g=9.81, render_mode="rgb_array")
            kernel = gaussian_kernel(sigma)
    else:
        print(f"Unknown environment: {env_name}")
        raise ValueError()


    def to_be_jit_kernel(X, Y):
        return kernel(X, Y)

    jit_kernel = jax.jit(to_be_jit_kernel)
    v_jit_kernel = jax.vmap(jit_kernel)

    if load_mdp_manager:
        with open("saves/mdp_manager.pickle", "rb") as f:
            mdp_manager = pickle.load(f)
    else:
        mdp_manager = FasterNyMDPManager(env, eta=eta, la=la, kernel=jit_kernel, gamma=gamma, n_subsamples=n_subsamples,
                                vkernel=v_jit_kernel)


        if n_warmup_episodes>0:
            if n_warmup_episodes < 10:
                batch_size = n_warmup_episodes
            else:
                batch_size = min(int(n_warmup_episodes/10), 100)
            for i in range(0, n_warmup_episodes, batch_size):
                print(f"Collecting data: {i}/{n_warmup_episodes}")
                _, timesteps = mdp_manager.run(batch_size, plot=False, collect_data=True)
                print(f"n_points: {mdp_manager.FTL.n}")

                if simplify:
                    print(f"simplifying...")
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
            test_result = mdp_manager.run(n_test_episodes, plot=False, collect_data=False)
            print("Test results:", test_results_list)

        print("Before", mdp_manager.FTL.n)
        print(f"Collecting data: {i*n_train_episodes + n_warmup_episodes}/{n_train_episodes*n_epochs + n_warmup_episodes}")
        
        train_result, timesteps = mdp_manager.run(n_train_episodes, plot=False, collect_data=True)
        total_timesteps+=timesteps
        

        if simplify:
            print(f"simplifying...")
            mdp_manager.simplify()

        

        print("Saving data ...")
        global_step = i 
        # Record rewards and data for plotting purposes
        writer.add_scalar("test reward", test_result, global_step)
        writer.add_scalar("train reward", train_result, global_step)
        writer.add_scalar("Sampling and Updating steps", n_train_episodes + i*(n_train_episodes + n_iter_pmd), global_step)
        writer.add_scalar("Epoch", i, global_step)
        writer.add_scalar("Train Episodes",  n_warmup_episodes + i*n_train_episodes, global_step)
        writer.add_scalar("timestep", total_timesteps, global_step)
        writer.add_scalar("Epoch and warmup ", i + n_warmup_episodes, global_step)  
        # x massima 
        if i % save_gif_every == 0:
            _, gif = mdp_manager.run(1, plot=True, collect_data=False, path = run_name)

            if args.offline is not True:
                wandb.log({"Epoch": global_step, "video": wandb.Video(run_name+"/"+gif)})
                # # check if the file exists
                # if os.path.isfile(run_name+"/"+gif):
                #     # remove the file
                #     os.remove(run_name+"/"+gif)

        mdp_manager.reset_Q()
        # mdp_manager.delete_Q_memory()

        # mdp_manager.gamma = min(mdp_manager.gamma*1.1, 0.99)
        # n_iter_pmd = min(n_iter_pmd*2, 30)
        # mdp_manager.eta = min(mdp_manager.eta*2, 100)















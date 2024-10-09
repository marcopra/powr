import os
import time
import warnings
import argparse
import jax.numpy as jnp
import gymnasium as gym
from powr.utils import *
from tqdm.rich import tqdm
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings(
    "ignore", category=TqdmExperimentalWarning
)  # Remove experimental warning
from powr.MDPManager import MDPManager

class POWR:
    def __init__(
            self, 
            env, 
            eval_env, 
            args,
            eta=0.1, 
            la=0.1, 
            gamma=0.99, 
            kernel=None,
            n_subsamples=100,
            n_warmup_episodes=2,
            n_epochs=100,
            n_iter_pmd=10,
            n_train_episodes=10,
            n_test_episodes=3,
            q_memories=10,
            save_gif_every=None,
            eval_every=1,
            save_checkpoint_every=None,
            delete_Q_memory=False,
            tensorboard_writer=None,
            starting_logging_epoch=0,
            run_path=None,
            seed=None,
            checkpoint=None,
            device='cpu',
            offline=False,
            ):
        """
        Initialize the POWR algorithm with the necessary parameters.
        
        Args:
            env (gym.Env): The training environment.
            eval_env (gym.Env): The evaluation environment.
            args (argparse.Namespace): The arguments for the POWR algorithm.
            eta (float): The learning rate.
            la (float): The regularization parameter.
            gamma (float): The discount factor.
            kernel (function): The kernel function to use for the MDP manager.
            n_subsamples (int): The number of subsamples to use for the kernel approximation.
            n_warmup_episodes (int): The number of episodes to use for warmup.
            n_epochs (int): The number of training epochs.
            n_train_episodes (int): The number of episodes to use for training.
            n_iter_pmd (int): The number of PMD iterations.
            n_test_episodes (int): The number of episodes to use for testing.
            q_memories (int): The number of Q-function memories to store.
            save_gif_every (int): Save a GIF every n episodes.
            eval_every (int): Evaluate the policy every n episodes.
            save_checkpoint_every (int): Save a checkpoint every n episodes.
            delete_Q_memory (bool): Whether to delete the Q-function memory after training.
            tensorboard_writer (SummaryWriter): The TensorBoard writer.
            starting_logging_epoch (int): The starting epoch for logging.
            run_path (str): The path to the run directory.
            seed (int): The random seed.
            checkpoint (str): The path to the checkpoint file.
            device (str): The device to run the algorithm on.
        """

        self.env = env
        self.eval_env = eval_env
        assert isinstance(env, gym.experimental.vector.sync_vector_env.SyncVectorEnv), f"env must be created with make_vec_env to obtain type gym.vector.VectorEnv, got {type(env)}"
        assert isinstance(eval_env, gym.experimental.vector.sync_vector_env.SyncVectorEnv), f"env must be created with make_vec_env to obtain type gym.vector.VectorEnv, got {type(eval_env)}"
        self.args = args # TODO cercare di rimuovere, bisogna vedere sul load checkpoint
        self.eta = eta
        self.la = la
        self.gamma = gamma
        assert kernel is not None, "Kernel function must be provided."
        self.n_subsamples = n_subsamples
        self.n_warmup_episodes = n_warmup_episodes
        self.n_epochs = n_epochs
        self.n_train_episodes = n_train_episodes
        self.n_iter_pmd = n_iter_pmd
        self.n_test_episodes = n_test_episodes
        self.q_memories = q_memories
        self.save_gif_every = save_gif_every
        self.eval_every = eval_every
        self.save_checkpoint_every = save_checkpoint_every
        self.delete_Q_memory = delete_Q_memory
        self.tensorboard_writer = tensorboard_writer
        self.starting_logging_epoch = starting_logging_epoch
        self.run_path = run_path
        self.log_file = open(os.path.join((self.run_path), "log_file.txt"), "a", encoding="utf-8")
        self.seed = seed        
        self.checkpoint = checkpoint
        self.device = device
        self.offline = offline



        # ** Initialize Logging **
        self.total_timesteps = 0
        self.starting_epoch = 0


        # ** MDP manager Settings**
        self.mdp_manager = MDPManager(
            self.env,
            self.eval_env,
            eta=self.eta,
            la=self.la,
            kernel=kernel,
            gamma=self.gamma,
            n_subsamples=self.n_subsamples,
            seed=self.seed,
        )


        if self.checkpoint is not None:
            mdp_manager_file = os.path.join(self.checkpoint, "mdp_manager.pkl")
            self.mdp_manager.load_checkpoint(mdp_manager_file)
            print("Loaded from Checkpoint")


    
    def train(self, 
              epochs=1000, 
              warmup_episodes=1, 
              training_episodes=1,
              test_episodes=1,
              iterations_pmd=1,
              eval_every=1,
              save_gif_every=None,
              save_checkpoint_every=None,
              args_to_save=None,

              ):
        """
        Train the POWR algorithm on the given environment.
        
        Args:
            epochs (int): The number of training epochs.
            warmup_episodes (int): The number of warmup episodes.
            training_episodes (int): The number of episodes to use for training.
            test_episodes (int): The number of episodes to use for testing.
            iterations_pmd (int): The number of PMD iterations.
            eval_every (int): Evaluate the policy every n epochs.
            save_gif_every (int): Save a GIF every n epochs.
            save_checkpoint_every (int): Save a checkpoint every n epochs.
            args_to_save (argparse.Namespace): The arguments to save in the checkpoint.
        """

        assert  warmup_episodes > 0, "Number of warmup episodes must be greater than 0"

        #** Warmup the models **
        if warmup_episodes > 0: 
            start_sampling = time.time()
            train_result, timesteps = self.mdp_manager.run(warmup_episodes, plot=False, collect_data=True)
            t_sampling = time.time() - start_sampling

        self.total_timesteps += timesteps


        for i in tqdm(range(epochs)):

            # ** Training the models with previously collected data**
            start_training = time.time()
            self.mdp_manager.train()
            t_training = time.time() - start_training

            # ** Applying Policy Mirror Descent to Policy**
            start_pmd = time.time()
            self.mdp_manager.policy_mirror_descent(iterations_pmd)
            t_pmd = time.time() - start_pmd

            # ** Evaluate the policy every <eval_every> epochs **
            if i%eval_every == 0:
                if self.n_test_episodes > 0:
                    start_test = time.time()
                    test_result = self.mdp_manager.test(
                        test_episodes
                    )
                    t_test = time.time() - start_test
                else:
                    t_test = None
                    test_result = None
            else:
                    t_test = None
                    test_result = None


            # ** Save gif ** # TODO testing
            if save_gif_every is not None and i % save_gif_every == 0:
                self.mdp_manager.test(
                    1,
                    plot=True,
                    wandb_log=(self.offline == False),
                    path=self.run_path,
                    current_epoch=i + self.starting_epoch,
                )
            
            execution_time = time.time() - start_sampling

            # ** Log data **
            log_epoch_statistics(
                self.tensorboard_writer,
                self.log_file,
                i + self.starting_epoch,
                test_result,
                train_result,
                self.n_train_episodes,
                self.n_iter_pmd,
                self.n_warmup_episodes,
                self.total_timesteps,
                t_sampling,
                t_training,
                t_pmd,
                t_test,
                execution_time,
            )

            
            # Save checkpoint every <save_checkpoint_every> epochs 
            if save_checkpoint_every is not None and i % save_checkpoint_every == 0 and i > 0:
                save_checkpoint(self.run_path , self.args, self.total_timesteps, i + self.starting_epoch, self.mdp_manager)

            # No need to store new data if it's the last epoch
            if i == self.n_epochs - 1:
                break
            
            # ** Collect data for the next epoch **
            start_sampling = time.time()
            train_result, timesteps = self.mdp_manager.collect_data(
                training_episodes
            )
            t_sampling = time.time() - start_sampling

            self.total_timesteps += timesteps

            self.mdp_manager.reset_Q(q_memories = self.q_memories)

            if self.delete_Q_memory:
                self.mdp_manager.delete_Q_memory()
            
        save_checkpoint(self.run_path , args_to_save, self.total_timesteps, i + self.starting_epoch if self.checkpoint is not None else i, self.mdp_manager)
    

    
    def evaluate(self, episodes):
        """
        Evaluate the policy on the environment without updating it.
        
        Args:
            episodes (int): Number of episodes to evaluate.
        
        Returns:
            avg_reward (float): The average reward over the evaluation episodes.
        """

        return self.mdp_manager.test(episodes)
        
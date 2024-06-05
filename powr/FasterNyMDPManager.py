import os
import cv2
import jax
import time
import wandb
import imageio
import numpy as np
import jax.numpy as jnp
import gymnasium as gym

from powr.FastQmodel import FastQmodel
from powr.FasterNyIncrementalRLS import FasterNyIncrementalRLS


class softmax:

    def __init__(self):
        pass

    def __call__(self, x):
        return jax.nn.softmax(x, axis=1)


class FasterNyMDPManager:

    def __init__(
        self,
        env,
        gamma=0.95,
        eta=1,
        la=1e-3,
        kernel=None,
        n_subsamples=None,
        plotting=False,
        eps_softmax=1e-9,
        vkernel=None,
        seed=None,
    ):

        assert kernel is not None
        assert n_subsamples is not None
        assert vkernel is not None

        self.kernel = kernel
        self.vkernel = vkernel
        self.n_subsamples = n_subsamples

        self.env = env
        # Environment seed
        if seed is not None:
            self.env.reset(seed=seed)
        
        self.n_actions = self.env.action_space.n

        self.gamma = gamma
        self.eta = eta
        self.la = la
        self.plotting = plotting

        self.eps_softmax = eps_softmax

        # to keep track of the training expoenents
        self.cum_train_exponent = None

        # previous  models for the cumulative Q functions
        self.f_prev_cumQ_models = []
        self.f_prev_exponents = None

        self.action_one_hot = jnp.eye(self.n_actions)

        self.softmax = softmax()

        self.f_cumQ_weights = None
        self.f_Q_mask = None
        self.FTL = FasterNyIncrementalRLS(
            kernel=self.kernel,
            n_actions=self.n_actions,
            la=self.la,
            n_subsamples=self.n_subsamples,
        )
    

    # Update the Q function
    def update_Q(self):

        assert self.f_cumQ_weights is not None

        f_exponents = (
            self.f_prev_exponents
            + self.eta * self.FTL.K_transitions_sub @ self.f_cumQ_weights
        )
        f_pi = self.softmax(f_exponents)

        pPit = self.FTL.K_transitions_sub.reshape(
            self.FTL.n, self.FTL.n_sub, 1
        ) * f_pi.reshape(self.FTL.n, 1, self.n_actions) # qui va out of memory
        pPit = pPit[:, jnp.arange(self.FTL.n_sub), self.FTL.A_sub]
        f_big_M = jnp.eye(self.FTL.n_sub) - (self.gamma) * self.FTL.B @ pPit
        f_tmp_Q = jnp.linalg.solve(f_big_M, self.FTL.r)
               
        self.f_cumQ_weights += f_tmp_Q * self.f_Q_mask

    def evaluate_pi(self, state):

        if self.f_cumQ_weights is None:
            return jnp.ones(self.n_actions) / self.n_actions

        jnp_state = jnp.array(state).reshape(1, -1)

        exponent = (
            self.eta
            * self.FTL.kernel(jnp.array(state).reshape(1, -1), self.FTL.X_sub) # TODO Controllare azioni
            @ self.f_cumQ_weights
        )
        for model in self.f_prev_cumQ_models:
            exponent += model.evaluate(jnp_state)

        pi = self.softmax(exponent)

        if jnp.isnan(pi).any():
            raise ValueError("Error: NaN in the results of the training")

        return pi

    def sample_action(self, state):

        p = self.evaluate_pi(state)

        p = np.asarray(p).astype("float64")
        p = p.squeeze()
        p = p / p.sum()

        try:
            action = np.random.choice(self.n_actions, p=p)
        except:
            raise ValueError("Error: NaN in the results of the training")

        return action, p

    # Delete the Q function from memory
    def delete_Q_memory(self):
        self.f_prev_cumQ_models = []
        self.f_prev_exponents = None

    # Reset the Q function
    def reset_Q(self):

        # Append the current Q function to the list of functions
        self.f_prev_cumQ_models.append(
            FastQmodel(
                kernel=self.kernel,
                Q=self.eta * self.f_cumQ_weights,
                X_sub=self.FTL.X_sub,
            )
        )
    
        # New transition learner
        new_FTL = FasterNyIncrementalRLS(
            kernel=self.kernel,
            n_actions=self.n_actions,
            la=self.la,
            n_subsamples=self.n_subsamples,
        )

        # Data collection
        new_FTL.collect_data(
            self.FTL.A, self.FTL.X, self.FTL.Y_transitions, self.FTL.Y_rewards
        )
        self.FTL = new_FTL

        self.f_cumQ_weights = None

    def subsample(self):

        self.FTL.subsample()

    def run(self, n_episodes=1, plot=False, collect_data=False, path=None, wandb_log=False, current_epoch=None):

        total_timesteps = 0
        if collect_data:
            f_X = []
            f_Y_transitions = []
            f_Y_rewards = []
            f_A = []

        cum_rewards = np.zeros(n_episodes)
        for episode_id in range(n_episodes):
            state, info = self.env.reset()

            images = []
            while True:
                action, pi = self.sample_action(state)
                total_timesteps += 1

                # save pi from jax vector into a list of floats (truncated at the third decimal)
                pi = [round(float(p), 3) for p in pi.squeeze()]


                new_state, reward, terminated, truncated, info = self.env.step(action)
                


                cum_rewards[episode_id] += reward

                if collect_data:
                    f_X.append(state)
                    f_Y_transitions.append(new_state)
                    f_Y_rewards.append(reward)
                    f_A.append(action)

                # update the state
                state = new_state

                if self.plotting or plot:
                    img = self.env.render()

                    # reduce the font size if the action is pi is too long
                    if len(str(pi)) > 10:
                        font_scale = 0.55
                    else:
                        font_scale = 1.0
                    # write the action on the right left corner of the image (in green) font 16 and thickness 2
                    img = cv2.putText(
                        img,
                        f"Action: {action} - Pi: {pi}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    images.append(img)

                if terminated or truncated:

                    # if the episode terminated, record the last state as a sink state
                    if terminated:
                        
                        if collect_data:
                            end_reward = 0
                            f_X.append(new_state)
                            f_Y_transitions.append(new_state)
                            f_Y_rewards.append(end_reward)
                            f_A.append(action)

                    # save the gif
                    if self.plotting or plot:
                        gif_name = f"Epoch={current_epoch}-Reward={cum_rewards[episode_id]}.gif"
                        if path is None:
                            path = "./gifs/tmp"
                            if os.path.isdir(path) is False:
                                os.mkdir(path)
                            imageio.mimsave(f"{path}/{gif_name}", images)
                        else:
                            imageio.mimsave(f"{path}/{gif_name}", images)

                        if wandb_log:

                            wandb.log(
                                {"Epoch": current_epoch, "video": wandb.Video(path + "/" + gif_name)}
                            )
                    break

        if collect_data:
            f_X = jnp.array(f_X)
            f_Y_transitions = jnp.array(f_Y_transitions)
            if f_X.ndim == 1:
                f_X = f_X.reshape(-1, 1)
                f_Y_transitions = f_Y_transitions.reshape(-1, 1)
            f_Y_rewards = jnp.array(f_Y_rewards).reshape(-1, 1)
            f_A = jnp.array(f_A)

            self.FTL.collect_data(
                f_A, f_X, f_Y_transitions, f_Y_rewards
            )

     
        return cum_rewards.mean(), total_timesteps

    def simplify(self):
        self.TransitionLearner.simplify()

    def train(self):

        self.FTL.train()

        if self.f_cumQ_weights is None and self.FTL.n_sub > 0:
            self.f_cumQ_weights = jnp.zeros((self.FTL.n_sub, self.n_actions))
            self.f_Q_mask = self.action_one_hot[self.FTL.A_sub]

        self.f_prev_exponents = jnp.zeros((self.FTL.n, self.n_actions))
        for model in self.f_prev_cumQ_models:
            self.f_prev_exponents += model.evaluate(self.FTL.Y_transitions)

    def policy_mirror_descent(self, n_iter):

        for _ in range(n_iter):
            self.update_Q()
        
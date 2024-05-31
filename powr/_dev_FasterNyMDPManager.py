import time

import cv2
import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import wandb
from powr._dev_NyRLS_solver import FasterNyIncrementalRLS
from powr.FastQmodel import FastQmodel


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
    ):

        assert kernel is not None
        assert n_subsamples is not None
        assert vkernel is not None

        self.kernel = kernel
        self.vkernel = vkernel
        self.n_subsamples = n_subsamples

        self.env = env

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

    def check_data_collected_but_not_trained(self):
        assert not self._DATA_COLLECTED_BUT_NOT_TRAINED

    def update_Q(self):

        self.check_data_collected_but_not_trained()

        assert self.f_cumQ_weights is not None

        f_exponents = (
            self.f_prev_exponents
            + self.eta * self.FTL.K_transitions_sub @ self.f_cumQ_weights
        )
        f_pi = self.softmax(f_exponents)

        pPit = self.FTL.K_transitions_sub.reshape(
            self.FTL.n, self.FTL.n_sub, 1
        ) * f_pi.reshape(self.FTL.n, 1, self.n_actions)
        pPit = pPit[:, jnp.arange(self.FTL.n_sub), self.FTL.A_sub]
        f_big_M = jnp.eye(self.FTL.n_sub) - (self.gamma) * self.FTL.B @ pPit
        # f_big_M = 1. *  jnp.eye(self.FTL.n_sub) - (self.gamma) * self.FTL.B @ (self.FTL.K_transitions_sub * f_pi.reshape(-1,1))
        f_tmp_Q = jnp.linalg.solve(f_big_M, self.FTL.r)
        #
        # piPt = (self.gamma) * self.FTL.B @ (self.FTL.K_transitions_sub * f_pi.reshape(-1,1))
        #
        # step = self.FTL.r
        # cum_step = step
        # step_val = self.FTL.K_transitions_sub @ (step * self.f_Q_mask)
        # # print(step_val)
        # for t in range(500):
        #     step = piPt @ step
        #     cum_step = cum_step + step
        #     step_val = self.FTL.K_transitions_sub @ (step * self.f_Q_mask)
        #     # print(step_val)
        current_q_vals = self.FTL.K_transitions_sub @ (f_tmp_Q * self.f_Q_mask)
        tmp_exp = self.eta * current_q_vals
        if (tmp_exp > 0).any():
            print(
                f"Max: {tmp_exp.max()} [WARNING][WARNING][WARNING][WARNING][WARNING][WARNING][WARNING][WARNING][WARNING][WARNING]"
            )

        self.f_cumQ_weights += f_tmp_Q * self.f_Q_mask

    def _evaluate_pi_legacy(self, state):

        if self.f_cumQ_weights is None:
            return jnp.ones(self.n_actions) / self.n_actions

        jnp_state = jnp.array(state).reshape(1, -1)

        exponent = (
            self.eta
            * self.FTL.kernel(jnp.array(state).reshape(1, -1), self.FTL.X_sub)
            @ self.f_cumQ_weights
        )
        for model in self.f_prev_cumQ_models:
            exponent += model.evaluate(jnp_state)

        pi = self.softmax(exponent)

        if jnp.isnan(pi).any():
            raise ValueError("Error: NaN in the results of the training")

        return pi

    def evaluate_pi(self, state):

        if self.f_cumQ_weights is None:
            return jnp.ones(self.n_actions) / self.n_actions

        jnp_state = jnp.array(state).reshape(1, -1)

        exponent = (
            self.eta
            * self.FTL.kernel(jnp.array(state).reshape(1, -1), self.FTL.X_sub)
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
            # round all elements of p at the 4 decimal floor
            print("Still???")

        return action, p

    def delete_Q_memory(self):
        self.f_prev_cumQ_models = []
        self.f_prev_exponents = None

    def reset_Q(self):

        self.f_prev_cumQ_models.append(
            FastQmodel(
                kernel=self.kernel,
                Q=self.eta * self.f_cumQ_weights,
                X_sub=self.FTL.X_sub,
            )
        )

        new_FTL = FasterNyIncrementalRLS(
            kernel=self.kernel,
            n_actions=self.n_actions,
            la=self.la,
            n_subsamples=self.n_subsamples,
        )
        new_FTL.collect_data(
            self.FTL.A, self.FTL.X, self.FTL.Y_transitions, self.FTL.Y_rewards
        )
        self.FTL = new_FTL

        self.f_cumQ_weights = None

    def subsample(self):

        self.FTL.subsample()

    def run(self, n_episodes=1, plot=False, collect_data=False, path=None, seed=None):
        assert seed is not None, "Use a seed for reproducible experiments"
        total_timesteps = 0
        if collect_data:
            f_X = []
            f_Y_transitions = []
            f_Y_rewards = []
            f_A = []

        cum_rewards = np.zeros(n_episodes)
        for episode_id in range(n_episodes):
            state, _ = self.env.reset(
                seed=seed
            )  # Important to seed here [PIE ??? WHY????]

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
                    # write the action on the right left corner of the image (in green) font 16 and thickness 2
                    img = cv2.putText(
                        img,
                        f"Action: {action} - Pi: {pi}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    images.append(img)

                if terminated or truncated:
                    # if the episode terminated, record the last state as a sink state
                    if terminated:
                        print(
                            "[TRAIN] !!!!!! TERMINATED !!!!! ", cum_rewards[episode_id]
                        )

                        if collect_data:
                            end_reward = 0

                            f_X.append(new_state)
                            f_Y_transitions.append(new_state)
                            f_Y_rewards.append(end_reward)
                            f_A.append(action)

                    if self.plotting or plot:
                        # save gif as a fast frame rate
                        print(
                            f"Episode {episode_id} - Reward: {cum_rewards[episode_id]}"
                        )
                        gif_name = f"{time.time()}-reward-{cum_rewards[episode_id]}.gif"
                        if path is None:
                            imageio.mimsave(f"./gifs/tmp/{gif_name}", images)
                        else:
                            imageio.mimsave(f"{path}/{gif_name}", images)
                        print("Saving gif")

                    break

        if collect_data:
            f_X = jnp.array(f_X)
            f_Y_transitions = jnp.array(f_Y_transitions)
            f_Y_rewards = jnp.array(f_Y_rewards).reshape(-1, 1)
            f_A = jnp.array(f_A)

            self.FTL.collect_data(
                f_A, f_X, f_Y_transitions, f_Y_rewards, print_error=True
            )

            self._DATA_COLLECTED_BUT_NOT_TRAINED = True
        if self.plotting or plot:
            return cum_rewards.mean(), gif_name
        elif collect_data:
            return cum_rewards.mean(), total_timesteps

        return cum_rewards.mean()

    def simplify(self):
        self.TransitionLearner.simplify()
        print(f"Sizes: {[l for l in self.TransitionLearner.n]}")

    def train(self):

        t = time.time()
        self.FTL.train()
        print("[Train] Time taken: ", time.time() - t)

        t = time.time()

        if self.f_cumQ_weights is None and self.FTL.n_sub > 0:
            self.f_cumQ_weights = jnp.zeros((self.FTL.n_sub, self.n_actions))
            self.f_Q_mask = self.action_one_hot[self.FTL.A_sub]

        self.f_prev_exponents = jnp.zeros((self.FTL.n, self.n_actions))
        for model in self.f_prev_cumQ_models:
            self.f_prev_exponents += model.evaluate(self.FTL.Y_transitions)

        if self.f_prev_exponents.max() > 0:
            print(
                f"Max={self.f_prev_exponents.max()} [WARNING][WARNING][WARNING][WARNING][WARNING][WARNING][WARNING][WARNING][WARNING][WARNING]"
            )

        self._DATA_COLLECTED_BUT_NOT_TRAINED = False

    def policy_mirror_descent(self, n_iter):

        t = time.time()
        for itr in range(n_iter):
            tinn = time.time()
            self.update_Q()
        #     print("[PMD STEP TIME]: ", time.time() - tinn)
        # print("[Total PMD STEP TIME]: ", time.time() - t)

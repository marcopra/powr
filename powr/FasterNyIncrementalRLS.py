import jax
import jax.numpy as jnp
import jax.random as jrandom
import os
import time
from powr.kernels import dirac_kernel
from jax import vmap


class FasterNyIncrementalRLS:
    def __init__(self, kernel=None, n_actions=None, la=1e-3, n_subsamples=None):

        assert kernel is not None
        assert n_actions is not None
        assert n_subsamples is not None

        self.kernel = kernel
        self.n_actions = n_actions
        self.n_subsamples = n_subsamples

        self.la = la

        # reset stuff
        self.n = 0
        self.n_sub = None

        self.reset()

    # verify that we have not called subsample yet
    def check_subsample(self):
        assert self._SUBSAMPLE_HAS_BEEN_CALLED == False

    def reset(self):
        self.A = None
        self.X = None
        self.Y_transitions = None
        self.Y_rewards = None
        self.X_sub = None
        self.A_sub = None

        self.K_full_sub = jnp.zeros((0, 0))
        self.K_transitions_sub = jnp.zeros((0, 0))
        self.K_sub_sub = None

        self.sub_indices = None

        self._SUBSAMPLE_HAS_BEEN_CALLED = False

    # collect data -> store in memory the data
    def collect_data(self, A, X, Y_transitions, Y_rewards):

        if self.n == 0:
            self.A = A
            self.X = X
            self.Y_transitions = Y_transitions
            self.Y_rewards = Y_rewards

        else:
            # check that the data is provided as a list of arrays one for each possible action
            self.X = jnp.vstack([self.X, X])
            self.Y_transitions = jnp.vstack([self.Y_transitions, Y_transitions])
            self.Y_rewards = jnp.vstack([self.Y_rewards, Y_rewards])
            self.A = jnp.hstack([self.A, A])

        self.n = self.X.shape[0]

        if self._SUBSAMPLE_HAS_BEEN_CALLED:
            self.update_kernels(A, X, Y_transitions)

    # update the kernels
    def update_kernels(self, A, X, Y_transitions):

        Knew = self.kernel(jnp.vstack([X, Y_transitions]), self.X_sub)

        self.K_full_sub = jnp.vstack(
            [self.K_full_sub, Knew[: X.shape[0]] * dirac_kernel(A, self.A_sub)]
        )

        self.K_transitions_sub = jnp.vstack(
            [self.K_transitions_sub, Knew[X.shape[0] :]]
        )

    def simplify(self):
        raise NotImplementedError("Not implemented yet")


    def subsample(self):

        self.check_subsample()

        if self.n == 0:
            return

        seed = int.from_bytes(os.urandom(4), "big")
        key = jrandom.PRNGKey(seed)

        # if the number of points is smaller than the number of subsamples, we just use all the points
        self.sub_indices = jnp.arange(self.n)
        if self.n > self.n_subsamples:
            self.sub_indices = jrandom.choice(
                key, int(self.n), (self.n_subsamples,), replace=False
            )
        self.n_sub = self.sub_indices.shape[0]

        self.X_sub = self.X[self.sub_indices]
        self.A_sub = self.A[self.sub_indices]

        self.K_full_sub = jnp.zeros((0, self.n_sub))
        self.K_transitions_sub = jnp.zeros((0, self.n_sub))

        self.update_kernels(self.A, self.X, self.Y_transitions)
        self.K_sub_sub = self.K_full_sub[self.sub_indices]

        self._SUBSAMPLE_HAS_BEEN_CALLED = True

    def train(self):

        if not self._SUBSAMPLE_HAS_BEEN_CALLED:
            self.subsample()

        L = jax.lax.linalg.cholesky(
            self.K_full_sub.T @ self.K_full_sub
            + self.n * self.la * self.K_sub_sub
            + 1e-6 * jnp.eye(self.K_full_sub.shape[1])
        )

        if jnp.isnan(L).any():
            raise ValueError("Error: NaN in the results of the training for Chol")

        W = jax.lax.linalg.triangular_solve(
            L,
            jax.lax.linalg.triangular_solve(
                L,
                jnp.hstack([self.K_full_sub.T, self.K_full_sub.T @ self.Y_rewards]),
                lower=True,
                left_side=True,
                transpose_a=False,
            ),
            lower=True,
            left_side=True,
            transpose_a=True,
        )

        self.r = W[:, -1].reshape(-1, 1)
        self.B = W[:, :-1]

        # check if the results contain nan
        if jnp.isnan(self.r).any() or jnp.isnan(self.B).any():
            raise ValueError("Error: NaN in the results of the training")

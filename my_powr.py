import jax
import timeit

import gymnasium as gym
from jax import vmap   
import jax.numpy as jnp
from jax.nn import softmax 
from jax.scipy.linalg import cho_factor, cho_solve

    
# define a one hot encoding function
def one_hot(x, k):
    return jax.nn.one_hot(x, k)

@jax.jit
def dirac_kernel(X, Y):
    return vmap(lambda x: vmap(lambda y: jnp.where(jnp.all(x == y), 1.0, 0.0))(Y))(X)

# gaussian kernel for matrices of n points and d dimensions
class gaussian_kernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Y):
        return jnp.exp(
            -(1 / self.sigma)
            * jnp.linalg.norm(
                X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1), axis=2
            )
        )

class Q:
    def __init__(self, kernel=None, Q=None, X=None):

        assert kernel is not None
        assert X is not None
        assert Q is not None

        self.kernel = kernel
        self.Q = Q
        self.X = X

    def evaluate(self, X=None):
        return self.kernel(X, self.X) @ self.Q
    
class pi:
    def __init__(self, env, kernel, eta = 0.1):
        self.kernel = kernel
        self.eta = eta
        self.n_actions = env.action_space.n
        self.Qs_weights = []
        self.Qs_weights.append((jnp.ones(self.n_actions) / self.n_actions))
        self.f_prev_cumQ_models = []
        self.H = None
        self.C = None
    
    def evaluate(self, state):

        if len(self.Qs_weights) == 1:
            return softmax(self.Qs_weights[-1])

        jnp_state = jnp.array(state).reshape(1, -1)

        exponent = 0
        for Q_weight in self.Qs_weights:
            exponent += (
            self.eta
            * self.kernel(jnp_state, jnp_state)
            @ Q_weight
        )

        pi = softmax(exponent)

        if jnp.isnan(pi).any():
            raise ValueError("Error: NaN in the results of the training")

        return pi
    
    def update(self, H, C):
        self.H = H
        self.C = C
        
        


    
def collect_data(env, n_episodes):
            states = []
            next_states = []
            actions = []
            rewards = []
            for episode in range(n_episodes):
                state, _ = env.reset()
                done = False
                while not done:
                    action = env.action_space.sample()
                    next_state, reward, terminated, truncated, info = env.step(action)

                    states.append(state)
                    next_states.append(next_state)
                    actions.append(action)
                    rewards.append(reward)  

                    done = terminated or truncated
                    state = next_state 
            states = jnp.array(states)
            next_states = jnp.array(next_states)
            actions = jnp.array(actions)
            rewards = jnp.array(rewards)
            return states.reshape(-1, states.shape[1]), jnp.array(next_states).reshape(-1, next_states.shape[1]), jnp.array(actions), jnp.array(rewards).reshape(-1, 1)

if __name__ == '__main__':
    # define a <env> environment from the gymnasium library
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    k = gaussian_kernel(0.2)
    delta = dirac_kernel
    la = 0.1


    n_actions = env.action_space.n
    n_episodes = 1
    n_epochs = 1

    for epoch in range(n_epochs):

        # loop over episodes
        # collect data (state, action, next_state, reward) from the environment
        states, next_states, actions, rewards = collect_data(env, n_episodes)

        H = k(states, next_states)
        K = k(states, states)*delta(actions, actions)
        # Compute the inverse using the Cholesky factor
        B_inv = K + la*jnp.eye(K.shape[1])
        B = cho_solve(cho_factor(B_inv), jnp.eye(B_inv.shape[1]))

        b = B @ rewards
        print(b.shape)
        exit()
       

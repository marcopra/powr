import jax.numpy as jnp

class FastQmodel:
    def __init__(self, kernel=None, Q=None, X_sub=None):

        assert kernel is not None
        assert X_sub is not None
        assert Q is not None

        self.kernel = kernel
        self.Q = Q
        self.X_sub = X_sub


    def evaluate(self, X=None):
        return self.kernel(X, self.X_sub) @ self.Q
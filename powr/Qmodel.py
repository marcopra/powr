import jax.numpy as jnp


class Qmodel:
    def __init__(self, kernel=None, Q=None, X_sub=None):

        assert kernel is not None
        assert X_sub is not None
        assert Q is not None

        self.kernel = kernel
        self.Q = Q
        self.X_sub = X_sub

    def evaluate(self, X=None):
        return self.kernel(X, self.X_sub) @ self.Q

    def __getstate__(self):
        """ Prepare the object for pickling by returning only necessary attributes. """
        state = {
            'Q': self.Q,
            'X_sub': self.X_sub
        }
        return state

    def __setstate__(self, state):
        """ Restore the object's state with only the necessary attributes. """
        self.Q = state['Q']
        self.X_sub = state['X_sub']
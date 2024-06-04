import jax
import jax.numpy as jnp
from jax import vmap

# compute the dirac kernel on batches of states
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


# gaussian kernel for matrices of n points and d dimension with a different sigma for each dimension
class gaussian_kernel_diag:
    def __init__(self, sigma):
        self.sigma = jnp.array(sigma).reshape(1, 1, -1)

    def __call__(self, X, Y):
        return jnp.exp(
            -jnp.sum(
                (X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1)) ** 2
                / (2 * self.sigma**2),
                axis=2,
            )
        )


class abel_kernel_diag:
    def __init__(self, sigma):
        self.sigma = jnp.array(sigma).reshape(1, 1, -1)

    def __call__(self, X, Y):
        return jnp.exp(
            -jnp.sum(
                jnp.abs(X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1))
                / (jnp.sqrt(2) * self.sigma),
                axis=2,
            )
        )

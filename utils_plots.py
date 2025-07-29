import jax
jax.config.update("jax_enable_x64", True)
from diffrax import *
import jax.random as jr
import jax.numpy as jnp
from jax import vmap
import time
import string
import matplotlib.pyplot as plt



def solve_single_path(key, d, y0, A, B):
    bm = VirtualBrownianTree(0, 1, tol=1e-5, shape=(d,), key=key)

    # Define drift and diffusion
    drift = lambda t, y, args: jnp.einsum('ij,j->i', B, y)
    diffusion = lambda t, y, args: jnp.einsum('ija,j->ia', A, y)

    term = MultiTerm(
        ODETerm(drift),
        ControlTerm(diffusion, bm),
    )

    return diffeqsolve(
        term,
        solver=Midpoint(),
        t0=0,
        t1=1,
        dt0=1e-3,
        y0=y0,
        saveat=SaveAt(t1=True),
        max_steps=None
    ).ys
    
    
def monte_carlo(sample_size, d, y0, A, B):
    seed = int(time.time() * 1e6) % (2**32 - 1)
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, sample_size)
    sols = jax.vmap(lambda key: solve_single_path(key, d, y0, A, B))(keys)
    return jnp.mean(sols, axis=0)


def mean_ODE(y0, A, B):

    # Define drift and diffusion
    AA = jnp.einsum('kic,ijc->kj', A, A)
    drift = lambda t, y, args: (B + 0.5 * AA) @ y

    term = ODETerm(drift)

    return diffeqsolve(
        term,
        solver=Midpoint(),
        t0=0,
        t1=1,
        dt0=1e-6,
        y0=y0,
        saveat=SaveAt(t1=True),
        max_steps=None
    ).ys


def A_to_the(A, n):
    result = A
    for i in range(1, n):
        subscripts = (
            "ab"
            + "".join(chr(100 + k) for k in range(i))
            + ",bc"
            + chr(100 + i)
            + "->ac"
            + "".join(chr(100 + k) for k in range(i + 1))
        )
        result = jnp.einsum(subscripts, result, A)
    return result


def Ans_tuple(A, N):
    "This outputs the tuple of powers of A from 1 to N. If vmapped across a batch outputs a list of jnp arrays with the batch on the first axis."
    return tuple(A_to_the(A, k) for k in range(1, N + 1))


def single_sum_euler(n, y0, An, Sn):
    indices_A = "ab" + "".join(chr(99 + i) for i in range(n - 1, -1, -1))
    indices_S = "".join(chr(99 + i) for i in range(n))
    output_indices = "ab"
    R = jnp.einsum(f"{indices_A},{indices_S}->{output_indices}", An, Sn)
    return jnp.dot(R, y0)


def one_step_euler(n, y0, S, *Ans):
    "Performs one step in the Euler scheme with Ans as variadic args for vmap-friendliness."
    return y0 + sum(
        single_sum_euler(k, y0, Ans[k-1], S[k]) for k in range(1, n + 1)
    )
    


# exp_points should be a list of length equal to the degree of cubature whose entries are
# of jnp arrays of shapes (no_pts,), (no_pts, d), (no_pts, d, d), ... where no_pts is the number of cubature points
# and exp_points[n][i] is the projection onto (R^d)^{⊗n} of the tensor exponential of the i-th Lie polynomial cubature point 
# viewed as a jnp array of shape (d, ..., d) (k d's)
# weights is a jnp array of shape (no_pts,)
# this will return the no_pts solutions to be fed into the next step or averaged
def one_step_cubature(y0, A, B, exp_pts):
    BA = jnp.concatenate([B[..., None], A], axis=-1)
    degree = len(exp_pts) - 1
    BAns = Ans_tuple(BA, degree)
    one_pt = lambda one_exp: one_step_euler(degree, y0, one_exp, *BAns)
    return jax.vmap(lambda *args: one_pt(list(args)), in_axes=0)(*exp_pts)


def weighted_sum(arr, w):
    # arr shape: (N, N, ..., N, e)
    # w shape: (N,)
    
    m = arr.ndim - 1  # number of N-axes
    assert m <= 26, "Too many N-dimensions for einsum"

    letters = string.ascii_lowercase
    axes = letters[:m]         # for N-dimensions
    batch = letters[m]         # the last letter represents 'e'
    eq = (
        f"{''.join(axes)}{batch}," +        # arr indices
        ",".join(axes) +                    # one 'w' for each axis
        f"->{batch}"                        # output shape (e,)
    )

    w_tuple = tuple(w for _ in range(m))  # one copy of w per axis
    return jnp.einsum(eq, arr, *w_tuple)


def inhom_scaling(arr_in, lam):
    shape = arr_in.shape
    k = len(shape) - 1  # number of trailing d dimensions

    if k == 0:
        return arr_in  # nothing to scale

    # Build index grid for the k trailing dimensions
    idx_grids = jnp.meshgrid(*[jnp.arange(shape[i+1]) for i in range(k)], indexing='ij')
    idx_stack = jnp.stack(idx_grids, axis=-1)  # shape: (d, ..., d, k)

    # Define weight function w(j): 1 if j == 0, 0.5 if j > 0
    def w(j):
        return jnp.where(j == 0, 1.0, 0.5)

    weights = w(idx_stack)  # shape: (d, ..., d, k)
    weight_sum = jnp.sum(weights, axis=-1)  # shape: (d, ..., d)

    # Expand weight_sum to match arr_in shape for broadcasting
    weight_sum = jnp.expand_dims(weight_sum, axis=0)  # shape: (1, d, ..., d)

    arr_out = arr_in * (lam ** weight_sum)
    return arr_out


def scale(cub_form, lam):
    scaled = []
    for k in range(len(cub_form)):
        #scaled.append((jnp.pow(lam, k/2)) * cub_form[k])
        scaled.append(inhom_scaling(cub_form[k], lam))
    return scaled


def multistep_cubature(y0, A, B, exp_pts, weights, no_intervals):
    ys = y0  # shape (e,)
    exp_pts = scale(exp_pts, 1/no_intervals)
    for _ in range(no_intervals):
        f = lambda y: one_step_cubature(y, A, B, exp_pts)  # (e,) → (N, e)

        for _ in range(ys.ndim - 1):  # vectorize over all but last axis
            f = jax.vmap(f)

        ys = f(ys)  # becomes (N, ..., N, e)
    return weighted_sum(ys, weights)

def plot_functions(fs, ms, labels=None, title="Function plots",
                   xlabel="x", ylabel="f(x)", scale="linear"):
    """
    Plots a list of functions fs with bullet markers and lines, using input range [1, m_k] for each.
    
    Args:
        fs: list of callables, each f_k(n) -> float
        ms: list of ints, length limits per function
        labels: optional list of strings
        title: plot title
        xlabel, ylabel: axis labels
        scale: "linear" (default) or "log" for log-y
    """
    assert len(fs) == len(ms), "fs and ms must be the same length"

    markers = ['^', 's', 'o', 'D', 'x']  # up to 5 styles

    for idx, (f, m) in enumerate(zip(fs, ms)):
        x = list(range(1, m + 1))
        y = [f(i) for i in x]
        label = labels[idx] if labels else f"f{idx+1}"
        plt.plot(x, y, marker=markers[idx % len(markers)], linestyle='-', label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if scale == "log":
        plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

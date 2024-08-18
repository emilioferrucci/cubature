import numpy as np
import roughpy as rp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from linear_solver import *
import jax.numpy as jnp
from diffrax import *

# define a path that goes round in circles
def circle(t, a, b, c):
    return a * np.exp(2 * b * np.pi * 1j * (t + c))


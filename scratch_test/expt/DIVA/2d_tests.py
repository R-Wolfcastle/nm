#1st party
import os
import sys


#3rd party
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

#local apps
nm_home = os.environ['NM_HOME']   

sys.path.insert(1, os.path.join(nm_home, 'utils'))
import constants_years as c
from vertical_grid import *
from standard_domains import wonky_stream



lx, ly, nr, nc,\
x, y, delta_x,\
delta_y, thk, b,\
C, mucoef_0, q,\
ice_mask, surface,\
grounded = wonky_stream(resolution=2000)









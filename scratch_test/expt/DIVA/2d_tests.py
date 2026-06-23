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








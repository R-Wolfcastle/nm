#1st party
import sys
import os

#3rd party
import jax
import jax.numpy as jnp

#local apps
nm_home = os.environ['NM_HOME']   

sys.path.insert(1, os.path.join(nm_home, 'utils'))
import constants_years as c




def pressure(z_coordinates)

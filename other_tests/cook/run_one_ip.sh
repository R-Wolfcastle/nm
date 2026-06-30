#!/bin/bash


#It was taking a loooong time to solve each LA problem, but this seems to fix it and stop
#PETSc trying to use n_threads=n_cores!
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16

#Force JAX to use CPU
export JAX_PLATFORMS=cpu

yr="2016"
nohup python -u cook_ip_for_year_data.py $yr > log_files/pp/msrs.log 2>&1 &
#-u flag for running in unbuffered mode so stdout written to files in right way


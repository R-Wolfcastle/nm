#!/bin/bash


#It was taking a loooong time to solve each LA problem, but this seems to fix it and stop
#PETSc trying to use n_threads=n_cores!
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

#Force JAX to use CPU
export JAX_PLATFORMS=cpu

yrs="2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026"
for yr in $yrs; do
    nohup python -u cook_ip_for_year_data.py $yr > log_files/pp/250m/$yr.log 2>&1 &
done
#-u flag for running in unbuffered mode so stdout written to files in right way


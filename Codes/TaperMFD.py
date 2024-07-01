#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:47:36 2023

@author: hkaveh
"""
#%%
import numpy as np
import random
import Taper
random.seed(2000)
#%%
N_beta=20           # Number of points to be sampled from beta distribution
N_zeta=20           # Number of points to be sampled from zeta distribution
N=5                # Number of points to be sampled from poisson distribution
N_dpoints=5        # Number of Points to be sampled fome model parameters
N_realization=10    # Number of realization of tapered process
# N_beta=10           # Number of points to be sampled from beta distribution
# N_zeta=10           # Number of points to be sampled from zeta distribution
# N=2                # Number of points to be sampled from poisson distribution
# N_dpoints=2        # Number of Points to be sampled fome model parameters
# N_realization=2    # Number of realization of tapered process

c=9.1           # global constant from Bourne, S. J., and S. J. Oates. 2020. “Stress-Dependent Magnitudes of Induced Earthquakes in the Groningen Gas Field.” Journal of Geophysical Research: Solid Earth 125(11): e2020JB020013.
d=1.5           # global constant from Bourne, S. J., and S. J. Oates. 2020. “Stress-Dependent Magnitudes of Induced Earthquakes in the Groningen Gas Field.” Journal of Geophysical Research: Solid Earth 125(11): e2020JB020013.
M_c=1.1        # Magnitude of completeness
DeltaM=0.0001
M_m=np.exp( (c+d*( M_c - 1/2 * DeltaM) ) *np.log(10) ) # Eq 15 from Bourne, S. J., and S. J. Oates. 2020. “Stress-Dependent Magnitudes of Induced Earthquakes in the Groningen Gas Field.” Journal of Geophysical Research: Solid Earth 125(11): e2020JB020013.
N_mesh=45
Cum_num=Taper.find_Cum_num(N_dpoints,N)
trace=Taper.OptimizeBeta_Zeta(c, d)
#%%
Year=2022     # Filter out numerical data and observational data with time greater than Year
Catalog=Taper.CleanCatalog()

mask = Catalog[:, 1] <= Year+1
Catalog=Catalog[mask]
Mean=Taper.Run(Year,N_beta,N_zeta,trace,M_m,N_mesh,N_realization,c,d,Catalog,Cum_num)


# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:22:52 2023

@author: hkaveh
"""
import numpy as np
import matplotlib.pylab as plt
import h5py
import Functions_spatial
import pandas as pd
import concurrent.futures
from pathlib import Path
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve


#%%
NumOfyearstoadd=3   # We add a few zeros to the catalog before the seismicity starts.
M = 4               # Number of paramteres to identify
h=.5e-8             # Parameter for the finite difference
taw=.1e-5           # Threshold for the cnvergence
mu1=1e-1            # Line search paramter 1
mu2=9e-1            # Line search parameter 2
sigma=2             # Line search parameter 3
RangeIncreaseFactor= 1.1 # Factor to increase the umax
Maxiter=600         # Optimization parameter
N_alpha_max=30      # Optimization parameter
LIKELIHOOD='Poisson'
NumOfInit=600#%
IsMonthly=1         # The current version of the code only works when IsMonthly=1
#%% Importing the resevoir geometry, well locations and so on.
f = h5py.File('./../Data/Groningen_Data_Example.hdf5', 'r')
# keys() shows the contents of the file
f.keys()
x_res = f['x_res']
y_res = f['y_res']
reservoir_outline = f['res_outline']
x_outline = f['x_outline']
y_outline = f['y_outline']
## The original well extraction data
extractions = f['extraction_data']
wells_locations = f['wells_locations_list']
wells_names = f['wells_names']
## The computed pressures from the linearized pressure diffusion model
pressures = f['pressures_list']
x_mesh = f['x_mesh']# The meshes used for the pressure diffusion model 
y_mesh = f['y_mesh']# The meshes used for the pressure diffusion model 
## The computed deformations and coulomb stress change from the BorisBlocks model
deformations = f['deformations_list']
x_reservoir_mesh = f['x_reservoir_mesh'] # The meshes used for the BorisBlocks model 
y_reservoir_mesh = f['y_reservoir_mesh'] # The meshes used for the BorisBlocks model
#%% Importing the sterss distribution, and the associated time, and then smoothing it.
if IsMonthly==1:
    Time_monthly=np.load("./../Data/time_monthly.npy")
# R_monthly=np.loadtxt('R_monthly.csv', delimiter=',')
    Stress_monthly=np.load("./../Data/max_coulomb_stresses.npy",allow_pickle=True).item()['-10m']
    max_coulomb_stress2 = np.nan_to_num(Stress_monthly) # Replacing NAN with zeros
    smoothed_coulomb_stress2 = np.zeros(max_coulomb_stress2.shape)
    kernel = Gaussian2DKernel(8)
    for i in range (Stress_monthly.shape[2]):
        smoothed_coulomb_stress2[:,:,i] = convolve(max_coulomb_stress2[:,:,i],kernel,nan_treatment='fill')# smoothen the coulomb stress
    
    Stress_monthly=smoothed_coulomb_stress2
    Time_monthly=Time_monthly[:Stress_monthly.shape[2]]
else:       
    # a=np.loadl()
    Time_monthly=np.load("Data/time_yearlysmoothed.npy")        
    Stress_monthly=np.load("Data/Coulomb_yearlysmoothed.npy")
        
DX,DY,a=np.shape(Stress_monthly)



fig, ax = plt.subplots(1,2,figsize=(12,5))

# the final map of pressures
quad1 = ax[0].scatter(np.array(x_res).flatten(), np.array(y_res).flatten(), c=np.mean(Stress_monthly,axis=2), cmap='coolwarm')
ax[0].plot(x_outline[:],y_outline[:],'k--')
ax[0].set_title('Average of stress over entire years')
cb1 = plt.colorbar(quad1,ax=ax[0])
cb1.set_label('Maximum Coulomb stress change (MPa)')
# the final map of deformation
quad2 = ax[1].scatter(np.array(x_res).flatten(), np.array(y_res).flatten(), c=Stress_monthly[:,:,-1], cmap='coolwarm')
ax[1].plot(x_outline[:],y_outline[:],'k--')
ax[1].set_title('Maximum coulomb stress change at the last data point')
cb1 = plt.colorbar(quad2,ax=ax[1])
cb1.set_label('Maximum Coulomb stress change (MPa)')
# the final map of coulomb stress change

# the map labels for the three top subplots
for ii in [0,1]:
  ax[ii].set_xlabel('X (km)')
  ax[ii].set_ylabel('Y (km)')
  ax[ii].set_xlim([230,270])
  ax[ii].set_ylim([565,615])
  ax[ii].set_aspect('equal')  
#%% Getting rid of the data outside the region with average stress below thresh. This is equivalent to applying the "map" of the Gas field.
# The red region you see on the right is the area that we consider for eq forcasting.
thresh=0.08
mask = Functions_spatial.Get_mask (Stress_monthly, thresh)

avg_ds = np.mean(Stress_monthly,axis=2)
fig, ax = plt.subplots(1,2,figsize=(12,5))

# the final map of pressures
quad1 = ax[0].scatter(np.array(x_res).flatten(), np.array(y_res).flatten(), c=avg_ds, cmap='coolwarm')
ax[0].plot(x_outline[:],y_outline[:],'k--')
ax[0].set_title('Maximum coulomb stress change')
cb1 = plt.colorbar(quad1,ax=ax[0])
cb1.set_label('Maximum Coulomb stress change (MPa)')
# the final map of deformation
quad2 = ax[1].scatter(np.array(x_res).flatten(), np.array(y_res).flatten(), c=mask, cmap='coolwarm')
ax[1].plot(x_outline[:],y_outline[:],'k--')
ax[1].set_title('Region of interest (threshold=0.1)')
cb1 = plt.colorbar(quad2,ax=ax[1])
# cb1.set_label('Maximum Coulomb stress change (MPa)')
# the final map of coulomb stress change

# the map labels for the three top subplots
for ii in [0,1]:
  ax[ii].set_xlabel('X (km)')
  ax[ii].set_ylabel('Y (km)')
  ax[ii].set_xlim([230,270])
  ax[ii].set_ylim([565,615])
  ax[ii].set_aspect('equal')    
  
#%% Import R0
if IsMonthly==1:
    df = pd.read_csv('./../Data/R_monthly.csv', header=None, names=['date', 'value'], skiprows=1)
else:
    df = pd.read_csv('./../Data/R_yearlysmoothed.csv', header=None, names=['date', 'value'], skiprows=1)
# Split the date column into year, month, and day columns
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
df['day'] = pd.to_datetime(df['date']).dt.day
data = np.array(df[['year', 'month', 'day', 'value']])
Time_R=  df['year']  +  df['month']/12-1/12 # Note that I have intentionally not indluded the "day" in data because the data correspond to the all the observed seismicity during that month

# Find index of element in A closest to B[0"
idx = np.abs(Time_monthly - Time_R[0]).argmin()
Monthly_R=np.array(df['value'])

#%%
def FindYearlyAve(time,values):
    years = (np.floor(time))
    yearly_averages = (np.zeros(len(np.unique(years))))
    for i, year in enumerate(np.unique(years)):
        mask = years == year
        yearly_averages[i] = np.mean(values[mask])
    return np.unique(years),yearly_averages
#%%

tpup,Rpup=FindYearlyAve(Time_R,Monthly_R)
    #plt.style.use('default')
fig = plt.figure(figsize=(3.7, 2.8))

plt.rc('font',family='Serif')
plt.rcParams.update({'font.family':'Serif'})

ax = fig.add_subplot(1, 1, 1)
ax.step(tpup,Rpup)


#%%
# preparing stress and seisimicity rate for inversion
Ds_t = Functions_spatial.Organize_Dst (Stress_monthly, mask)
Totnum_month=int(Monthly_R.size+NumOfyearstoadd*12)
idx = np.abs(Time_monthly - Time_R[0]).argmin() # This is the index of the time of stress  that the time of seimicity and time of stress are equal
N_z=NumOfyearstoadd*12
Ds_t_final = Ds_t[idx-N_z:,:]*1e6 # Note that this contains stresses up to almost 2030
R0=np.append(np.zeros((1,N_z)),Monthly_R) # But this one contains rate untill 2022

time = np.array(range(R0.size))


plt.plot(time,R0)
time_particle=time
time_total=np.array(range(Ds_t_final.shape[0]))

#%% Priors (ranges)

# Defining ranges of priors

# r_min= 5e-7/12 # 6.2e-7 # 4.9e-6
# r_max= 4e-5/12 #2.5e-5 5.1e-6

# t_a_min= 500*12 # 6000
# t_a_max= 1e5*12

# Asig_min= 1e3 #1e3 #
# Asig_max= 0.01e6#.01e6 #1e6 #

# Ds_c_min= 0.07e6 # 0.1e2 #0
# Ds_c_max= 0.3e6#0.3e6 #.5e6
#%%
r_min= 1e-7/12 # 6.2e-7 # 4.9e-6
r_max= 4.5e-5/12 #2.5e-5 5.1e-6

t_a_min= 1*12 # 6000
t_a_max= 1e5*12

Asig_min= 1e3 #1e3 #
Asig_max= .03e6#.01e6 #1e6 #

Ds_c_min= 1e3 # 0.1e2 #0
Ds_c_max= .35e6#0.3e6 #.5e6
#%%

r_range=r_max-r_min
t_a_range=t_a_max-t_a_min
Asig_range=Asig_max-Asig_min
Ds_c_range=Ds_c_max-Ds_c_min


# u=np.array([r,t_a,Asig,Ds_c])
u_min= np.array([r_min,t_a_min,Asig_min,Ds_c_min])
u_max = np.array([r_max,t_a_max,Asig_max,Ds_c_max])
#%%
import Functions_Steepest
# These are normalization constant
s_r=1e-6/12
s_t_a=1e4*12
s_Asigma=5e3
s_Ds_c=10e7

r_init=3.5e-6
t_a_init=7e4
Asigma0=4800
Delta_s_c0=290000
LIKELIHOOD="Poisson"
s=np.array([s_r,s_t_a,s_Asigma,s_Ds_c]).reshape((4,1))
R_o=R0

time=time_particle
u=np.array([r_init,t_a_init,Asigma0,Delta_s_c0]).reshape((4,1))
Ds_t=Ds_t_final

gamma=100
if LIKELIHOOD=='Gaussian':
    Typeofliklihood=1
    #gamma=input('Enter number gamma for Gaussain Process')
    gamma=float(gamma)
    
elif LIKELIHOOD=='Poisson':
    Typeofliklihood=2 
else:
    raise Exception("LIKELIHOOD should either be Gaussian or Poisson")         

#%%
N_parameters=4

def FiniteDifference(N_parameters,f,u_bar,h):
    grad_f=np.zeros((N_parameters,1))
    for i in range (N_parameters):
        u_add=np.zeros_like(u_bar)
        u_add[i]=h        
        grad_f[i]=(f(u_bar+u_add)-f(u_bar-u_add))/(2*h)
    
    return grad_f
#%%
Starting_index=0 # Using data from the begining
LastYeartr=2022 # Last year that is used in the training set, 2012 is included
idx2 = np.abs(LastYeartr +1 - Time_R).argmin()
Tmax=idx2+N_z
YearInit=1992-NumOfyearstoadd
YearMax=LastYeartr
YearMin=YearInit
# Number of months to be considered in the dataset
#%% BFGS
def BFGS_OneInit(u,s,R_o,Tmax,Ds_t_final,time_particle,Maxiter,LIKELIHOOD,N_z,N_alpha_max,h,taw,mu1,mu2,sigma,Starting_index,YearInit,gamma):
# Observations: R_o[Starting_index:Tmax] and Ds_t_final[Starting_index:Tmax] where Tmax and Starting_index are index
# Model parameter: u
# time_particle[Starting_index:Tmax] is the time over which we integrate, note that it is incremented by one so we multiply it by 1/12

    N_parameters=4
    alpha_new=1
    I=np.eye((N_parameters))
    f=lambda u_bar : -Functions_spatial.FindLogLikelihood(u_bar*s,R_o[Starting_index:Tmax],Ds_t_final[Starting_index:Tmax],time_particle[Starting_index:Tmax],LIKELIHOOD,gamma)
    df = lambda u_bar: FiniteDifference(N_parameters,f,u_bar,h)
    u_bar=u/s
    Cost=np.array([-f(u_bar)])
    XK=u

    xk=u_bar
    df_new=df(xk)
    V_new=I/np.linalg.norm(df_new);
    pk=-np.matmul(V_new,df_new)
    phi =  lambda alpha : f(xk+alpha*pk)
    phi_p= lambda alpha : np.dot(df(xk+alpha*pk).T,pk) # Phi_p of alpha
    alpha_init=1
    phi_0=phi(0)
    Dphi_0=phi_p(0)
    alpha_max=.01
    # Functions_Steepest.plot_linesearch(phi,alpha_max)
    alf,Diverged=Functions_Steepest.Bracketing(phi,phi_p,xk,pk,alpha_init,phi_0,Dphi_0,mu1,mu2,sigma,N_alpha_max)
    Dummy=xk+alf[-1]*pk
    while np.any(xk+alf[-1]*pk<0) or np.max(Ds_t_final)<Dummy[3,0]*s[3,0]: # Ensure every iteration is positive
        Dummy=xk+alf[-1]*pk     
        alf[-1]=.8*alf[-1]
    xkp1=xk+alf[-1]*pk
    k=1
    K=np.array([k])
    while np.linalg.norm(df_new)>taw and k<Maxiter and Diverged==False:

        reset=0
        if np.mod(k,5)==0:
            reset=1
        xk_old=xk
        xk=xkp1
        df_old=df_new
        df_new=df(xkp1)
        V_old=V_new
        if reset==1:
            V_new=I/np.linalg.norm(df_new)
        else:
            ss=xk-xk_old
            y=df_new-df_old
            Sigma=1/(np.dot(ss.T,y))
            V_new=(I-Sigma*(ss@(y.T)))@V_old@(I-Sigma*y@(ss.T))+Sigma*(ss@ss.T)    
        pk=-V_new@df_new
        phi =  lambda alpha : f(xk+alpha*pk)
        phi_p= lambda alpha : np.dot(df(xk+alpha*pk).T,pk) # Phi_p of alpha
        phi_0=phi(0)
        Dphi_0=phi_p(0)
        # alpha_max=1
        # Functions_Steepest.plot_linesearch(phi,alpha_max)
        # plt.show()
        alf,Diverged=Functions_Steepest.Bracketing(phi,phi_p,xk,pk,alpha_new,phi_0,Dphi_0,mu1,mu2,sigma,N_alpha_max)
        Dummy=xk+alf[-1]*pk 
        while np.any(xk+alf[-1]*pk<0) or np.max(Ds_t_final)<Dummy[3,0]*s[3,0]: # Ensure every iteration is positive   
            alf[-1]=.5*alf[-1]
            Dummy=xk+alf[-1]*pk

        xkp1=xk+alf[-1]*pk

        k=k+1
        # print(k)
        Cost=np.append(Cost,-f(xkp1))
        XK=np.append(XK,xkp1*s,axis=1) # Recording the data
        K=np.append(K,k)
    Converged=k<Maxiter
    Data=pd.DataFrame(XK.T,columns=['x_0','x_1','x_2','x_3'])
    Data['Cost']=Cost
    Data['Converged']=Converged
    Data['LastDateUsed']=(Tmax-Starting_index)//12+Starting_index+YearInit
    Data['Liklihood']=LIKELIHOOD
    Data['NumOfiter']=K
    Data['FirstDateUsed']=Starting_index+YearInit
    Data['Diverged']=Diverged
    return Data
#%% Generate Initial conditions in ranges



U0 = 1*u_max.reshape((4,1))
for i in range(NumOfInit-1):
    u1 = Functions_spatial.GetU0_Uniform (1,M,u_min,RangeIncreaseFactor*u_max).T
    R_dummy=Functions_spatial.FindR(Ds_t[0:Tmax],u1,time[0:Tmax])
    while R_dummy[0]>1e10:
        u1 = Functions_spatial.GetU0_Uniform (1,M,u_min,RangeIncreaseFactor*u_max).T
        R_dummy=Functions_spatial.FindR(Ds_t[0:Tmax],u1,time[0:Tmax])
        #print(R_dummy) 
    U0=np.append(U0,u1,axis=1)
#%%
#%% Parallel Computing
from multiprocessing import freeze_support
NumProcessor=30
results = []
ConcatData = pd.DataFrame(columns=['x_0','x_1','x_2','x_3','Cost','Converged','LastDateUsed','Liklihood','NumOfiter','Diverged'])

def run_process_pool_executor(start, end):
    global ConcatData
    results = []
    if __name__ == '__main__':
        freeze_support()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(start, end):
                print("iteration number"+str(i)+"started!")
                results.append(executor.submit(BFGS_OneInit,U0[:,i].reshape(4,1),s,R_o,Tmax,Ds_t_final,time_particle,Maxiter,LIKELIHOOD,N_z,N_alpha_max,h,taw,mu1,mu2,sigma,Starting_index,YearInit,gamma))

            for f in concurrent.futures.as_completed(results):
                Data=f.result()
                ConcatData=pd.concat([ConcatData,Data],ignore_index=True)

# Define ranges for the loop

ranges = []
for i in range(0, NumOfInit, NumProcessor):
    ranges.append((i, i+NumProcessor))
    
    
# Execute the loop
for start, end in ranges:
    run_process_pool_executor(start, end)


                  
IsPoisson=0
if LIKELIHOOD=='Poisson':
    IsPoisson=1
    
filepath = Path(r'./../Output/Yearmax%dYearmin%dIsItPoisson%dIsItMonthly%d.csv'%(YearMax,YearMin,IsPoisson,IsMonthly))  
# filepath.parent.mkdir(parents=True, exist_ok=True) 
ConcatData.to_csv(filepath) 
#%%
# # 	x_0	x_1	x_2	x_3
# # 296	5.0202222547100045e-06	275624.6261331457	142.1895878546026	284803.4806973911
# # 	x_0	x_1	x_2	x_3
# # 456	5.052957962750759e-07	1137500.1607795944	22686.472327021296	111945.31845479677

# # 1312	3.7974232671250187e-06	85529.45337705342	16351.602200731737	195341.81638442885
# u_hojj_y=np.array([3.7974232671250187e-06,85529.45337705342,	16351.602200731737,	195341.81638442885])
# u_mateo_y=np.array([4.92e-5 /12,64.5 *12,1.08e3,2.93e5])
# u_mateo=np.array([4.75e-3/12,2.53*12,2.80e+3,3e5])
# u_hojj=np.array([5.052957962750759e-07,1137500.1607795944,	22686.472327021296,	111945.31845479677])
# R_mateo=Functions_spatial.FindR(Ds_t_final[Starting_index:Tmax],u_mateo,time_particle[Starting_index:Tmax])

# R_hojj=Functions_spatial.FindR(Ds_t_final[Starting_index:Tmax],u_hojj,time_particle[Starting_index:Tmax])
# plt.plot(R_hojj,label='H') 
# plt.plot(R_mateo,label='M')
# plt.plot(R_o[Starting_index:Tmax])
# plt.legend()
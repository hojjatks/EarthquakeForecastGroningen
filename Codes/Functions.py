#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 20:08:20 2023

@author: hkaveh
"""

import sys
sys.path.append("../..")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rc
import numpy as np 
import Functions_spatial

# Array and Dataformating
import h5py
# Plotting functions

from pathlib import Path
import math

# Array and Dataformating
import pickle
# Plotting functions

import matplotlib.pylab as plt
from matplotlib.patches import Circle, Wedge, Polygon
import Functions_spatial
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from pathlib import Path
import concurrent.futures
def MakeRforMmax(N_dpoints):
    
    
    YearMax=2021
    YearMin=1979
    LIKELIHOOD='Poisson'
    N_z=int(1990-YearMin)
    Tmax=int(YearMax-1989+N_z)
    # LIKELIHOOD=input("Enter the name of the Liklihood\n")# Name of the Liklihood
    IsPoisson=0
    if LIKELIHOOD=='Poisson':
        IsPoisson=1
        
    filepath = Path(r'../../Output/Yearmax%dYearmin%dIsItPoisson%d.csv'%(YearMax,YearMin,IsPoisson))  
    filepath.parent.mkdir(parents=True, exist_ok=True) 
    Data=pd.read_csv(filepath)
    
    ata=Data.dropna()
    #%
    Data['Likelihood']=(np.exp(Data['Cost']))
    #%
    # Data1=np.loadtxt('ParticlesAtIter1.csv', delimiter=",")
    Data['PDF']=(Data['Likelihood'])/np.max(Data['Likelihood'])

    Data['Scaled_r']=Data['x_0']
    Data['Scaled_ta']=Data['x_1']/1000                  #Kilo year
    Data['Scaled_Asigma']=Data['x_2']/1000              #Kilo Pascal
    Data['Scaled_DeltaSc']=Data['x_3']/1000000          #Mega Pascal
    #% Importing R0
    df=pd.read_excel('../../eventcatalogue23May2022.xlsx')
    A=df[["Datum","Magnitude"]]
    b=pd.to_datetime(df["Datum"],format='%d%m%Y')
    A['year'] = pd.DatetimeIndex(b).year
    A=A[["Magnitude","year"]]
    Year=2022 # Removing year 2022 (Because this year has incomplete measures)
    A=A.drop(A[A["year"]==Year].index)

    Mc=1.5
    A=A.drop(A[A["Magnitude"]<Mc].index)
    Yearly_R0 = np.zeros((2021-1990+1,1))

    for i in range(1,Yearly_R0.size):

        year = i+1990
        print(year)
            
        start = np.min(np.where(A['year']>=year))
        end = np.max(np.where(A['year']<year+1))
        
        Yearly_R0[i] = end-start+1
        print(end-start+1)
        Ds_t_final=np.load('../../Ds_t_final.npy')

    R0=np.append(np.zeros((1,N_z)),Yearly_R0)
    time = np.array(range(R0.size))
    plt.plot(time,R0)
    time_particle=time
    time_total=np.array(range(60))
    
    Start_Time=1990-N_z

    
    
    
    # Normalize the likelihood so it sums to 1
    Data['Norm_likelihood']=Data['Likelihood']/sum(Data['Likelihood'])
    sampled_indices = np.random.choice(len(Data['Norm_likelihood']), size=N_dpoints, replace=False, p=Data['Norm_likelihood'])
    MLE=np.max(Data['Likelihood'])
    flag=0
    fig = plt.figure(figsize=(3.7, 2.8))
    ax = fig.add_subplot(1, 1, 1) 
    R_final=np.empty((60, 0))
    for i in range (Data.shape[0]):
        if Data.iloc[i]['Likelihood']/MLE>=1 and flag==0:
            flag=1
            u=np.array([Data.iloc[i]['x_0'],Data.iloc[i]['x_1'],Data.iloc[i]['x_2'],Data.iloc[i]['x_3']]).reshape(4,1)
            R_pred=Functions_spatial.FindR(Ds_t_final,u,time_total)
            ax.step(time_total+Start_Time,R_pred,alpha=1,color='cyan',label=r'$\mathbf {h}^{MLE}$',linewidth=1.5)    
    for i in sampled_indices:
       
        u=np.array([Data.iloc[i]['x_0'],Data.iloc[i]['x_1'],Data.iloc[i]['x_2'],Data.iloc[i]['x_3']]).reshape(4,1)
        R_pred=Functions_spatial.FindR(Ds_t_final,u,time_total)
        R_final=np.append(R_final,R_pred.reshape(-1,1),axis=1)
        # np.savetxt('R_pred.csv', R_pred, delimiter=',')
        # np.savetxt('Time_pred.csv', time_total, delimiter=',')
        ax.step(time_total+Start_Time,R_pred,alpha=.1,color='red',label=r'$\mathbf {h}^{MLE}$',linewidth=.5,marker='o')    
       
    
    return R_final



#%%


def Sample_N(R,N):
    # R is a vector consists of number of predicted earthquake in each year
    # We assume that R is the mean of te poisson process and we are going to sample many times from it
    # N is the number of samples
    Gen_Rate=np.random.poisson(R,size=(N,R.size))
    # Gen_Rate is a N by R.size matrix 
    Cum_num=np.cumsum(Gen_Rate,axis=1)
    # Cum_num is the cumulative number at each year
    return Cum_num    

def Sample_N2(R_final,N,N_dpoints):
    # The difference with Sample_N is that here R_final is matrix considering different realization of the parameter space
    # R is a vector consists of number of predicted earthquake in each year
    # We assume that R is the mean of te poisson process and we are going to sample many times from it
    # N is the number of samples
    Gen_Rate=np.random.poisson(R_final[:,0],size=(N,R_final.shape[0]))
    for i in range(N_dpoints-1):
        Gen_Rate2=np.random.poisson(R_final[:,i+1],size=(N,R_final.shape[0]))
        Gen_Rate=np.append(Gen_Rate,Gen_Rate2,axis=0)
    # Gen_Rate is a N by R.size matrix 
    Cum_num=np.cumsum(Gen_Rate,axis=1)
    # Cum_num is the cumulative number at each year
    return Cum_num    

def Sample_bplus(M):
    b_plus=np.loadtxt('b_plus.csv',delimiter=',')
    #ax=sns.ecdfplot(b_plus)
    Uniform_num=np.random.uniform(size=(M,1))
    x,y=ecdf(b_plus)
    Sampled_b_plus=np.zeros((M,1))
    for i in range(M):
        Dump=np.absolute(y-Uniform_num[i])
        index = Dump.argmin()
        Sampled_b_plus[i]=x[index]
    #sns.ecdfplot(Sampled_b_plus)
    return Sampled_b_plus

def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)

def FindP(M_thresh,Mc,Cum_num,N,M,Sampled_b_plus,m,N_dpoints):
    # N is size of realization of Poisson Process
    # M is the size of sampled b_plus
    # m is the number of years
    q=np.zeros((M,N*N_dpoints,m))
    for i in range (M):
        # for each b value we calculate the process for all realization of Poisson Process
        # So i iterates over b_plus
        Mmax=Mc+np.log10(Cum_num)/Sampled_b_plus[i]
        alpha=Sampled_b_plus[i]*(Mmax-M_thresh)
        # Used equation 3 in \cite{VanderElst2016InducedExpected} to find p as the probability of having an earthquake above certain magnitude.
        # q[i,:,:]=np.exp(N*np.log(1-(10**alpha)/N))
        q[i,:,:]=np.exp(-10**alpha)
    P=1-q
    return P,q,Mmax,(1-(10**alpha)/N)
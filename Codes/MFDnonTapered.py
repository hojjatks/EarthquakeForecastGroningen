#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:50:06 2023

@author: hkaveh
"""
#%%
import sys
sys.path.append("../..")
#%%
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

import pickle
# Plotting functions

import matplotlib.pylab as plt
from matplotlib.patches import Circle, Wedge, Polygon
import Functions_spatial
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from pathlib import Path
import concurrent.futures

#%%

def MakeRforMmax(N_dpoints):
    
    # What you need to simulate this function:
    # Here I first load the likelihood of the model parameters, I load the time matrix and the stress history
    # Then I sample from the posterior of the model parameters and run the forward model to find the seismicity rate
    # N_dpoints is the number realizations of the model parameters to be considered.
    # To generate rate for different cases you need to import model parameters, time and stress history of your case
    # At the end, R_final has the following size: (NTime,N_dpoints)
    YearMax=2022
    YearMin=1989
    LIKELIHOOD='Poisson'
    N_z=3
    # LIKELIHOOD=input("Enter the name of the Liklihood\n")# Name of the Liklihood
    IsPoisson=0
    IsItmonthly=1
    if LIKELIHOOD=='Poisson':
        IsPoisson=1
        
    filepath = Path(r'../Output/Yearmax%dYearmin%dIsItPoisson%dIsItMonthly%d.csv'%(YearMax,YearMin,IsPoisson,IsItmonthly))  
    filepath.parent.mkdir(parents=True, exist_ok=True) 
    Data=pd.read_csv(filepath)
    
    Data=Data.dropna()
    #%
    Data = Data[Data['Cost'] != 0]

    Data['Likelihood']=(np.exp(Data['Cost']))
    #%
    # Data1=np.loadtxt('ParticlesAtIter1.csv', delimiter=",")
    Data['PDF']=(Data['Likelihood'])/np.max(Data['Likelihood'])

    Data['Scaled_r']=Data['x_0']
    Data['Scaled_ta']=Data['x_1']/1000                  #Kilo year
    Data['Scaled_Asigma']=Data['x_2']/1000              #Kilo Pascal
    Data['Scaled_DeltaSc']=Data['x_3']/1000000          #Mega Pascal
    #% Importing R0

    Ds_t_final=np.load('../Data/Ds_t_final.npy')
    time_total=np.load('../Data/time_total.npy')

    
    Start_Time=1992-N_z

    
    
    
    # Normalize the likelihood so it sums to 1
    Data['Norm_likelihood']=Data['Likelihood']/sum(Data['Likelihood'])
    sampled_indices = np.random.choice(len(Data['Norm_likelihood']), size=N_dpoints, replace=False, p=Data['Norm_likelihood'])
    MLE=np.max(Data['Likelihood'])
    flag=0
    fig = plt.figure(figsize=(3.7, 2.8))
    ax = fig.add_subplot(1, 1, 1) 
    R_final=np.empty((519, 0))
    for i in range (Data.shape[0]):
        if Data.iloc[i]['Likelihood']/MLE>=1 and flag==0:
            flag=1
            u=np.array([Data.iloc[i]['x_0'],Data.iloc[i]['x_1'],Data.iloc[i]['x_2'],Data.iloc[i]['x_3']]).reshape(4,1)
            R_pred=Functions_spatial.FindR(Ds_t_final,u,time_total)
            ax.step(time_total/12+Start_Time,R_pred,alpha=1,color='cyan',label=r'$\mathbf {h}^{MLE}$',linewidth=1.5)    
    for i in sampled_indices:
       
        u=np.array([Data.iloc[i]['x_0'],Data.iloc[i]['x_1'],Data.iloc[i]['x_2'],Data.iloc[i]['x_3']]).reshape(4,1)
        R_pred=Functions_spatial.FindR(Ds_t_final,u,time_total)
        R_final=np.append(R_final,R_pred.reshape(-1,1),axis=1)
        ax.step(time_total/12+Start_Time,R_pred,alpha=.01,color='red',label=r'$\mathbf {h}^{MLE}$',linewidth=.5,marker='o')    
       
    
    return R_final
#%%

def Sample_N2(R_final,N,N_dpoints):
    # The difference with Sample_N is that here R_final is matrix considering different realization of the parameter space
    # R_final is a vector consists of number of predicted earthquake in each time bin
    # We assume that R_final is the mean of te poisson process and we are going to sample many times from it
    # N is the number of samples
    Gen_Rate=np.random.poisson(R_final[:,0],size=(N,R_final.shape[0]))
    for i in range(N_dpoints-1):
        Gen_Rate2=np.random.poisson(R_final[:,i+1],size=(N,R_final.shape[0]))
        Gen_Rate=np.append(Gen_Rate,Gen_Rate2,axis=0)
    # Gen_Rate is a N by R.size matrix 
    Cum_num=np.cumsum(Gen_Rate,axis=1)
    # Cum_num is the cumulative number at each time bin and has the size of (N*N_dpoints,NTime) 
    return Cum_num 

#%%
def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)

def Sample_bplus(M):
    # Here b_plus2.csv is a file containing the time series of the b value (you need this as an input)
    # M is the number of samples for the bvalue
    b_plus=np.loadtxt('../Data/b_plus2.csv',delimiter=',')
    Uniform_num=np.random.uniform(size=(M,1))
    x,y=ecdf(b_plus)
    Sampled_b_plus=np.zeros((M,1))
    for i in range(M):
        Dump=np.absolute(y-Uniform_num[i])
        index = Dump.argmin()
        Sampled_b_plus[i]=x[index]
    return Sampled_b_plus
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx
#%% Plotting Guttenberg and Distribution on top of each other
from scipy import stats as st
from scipy.stats import mode

def PlotGuttenberg_PdfMmax(Sampled_b_plus,Cum_num,N,M,m,Mc,N_mesh,N_realization,Catalog,Year,N_dpoints):
    # This is the last function to plot the MFD and the PDF of M_{max}
    # Inputs:
        # Sampled_b_plus is an array of realization of b-value
        # Cum_num is the output of Sample_N2 function which already has considered the uncertainity in the model parameters and the Poisson process
        # N is the Number of Poisson Process Realization
        # M is the Number of b-plus Realization
        # m is the size of the time vector
        # Mc is the cutoff Magnitude
        # N_mesh is the number of grids in the magnitude, the higher the better
        # N_realization is the number of realizations of the MFD
        # Catalog consists of magnitude of events
        # Year is the last year to be included in the plots
        # N_dpoints is the number of realizations of the model parameters
    ###############################################
    left, width = 0.1, 0.65
    bottom, height_top = 0.1, 0.6
    height_below=.3
    spacing = 0.12
    custom_font='serif'
    FontSize=8   
    rect_top = [left, bottom+height_below+spacing, width, height_top]
    #rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_bottom= [left, bottom,width, height_below]
    fig = plt.figure(figsize=(3.7*2, 3.7))

    ax_top = fig.add_axes(rect_top)
    ax_bottom = fig.add_axes(rect_bottom, sharex=ax_top)
    ##################################################
    Cutend=(Year-1989+1)*12 # Data starts from early 1989 and we want remove data after Year, data in Year is included
    Num=Cum_num[:,Cutend]   # Throwing away data after Year


    M_mesh=np.linspace(Mc,7,N_mesh) # griding the Moments
    np.savetxt('../Data/MeshM.txt',M_mesh) # Saving for Shapiro plot
    A_mean=np.zeros_like(M_mesh)    # initializing an array for expected value line
    Max_mag_array=np.array([])      # initializing an array for recording Maximum magnitudes
    A2_array = np.empty((N*N_dpoints * M * N_realization, N_mesh))
    p=0
    for i in range(N*N_dpoints): # for each Poisson realization and model parameter realization
        for j in range(M):       # and for each b-value realization
            for l in range(N_realization): # for each MFD realization
                A=np.random.exponential(scale=(1/(Sampled_b_plus[j]*np.log(10))),size=(Num[i],1))+Mc # Finding the Magnitudes for one realization
                A2=np.zeros_like(M_mesh)   
                Max_mag_array=np.append(Max_mag_array,np.max(A)) # recording the M_max of the current realization
                
                mask = A[:, np.newaxis] >= M_mesh  # Counting
                A2 = np.sum(mask, axis=0)                  
                A2_array[p] = A2.reshape(M_mesh.shape)  
                p+=1
                ax_top.plot(M_mesh,A2.reshape(M_mesh.shape),linewidth=1,alpha=.002,color="black") # Plotting the black lines in MFD
    A_mean=np.mean(A2_array,axis=0)
    np.savetxt('../Data/Mags'+str(Year)+'NONTaper.txt',A_mean) # Saving for Shapiro plot
    percentile_3 = np.percentile(A2_array, 3, axis=0)
    percentile_97 = np.percentile(A2_array, 97, axis=0)
    ax_top.plot(M_mesh,percentile_3,linewidth=2,color="green",label=r"$3^{rd}$ and $97^{th}$ percentile")
    ax_top.plot(M_mesh,percentile_97,linewidth=2,color="green")

    ax_top.plot(M_mesh,A2.reshape(M_mesh.shape),linewidth=1,alpha=.001,color="black",label="Realizations up to %d"%(Year))
    ax_top.plot(M_mesh,A_mean.reshape(M_mesh.shape),linewidth=3,color="blue",label="$E[N_{>M_w}]$ up to %d"%(Year))

    ax_top.set_yscale("log")
    A2=np.zeros_like(M_mesh)
    # Plotting the observed MFD
    A=Catalog[:,0]
    for k in range(N_mesh):
        A2[k]=np.size(A[A>=M_mesh[k]])    
    ax_top.plot(M_mesh,A2,linewidth=3,color="red",label="Observed $N_{>M_w}$  up to"+str(Year))
    # ax_top.plot(M_mesh,A2,linewidth=3, linestyle='--',color="red",label="Observed $N_{>M_w}$  up to 2022")
    
    ax_top.set_ylabel(r"Number of Events above $M_w$",fontname="Serif",fontsize=12)
    leg = ax_top.legend(fontsize=FontSize,frameon=0)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    ##################################################%%%%%%%%% The other plot


    Percent97=np.percentile(Max_mag_array, 97) 
    Percent3=np.percentile(Max_mag_array, 3) 
    Percent50=M_mesh[find_nearest(A_mean,1)] # The value which the top plot hit one
    
    
 

    ax_top.set_ylabel(r"Number of events greater than $M_w$ ($N_{>M_w}$)",fontname=custom_font,fontsize=FontSize)

    sns.distplot(Max_mag_array,hist=False,label="PDF of $M_{max}$ up to %d"%(Year),ax=ax_bottom,color='black')
    plt.axvline(x = np.max(A), linewidth=2,color = 'red', label = r'$M_{max}^o$ untill ' + str(Year) +' is %1.2f'%np.max(A))
    # plt.axvline(x = np.max(A), linewidth=2,linestyle='--',color = 'red', label = r'$M_{max}^o$ untill 2022' + ' is %1.2f'%np.max(A))

    plt.axvline(x = Percent97, linewidth=2,color = 'green', label = r'$\Delta M^{0.97}$=%1.2f'%Percent97)
    plt.axvline(x = Percent3, linewidth=2,color = 'green', label = r'$\Delta M^{0.03}$=%1.2f'%Percent3)
    plt.axvline(x = Percent50, linewidth=2,color = 'blue', label = r'$\hat{M}_{max}=$%1.2f'%Percent50)

    leg = ax_bottom.legend(fontsize=FontSize,frameon=0,loc="lower left")

    xticks = ax_top.get_xticklabels()
    for ticks in xticks:     
        ticks.set_visible(False)
    ax_top.set_xlim(left=1.1,right=7)
    ax_bottom.invert_yaxis()
    ax_top.set_ylim(bottom=.8,top=1000)
    ax_bottom.set_xlabel(r"Magnitude($M_w$)",fontname=custom_font,size=FontSize)
#Set axis to top
    ax_bottom.xaxis.tick_top()
#Set x axis lable to top
    ax_bottom.xaxis.set_label_position('top') 
    # ax_bottom.set_ylim(bottom=.9)
    ax_bottom.set_ylabel('PDF',fontname=custom_font,size=FontSize)
    for ticks in ax_bottom.get_xticklabels():
        ticks.set_fontname(custom_font)
        ticks.set_fontsize(FontSize)

    for ticks in ax_bottom.get_yticklabels():
        ticks.set_fontname(custom_font)  
        ticks.set_fontsize(FontSize)

    for ticks in ax_top.get_yticklabels():
        ticks.set_fontname(custom_font)  
        ticks.set_fontsize(FontSize)
    # ax_top.text(5.75,20,"(a1)",fontname=custom_font,size=FontSize-1)
    # ax_bottom.text(5.75,.45,"(a2)",fontname=custom_font,size=FontSize-1)
    
    fig.savefig("../Figs/GuttenbergAndPDF_YearMax="+str(Year)+".png", bbox_inches = 'tight',dpi=600)
    return A,Num,A2


#%%
import pandas as pd
Year=2030 # Included
Mc=1.1
N_mesh=100
N_realization=10

file_path = '../Data/KNMI_CAT_1991-12to2023-01_Polygon1.15Reservoir.csv'

data =  pd.read_csv(file_path, delimiter=',')
A=data[["magnitude","DecDates"]]
B=A.drop(A[A["DecDates"]>Year+1].index)
B=B.drop(B[B["magnitude"]<1.1].index)

Catalog=B.to_numpy()

# N_realization=5  # Number of GR realization
# N=5 # Number of Poisson Process Realization
# N_dpoints=5 # Number of model parameters to include
# M=5 # Number of b-plus Realization
# Sampled_b_plus=Sample_bplus(M)

N_realization=10  # Number of GR realization
N=20 # Number of Poisson Process Realization
N_dpoints=20 # Number of model parameters to include
M=50 # Number of b-plus Realization
Sampled_b_plus=Sample_bplus(M)


R_final=MakeRforMmax(N_dpoints)

Cum_num=Sample_N2(R_final,N,N_dpoints)
m=R_final.shape[0]

A,Num,A2=PlotGuttenberg_PdfMmax(Sampled_b_plus,Cum_num,N,M,m,Mc,N_mesh,N_realization,Catalog,Year,N_dpoints)

# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:35:07 2024

@author: hkaveh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: hkaveh
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import Functions_spatial
import matplotlib
from pathlib import Path
from scipy.stats import chi2
import matplotlib.font_manager
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

#%% Loading the best Parameters that are learned.

Move_crossvalid_window=0
YearMax=2022+Move_crossvalid_window
YearMin=1989+Move_crossvalid_window
LIKELIHOOD='Poisson'
YearInit=1979
IsPoisson=1
if LIKELIHOOD=='Poisson':
    IsPoisson=1
IsMonthly=1   
filepath = Path(r'../Output/Yearmax%dYearmin%dIsItPoisson%dIsItMonthly%d.csv'%(YearMax,YearMin,IsPoisson,IsMonthly))   # Directory of the parameters that are found as output of FindParameters.py
filepath.parent.mkdir(parents=True, exist_ok=True) 
Data=pd.read_csv(filepath)

#%% Cleaning the imported data and adding a few columns
Data=Data.dropna()
Data = Data[Data['Cost'] != 0]

Data['Likelihood']=(np.exp(Data['Cost']))
# Data1=np.loadtxt('ParticlesAtIter1.csv', delimiter=",")
Data['PDF']=(Data['Likelihood'])/np.max(Data['Likelihood'])

Data['Scaled_r']=Data['x_0']
Data['Scaled_ta']=Data['x_1']/1000                  #Kilo year
Data['Scaled_Asigma']=Data['x_2']/1000              #Kilo Pascal
Data['Scaled_DeltaSc']=Data['x_3']/1000000          #Mega Pascal
#%% Import R0, time, stress and plottimg 
NumOfyearstoadd=3
if IsMonthly==1:
    df = pd.read_csv('../Data/R_monthly.csv', header=None, names=['date', 'value'], skiprows=1)
    Time_monthly=np.load("../Data/time_monthly.npy")
# R_monthly=np.loadtxt('R_monthly.csv', delimiter=',')
    Stress_monthly=np.load("../Data/max_coulomb_stresses.npy",allow_pickle=True).item()['-10m']
    max_coulomb_stress2 = np.nan_to_num(Stress_monthly) # Replacing NAN with zeros
    smoothed_coulomb_stress2 = np.zeros(max_coulomb_stress2.shape)
    kernel = Gaussian2DKernel(8)
    for i in range (Stress_monthly.shape[2]):
        smoothed_coulomb_stress2[:,:,i] = convolve(max_coulomb_stress2[:,:,i],kernel,nan_treatment='fill')# smoothen the coulomb stress
    
    Stress_monthly=smoothed_coulomb_stress2
    Time_monthly=Time_monthly[:Stress_monthly.shape[2]]
else:
    Time_monthly=np.load("../../Data/time_yearlysmoothed.npy")
# R_monthly=np.loadtxt('R_monthly.csv', delimiter=',')
    Stress_monthly=np.load("../../Data/Coulomb_yearlysmoothed.npy")

    df = pd.read_csv('../../Data/R_yearlysmoothed.csv', header=None, names=['date', 'value'], skiprows=1)
    
# Split the date column into year, month, and day columns
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
df['day'] = pd.to_datetime(df['date']).dt.day
data = np.array(df[['year', 'month', 'day', 'value']])
Time_R=  df['year']  +  df['month']/12-1/12

# Find index of element in A closest to B[0]
idx = np.abs(Time_monthly - Time_R[0]).argmin()
Monthly_R=np.array(df['value'])



thresh=0.08 # Should be same as in FindParameters.py
mask = Functions_spatial.Get_mask (Stress_monthly, thresh)

Ds_t = Functions_spatial.Organize_Dst (Stress_monthly, mask)
idx = np.abs(Time_monthly - Time_R[0]).argmin() # This is the index of the time of stress  that the time of seimicity and time of stress are equal
N_z=NumOfyearstoadd*12
Ds_t_final = Ds_t[idx-N_z:,:]*1e6 # Note that this contains stresses up to almost 2030


R0=np.append(np.zeros((1,N_z)),Monthly_R)
time = np.array(range(R0.size))
plt.plot(time,R0)
time_particle=time
time_total=np.array(range(Ds_t_final.shape[0]))
plt.plot(time,R0)
plt.title("Catalog")
plt.xlabel("index of month")

#%%

#%%


Start_Time=Time_R[0]-NumOfyearstoadd
def PostProcessV2(Data,alpha,time_total,Ds_t_final,Start_Time,time_particle,R0,YearMax,IsMonthly):
    custom_font='Serif'
    FontSize=8

    NumInInterval=0 # Number of Points in the alpha interval
    MLE=np.max(Data['Likelihood'])
    matplotlib.rcParams['font.family'] = 'serif'
    #plt.style.use('default')
    fig = plt.figure(figsize=(3.7, 2.8))

    SizeTime=np.size(time_total)
    plt.rc('font',family='Serif')
    plt.rcParams.update({'font.family':'Serif'})

    ax = fig.add_subplot(1, 1, 1)

    R_concat=np.empty_like(time_total.reshape(SizeTime,1))
    flag=0
    for i in range (Data.shape[0]):
        if Data.iloc[i]['Likelihood']/MLE>=alpha:
            NumInInterval+=1
            u=np.array([Data.iloc[i]['x_0'],Data.iloc[i]['x_1'],Data.iloc[i]['x_2'],Data.iloc[i]['x_3']]).reshape(4,1)
            R_pred=Functions_spatial.FindR(Ds_t_final,u,time_total)
            ax.step(time_total/12+Start_Time,R_pred,color='red',alpha=.01)
            R_concat=np.append(R_concat,R_pred.reshape(SizeTime,1),axis=1)
            #print(R_pred.shape)
    ax.step(time_total/12+Start_Time,R_pred,alpha=0.01, label=r'$\mathbf {h}^{99\%}$',color='red',linewidth=1.5)
    for i in range (Data.shape[0]):
        if Data.iloc[i]['Likelihood']/MLE>=1 and flag==0:
            flag=1
            u=np.array([Data.iloc[i]['x_0'],Data.iloc[i]['x_1'],Data.iloc[i]['x_2'],Data.iloc[i]['x_3']]).reshape(4,1)
            print(u)
            R_pred=Functions_spatial.FindR(Ds_t_final,u,time_total)

            ax.step(time_total/12+Start_Time,R_pred,alpha=1,color='cyan',label=r'$\mathbf {h}^{MLE}$ non-seasonal',linewidth=1.5)    
    

    MinR=np.min(R_concat[:,1:],axis=1)            
    MaxR=np.max(R_concat[:,1:],axis=1)

    Confmin=np.zeros(np.shape(MinR))
    Confmax=np.zeros(np.shape(MaxR))
    Confinter=.91    #*100= percent confidence in the aleatoric uncertainty
    alpha_poisson=1-Confinter
    
    for i in range(MinR.size):
        
        Confmin[i] =   +chi2.ppf(alpha_poisson/2,2*MinR[i])/2
        
        Confmax[i]=   +  chi2.ppf(1-alpha_poisson/2,2*(MaxR[i]+1))/2
    
    
    
    Minus2Sigma=MinR-2*np.sqrt(MinR)
    Plus2Sigma=MaxR+2*np.sqrt(MaxR)
    for i in range(Minus2Sigma.size):
        if Minus2Sigma[i]<0:
            Minus2Sigma[i]=0
    # ax.step(time_total+Start_Time,MinR,color='green',label=r"alpha=%1.2f"%alpha)    
    # ax.step(time_total+Start_Time,MaxR,color='green') 
    #ax.step(time_total+Start_Time,Minus2Sigma,'Black',label=r"$90\%$ Confidence Interval")
    #ax.step(time_total+Start_Time,Plus2Sigma,'Black')

    #ax.step(time_total/12+Start_Time,Confmin,'Black',label=r"$90\%$ Confidence")
    #ax.step(time_total/12+Start_Time,Confmax,'Black')
    
    
    ax.step(x=time_particle/12+Start_Time, y=R0,linewidth=2,color='blue',label='Catalog')
    
    plt.axvspan(YearMax+1,2030, color='black', alpha=0.5, lw=0)
    EndingX=2030
    plt.xlim([time_total[0]+Start_Time,EndingX])
    #plt.xlim([2010,2016])
    leg = ax.legend(fontsize=FontSize,frameon=0)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    #ax.set_xticks(fontname="serif")
    plt.yticks(fontname="serif")
    ax.set_xlabel(r"Time (year)",fontname=custom_font,size=FontSize)
    ax.set_ylabel(r"Number/rate of events",fontname=custom_font,size=FontSize)
    for ticks in ax.get_xticklabels():
        ticks.set_fontname(custom_font)
    for ticks in ax.get_yticklabels():
        ticks.set_fontname(custom_font)   

    plt.show()
    fig.savefig('../Figs/PredictionBeyondIsMonthly%d_1.png'%(IsMonthly), bbox_inches = 'tight',dpi=600)
    return NumInInterval,R_concat,MaxR,R_pred
#%% Adjusting the level of uncertainty
 
Beta=.01 # determines the level of uncertainty in the model parameters (99 percent confidence)
expr= lambda x:  1-Beta-chi2.cdf(x,4)
from scipy.optimize import fsolve
y =  fsolve(expr, 1)
alpha=np.exp(-1/2*y)
#%%
N,R_concat,MaxR,R_MLE=PostProcessV2(Data, alpha,time_total,Ds_t_final,Start_Time,time_particle,R0,YearMax,IsMonthly)    
#%%

def FindYearlyAve(time,values):
    years = (np.floor(time))
    yearly_averages = (np.zeros(len(np.unique(years))))
    for i, year in enumerate(np.unique(years)):
        mask = years == year
        yearly_averages[i] = np.mean(values[mask])
    return np.unique(years),yearly_averages
#%%
def PlotAverage(Data,alpha,time_total,Ds_t_final,Start_Time,time_particle,R0,YearMax,IsMonthly):
    custom_font='Serif'
    FontSize=8
    # Finding number of unique years:
    T=time_total/12+Start_Time
    Num=int(int(T.max())-int(T.min()))+1 # Number of unique years

    NumInInterval=0 # Number of Points in the alpha interval
    MLE=np.max(Data['Likelihood'])
    matplotlib.rcParams['font.family'] = 'serif'
    #plt.style.use('default')
    fig = plt.figure(figsize=(3.7, 2.8))

    SizeTime=np.size(time_total)
    plt.rc('font',family='Serif')
    plt.rcParams.update({'font.family':'Serif'})

    ax = fig.add_subplot(1, 1, 1)

    flag=0
    
    R_concat=np.empty((Num,1))
    for i in range (Data.shape[0]):
        if Data.iloc[i]['Likelihood']/MLE>=alpha:
            NumInInterval+=1
            u=np.array([Data.iloc[i]['x_0'],Data.iloc[i]['x_1'],Data.iloc[i]['x_2'],Data.iloc[i]['x_3']]).reshape(4,1)
            R_pred=Functions_spatial.FindR(Ds_t_final,u,time_total)
            
            TPlot,RPlot=FindYearlyAve(time_total/12+Start_Time,R_pred)
            
            ax.step(TPlot,RPlot,color='red',alpha=.01)
            R_concat=np.append(R_concat,RPlot.reshape(Num,1),axis=1)
    TPlot,RPlot=FindYearlyAve(time_total/12+Start_Time,R_pred)
    ax.step(TPlot,RPlot,alpha=0.01, label=r'$\mathbf {h}^{99\%}$',color='red',linewidth=1.5)


    for i in range (Data.shape[0]):
        if Data.iloc[i]['Likelihood']/MLE>=1 and flag==0:
            flag=1
            u=np.array([Data.iloc[i]['x_0'],Data.iloc[i]['x_1'],Data.iloc[i]['x_2'],Data.iloc[i]['x_3']]).reshape(4,1)
            print(u)
            R_pred=Functions_spatial.FindR(Ds_t_final,u,time_total)
            TPlot,RPlot=FindYearlyAve(time_total/12+Start_Time,R_pred)
            ax.step(TPlot,RPlot,alpha=1,color='cyan',label=r'$\mathbf {h}^{MLE}$ seasonal',linewidth=1.5)    
    


    MinR=np.min(R_concat[:,1:],axis=1)            
    MaxR=np.max(R_concat[:,1:],axis=1)

    Confmin=np.zeros(np.shape(MinR))
    Confmax=np.zeros(np.shape(MaxR))
    Confinter=.91 #*100= percent confidence in the aleatoric uncertainty
    alpha_poisson=1-Confinter
   
    for i in range(MinR.size):
       
        Confmin[i] =   +chi2.ppf(alpha_poisson/2,2*MinR[i])/2
       
        Confmax[i]=   +  chi2.ppf(1-alpha_poisson/2,2*(MaxR[i]+1))/2
    Minus2Sigma=MinR-2*np.sqrt(MinR)/np.sqrt(12)
    Plus2Sigma=MaxR+2*np.sqrt(MaxR)/np.sqrt(12)
    for i in range(Minus2Sigma.size):
        if Minus2Sigma[i]<0:
            Minus2Sigma[i]=0
            
    ax.step(TPlot,Minus2Sigma,'Black',label=r"$90\%$ Confidence" "\n" "Interval") # 0.9=0.91*0.99
    ax.step(TPlot,Plus2Sigma,'Black')

    
    TPlot,RPlot=FindYearlyAve(time_particle/12+Start_Time,R0)
    ax.step(x=TPlot[:-1], y=RPlot[:-1],linewidth=2,color='blue',label='Catalog')
    
    plt.axvspan(YearMax,2030, color='black', alpha=0.5, lw=0)
    EndingX=2030
    plt.xlim([time_total[0]+Start_Time,EndingX])
    leg = ax.legend(fontsize=FontSize,frameon=0)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    plt.yticks(fontname="serif")
    ax.set_xlabel(r"Time (year)",fontname=custom_font,size=FontSize)
    ax.set_ylabel(r"Number/rate of events",fontname=custom_font,size=FontSize)
    for ticks in ax.get_xticklabels():
        ticks.set_fontname(custom_font)
    for ticks in ax.get_yticklabels():
        ticks.set_fontname(custom_font)   

    plt.show()
    fig.savefig('../Figs/PredictionBeyondIsMonthly%d_2.png'%(IsMonthly), bbox_inches = 'tight',dpi=600)
    return NumInInterval,R_concat,R_pred

N,R_concat,R_MLE=PlotAverage(Data, alpha,time_total,Ds_t_final,Start_Time,time_particle,R0,YearMax,IsMonthly)    
#%%

#%%
np.save("../Data/Ds_t_final.npy",Ds_t_final )
np.save("../Data/time_total.npy",time_total )
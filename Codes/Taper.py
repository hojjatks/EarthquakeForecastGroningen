#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:15:03 2023

@author: hkaveh
"""
import sys
sys.path.append("../..")
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy.optimize import minimize
import Functions
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import seaborn as sns
from pathlib import Path
import Functions_spatial
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
DeltaM=.00001       # discritization level
c=9.1           # global constant from Bourne, S. J., and S. J. Oates. 2020. “Stress-Dependent Magnitudes of Induced Earthquakes in the Groningen Gas Field.” Journal of Geophysical Research: Solid Earth 125(11): e2020JB020013.
d=1.5           # global constant from Bourne, S. J., and S. J. Oates. 2020. “Stress-Dependent Magnitudes of Induced Earthquakes in the Groningen Gas Field.” Journal of Geophysical Research: Solid Earth 125(11): e2020JB020013.
M_c=1.1         # Cutoff Magnitude for sampling
M_m=np.exp( (c+d*( M_c - 1/2 * DeltaM) ) *np.log(10) ) # Eq 15 from Bourne, S. J., and S. J. Oates. 2020. “Stress-Dependent Magnitudes of Induced Earthquakes in the Groningen Gas Field.” Journal of Geophysical Research: Solid Earth 125(11): e2020JB020013.
  
def loglikelihood(M_m,Data,beta,zeta):
  # Import Data
  # Import Data
    # Data=ImportData(c, d)
    ll=0
    N=len(Data)
    Ms=np.asarray(Data["Mag"])
    
    for i in range(N):
        ll+= np.log(beta+zeta*Ms[i]/M_m)  -(1+beta)*np.log(Ms[i]/M_m)-zeta*(Ms[i]/M_m-1)
    # ll=np.sum(np.log(beta+zeta*Ms/M_m)  -(1+beta)*np.log(Ms/M_m)-zeta*(Ms/M_m-1))
    return ll

def ImportData(c,d):
    # Importing Catalog and find Mag from Bourne, S. J., and S. J. Oates. 2020.
    file_path = '../Data/KNMI_CAT_1991-12to2023-01_Polygon1.15Reservoir.csv'
    data =  pd.read_csv(file_path, delimiter=',')
    
    A2=data[["magnitude","DecDates"]]

    # df=pd.read_excel('../../eventcatalogue23May2022.xlsx')
    # A=df[["Datum","Magnitude"]]
    # b=pd.to_datetime(df["Datum"],format='%d%m%Y')
    # A['year'] = pd.DatetimeIndex(b).year
    # A=A[["Magnitude","year"]]
    # Year=2022 # Removing year 2022 (Because this year has incomplete measures)
    # A=A.drop(A[A["year"]==Year].index)
    Mc=1.1
    A2=A2.drop(A2[A2["magnitude"]<Mc].index)
    A2["Mag"]=10**(c+d*A2["magnitude"])
    return A2

#%%
def OptimizeBeta_Zeta(c,d):
    Data=ImportData(c,d)

    with pm.Model() as model:
    
        beta = pm.Uniform("beta", lower=.3,upper=1)
        zeta = pm.Uniform("zeta", lower=0, upper=.1)
    # Define the log-likelihood as a PyMC3 stochastic variable
        likelihood_var = pm.Potential("likelihood", loglikelihood(M_m,Data,beta, zeta))
        trace = pm.sample(2000, tune=1000)

# Plot the posterior distributions of the model parameters
    az.plot_posterior(trace, var_names=["beta", "zeta"])
    return trace

#%%
# beta_init = .64
# zeta_init = 1.2e-3

# # Use the Nelder-Mead optimization algorithm to maximize the log likelihood
# result = minimize(lambda x: -loglikelihood(Data,x[0], x[1]), [beta_init, zeta_init], method='nelder-mead')

# # Extract the optimal values of beta and zeta
# beta_opt = result.x[0]
# zeta_opt = result.x[1]

# # Print the optimal values
# print("Optimal beta: ", beta_opt)
# print("Optimal zeta: ", zeta_opt)    

def FindP2(N_beta,N_zeta,N,N_dpoints,m,Beta,Zeta,Mag,M_m,Cum_num):
    # N is size of realization of Poisson Process
    # N_beta is the size of beta
    # N_zeta is the size of zeta
    # m is the number of years
    # N_dpoints is the number of parameter space
    P=np.zeros((N_beta,N_zeta,N*N_dpoints,m))
    
    for i in range (N_beta):
        for j in range (N_zeta):
            p1=(Mag/M_m)**(-Beta[i])*np.exp(-Zeta[j]*(Mag/M_m-1)) # Probability of one realization to be greater than M
            print(p1)
            q1=1-p1 # Probability that one realizaion is smaller than M
            Q1=(q1)**Cum_num # Probability that all the events have magnitude smaller than M
            P[i,j,:,:]=1-Q1 # Probability that at least one event has magnitude greater than M
        
   
    return P
#%% Importing Catalog for plots:
import pandas as pd
def CleanCatalog():
    file_path = '../Data/KNMI_CAT_1991-12to2023-01_Polygon1.15Reservoir.csv'
    data =  pd.read_csv(file_path, delimiter=',')
    
    A=data[["magnitude","DecDates"]]
    
    B=A
    B=B.drop(B[B["magnitude"]<1.1].index)
    Catalog=B.to_numpy()
    return Catalog
#%%
# N=5 # Number of PoissonProcessRealization
# N_dpoints=1 # Number of model parameters to include
# R_final=MakeRforMmax(N_dpoints)
# N_beta=10
# N_zeta=10
# m=R_final.shape[0]
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

def find_Cum_num(N_dpoints,N):
    
    R_final=MakeRforMmax(N_dpoints)
    
    Cum_num=Sample_N2(R_final,N,N_dpoints)
    return Cum_num
 
# Cum_num=find_Cum_num(N_dpoints,N)
# M_thresh=3

# Mag=10**(c+d*M_thresh)
#%%
# P=FindP2(N_beta,N_zeta,N,N_dpoints,m,Beta,Zeta,Mag,M_m,Cum_num)
# Average_P=np.mean(P,axis=(0,1,2))
# Time=np.loadtxt('Time_pred.csv', delimiter=",")

def PlotMultipleP(M_thresh_vector,Time,N_beta,N_zeta,N,N_dpoints,m,Mag,M_m,Cum_num):
    trace=OptimizeBeta_Zeta(c, d)
    beta_samples = trace.posterior['beta'].values.flatten()[:N_beta]
    zeta_samples = trace.posterior['zeta'].values.flatten()[:N_zeta]
    FontName="Serif"
    FontSize=8
    fig=plt.figure(figsize=(3.7,3))
    ax=fig.add_axes([.1,.1,.9,.9])
    plt.rcParams.update({'font.family':'serif'})
    Time=Time+1979
    Size_M_thresh=M_thresh_vector.size
    # text=
    for j in range (Size_M_thresh):
        Mag=10**(c+d*M_thresh_vector[j])

        P=FindP2(N_beta,N_zeta,N,N_dpoints,m,Beta,Zeta,Mag,M_m,Cum_num)
        Average_P=np.mean(P,axis=(0,1,2))
        ax.plot(Time,Average_P,linewidth=2,linestyle='--')
        # text=text.append(r"M>%1.1f"%(M_thresh_vector[j]))
#        ax.legend(r"M>%1.1f" %(M_thresh_vector[j]))
    ax.set_xlabel("Time (year)",fontname=FontName,fontsize=FontSize)
    
    ax.set_ylabel(r"$E[P(\hat{M}_{max}>\hat{M}_q)] $",fontname=FontName,fontsize=FontSize)
  #  ax.tick_params(fontname="Times new roman")
#    ax.set_title("M>%1.1f" %(M_thresh),fontname="Times new roman")
    plt.xticks(fontname="Serif")
    plt.yticks(fontname="Serif")

    # plt.legend(text)
    ax.set_xlim(left=1985,right=2030)    
    ax.legend([r"$\hat{M}_q=$%1.1f"%(M_thresh_vector[0]),r"$\hat{M}_q=$%1.1f"%(M_thresh_vector[1]),r"$\hat{M}_q=$%1.1f"%(M_thresh_vector[2]),r"$\hat{M}_q=$%1.1f"%(M_thresh_vector[3]),r"$\hat{M}_q=$%1.1f"%(M_thresh_vector[4])],fontsize=FontSize,frameon=0)
    
    
    
    
    for ticks in ax.get_xticklabels():
        ticks.set_fontname(FontName)
        ticks.set_fontsize(FontSize)
    
    for ticks in ax.get_yticklabels():
        ticks.set_fontname(FontName)  
        ticks.set_fontsize(FontSize)
    fig.savefig("TestTaper.png", bbox_inches = 'tight',dpi=700)

    return
M_thresh_vector=np.array([3,3.3,3.6,3.8,4.2])
# PlotMultipleP(M_thresh_vector,Time,N_beta,N_zeta,N,N_dpoints,m,Mag,M_m,Cum_num)
#%%

def CDF_taper(x,M_m,beta,zeta):
    # CDF = 1-P(>=x)
    return 1- ((x/M_m)**(-beta)) * np.exp(-zeta*((x/M_m)-1))

def FindMags(NumEvent,M_m,beta,zeta,c,d):
    # Finding Number Of events
    
    rnds=np.random.uniform(low=0.0, high=1.0, size=NumEvent) # Randomely sample from zero to one
    Mags=np.zeros((NumEvent,1))
    for i in range(NumEvent):
        
        Mags[i]=fsolve(lambda x: CDF_taper(x,M_m,beta,zeta)-rnds[i],M_m)
        
    return (1/d)*(np.log10(Mags)-c)




def Run(Year,N_beta,N_zeta,trace,M_m,N_mesh,N_realization,c,d,Catalog,Cum_num):
    Cutend=(Year-1989+1)*12 # Data starts from early 1989 and we want remove data after Year, data in Year is included

    TotNumEventsRealiz=Cum_num[:,Cutend] # Realiozations of total number of events Including Model Parameters and poisson process
    Betas = trace.posterior['beta'].values.flatten()[:N_beta]
    Zetas = trace.posterior['zeta'].values.flatten()[:N_zeta]
 
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
    M_c=(1/d)*(np.log10(M_m)-c) # This is the M_c corresponds to M_m it is close to M_c
    M_mesh=np.linspace(M_c,6,N_mesh)
    np.savetxt('../Data/MeshMtaper.txt',M_mesh) # Saving for Shapiro plot
    A_mean=np.zeros_like(M_mesh)    
    Max_mag_array=np.array([]) # Array containing MMaxs of all realizations
    p=0
    A2_array = np.empty((TotNumEventsRealiz.size*N_beta * N_zeta* N_realization, N_mesh))
    for i in range(TotNumEventsRealiz.size):
        for j in range(N_beta):
            for h in range(N_zeta):
                for l in range(N_realization):
                    A=FindMags(TotNumEventsRealiz[i],M_m,Betas[j],Zetas[h],c,d)
                    A2=np.zeros_like(M_mesh)
                    Max_mag_array=np.append(Max_mag_array,np.max(A))
                    mask = A[:, np.newaxis] >= M_mesh  # Counting
                    A2 = np.sum(mask, axis=0)  
                    A2_array[p] = A2.reshape(M_mesh.shape)  
                    p+=1
                    
                    # for k in range(N_mesh):
                    #     A2[k]=np.size(A[A>=M_mesh[k]])
                    #     A_mean[k]+=A2[k]/(TotNumEventsRealiz.size*N_beta*N_zeta*N_realization)
                        
                    ax_top.plot(M_mesh,A2.reshape(M_mesh.shape),linewidth=1,alpha=0.005,color="black")
    A_mean=np.mean(A2_array,axis=0)
    np.savetxt('../Data/Mags'+str(Year)+'Taper.txt',A_mean) # Saving for Shapiro plot
    percentile_3 = np.percentile(A2_array, 3, axis=0)
    percentile_97 = np.percentile(A2_array, 97, axis=0)
    ax_top.plot(M_mesh,percentile_3,linewidth=2,color="green",label=r"$3^{rd}$ and $97^{th}$ percentile")
    ax_top.plot(M_mesh,percentile_97,linewidth=2,color="green")
    
    ax_top.plot(M_mesh,A2.reshape(M_mesh.shape),linewidth=1,alpha=.005,color="black",label="Realizations up to %d"%(Year))
    ax_top.plot(M_mesh,A_mean,linewidth=2,alpha=.9,color="blue",label="$E[N_{>M_w}]$ up to %d"%(Year))
    ax_top.set_yscale("log")
    A2=np.zeros_like(M_mesh)
    A=Catalog[:,0]
    for k in range(N_mesh):
        A2[k]=np.size(A[A>=M_mesh[k]])    
    # ax_top.plot(M_mesh,A2,linewidth=3, linestyle='--',color="red",label="Observed $N_{>M_w}$  up to 2022")
    ax_top.plot(M_mesh,A2,linewidth=3,color="red",label="Observed $N_{>M_w}$  up to" + str(Year))
    #ax_top.set_xlabel("Magnitude",fontname="Times new roman",fontsize=12)
    ax_top.set_ylabel(r"Number of events greater than $M_w$ ($N_{>M_w}$)",fontname=custom_font,fontsize=FontSize)
    leg = ax_top.legend(fontsize=FontSize,frameon=0)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    
        
    Percent95=np.percentile(Max_mag_array, 97) 
    Percent5=np.percentile(Max_mag_array, 3) 
    Mean=np.mean(Max_mag_array) 
    sns.distplot(Max_mag_array,hist=False,label="PDF of $M_{max}$ up to %d"%(Year),ax=ax_bottom,color='black')
    plt.axvline(x = np.max(A), linewidth=2,color = 'red', label = r'$M_{max}^o$ untill ' + str(Year) +' is %1.2f'%np.max(A))
    # plt.axvline(x = np.max(A), linewidth=2,color = 'red', linestyle='--', label = r'$M_{max}^o$ untill 2022 is %1.2f'%np.max(A))

    plt.axvline(x = Percent95, linewidth=2,color = 'green', label = r'$\Delta M^{0.97}$=%1.1f'%Percent95)
    plt.axvline(x = Percent5, linewidth=2,color = 'green', label = r'$\Delta M^{0.03}$=%1.1f'%Percent5)
    plt.axvline(x = Mean, linewidth=2,color = 'blue', label = r'$\hat{M}_{max}$=%1.2f'%Mean)
    leg = ax_bottom.legend(fontsize=FontSize,frameon=0,loc='upper left')

    xticks = ax_top.get_xticklabels()
    for ticks in xticks:     
        ticks.set_visible(False)
    ax_top.set_xlim(left=1.1,right=6)
    ax_bottom.invert_yaxis()
    ax_top.set_ylim(bottom=.8,top=1000)
    ax_bottom.set_xlabel(r"Magnitude($M_w$)",fontname=custom_font,size=FontSize)
#Set axis to top
    ax_bottom.xaxis.tick_top()
#Set x axis lable to top
    ax_bottom.xaxis.set_label_position('top') 
    ax_bottom.set_ylabel('PDF',fontname=custom_font,size=FontSize)
    #ax_bottom.xaxis.set_tick_params(labeltop='on',labelbottom='off')
    for ticks in ax_bottom.get_xticklabels():
        ticks.set_fontname(custom_font)
        ticks.set_fontsize(FontSize)

    for ticks in ax_bottom.get_yticklabels():
        ticks.set_fontname(custom_font)  
        ticks.set_fontsize(FontSize)

    for ticks in ax_top.get_yticklabels():
        ticks.set_fontname(custom_font)  
        ticks.set_fontsize(FontSize)
    # ax_top.text(5.75,20,"(b1)",fontname=custom_font,size=FontSize-1)
    # ax_bottom.text(5.75,.45,"(b2)",fontname=custom_font,size=FontSize-1)
    
    fig.savefig("../Figs/GuttenbergTaperedAndPDF_YearMax="+str(Year)+".png", bbox_inches = 'tight',dpi=600)     
    
    return Mean

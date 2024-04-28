# -*- coding: utf-8 -*-


import numpy as np
from scipy import integrate
import matplotlib.pylab as plt
import math
def Get_mask (DSt, thresh):
    # Dst has size X*Y*t
    
    avg_ds = np.mean(DSt,axis=2)
    mask = np.ones_like(avg_ds)
    mask[avg_ds<thresh] = 0
    
    return mask
def Get_mask_notsmoothdata (DSt, thresh):
    # Dst has size X*Y*t
    
    #avg_ds = np.mean(DSt,axis=2)
    mask = np.ones_like(DSt[:,:,1])
    mask[np.isnan(DSt[:,:,-1])] = 0
    
    return mask
def Organize_Dst (Dst, mask):
    # Dst has size X*Y*t
    # mask is 0/1 mattrix of size X*Y 

    Dst_new = Dst[mask==1]
    
    return Dst_new.T

def GetU0_Normal (N,u,u_min,u_max):
    
    u_range=u_max-u_min 
    u_sigma=u_range/5
    
    cov = np.diag(u_sigma)
    
    U0 = np.random.multivariate_normal(u,cov,N)
    
    return U0

def GetU0_Uniform (N,M,u_min,u_max):
    
    return np.random.uniform(u_min,u_max,[N,M])

def ParticleFilter (u_min,u_max,U0,R0,Ds_t,time,iters,likelihood):
    N,M = U0.shape
    # U_history = np.zeros((iters+1,N,M))
    # U_history[0,:,:] = U0
    # for i in range (iters):
    #     print(i)
    #     U_history[i+1,:,:] = ParticleFilterStep (U_history[i,:,:],R0,Ds_t,time,u_min,u_max)
    
    # return U_history

    U = U0
    i=1
    np.savetxt('ParticlesAtIter%d.csv'%(i), U, delimiter=",")
    for i in range (iters):
        print(i)
        # This was originally 40
        u_sigma = (u_max-u_min)/(40*np.sqrt(np.sqrt(i+1)))
        U = ParticleFilterStep (U,R0,Ds_t,time,u_min,u_max,u_sigma,likelihood)
    
        if i%50==0:
            for m in range(4):
                if i!=0:
                    np.savetxt('ParticlesAtIter%d.csv'%(i), U, delimiter=",")
                plt.show()
                plt.hist(U[:,m],bins=100, range=(u_min[m], u_max[m]))
                plt.title( "Variable: %d Iteration: %d" %(m,i))
                

                plt.show()
                
                
            wts = FindWeights (U,R0,Ds_t,time,likelihood)
            u_pred = np.dot(wts,U)
            
            R_pred = FindR(Ds_t,u_pred,time)
            
            plt.plot(time,R0, label='Actual')
            plt.plot(time,R_pred, label='Predicted')
            plt.xlabel("Time (Years)")
            plt.ylabel("R")
            plt.title("Prediction at Iteration: %d" %(i))
            plt.legend()
            plt.show()


    return U

def ParticleFilterStep (U,R0,Ds_t,time,u_min,u_max,u_sigma,likelihood):
    N = U.shape[0]
    U1 = np.zeros(U.shape)
    for i in range(N):
        U1[i,:] = StatePerturb(U[i,:],u_min,u_max,u_sigma)
    
    # U1 = StatePerturb(U,u_min,u_max,u_sigma)
    
    w = FindWeights (U1,R0,Ds_t,time,likelihood)
    
    U2 = ResampleParticles (U1,w)
    
    return U2

def ResampleParticles (U,weights):
    N = U.shape[0]

    indices = np.random.choice(N, N, replace=True, p=weights)
    
    U_new = U[indices]
    
    return U_new


def StatePerturb(u,u_min,u_max,u_sigma):
    # u_range=u_max-u_min 
    # u_sigma= u_range/200 # the sigma of the purterbation in Particle filter
    # We can change the value and specify for each u[i]

    Noise=np.zeros(u.shape)
    u_new=np.zeros(u.shape)
    
    
    # U0 = np.random.multivariate_normal(u,cov,N)
    
    for i in range(0,u.size):
        Noise[i]=np.random.normal(0,u_sigma[i])
        u_new[i]=u[i]+Noise[i]
        # while u_new[i]<u_min[i] or u_new[i]>u_max[i]: # To ensure to be inside the range
        #     Noise[i]=np.random.normal(0,u_sigma[i])
        #     u_new[i]=u[i]+Noise[i]
        
    # [N,M] = U.shape
    # noise = np.random.multivariate_normal(np.zeros((M,)),np.diag(u_sigma),N)
    # return U +noise
    
    return u_new

def FindLikelihood (u,R0,Ds_t,time,likelihood):
    # The following lines are gaussian
    # R = FindR(Ds_t,u,time)
    # w= np.exp(-0.5*(np.linalg.norm(R-R0,ord=1))**2/(100**2)) #1351*0.25*
    # if np.isnan(w):
    #     w=0
    #     print('yes')
    if likelihood=="Gaussian":
        R = FindR(Ds_t,u,time)
        w= np.exp(-0.5*(np.linalg.norm(R-R0,ord=1))**2/(100**2)) #1351*0.25*
        if np.isnan(w):
            w=0
            print('yes')
    if likelihood=="Poisson":
        R = FindR(Ds_t,u,time) # This is a vector predicted by the model, we need log of this vector
    
        log_R_model=np.log(R)
        dum1=np.dot(R0,log_R_model)
        dum2=np.sum(R)
        w=np.exp(dum1-dum2)
    #w=np.exp(-np.abs(np.sum(Log_R_o)-np.sum(R)))
    #print(np.sum(R))
    #print(np.sum(np.sum(Log_R_o)))
    
        if np.isnan(w):
            w=0
            # print('yes')   
            # print(np.sum(R))
    return w

def FindWeights (U,R0,Ds_t,time,likelihood):
    N = U.shape[0]
    weights = np.zeros(N)
    for i in range(N):
        weights[i] = FindLikelihood (U[i,:],R0,Ds_t,time,likelihood)
    
    norm_weights = weights/np.sum(weights)
    
    return norm_weights

def FindR(Ds_t,u,time):
    # u s is parameters
    r=u[0]
    t_a=u[1]
    Asigma=u[2]
    Ds_c=u[3]
    R=np.zeros(Ds_t.shape)
    y=np.exp((Ds_t-Ds_c)/Asigma)
    boolian=Ds_t>Ds_c
    temp1 = integrate.cumtrapz(y*boolian, time , initial=0, axis = 0)# The multiplication of 1/12 is that we are analysing things monthly
    temp2 = temp1/t_a +1
    
    R = r*y/temp2 # To check that should I also multiply the numerator with Hevicide or not
    # R=r*y*boolian/temp2
    R_avg = 0.25* np.sum(R,axis=1)

    # for i in range(0,Ds_t.size):
    #     Num=r*y[i]
    #     boolian=Ds_t[0:i+1]>Ds_c
    #     Den=(np.trapz(boolian*y[0:i+1],time[0:i+1])/t_a)+1
    #     R[i]=Num/Den
    return R_avg

def GenLinDs_t(time,u,m):
    Ds_t=m*time
    return Ds_t

def GenbiLinDs_t(time,u,m):
    t_max= time.size
    t_half=np.int(t_max/2)
    print(t_half)
    Ds_t= np.zeros(time.shape)
    Ds_t[0:t_half]=m*time[0:t_half]
    Ds_t[t_half:] = Ds_t[t_half-1]
    return Ds_t

def PostProcess(wts,U_history,Ds_t_final,time_particle,time_total,R0,R_pred,method,T_max,N):
    # Ds_t_final is the stress history in all times
    if method==1:
        
        Start_Time=2021-time_particle[-1]
        fig = plt.figure(figsize=(4, 3), dpi=300)

        plt.style.use('default')
        ax = fig.add_subplot(1, 1, 1)
        ax.step(x=time_particle+Start_Time, y=R0,linewidth=3.0,color='blue',label='Catalog')
        plt.axvspan(time_particle[0]+Start_Time, T_max+Start_Time,color='white', alpha=0.2, lw=0)
        plt.axvspan(T_max+Start_Time, time_particle[-1]+Start_Time, color='grey', alpha=0.5, lw=0)
        plt.axvspan(time_particle[-1]+Start_Time,time_total[-1]+Start_Time , color='black', alpha=0.5, lw=0)

        plt.rc('font',family='Times New Roman')
        plt.rcParams.update({'font.family':'Times new roman'})

        #time_dummy=np.linspace(time_particle[0]+1992,time_particle[-1]+1992,(time_particle[-1]-time_particle[0])*10)
        #y_dummy=np.ones(((time_particle[-1]-time_particle[0])*10,1))
        
        # ax.step(x=time_particle+Start_Time, y=R_pred,linewidth=3.0,color='red')
        
        #ax.step(x=time_dummy, y=np.dot(R_pred,y_dummy),linewidth=3.0,color='red')
        MeanParameters=np.mean(U_history,axis=0) 
        R_mean=np.zeros(time_total.shape)
        Sigma_timeseries=np.zeros(time_total.shape)
        Sigma_i=np.zeros(time_total.shape)
        R_matrix=np.zeros((N,time_total.size))
        counter_nan=0
        # print(R_matrix.shape)
        for p in range(N):
            
            R_pred=FindR(Ds_t_final,U_history[p,:],time_total)
            a=np.sum(R_pred)
            # Removing Nan from R_pred
            if (np.isnan(a)):
                counter_nan+=1
                print(counter_nan)
            else:
                R_matrix[p,:]=R_pred
                R_mean+=R_pred

            ax.step(time_total+Start_Time,R_pred,alpha=0.01,color='red')
        # To make the legend Different Particles
        R_mean=R_mean/(N-counter_nan)        
        ax.step(time_total+Start_Time,R_pred,alpha=0.01, label=r'Particles',color='red')
        print(R_mean)
        for p in range(N):
            a=np.sum(R_matrix[p,:])
            if a!=0:    
                Sigma_i+=np.square(R_matrix[p,:]-R_mean)+R_matrix[p,:]
        Sigma_timeseries=np.sqrt(Sigma_i/(N-counter_nan-1))
        
        #R_pred=FindR(Ds_t_final,MeanParameters,time_particle)
        #ax.step(time_particle+Start_Time,R_pred, label='Mean Prediction',color='yellow')
        ax.step(time_total+Start_Time,R_mean,linewidth=3.0 ,label='Mean Predicted',color='cyan')
        ax.step(time_total+Start_Time,R_mean-2*Sigma_timeseries, label=r"$2\sigma$ lines",color='black')
        ax.step(time_total+Start_Time,R_mean+2*Sigma_timeseries,color='black')
        
        leg = plt.legend()
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
        EndingX=2030
        plt.xlim([time_total[0]+Start_Time,EndingX])
        #plt.ylim([0,25])
        plt.xlabel(r"Time (year)",fontname="Times new roman")
        plt.ylabel(r"Seismicity",fontname="Times new roman")
        plt.rcParams.update({'font.family':'Times new roman'})
        plt.xticks(fontname="Times new roman")
        plt.yticks(fontname="Times new roman")
        # SMALL_SIZE = 8
        # MEDIUM_SIZE = 10
        # BIGGER_SIZE = 12

        # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        fig.savefig('CatalogSmallpriorRangePoisson.png', bbox_inches = 'tight',dpi=500)

    return MeanParameters,R_mean
def Distribution(wts,U_history,Ds_t_final,time_particle,R0,R_pred,method,T_max,N):
    import pandas as pd
    import seaborn as sns
    from matplotlib import rc
    Y=pd.DataFrame(U_history,columns=["r", "ta","Asigma","DeltaSc"])

# Apply the default theme
    sns.set_theme()

    g = sns.PairGrid(Y,corner=True)
    g.map_lower(sns.kdeplot, levels=4, color=".2")
    g.map_lower(sns.scatterplot, marker="+")
    g.map_diag(sns.histplot, element="bars", linewidth=1, kde=True)
    g.axes[3,3].xaxis.set_label_text(r"$\Delta S_c$")
    g.axes[3,2].xaxis.set_label_text(r"$A\sigma_0$")
    g.axes[3,1].xaxis.set_label_text(r"$t_a$")
    g.axes[3,0].xaxis.set_label_text(r"$r$")
    g.axes[3,0].yaxis.set_label_text(r"$\Delta S_c$")
    g.axes[2,0].yaxis.set_label_text(r"$A\sigma_0$")
    g.axes[1,0].yaxis.set_label_text(r"$t_a$")
    g.axes[0,0].yaxis.set_label_text(r"$r$")
    sns.set(font="serif")
    rc('text', usetex=True)
    return 0
def GenRateSynthetic(Ds_t,N_years,MeanParameters,Scaling):
    # This function find the seismicity rate for the last N_years data point of the stress distribution and multiply it by Scaling
    # Ds_t Stress in all years
    # N_years is the number of years that I want to have a seismicity for
    Ds_t_synthetic=Ds_t[-N_years:,:]
    time_synthetic=np.array(range(N_years))
    R_synthetic=FindR(Ds_t_synthetic,MeanParameters,time_synthetic)
    R_synthetic*=Scaling
    # R_synthetic is the seismicity rate of the synthetic catalog calculate using forward model and the mean parameter
    # Ds_t_synthetic nothing is changed here for Ds_t_synthetic only the number of years is adjusted based on N_years
    return R_synthetic,Ds_t_synthetic,time_synthetic

from scipy.stats import poisson
def GenOneSyntheticCatalog(R_synthetic,time_synthetic):
    Lambda_star=np.max(R_synthetic)
    numPoints=poisson.rvs(mu=Lambda_star*(time_synthetic[-1]+1-time_synthetic[0]))
    # Generating Homogeous Poisson Process
    xx=np.random.uniform(low=np.min(time_synthetic), high=np.max(time_synthetic)+1, size=numPoints)
    ## xx is the years in which events (numPoints number of events) have happened, next step is to reject some of them, because we really do not want numPoints number of events, in each year a proportion of lambda(t)/lambda_max is enougth
    Index=(np.floor(xx)-np.min(time_synthetic))
    p=np.array([])
    for i in range(numPoints):
        index=int(Index[i])
        p=np.append(p,R_synthetic[index]/Lambda_star)
        #p[i]=R_synthetic[index]/Lambda_star
        # p is the acceptance probability
    # Generate Bernoulli variables (ie coin flips) for thinning
    booleRetained=np.random.uniform(size=numPoints)<p
    Events=xx[booleRetained]
    Event_years=np.floor(Events)
    # Next line is added because of definition of histogram
    TTT=np.append(time_synthetic,time_synthetic[-1]+1)
    GeneratedR,_=np.histogram(Event_years,TTT)
    
    #plt.plot(time_synthetic,GeneratedR)
    return GeneratedR

def GenFinalSyntheticCatalog(Ds_t,N_years,MeanParameters,Scaling,N_cats):
    R_synthetic,Ds_t_synthetic,time_synthetic=GenRateSynthetic(Ds_t,N_years,MeanParameters,Scaling)
    Concat_cats=GenOneSyntheticCatalog(R_synthetic,time_synthetic)
    fig = plt.figure(figsize=(4, 3), dpi=300)

    #plt.style.use('default')
    ax = fig.add_subplot(1, 1, 1)
    plt.rcParams.update({'font.family':'Times new roman'})

    for i in range(N_cats-1):
        R0=GenOneSyntheticCatalog(R_synthetic,time_synthetic)
        Concat_cats=np.vstack((Concat_cats,R0))
        #plt.rcParams.update({'font.family':'Times new roman'})

        ax.step(np.linspace(1979,2038,60),R0,alpha=0.03,color='red')
    ax.step(np.linspace(1979,2038,60),R0,alpha=0.03,color='red',label="Sample")

        # Plotting the syntetic catalog
    R_final=np.mean(Concat_cats,axis=0) 
    print(R_final)
    for i in range (N_years-1):
        reminder=R_final[i]-np.floor(R_final[i])
        R_final[i+1]=R_final[i+1]+reminder
        R_final[i]=int(R_final[i])
    R_final[-1]=int(R_final[-1])    
    ax.step(np.linspace(1979,2038,60),R_final, label='Mean',color='black')
    #ax.step(+1992+T_real,R_real, label='Real Catalog',color='blue')
    #plt.title("Scale is %0.1f" %(Scaling))

    leg = plt.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    plt.xlabel(r"Time (year)")
    plt.ylabel(r"Seismicity")
    plt.rcParams.update({'font.family':'Times new roman'})

    fig.savefig("synth%0.1f.pdf" %(Scaling), bbox_inches = 'tight',dpi=300)

    return R_final,Ds_t_synthetic,time_synthetic

def FindLogLikelihood (u,R0,Ds_t,time,LIKELIHOOD,gamma):
    # The following lines are gaussian
    # R = FindR(Ds_t,u,time)
    # w= np.exp(-0.5*(np.linalg.norm(R-R0,ord=1))**2/(100**2)) #1351*0.25*
    # if np.isnan(w):
    #     w=0
    #     print('yes')
    if LIKELIHOOD=='Poisson':
        R = FindR(Ds_t,u,time) # This is a vector predicted by the model, we need log of this vector
    
        log_R_model=np.log(R)
        dum1=np.dot(R0,log_R_model)
        dum2=np.sum(R)
        w=(dum1-dum2)
        #w=np.exp(-np.abs(np.sum(Log_R_o)-np.sum(R)))
        #print(np.sum(R))
        #print(np.sum(np.sum(Log_R_o)))
    
        if np.isnan(w):
            w=0
            # print('yes')   
            #print(np.sum(R))
    if LIKELIHOOD=='Gaussian':
        R=FindR(Ds_t,u,time)
        No_zero_R0=R0[R0 != 0]
        Nonzeros=np.size(No_zero_R0)
        gamma=np.sqrt(np.mean(No_zero_R0))
#        w=-0.5*(np.linalg.norm(R-R0,ord=1))**2/(gamma**2)
        w=-0.5*(np.linalg.norm(R-R0))**2/(Nonzeros*gamma**2)
        
    return w

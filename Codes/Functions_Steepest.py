# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:08:01 2022

@author: Hojjat
"""

import numpy as np
from scipy import integrate
import matplotlib.pylab as plt
import math
from scipy import integrate

def Find_Dh_i_Du_j(Ds_t,u,time):
    # dR_du1 is a matrix of M*X where M is the length in time and M is length in space
    # The plan is to consider position integration in the very last section
    r=u[0]
    t_a=u[1]
    Asigma=u[2]
    Ds_c=u[3]
    dR_du1=np.zeros(Ds_t.shape) # In time and space
    dR_du2=np.zeros(Ds_t.shape)
    dR_du3=np.zeros(Ds_t.shape)
    dR_du4=np.zeros(Ds_t.shape)
    #%% Defining f1,f2,f3
    # f_1 and f_2 are defined in the report, but it is easier to find f3 instead of f2
    # f_4 is for the partial derivitive wrt Asigma0
    f_1=np.exp((Ds_t-Ds_c)/Asigma)
    boolian=Ds_t>Ds_c
    temp1 = integrate.cumtrapz(f_1*boolian, time , initial=0, axis = 0)
    f_2 = temp1/t_a +1
    f_3=temp1 # this is Integral of f1 in time from t_b to t
    f_4=integrate.cumtrapz(f_1*(Ds_c-Ds_t)*boolian, time , initial=0, axis = 0)
    
    #%% Defining dR_dui based on equations in the note.
    dR_du1=f_1/f_2
#    dR_du2=(r*f_1*f_3)/((t_a**2)*(f_2**2))
    dR_du2=((r*f_1)/(f_2))*((f_3)/(f_2))/(t_a**2)
    term1=(r*f_1*((Ds_c-Ds_t)))/(Asigma**2*f_2)
#    term2=-(r*f_1*f_4)/(t_a*(f_2**2)*(Asigma**2))
    term2=-((r*f_1)/(f_2))*((f_4)/(f_2))/(t_a*Asigma**2)
    dR_du3=term1+term2
    term3=(-r*f_1)/(Asigma*f_2)
#    term4=(r*f_1*f_3)/(t_a*Asigma*f_2**2)
    term4=((r*f_1)/(f_2))*((f_3)/(t_a*Asigma*f_2))
    dR_du4=term3+term4
    #%% Integrate in space
    dh_du1=.25*np.sum(dR_du1,axis=1)
    dh_du2=.25*np.sum(dR_du2,axis=1)
    dh_du3=.25*np.sum(dR_du3,axis=1)
    dh_du4=.25*np.sum(dR_du4,axis=1)
    Dh_i_Du_j=np.array([dh_du1,dh_du2,dh_du3,dh_du4]).T
    # Here dh_du1 is the derivitive of each observation in time with the prespective parameter
    # So dh_dui is a M*1 vector
    # Dh_i_Du_j is a M*4 Matrix
#%%
    return Dh_i_Du_j
#%%

def DiffGaussian_dh(R_o,R_pred,gamma):
    # Gamma is the vairance of the liklihood
    # Just a quick note about the sign, This equation has a negative derivitve with respect to h_i so if you want to minimize something you should add a minus sign at the end, this is true for both the poisson liklihood and the gaussain likelihood
    DiffGaussian_dh_vec=(R_o-R_pred)/gamma
    
    return DiffGaussian_dh_vec


def DiffPoissson_dh(R_o,R_pred):
    # Just a quick note about the sign, This equation has a negative derivitve with respect to h_i so if you want to minimize something you should add a minus sign at the end, this is true for both the poisson liklihood and the gaussain likelihood
   
    DiffPoissson_dh_vec=R_o/R_pred-1 
    
    return DiffPoissson_dh_vec



#%%
def DiffLLK_uj(Typeofliklihood,Ds_t,u,time,R_o,R_pred,gamma):
    # Type of liklihood equal to one is Gaussian
    # Type of liklihood equal to two is Poisson
    Dh_i_Du_j=Find_Dh_i_Du_j(Ds_t,u,time)
    DiffLLK_uj_vec=np.zeros((4,1))
    
    # Here we delete the first of the vectors, the reason is that we have an integration in the denomenator and we want to avoid zero in the denomerator
    Dh_i_Du_j=Dh_i_Du_j[1:]
    #print(Dh_i_Du_j)
    if Typeofliklihood==1:
        DiffGaussian_dh_vec=DiffGaussian_dh(R_o,R_pred,gamma)
        DiffGaussian_dh_vec=DiffGaussian_dh_vec[1:]
        for j in range(4):

            DiffLLK_uj_vec[j]=np.dot(DiffGaussian_dh_vec,Dh_i_Du_j[:,j])
        
    if Typeofliklihood==2:
        DiffPoisson_dh_vec=DiffPoissson_dh(R_o,R_pred)
        DiffPoisson_dh_vec=DiffPoisson_dh_vec[1:]
        for j in range(4):
            DiffLLK_uj_vec[j]=np.dot(DiffPoisson_dh_vec,Dh_i_Du_j[:,j])
        
    return DiffLLK_uj_vec
#%%
def DiffLLK_ubarj(Typeofliklihood,Ds_t,u,time,R_o,R_pred,gamma,s):
    DiffLLK_uj_vec=DiffLLK_uj(Typeofliklihood,Ds_t,u,time,R_o,R_pred,gamma)
    DiffLLK_ubarj_vector=DiffLLK_uj_vec*s
    return DiffLLK_ubarj_vector



# def SteepestDescent(u0,LineSearchMethod,N,gradient,criterian):
#     # N is the number of Points in the steepest descent
#     for i in range(N):
#         Delta_u=Gradient()
#         u_history[:,i+1]=u_history[:,i]+alpha[i]*Delta_u
        
    
    
#     return u_history


def find_alpha_p(alpha_low,alpha_high):
    alpha_p=(alpha_low+alpha_high)/2
    return alpha_p

def Pinpoint(phi,phi_p,alpha_low,alpha_high,phi_0,phi_low,phi_high,Dphi_0,Dphi_low,Dphi_high,mu1,mu2,N_alpha_max):
    #print("In the begining alpha is:",alpha_low)
    alf=np.array([0])
    k=0
    while k<N_alpha_max:
        k=k+1
        
        alpha_p=find_alpha_p(alpha_low,alpha_high)
        alf=np.append(alf,alpha_p)
        phi_at_p=phi(alpha_p)
        if phi_at_p>phi_0+mu1*alpha_p*Dphi_0 or phi_at_p>phi_low:
            #print("Case 1")
            #print("alpha_high is ",alpha_high)
            #print("alpha_low is ",alpha_low)
            alpha_high=alpha_p
            #print("Now high is",alpha_high)
            
        else:
            #print("Hi")
            #print(alpha_p)
            Dphi_p=phi_p(alpha_p)
            #print("Dphi_0 is",Dphi_0)
            if np.linalg.norm(Dphi_p)<=-mu2*Dphi_0:
                alpha_star=alpha_p # Here the program gets out of the loop the last element of alf is the answer
                #print ("Case 2")
                break
            elif Dphi_p*(alpha_high-alpha_low)>0:
                alpha_high=alpha_low
                #print("Case 3")
            alpha_low=alpha_p;    
            #print(alpha_low,alpha_high)

    return alf

def Bracketing(phi,phi_p,xk,pk,alpha_init,phi_0,Dphi_0,mu1,mu2,sigma,N_alpha_max):
    # phi = @(a) f(xk+a*pk);
    # phi_p  = @(a) df(xk+a*pk)'*pk;  % Phi_p of alpha
    alpha1=0
    alpha2=alpha_init;
    # phi1=phi_0
    Dphi1=Dphi_0
    first=1
    Diverged=False
    # I do not want the program to stuck here
    while 1:
        phi2=phi(alpha2)
        phi1=phi(alpha1)
        if phi2>phi_0+mu1*alpha2*Dphi_0 or (first==0 and phi2>phi1):
            phi_low=phi1
            phi_high=phi2
            Dphi_low=phi_p(alpha1)
            Dphi_high=phi_p(alpha2)
            #print("I start from here","and alpha1 is",alpha1)
            alf=Pinpoint(phi,phi_p,alpha1,alpha2,phi_0,phi_low,phi_high,Dphi_0,Dphi_low,Dphi_high,mu1,mu2,N_alpha_max)
            break
        Dphi2=phi_p(alpha2)
        #print("When alpha2 is ",alpha2)
        #print("Dphi2=",Dphi2)
        
        if np.linalg.norm(Dphi2)<=-mu2*Dphi_0:
            alf=np.append(0,alpha2)
            break
        elif Dphi2>=0:
            #print("Now I am here")
            phi_low=phi2
            phi_high=phi1
            Dphi_low=phi_p(alpha2)
            Dphi_high=phi_p(alpha1)            
            alf=Pinpoint(phi,phi_p,alpha2,alpha1,phi_0,phi_low,phi_high,Dphi_0,Dphi_low,Dphi_high,mu1,mu2,N_alpha_max);
            break
        else:
            #print("When I am here alpha1 is",alpha1)
            #print("When I am here alpha2 is",alpha2)

            alpha1=alpha2
            alpha2=sigma*alpha2
            if alpha2>1e6:
                Diverged=True
                alf=np.zeros((1,1))
                break
        first=0
    return alf,Diverged


#%% 
import matplotlib.pyplot as plt

def plot_linesearch(phi,alpha_max):
    
    N=100
    A=np.linspace(0,alpha_max,N)
    B=np.array([])
    for i in range(N):    
        B=np.append(B,phi(A[i]))
    
    fig, ax = plt.subplots()
    plt.plot(A,B)

    return 0

def CheckSteepestDescent():
    x0=np.array([10,1]).T
    taw=1e-2 # Threshold
    mu1=1e-4
    mu2=1e-2
    sigma=5
    beta=15
    f= lambda x : x[0]**2+beta*x[1]**2
    df= lambda x : np.array([2*x[0],2*beta*x[1]])
    xk=x0
    df_new=df(xk)
    V_new=np.eye(2)/np.linalg.norm(df_new);
    pk=-np.matmul(V_new,df_new)
    #print(pk.shape)
    phi =  lambda alpha : f(xk+alpha*pk)
    phi_p= lambda alpha : np.dot(df(xk+alpha*pk),pk) # Phi_p of alpha
    #print(phi_p(20))
    # alpha_max=3
    # plot_linesearch(phi,alpha_max)
    alpha_init=.1
    phi_0=phi(0)
    Dphi_0=phi_p(0)
    
    alf=Bracketing(phi,phi_p,xk,pk,alpha_init,phi_0,Dphi_0,mu1,mu2,sigma)
    xkp1=xk+alf[-1]*pk
    k=0
    while np.linalg.norm(df_new)>taw:
        k=k+1
        alf_old=alf[-1]
        pk_old=pk
        df_old=df_new
        xk=xkp1
        df_new=df(xk)
        pk=-df_new/np.linalg.norm(df_new)
        phi =  lambda alpha : f(xk+alpha*pk)
        phi_p= lambda alpha : np.dot(df(xk+alpha*pk),pk) # Phi_p of alpha
        phi_0=phi(0)
        Dphi_0=phi_p(0)
        alpha_new=alf_old*(np.dot(df_old,pk_old))/np.dot(df_new,pk)
        alf=Bracketing(phi,phi_p,xk,pk,alpha_new,phi_0,Dphi_0,mu1,mu2,sigma)
        #print("Hi")
        #print(alf.shape)
        xkp1=xk+alf[-1]*pk;
    print(k)
    return xkp1
#xkp1=CheckSteepestDescent()
#%%       
def CheckBFGS():
    x0=np.array([10,1]).T
    # Defining Some paramters in 
    taw=1e-2 # Threshold
    mu1=1e-4
    mu2=1e-2
    sigma=5
    beta=15
    N_parameters=2
    alpha_new=1
    I=np.eye((N_parameters))
    f= lambda x : x[0]**2+beta*x[1]**2
    df= lambda x : np.array([2*x[0],2*beta*x[1]])
    xk=x0
    df_new=df(xk)
    V_new=I/np.linalg.norm(df_new);
    pk=-np.matmul(V_new,df_new)

    phi =  lambda alpha : f(xk+alpha*pk)
    phi_p= lambda alpha : np.dot(df(xk+alpha*pk),pk) # Phi_p of alpha

    alpha_init=1
    phi_0=phi(0)
    Dphi_0=phi_p(0)
    
    alf=Bracketing(phi,phi_p,xk,pk,alpha_init,phi_0,Dphi_0,mu1,mu2,sigma)
    xkp1=xk+alf[-1]*pk
    k=1
    #print (xkp1)
    while np.linalg.norm(df_new)>taw:
        reset=0
        if np.mod(k,5)==0:
            reset=1
        xk_old=xk
        xk=xkp1
        df_old=df_new # you need to remove 
        df_new=df(xk)
        V_old=V_new

        if reset==1:
            V_new=I/np.linalg.norm(df_new)
     
        else:
            s=xk-xk_old
            y=df_new-df_old
            Sigma=1/(np.dot(s.T,y))
            V_new=(I-Sigma*(s@(y.T)))@V_old@(I-Sigma*s@(y.T))+Sigma*(s@s.T)
        pk=-V_new@df_new
        phi =  lambda alpha : f(xk+alpha*pk)
        phi_p= lambda alpha : np.dot(df(xk+alpha*pk),pk) # Phi_p of alpha
        phi_0=phi(0)
        Dphi_0=phi_p(0)
        print(pk.shape)
        alf=Bracketing(phi,phi_p,xk,pk,alpha_new,phi_0,Dphi_0,mu1,mu2,sigma)
        xkp1=xk+alf[-1]*pk
            
        

        k=k+1
        print(k)
    return xkp1 
#xkp1=CheckBFGS()
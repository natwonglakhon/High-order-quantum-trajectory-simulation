#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:34:30 2025

@author: Nattaphong
"""

import numpy as np
import math
import time
from numpy import matmul as mul

#Set parameters
Tf = 1 # Final time
dt = 0.01 # true time step
Ga = 1 #coupling rate
n = np.int(np.round(Tf/dt)) #Number of step per quantum trajectory
r = 5000  # Number of realizations
eta = 1 # measurement efficientcy

#Define Puali matrices (spin 1/2)
sx = np.array([[0, 1],[ 1, 0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1, 0],[0, -1]])
II = np.array([[1, 0],[0, 1]])

#Define Puali matrices (spin 1)
sx = np.array([[0, 1],[ 1, 0]])
sx = np.array([[0, 1, 0],[ 1, 0, 1],[ 0, 1, 0]])/math.sqrt(2)
sy = np.array([[0, -1j, 0],[ 1j, 0, -1j],[ 0, 1j, 0]])/math.sqrt(2)
sz = np.array([[1, 0, 0],[0, 0, 0],[0, 0, -1]])
II = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

#Define matrices (spin 3/2)
s_m =  np.array([[0, 0, 0, 0],[np.sqrt(3), 0, 0, 0],[ 0, 2, 0, 0], [ 0, 0, np.sqrt(3), 0]])/2
II = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


#Setting initial state (choose upon the spin system)
psiInqb = 1/math.sqrt(2)*np.array([[1],[1]]) #Qubit initial state
# psiInsp1 = 1/2*np.array([[1],[math.sqrt(2)],[1]]) #Spin 1 initial state
# psiInsp32 = 1/math.sqrt(8)*np.array([[1], [np.sqrt(3)], [np.sqrt(3)],[1]]) #Spin 3/2 initial state

psiIn = psiInqb # initial state


###### Define measurement operator ######

#Qubit system
K = math.sqrt(Ga/2)*sz # Measurement operator for z measurement
Kd = math.sqrt(Ga/2)*sz # Measurement operator for z measurement (Conjugate transpose)
# K = math.sqrt(Ga)*(sx - 1j*sy)/2 # Measurement operator for qubit fluorescent 
# Kd =math.sqrt(Ga)*(sx + 1j*sy)/2 # Measurement operator for qubit fluorescent (Conjugate transpose)

#Spin 1 system
K = math.sqrt(Ga/2)*sz # Measurement operator for z measurement
Kd = math.sqrt(Ga/2)*sz # Measurement operator for z measurement (Conjugate transpose)
# K = math.sqrt(Ga)*(sx - 1j*sy)/2 # Measurement operator for qubit fluorescent 
# Kd =math.sqrt(Ga)*(sx + 1j*sy)/2 # Measurement operator for qubit fluorescent (Conjugate transpose)

#Spin 3/2 system
K = math.sqrt(Ga)*s_m # lowering operator
Kd = math.sqrt(Ga)*(s_m.conj().T) # creation operator


#useful operators
k = mul(Kd,K)
kk = mul(K,Kd)
K2 = mul(K,K)
K3 = mul(mul(K,K),K)


start_time = time.time()

#Import readout (course-grained data Y_t)
filename = 'ycg.txt'
data = np.loadtxt(filename)
y = np.reshape(data, [r,n])

#Import readout (course-grained data Z_t)
filename = 'zcg.txt'
data = np.loadtxt(filename)
Z = np.reshape(data, [r,n])

#Define a function for computing the normalisation factor
def nor(M, psi):
    rho = mul(psi,psi.conj().T)
    return np.trace(mul(M,mul(rho,M.conj().T)))


###### Simulating quantum trajectories ###### 
# WWC map
psiW =  np.zeros((r,n, 2, 1), dtype =complex)
for j in range(0,r):
    psi = psiIn # initial state
    for i in range(0,n):
        psiW[j,i] = psi
        WM1 = - 0.5*eta*k + math.sqrt(eta)*K*y[j,i] - 0.5*eta*K2 # first order in Dt
        WM2 = 0.5*eta*y[j,i]**2*K2 + (1/8)*eta**2*mul(k,k) -(1/4)*eta**(3/2)*y[j,i]*(mul(Kd,K2) + mul(K,k)) #Second order in Dt
        MW = II + WM1*dt + dt**2*(WM2) # combine all orders
        p = nor(MW,psi) # This is to compute the normalisation factor
        psi = mul(MW,psi)/math.sqrt(p)

        
# Rouchon and Ralph map
psiRR =  np.zeros((r,n, 2, 1), dtype =complex)
for j in range(0,r):
    psi = psiIn # initial state
    for i in range(0,n):
        psiRR[j,i] = psi
        MR = II  - 1/2*k*dt + math.sqrt(eta)*y[j,i]*K*dt + 1/2*eta*K2*(y[j,i]**2*dt**2 -dt) # RR MO
        p = nor(MR,psi) # This is to compute the normalisation factor
        psi = mul(MR,psi)/math.sqrt(p)

# Ito map
psiI =  np.zeros((r,n, 2, 1), dtype =complex)
for j in range(0,r):
    psi = psiIn # initial state
    for i in range(0,n):
        psiI[j,i] = psi
        MI = II - 1/2*eta*k*dt + math.sqrt(eta)*y[j,i]*K*dt # Ito Mo
        p = nor(MI,psi) # This is to compute the normalisation factor
        psi = mul(MI,psi)/math.sqrt(p)

# Robinet
psiRB =  np.zeros((r,n, 2, 1), dtype =complex)
for j in range(0,r):
    psi = psiIn # initial state
    for i in range(0,n):
        psiRB[j,i] = psi
        Mrb = II + y[j,i]*dt*K + dt*(-k/2+(y[j,i]**2*dt-1)/2*K2) + dt**2*(-y[j,i]*(mul(k,K)+mul(K,k))/4 +(y[j,i]**3*dt-3*y[j,i])/6*K3 ) + dt**2*(mul(k,k)/8 - (y[j,i]**2*dt-1)/12*(mul(k,K2)+mul(kk,K2)+mul(K2,k))+(y[j,i]**4*dt**2-6*y[j,i]**2*dt+3)/24*(mul(K2,K2))) 
        p = nor(Mrb,psi) # This is to compute the normalisation factor
        psi = mul(Mrb,psi)/math.sqrt(p)

# Nearly exact map (\Phi)
psiPhi =  np.zeros((r,n, 2, 1), dtype =complex)
for j in range(0,r):
    psi = psiIn # initial state
    for i in range(0,n):
        psiPhi[j,i] = psi
        MPhi = II + y[j,i]*dt*K + dt*( -k/2 + 0.5*(y[j,i]**2*dt-1)*K2) + dt**2*( -y[j,i]*( mul(k,K)+ mul(K,k) )/4 + ( y[j,i]**3*dt-3*y[j,i] )/6*K3 ) + dt**2*( mul(k,k)/8 ) - 0.5*Z[j,i]*( mul(K,k) - mul(k,K) ) 
        p = nor(MPhi,psi) # This is to compute the normalisation factor
        psi = mul(MPhi,psi)/np.sqrt(p)

print("--- %s seconds ---" % (time.time() - start_time))       


################# Analysing the distances #########################

#Define a function for trace-absolute distance (for pure states)
def td(psi1, psi2):
    overlap = mul(psi2.conj().T, psi1)*mul(psi1.conj().T, psi2)
    return np.sqrt(1 - np.trace(overlap))

#Define a function for trace-squared distance (for pure states)
def td2(psi1, psi2):
    overlap = mul(psi2.conj().T, psi1)*mul(psi1.conj().T, psi2)
    return 1 - np.trace(overlap)


#Set parameters
Tf = 1 # Final time
dt = 0.0001 # True time step
Dt = 0.01 #Finite time step
r = 5000  # Number of realisations

n = np.int(np.round(Tf/Dt)) # Number of steps per finite evolution
ng = np.int(np.round(Tf/dt)) # Number of steps per in true evolution
N = np.int(np.round(Dt/dt)) # Number of data per in finite interval

#Importing the true state (choose the system)
psiTrue = np.load('psi.npy')

start_time = time.time()

# Absolute Trace distance with true state
Dw = np.zeros([r,n], dtype = float)  #  WWC map
Dr = np.zeros([r,n], dtype = float)  #  RR map
Drb = np.zeros([r,n], dtype = float)  # Robinet map
Dphi = np.zeros([r,n], dtype = float)  # Phi map
Di = np.zeros([r,n], dtype = float)  # Ito map

# Squared Trace distance with true state
Dw2 = np.zeros([r,n], dtype = float)  # WWC map
Dr2 = np.zeros([r,n], dtype = float)  # RR map
Drb2 = np.zeros([r,n], dtype = float) # Robinet map
Dphi2 = np.zeros([r,n], dtype = float) # Phi map
Di2 = np.zeros([r,n], dtype = float)  #Ito map

# Calculate trace distance
for j in range(0,r):
    for i in range(0,n):
        Dphi[j,i] = td(psiTrue[j,i],psiPhi[j,i])
        Drb[j,i] = td(psiTrue[j,i],psiRB[j,i])
        Dw[j,i] = td(psiTrue[j,i],psiW[j,i])
        Dr[j,i] = td(psiTrue[j,i],psiRR[j,i])
        Di[j,i] = td(psiTrue[j,i],psiI[j,i])
        Dphi2[j,i] = td2(psiTrue[j,i],psiPhi[j,i])
        Drb2[j,i] = td2(psiTrue[j,i],psiRB[j,i])
        Dw2[j,i] = td2(psiTrue[j,i],psiW[j,i])
        Dr2[j,i] = td2(psiTrue[j,i],psiRR[j,i])
        Di2[j,i] = td2(psiTrue[j,i],psiI[j,i])

# Computing average for the absolute distance
Wa = np.sum(np.sum(Dw,1)/(n))/r; print("WWC", Wa)  
Ra = np.sum(np.sum(Dr,1)/(n))/r;  print("RR", Ra)
RBa = np.sum(np.sum(Drb,1)/(n))/r;  print("Robinet", RBa)
Phia = np.sum(np.sum(Dphi,1)/(n))/r;  print("Nearly exact", Phia)
Ia = np.sum(np.sum(Di,1)/(n))/r;  print("Ito", Ia,"\n") 

# Computing average for the squared distance
W2a = np.sum(np.sum(Dw2,1)/(n))/r; print("WWC", W2a)  
R2a = np.sum(np.sum(Dr2,1)/(n))/r;  print("RR", R2a)
RB2a = np.sum(np.sum(Drb2,1)/(n))/r;  print("Robinet", RB2a)
Phi2a = np.sum(np.sum(Dphi2,1)/(n))/r;  print("Nearly exact", Phi2a)
I2a = np.sum(np.sum(Di2,1)/(n))/r;  print("Ito", I2a, "\n")

print("--- %s seconds ---" % (time.time() - start_time)) 

## Save data

# Trace abolute distance
np.savetxt('Di.txt', np.real(Di), fmt='%.8e', delimiter=' ', newline='\t')
np.savetxt('Dr.txt', np.real(Dr), fmt='%.8e', delimiter=' ', newline='\t')
np.savetxt('Drb.txt', np.real(Drb), fmt='%.8e', delimiter=' ', newline='\t')
np.savetxt('Dw.txt', np.real(Dw), fmt='%.8e', delimiter=' ', newline='\t')
np.savetxt('Dphi.txt', np.real(Dphi), fmt='%.8e', delimiter=' ', newline='\t')

# Trace squared distance
np.savetxt('Di2.txt', np.real(Di2), fmt='%.12e', delimiter=' ', newline='\t')
np.savetxt('Dr2.txt', np.real(Dr2), fmt='%.12e', delimiter=' ', newline='\t')
np.savetxt('Drb2.txt', np.real(Drb2), fmt='%.12e', delimiter=' ', newline='\t')
np.savetxt('Dw2.txt', np.real(Dw2), fmt='%.12e', delimiter=' ', newline='\t')
np.savetxt('Dphi2.txt', np.real(Dphi2), fmt='%.12e', delimiter=' ', newline='\t')   

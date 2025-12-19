#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:19:44 2025

@author: Nattaphong
"""

import numpy as np
import math
import time
from numpy import matmul as mul

#Set parameters
Tf = 1 # Final time
dt = 0.0001 # true time step
Ga = 1 #coupling rate
n = np.int(np.round(Tf/dt)) #Number of step per quantum trajectory
r = 5000  # Number of realizations
eta = 1 # measurement efficientcy

#Define Puali matrices
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
Kd2 = mul(Kd,Kd)
K3 = mul(mul(K,K),K)
Kd3 = mul(mul(Kd,Kd),Kd)


#Generate Wiener increment
dw = np.random.normal(0.0, math.sqrt(dt),[r,n]) #With zero mean, 1/sqrt(dt) variance

start_time1 = time.time()

############# Generate records via RR approach ################## 
psit = np.zeros((r,n, 2, 1), dtype =complex)
y_out = np.zeros([r,n],dtype =complex) #keep the readouts
for j in range(0,r):
    psi = psiIn # initial state
    for i in range(0,n):
        psit[j,i] = psi
        rho = mul(psi,psi.conj().T)
        y_out[j,i] = math.sqrt(eta)*np.trace(mul(rho,Kd) + mul(K,rho)) + dw[j,i]/dt #Measurement record definition
        M = II  - (1/2)*eta*k*dt + math.sqrt(eta)*K*dt*y_out[j,i] + 1/2*eta*K2*(y_out[j,i]**2*dt**2 -dt) #Adding Rouchon correction MO
        Md = II - (1/2)*eta*k*dt + math.sqrt(eta)*Kd*dt*y_out[j,i] + 1/2*eta*Kd2*(y_out[j,i]**2*dt**2 -dt) # Adding Rouchon correction MO
        MapR = mul(M,mul(rho,Md)) 
        psi = mul(M,psi)/math.sqrt(np.trace(MapR))

print("--- %s seconds ---" % (time.time() - start_time1))       


start_time2 = time.time()

######## Coarse graining records, weighted equally (Y_t) ##########
Dt = 0.01 # Changing size of dt
ng = np.int(np.round(Tf/Dt)) # number of time step per trajectory for Dt
N = np.int(Dt/dt) #number of record in the increment Dt
yg_out = np.zeros([r,ng],dtype=float) # to collect course-grained records
#yz_cg = np.zeros([r,ng],dtype=float) # to collect course-grained records
zcg = np.zeros([r,ng],dtype=float) # to collect course-grained records
for j in range(0,r):
    for i in range(0,ng):
        for k in range(0,N):
            zcg[j,i] += dt*np.real(y_out[j, N*i + k])*(k*dt - (Dt/2))
            yg_out[j,i] += 1/N*np.real(y_out[j, N*i + k])

# ######### Collect relevant rhot ##########
psi_true = np.zeros((r,ng, 2, 1), dtype =float)
for j in range(0,r):
    for i in range(0,ng):
        psi_true[j,i] = psit[j,N*i]

print("--- %s seconds ---" % (time.time() - start_time2))  

###### Collect data #########

np.save('psi_true.npy', psi_true) # Collecing the true quantum states            
np.savetxt('ycg.txt', np.real(yg_out), fmt='%.8e', delimiter=' ', newline='\t') # Collecting equally coarse-grained records (Y_t)
np.savetxt('zcg.txt', np.real(zcg), fmt='%.8e', delimiter=' ', newline='\t') # Collecting weighted coarse-grained records (Z_t)





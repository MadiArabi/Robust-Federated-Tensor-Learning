import numpy as np
from tensorly import unfold
from tensorly.tenalg import mode_dot
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

mat_data = loadmat('your_dataset.mat')
data = mat_data['A']

# Data Genration 
np.random.seed(1234)
I1 =20
I2 =30
M1 =100
M2 =200
P1 =10
P2 = 25
iterations = 50
epsilon = 0.0001
percentage = 0.70
#A = np.random.rand(M1,I1,I2)
A = np.array(data)
'''
for i in range(M1):
    for j in range(I1):
        for k in range(I2):
            A[i,j,k] = float(str(A[i,j,k])[0:6])
'''
Abar =  A - np.mean(A,0)
x = Abar.copy()

def Uinitial1(x):
    scatter =  np.zeros((I1,I1))
    for i in range(x.shape[0]):
        scatter +=unfold(x[i,:,:], mode=0)@unfold(x[i,:,:], mode=0).T  
   
    eitgenvalue, u = np.linalg.eig(scatter)
    sorted_indices = np.argsort(eitgenvalue)[::-1]  # Indices for sorting in descending order
    sorted_eigenvectors = u[:, sorted_indices]
    
    lam1 = eitgenvalue[sorted_indices]
    cumulative =[lam1[0]]
    for index in range(1,len(lam1)):
        value = lam1[index]+cumulative[-1]
        cumulative.append(value)
    cumulative = cumulative/cumulative[-1]
    return sorted_eigenvectors, cumulative

def Uinitial2(x):
    
    scatter =  np.zeros((I2,I2))
    for i in range(x.shape[0]):
        scatter +=unfold(x[i,:,:], mode=1)@unfold(x[i,:,:], mode=1).T
        
    eitgenvalue,u = np.linalg.eig(scatter)
    sorted_indices = np.argsort(eitgenvalue)[::-1]  # Indices for sorting in descending order
    sorted_eigenvectors = u[:, sorted_indices]
    lam2 = eitgenvalue[sorted_indices]
    cumulative =[lam2[0]]
    for index in range(1,len(lam2)):
        value = lam2[index]+cumulative[-1]
        cumulative.append(value)
    cumulative = cumulative/cumulative[-1]
    return sorted_eigenvectors, cumulative

def U1(x1,u):
    
    scatter =np.zeros((I1,I1))
    print(scatter.shape)
    print(u.shape)
    print(unfold(x1[0,:,:],mode=0).shape)
    for i in range(x1.shape[0]):
        scatter +=unfold(x1[i,:,:],mode=0)@u@u.T@unfold(x1[i,:,:],mode=0).T
        
    
    eitgenvalue,u= np.linalg.eig(scatter)
    sorted_indices = np.argsort(eitgenvalue)[::-1]  # Indices for sorting in descending order
    sorted_eigenvectors = u[:, sorted_indices]
    return sorted_eigenvectors
def U2(x1,u):
    
    scatter =np.zeros((I2,I2))
    for i in range(x1.shape[0]):
        scatter +=unfold(x1[i,:,:],mode=1)@u@u.T@unfold(x1[i,:,:],mode=1).T
        
    eitgenvalue,u = np.linalg.eig(scatter)
    sorted_indices = np.argsort(eitgenvalue)[::-1]  # Indices for sorting in descending order
    sorted_eigenvectors = u[:, sorted_indices]
    return sorted_eigenvectors


u1, cum1 = Uinitial1(Abar)
u2,cum2 = Uinitial2(Abar)

def phi_calculator(x1,u1,u2):
    phi_frob1 =0
    for i in range(x1.shape[0]):
        first = mode_dot(x1[i,:,:], u1.T, mode=0)
        phi =mode_dot(first, u2.T, mode=1)
        phi_frob1 += np.linalg.norm(phi, ord ='fro')
    return phi_frob1
    
phi_old = phi_calculator(Abar,u1,u2)
phi_record = np.zeros(iterations)
P1=np.where(cum1>=percentage)[0][0]+1
P2=np.where(cum2>=percentage)[0][0]+1
u1 = u1[:,0:P1]
u2 = u2[:,0:P2]
for index in range(iterations):
    
    u1 = U1(x,u2)
    if P1< I1:
        u1 = u1[:,0:P1]
    u2 = U2(x,u1)
    if P2< I2:
        u2 = u2[:,0:P2]
    print(index)
    phi_curr = phi_calculator(x,u1,u2)
    phi_old = phi_curr
    phi_record[index] = phi_old
    
    

plt.plot(phi_record)
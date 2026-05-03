import numpy as np

A = np.random.rand(100,20,30)
B = np.random.rand(30,30)
u,s,v = np.linalg.svd(B,full_matrices=False)
check = u@u.T
Abar = A - np.mean(A,0)
before= Abar@u
projected = np.matmul(A,u)
after = projected- np.mean(projected,0)


B = np.random.rand(30,30)
u,s,v = np.linalg.svd(B,full_matrices=False)

C= np.random.rand(30,20)
uc,s,vc = np.linalg.svd(C,full_matrices=False)
checkc = vc@vc.T
D= np.random.rand(30,20)
ud,s,vd = np.linalg.svd(D,full_matrices=False)
checkd = vd@vd.T
v = np.linalg.inv(vd@vd.T+0.00001*np.eye(20))@vd@vc.T
checkv= v@v.T


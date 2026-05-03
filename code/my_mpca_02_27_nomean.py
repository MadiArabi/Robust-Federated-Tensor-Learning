import numpy as np
import tensorly
from tensorly import unfold
from tensorly.tenalg import mode_dot, multi_mode_dot
import copy
test_size = 20
I1, I2, I3 = 10, 10, 16
percentage, iterations, lam, epsilon = 0.90, 10, 0.00001, 0.0001

class MPCA_FD():
        

    def __init__(self,I,P,percentage=0.9):
        #self.M = size
        self.percentage = percentage
        self.iterations = 150
        self.I = I
        self.P = P
        
        
    def projection(self, data, matrix):
        return multi_mode_dot(data, [m.T for m in matrix], modes=[1, 2, 3])
    
    def train(self,Abar,Bbar,Cbar):

                
        def Vinitial(x,MODE,S):
            unfolded_x = np.array([unfold(x[i], mode=MODE).T for i in range(x.shape[0])])
            scatter = np.einsum("nij,nik->jk", unfolded_x, unfolded_x)
              
            eitgenvalue, u = np.linalg.eig(scatter)
            u = np.real_if_close(u, tol=1)
            eitgenvalue = np.real_if_close(eitgenvalue, tol=1)
            sorted_indices = np.argsort(eitgenvalue)[::-1]  
            sorted_eigenvectors = u[:, sorted_indices]
            if np.any(np.iscomplex(sorted_eigenvectors)):
                print("Vinitial",'complex')
            return sorted_eigenvectors[:,0:S]
            
        
        def Uinitial(data,MODE,P):
            x,y,z = [arr.copy() for arr in data]
            I1 = x.shape[MODE+1]
            X = unfold(x[0], mode=MODE)
            Y = [unfold(x[i], mode=MODE) for i in range(1, x.shape[0])] + \
                [unfold(y[i], mode=MODE) for i in range(y.shape[0])] + \
                [unfold(z[i], mode=MODE) for i in range(z.shape[0])]
            u,s,_ = np.linalg.svd(X,full_matrices=True)  
            sigma = np.zeros_like(X, dtype=float)
            sigma[:min(X.shape[0], X.shape[1]), :min(X.shape[0], X.shape[1])] = np.diag(s[:min(X.shape[0], X.shape[1])])
            for sample in Y:
                
                W = sample - u@u.T@sample
                M = np.block([[sigma, u.T@sample],[np.zeros((sample.shape[1],sigma.shape[1])), np.diag(np.linalg.norm(W,axis=0)**2)]]) 
                u_prime, sigma_prime, _ =np.linalg.svd(M,full_matrices=True)
                norm_w = W / np.linalg.norm(W, axis=0)
                u = (np.hstack((u,norm_w))@u_prime)[:,0:I1]
                sigma = np.zeros_like(sample, dtype=float)
                sigma[:min(sample.shape[0], sample.shape[1]), :min(sample.shape[0], sample.shape[1])] = np.diag(sigma_prime[0:min(sample.shape[0], sample.shape[1])])
            
            return u[:,0:P]
            
            
        def V(x2,U_mat,V_mat,P1,MODE,S):
            u1,u2,u3 = U_mat
            first, second = min((MODE+1)%3,(MODE+2)%3), max((MODE+1)%3,(MODE+2)%3)
            v1,v2 = V_mat[first],V_mat[second]
            if MODE==0:
                c2=v1@u2
                c3 =v2@u3
                c = np.kron(c2, c3)
            elif MODE==1:
                c1=v1@u1
                c3 =v2@u3
                c = np.kron(c1, c3)
            else:
                c1=v1@u1
                c2 =v2@u2
                c = np.kron(c1, c2)
            unfolded_x = np.array([(unfold(x2[i], mode=MODE)@c).T for i in range(x2.shape[0])])
            scatter = np.einsum("nij, nik -> jk", unfolded_x , unfolded_x)
                
            eitgenvalue,u= np.linalg.eig(scatter)
            u = np.real_if_close(u, tol=1)
            eitgenvalue = np.real_if_close(eitgenvalue, tol=1)
            sorted_indices = np.argsort(eitgenvalue)[::-1]  # Indices for sorting in descending order
            sorted_eigenvectors = u[:, sorted_indices]
            u = sorted_eigenvectors
            u=u[:,0:P1]
            I2 = S
            if MODE==0:
                v = u@u1.T@ np.linalg.inv((u1@u1.T)+lam*np.eye(I2))
            elif MODE==1:
                v = u@u2.T@ np.linalg.inv((u2@u2.T)+lam*np.eye(I2))
            else:
                v = u@u3.T@ np.linalg.inv((u3@u3.T)+lam*np.eye(I2))
                
            if np.any(np.iscomplex(sorted_eigenvectors)):
                print("V",'complex')
            return v
        
        
        def U(data,U_mat,MODE,P):
            first, second = min((MODE+1)%3,(MODE+2)%3), max((MODE+1)%3,(MODE+2)%3)
            u1,u3 = U_mat[first],U_mat[second]
            x,y,z = [arr.copy() for arr in data]
            I1 = x.shape[MODE+1]
            u = np.kron(u1, u3)
            #print(f" u itself, y, z, {y.shape}, {z.shape}")
            #print(f'I1, u1, u3, u, {I1}, {u1.shape}, {u3.shape}, {u.shape}')
            X = unfold(x[0,:,:,:], mode=MODE)@u
            Y = [unfold(x[i], mode=MODE)@u for i in range(1, x.shape[0])] + \
                [unfold(y[i], mode=MODE)@u for i in range(y.shape[0])] + \
                [unfold(z[i], mode=MODE)@u for i in range(z.shape[0])]
            
            u,s,_ = np.linalg.svd(X,full_matrices=True)  
            sigma = np.zeros_like(X, dtype=float)
            sigma[:min(X.shape[0], X.shape[1]), :min(X.shape[0], X.shape[1])] = np.diag(s[:min(X.shape[0], X.shape[1])])
            #print(f"sigma, s, {sigma.shape}, {s.shape}, {min(X.shape[0], X.shape[1])}")
            for sample in Y:
                #print('sample', sample.shape)
                W = sample - u@u.T@sample
                M = np.block([[sigma, u.T@sample],[np.zeros((sample.shape[1],sigma.shape[1])), np.diag(np.linalg.norm(W,axis=0)**2)]]) 
                #if np.any(np.iscomplex(M)):
                 #   print("V",'complex')
                #print(M.shape)
                #print(M[0])
                u_prime, sigma_prime, _ =np.linalg.svd(M,full_matrices=True)
                norm_w = W / np.linalg.norm(W, axis=0)
                u = (np.hstack((u,norm_w))@u_prime)[:,0:I1]
                #print('SIGMA PRIME', sigma_prime.shape)
                sigma = np.zeros_like(sample, dtype=float)
                sigma[:min(sample.shape[0], sample.shape[1]), :min(sample.shape[0], sample.shape[1])] = np.diag(sigma_prime[0:min(sample.shape[0], sample.shape[1])])
                #print("sigma",sigma.shape)
            return u[:,0:P]
        
        def phi_calculator(Data,U_mat,V_mat):
            phi_frob =0
            for data in Data:
                projected2 = self.projection(data, U_mat)
                phi_frob += np.sqrt(np.sum(projected2**2))
            return phi_frob
        
        
        V_mat = [None]*3
        for i, data in enumerate([Abar, Bbar, Cbar]):
            V_mat[i]= [np.empty(0),np.empty(0),np.empty(0)]
            for j in range(3):
                V_mat[i][j]=Vinitial(data,j,self.I[j])
        
        projected = []
        for i, data in enumerate([Abar, Bbar, Cbar]):
            projected.append(self.projection(data,V_mat[i]))
        
        
        U_mat = [None]*3
        for j in range(3):
            U_mat[j]=Uinitial(projected,j,self.P[j])
        phi_old = phi_calculator(projected,U_mat,V_mat)
        phi_record =np.zeros(self.iterations)
        
        for index in range(self.iterations):

            for i in range(3):
                U_mat[i] = U(projected,U_mat,i,self.P[i])
            for i, data in enumerate([Abar, Bbar, Cbar]):
                for j in range(0,3):
                    V_mat[i][j]=V(data,U_mat,V_mat[i],self.P[j],j,self.I[j])

            for i, data in enumerate([Abar, Bbar, Cbar]):
                projected[i]= self.projection(data,V_mat[i])
            phi_curr = phi_calculator(projected,U_mat,V_mat)
            phi_old = phi_curr
            phi_record[index] = phi_old
        prime= self.compacter([Abar, Bbar, Cbar], U_mat, V_mat)
        prime = np.array(prime)
        self.U_mat = U_mat
        self.V_mat = V_mat
        return prime, self.U_mat
    
    def compacter(self,matrix,u_mat,v_mat):
        #print('compact')
        #print(matrix[0].shape,matrix[1].shape,matrix[2].shape)
        projected_prime = [None]*3
        for i, data in enumerate(matrix):
            projected_prime[i]= self.projection(data,v_mat[i])
        #print(projected_prime[0].shape,projected_prime[1].shape,projected_prime[2].shape)
        prime = []
        for i, data in enumerate(projected_prime):
            prime.extend(self.projection(data,u_mat))
        #print(len(prime), prime[0].shape)
        return prime
        
    def test(self,A_test,B_test,C_test):
        prime= self.compacter([A_test, B_test, C_test], self.U_mat, self.V_mat)
        prime = np.array(prime)
        return prime
    
    def min_max(self,prime):
        self.Min = np.min(prime,axis=0)
        self.Max = np.max(prime,axis=0)
        
    def scale(self,prime):
        prime = (prime-self.Min)/(self.Max-self.Min)
        return prime
    





class MPCA_beta():
    
    def __init__(self,P,size,percentage):
        self.P = P
        self.M = size
        self.iterations = 50
        self.percentage = percentage
        
    def train(self,matrix):
        
        def Uinitial(x,MODE):
            unfolded_x = np.array([unfold(x[i], mode=MODE).T for i in range(x.shape[0])])
            scatter = np.einsum("nij,nik->jk", unfolded_x, unfolded_x)
    
            eitgenvalue,u = np.linalg.eig(scatter)
            u = np.real_if_close(u, tol=1)
            eitgenvalue = np.real_if_close(eitgenvalue, tol=1)
            sorted_indices = np.argsort(eitgenvalue)[::-1] 
            sorted_eigenvectors = u[:, sorted_indices]
            lam1 = eitgenvalue[sorted_indices]
            cumulative =[lam1[0]]
            for index in range(1,len(lam1)):
                value = lam1[index]+cumulative[-1]
                cumulative.append(value)
            cumulative = cumulative/cumulative[-1]
            if np.any(np.iscomplex(sorted_eigenvectors)):
                print("uinitial",'complex')
            return sorted_eigenvectors, cumulative
            
            
    
        def U(x1,U_mat,MODE):
            first, second = min((MODE+1)%3,(MODE+2)%3), max((MODE+1)%3,(MODE+2)%3)
            u1,u3 = U_mat[first],U_mat[second]
            u = np.kron(u1, u3)
            if np.any(np.iscomplex(u1)):
                print("u1",'complex')
            if np.any(np.iscomplex(u3)):
                print("u3",'complex')
                
            if np.any(np.iscomplex(u)):
                 print("u1u3",'complex')
            unfolded_x = np.array([(unfold(x1[i], mode=MODE)@u).T for i in range(x1.shape[0])])
            scatter = np.einsum("nij, nik -> jk", unfolded_x , unfolded_x)
            if np.any(np.iscomplex(scatter)):
                print("scatter",'complex',MODE)
        
            eitgenvalue,u= np.linalg.eig(scatter)
            u = np.real_if_close(u, tol=1)
            eitgenvalue = np.real_if_close(eitgenvalue, tol=1)
            #print(self.P1,self.P2,self.P3, MODE,np.linalg.norm(scatter))
            if np.any(np.iscomplex(u)):
                print("u_1",'complex',MODE)
            sorted_indices = np.argsort(eitgenvalue)[::-1]  
            sorted_eigenvectors = u[:, sorted_indices]
            if np.any(np.iscomplex(sorted_eigenvectors)):
                print("sorted_eigenvectors",'complex',MODE)
            if np.any(np.iscomplex(sorted_eigenvectors)):
                print("u",'complex',MODE)
            return sorted_eigenvectors
    
    
    
        def phi_calculator(x1,U_mat):
            projected = multi_mode_dot(x1, [m.T for m in U_mat], modes=[1, 2, 3])
            phi_frob = np.sqrt(np.sum(projected**2))
            return phi_frob

        
        U_mat = [None]*3
        cum = [None]*3
        for j in range(3):
            U_mat[j], cum[j]=Uinitial(matrix,j)
        #print('U_mat', len(U_mat), U_mat[0].shape)
        phi_old = phi_calculator(matrix,U_mat)
        phi_record =np.zeros(self.iterations)
        self.P1_beta = np.zeros(3)
        for i in range(3):
            self.P1_beta[i]=np.where(cum[i]>=self.percentage)[0][0]+1
            U_mat[i] = U_mat[i][:,0:int(self.P1_beta[i])]
        
        for index in range(self.iterations):
            for j in range(3):
                U_mat[j] =U(matrix,U_mat, j)
                if self.P1_beta[j]< self.P[j]:
                    U_mat[j] = U_mat[j][:,0:int(self.P1_beta[j])]

            phi_curr = phi_calculator(matrix,U_mat)
            phi_old = phi_curr
            phi_record[index] = phi_old
        
        prime = multi_mode_dot(matrix, [m.T for m in U_mat], modes=[1, 2, 3])
        
        self.U_mat = U_mat
        return prime, self.U_mat
    
    def test(self,test):
       
        prime_test =  multi_mode_dot(test, [m.T for m in self.U_mat], modes=[1, 2, 3])
        return prime_test
    

class MPCA():

    def __init__(self,P,percentage=0.9):
        #self.M = size
        self.percentage = percentage
        self.iterations = 150
        self.P = P

    def train(self,matrix):
        
        def Uinitial(x,MODE,P):
            unfolded_x = np.array([unfold(x[i], mode=MODE).T for i in range(x.shape[0])])
            scatter = np.einsum("nij,nik->jk", unfolded_x, unfolded_x) 
    
            eitgenvalue,u = np.linalg.eig(scatter)
            u = np.real_if_close(u, tol=1)
            eitgenvalue = np.real_if_close(eitgenvalue, tol=1)
            sorted_indices = np.argsort(eitgenvalue)[::-1]  # Indices for sorting in descending order
            sorted_eigenvectors = u[:, sorted_indices]
            return sorted_eigenvectors[:,0:P]
                    
        def U(x1,U_mat,MODE,P):
            first, second = min((MODE+1)%3,(MODE+2)%3), max((MODE+1)%3,(MODE+2)%3)
            u1,u3 = U_mat[first],U_mat[second]
            u = np.kron(u1, u3)
            unfolded_x = np.array([(unfold(x1[i], mode=MODE)@u).T for i in range(x1.shape[0])])
            scatter = np.einsum("nij, nik -> jk", unfolded_x , unfolded_x)
            
            eitgenvalue,u= np.linalg.eig(scatter)
            u = np.real_if_close(u, tol=1)
            eitgenvalue = np.real_if_close(eitgenvalue, tol=1)
            sorted_indices = np.argsort(eitgenvalue)[::-1]  
            sorted_eigenvectors = u[:, sorted_indices]
            return sorted_eigenvectors[:,0:P]
            
        def phi_calculator(data,U_mat):
            projected = multi_mode_dot(data, [m.T for m in U_mat], modes=[1, 2, 3])
            phi_frob = np.sqrt(np.sum(projected**2))
            return phi_frob
                
        U_mat = [None]*3
        for j in range(3):
            U_mat[j]=Uinitial(matrix,j,self.P[j])        
                
        phi_old = phi_calculator(matrix,U_mat)
        phi_record = np.zeros(self.iterations)
                
        for index in range(self.iterations):

            for j in range(3):
                U_mat[j]=U(matrix,U_mat,j,self.P[j]) 

            phi_curr = phi_calculator(matrix,U_mat)
            phi_old = phi_curr
            phi_record[index] = phi_old

        prime = multi_mode_dot(matrix, [m.T for m in U_mat], modes=[1, 2, 3])            
        self.U_mat = U_mat
        return prime, self.U_mat
            
    def test(self,test):
        prime_test = multi_mode_dot(test, [m.T for m in self.U_mat], modes=[1, 2, 3]) 
        return prime_test
                  
    def min_max(self,prime):
        self.Min = np.min(prime,axis=0)
        self.Max = np.max(prime,axis=0)
        
    def scale(self,prime):
        prime = (prime-self.Min)/(self.Max-self.Min)
        return prime
    

    
def train_test(arr,y,test_size):
    size =  arr.shape[0]
    shuffled = np.arange(size)
    np.random.shuffle(shuffled)
    train = arr[shuffled[0:(size-test_size)]]
    test= arr[shuffled[(size-test_size):size]]
    y_train = y[shuffled[0:(size-test_size)]]
    y_test = y[shuffled[(size-test_size):size]]
    Mean = np.mean(train,0)
    train_mean =  train - Mean
    test_mean = test - Mean
    return train_mean,test_mean,y_train,y_test



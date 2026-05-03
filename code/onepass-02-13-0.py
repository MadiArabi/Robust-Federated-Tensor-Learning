import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Loading packages ...")
import numpy as np
import tensorly
from tensorly import unfold
from tensorly.tenalg import mode_dot
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from tensorly.regression.tucker_regression import TuckerRegressor
import os
import tucker_regression0
from sklearn.linear_model import LinearRegression
import sklearn
import random
import time
import multiprocessing
import my_mpca_02_27_nomean
import copy
from my_mpca_02_27_nomean import MPCA, MPCA_FD, MPCA_beta, train_test
#path = r"/share/iselingzhang/sarabi/tensor/Simulated Data"
#path = r""G:/Shared drives/Madi Project/chapter3/Simulated Data"
test_size = 30
percentage, iterations, lam, epsilon = 0.90, 10, 0.00001, 0.0001

def prepare_data(size, data, u_matrix):
        prepared_data = [[np.dot(img, u_matrix) for img in data_set] for data_set in data[:size]]
        return np.transpose(prepared_data, (0,2,3,1))
def load_data(path):
    files = os.listdir(path)
    files.sort()
    data = []
    for file in files:
        meta_data = loadmat(os.path.join(path, file))
        imgs = np.array(meta_data['SimulateData'])
        data.append([np.array(img[0]) for img in imgs])
    return np.array(data)

def inidividuals(INDEX, index, USER, Train, Test, y_Train, y_Test):
    AIC_best  = np.inf
    predicted_best = None
    AIC_optimal = {}
    Rank = [[2,2,2],[3,3,2],[4,4,3],[5,5,4],[6,6,5],[7,7,6],[8,8,7],[9,9,8],[10,10,7],[11,11,9],[12,12,7],[13,13,8],[14,14,9],[16,16,10],[18,18,10]]
    #Rank = [[3,3,2]]
    for rank in Rank:
        logger.info('Individual rank:{}'.format(rank))
        P1,P2,P3 = rank
        mpca = MPCA(rank)
        Prime, _ = mpca.train(Train)
        Prime_test = mpca.test(Test)
        mpca.min_max(Prime)
        Prime = mpca.scale(Prime)
        Prime_test = mpca.scale(Prime_test)
        
        clf = sklearn.linear_model.Ridge(alpha=0.0001)
        clf.fit(np.reshape(Prime,(Prime.shape[0],-1)), y_Train)
        beta = clf.coef_
        beta = np.reshape(beta, (1,P1,P2,P3))
        
        try:
            mpca_beta = MPCA_beta(rank,1,0.9)
            G,U_beta = mpca_beta.train(beta)
            P1_beta,P2_beta,P3_beta = U_beta[0].shape[1], U_beta[1].shape[1], U_beta[2].shape[1]
            #print('individual',USER,rank,U_beta[0].shape,U_beta[1].shape,U_beta[2].shape)
            estimator = tucker_regression0.TuckerRegressor(weight_ranks=[P1_beta,P2_beta,P3_beta],G=np.squeeze(G),U = U_beta,tol=10e-7, n_iter_max=100, reg_W=0)
            estimator.fit(Prime, np.log(y_Train))
            predicted = estimator.predict(Prime_test)
            abs_diff = np.abs(predicted) - np.abs(np.log(y_Test))
            RSS = np.mean(abs_diff ** 2)
            AIC_curr = len(predicted) * np.log(RSS) + P1_beta * P2_beta * P3_beta
            predicted = np.abs(abs_diff) / np.abs(np.log(y_Test))
            predicted = pd.DataFrame(predicted)
            if AIC_curr < AIC_best:
                predicted_best = predicted
                AIC_best = AIC_curr
                AIC_optimal[(INDEX, index, USER)] = rank
        except Exception as e:
            logger.info('This rank {} is unstable'.format(rank))
            print(rank,"this rank is unstable in regression")
            print(f"An error occurred: {e}")
            logger.info('An error occured{}'.format(e))
    predicted_best.to_csv(r"/share/iselingzhang/sarabi/tensor/MPCA_"+str(USER)+".2log."+str(INDEX)+".csv", encoding='utf-8', index=False, mode='a')
    #predicted_best.to_csv(r"G:\My Drive\TensorImage\Tensor_suspace\MPCA_real_"+str(USER)+".11."+str(INDEX)+".csv", encoding='utf-8', index=False, mode='a')
    return pd.DataFrame(AIC_optimal)



def paralizer(INPUT):
    AIC_best_FD  = np.inf
    predicted_best_FD = None
    AIC_optimal={}
    index, INDEX, RANDS = INPUT
    start_time = time.time()
    print(RANDS)   
    np.random.seed(RANDS)
    random.seed(RANDS)
    data = load_data(path)
    meta_y=loadmat(os.path.join(path,'SimulateData_2000.mat'))
    y = np.squeeze(np.array(meta_y['domainalp']))
    size1, size2, size3 = sizes[INDEX]
    sample = np.arange(len(data))
    np.random.shuffle(sample)

    user1 = np.transpose((data[sample])[0:size1],( 0,2,3,1))

    u = np.linalg.svd(np.random.rand(21, 30), full_matrices=True)[2][:21, :]
    user2 = prepare_data(size2, (data[sample])[size1:], u)
    
    u = np.linalg.svd(np.random.rand(21, 30), full_matrices=True)[2][:, :21]
    u2 = np.linalg.svd(np.random.rand(21, 30), full_matrices=True)[2][:21,:]
    user3_step1 = prepare_data(size3, (data[sample])[size1+size2:], u2)
    prepared_user3 = [[np.dot(u,data_set[:,:,img]) for img in range(10)] for data_set in user3_step1]
    user3 = np.transpose(prepared_user3,  (0,2,3,1))
    
    y_A, y_B, y_C = (y[sample])[:size1], (y[sample])[size1:size1+size2], (y[sample])[size1+size2:size1+size2 + size3]
    Abar, A_test, y_A_train, y_A_test = train_test(user1,y_A,30)
    Bbar, B_test, y_B_train, y_B_test = train_test(user2,y_B,30) 
    Cbar, C_test, y_C_train, y_C_test = train_test(user3,y_C,30)
    Abar_deepcopy = copy.deepcopy(Abar)
    Rank = [[2,2,2],[3,3,2],[4,4,3],[5,5,4],[6,6,5],[7,7,6],[8,8,7],[9,9,8],[10,10,7],[11,11,9],[12,12,7],[13,13,8],[14,14,9],[16,16,10],[18,18,10]]
    #Rank = [[3,3,2]]
    AIC_optimal_A = inidividuals(INDEX, index, 'A', Abar, A_test, y_A_train, y_A_test)
    AIC_optimal_B = inidividuals(INDEX, index, 'B', Bbar, B_test, y_B_train, y_B_test)
    AIC_optimal_C = inidividuals(INDEX, index, 'C', Cbar, C_test, y_C_train, y_C_test)
    print('if they are the same before fedearted', np.array_equal(Abar, Abar_deepcopy))
    logger.info('After Individuals. Are they the same?{}'.format(np.array_equal(Abar, Abar_deepcopy)))
    for rank in Rank:
        logger.info('Federated Rank: {}'.format(rank))
        P1,P2,P3 = rank
        #try:
        mpca_FD = MPCA_FD([21,21,10],rank)
        Prime_FD, _ = mpca_FD.train(Abar,Bbar,Cbar)
        Prime_test_FD = mpca_FD.test(A_test,B_test,C_test)
        mpca_FD.min_max(Prime_FD)
        Prime_FD = mpca_FD.scale(Prime_FD)
        Prime_test_FD = mpca_FD.scale(Prime_test_FD)
        concat_y_FD= np.concatenate((y_A_train, y_B_train,y_C_train),axis=0)
        logger.info('After Federated. Are they the same?{}'.format(np.array_equal(Abar, Abar_deepcopy)))
        try: 
            clf = sklearn.linear_model.Ridge(alpha=0.0001)
            clf.fit(np.reshape(Prime_FD,(Prime_FD.shape[0],-1)), concat_y_FD)
            beta_FD = clf.coef_
            beta_FD = np.reshape(beta_FD, (1,P1,P2,P3))
            mpca_beta_FD = MPCA_beta(rank,1, 0.90)
            G_FD,U_beta_FD = mpca_beta_FD.train(beta_FD)
            P1_beta,P2_beta,P3_beta = U_beta_FD[0].shape[1], U_beta_FD[1].shape[1], U_beta_FD[2].shape[1]
            #print(U_beta_FD[0].shape,P1_beta,P2_beta,P3_beta)
            #print('here2')
            estimator = tucker_regression0.TuckerRegressor(weight_ranks=[P1_beta,P2_beta,P3_beta],G=np.squeeze(G_FD),U = U_beta_FD,tol=10e-7, n_iter_max=100, reg_W=0)
            estimator.fit(Prime_FD, np.log(concat_y_FD))
            predicted_FD = estimator.predict(Prime_test_FD)
            concat_y_test_FD= np.concatenate((y_A_test, y_B_test, y_C_test),axis=0)
            abs_diff_FD = np.abs(predicted_FD) - np.abs(np.log(concat_y_test_FD))
            RSS_FD = np.mean(abs_diff_FD ** 2)
            AIC_curr_FD = len(predicted_FD) * np.log(RSS_FD) + P1_beta * P2_beta * P3_beta
            predicted_FD = np.abs(abs_diff_FD) / np.abs(np.log(concat_y_test_FD))
            predicted_FD = pd.DataFrame(predicted_FD)
            if AIC_curr_FD < AIC_best_FD:
                AIC_best_FD = AIC_curr_FD
                predicted_best_FD = predicted_FD
                AIC_optimal[(INDEX, index, 'FD')] = rank
        except Exception as e:
            logger.info('rank {}is unstable'.format(rank))
            print(rank,"this rank is unstable in regression")
            logger.info('the error is: {}'.format(e))
            print(f"An error occurred: {e}")
    print(AIC_best_FD)
    predicted_best_FD.to_csv(r"/share/iselingzhang/sarabi/tensor/MPCA_FD.2log."+str(INDEX)+".csv", encoding='utf-8', index=False,mode='a')
    #predicted_best_FD.to_csv(r"G:\My Drive\TensorImage\Tensor_suspace\MPCA_FD_real.11."+str(INDEX)+".csv", encoding='utf-8', index=False,mode='a')
    logger.info('After all. Are they the same?{}'.format(np.array_equal(Abar, Abar_deepcopy)))
    AIC_optimal = pd.DataFrame(AIC_optimal)
    #AIC_optimal_A = inidividuals(INDEX, index, 'A', Abar, A_test, y_A_train, y_A_test)
    #pd.DataFrame(Abar[:,0,0,0]).to_csv(r"/share/iselingzhang/sarabi/tensor_subspace/users"+str(index)+".csv", encoding='utf-8', index=False, mode='a') 
    #AIC_optimal_B = inidividuals(INDEX, index, 'B', Bbar, B_test, y_B_train, y_B_test)
    #pd.DataFrame(Bbar[:,0,0,0]).to_csv(r"/share/iselingzhang/sarabi/tensor_subspace/users"+str(index)+".csv", encoding='utf-8', index=False, mode='a') 
    #AIC_optimal_C = inidividuals(INDEX, index, 'C', Cbar, C_test, y_C_train, y_C_test)
    #AIC_optimal_A.to_csv(r"/share/iselingzhang/sarabi/tensor_subspace/LOG_real.csv", encoding='utf-8', index=False, mode='a')
    #AIC_optimal_B.to_csv(r"/share/iselingzhang/sarabi/tensor_subspace/LOG_real.csv", encoding='utf-8', index=False, mode='a')
    #AIC_optimal_C.to_csv(r"/share/iselingzhang/sarabi/tensor_subspace/LOG_real.csv", encoding='utf-8', index=False, mode='a')
    #AIC_optimal.to_csv(r"/share/iselingzhang/sarabi/tensor_subspace/LOG_real.csv", encoding='utf-8', index=False, mode='a')
    end_time = time.time()
    #print('time: ', (end_time-start_time)/3600)
    return AIC_optimal
        
sizes = [[70,100,130],[110,170,230],[150,240,330]]
if __name__ == '__main__':
    # Create a pool of processes
    pool = multiprocessing.Pool(processes=4)  # Use 4 CPU cores
    
    np.random.seed(23420)
    random.seed(1234)
    rands = np.random.randint(1,100000,50)
    List = []
    for i in range(10):
        List.append((i,0,int(rands[i])))
    AIC_optimal = pool.map(paralizer, List)
    print('I am done')
    
    




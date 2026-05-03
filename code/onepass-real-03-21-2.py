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

path = r"/share/iselingzhang/sarabi/tensor/Simulated Data"

#path = r"G:\My Drive\TensorImage\Tensor_suspace"

test_size = 20

percentage, iterations, lam, epsilon = 0.90, 10, 0.00001, 0.0001



def prepare_data(X, size, which):

    if which == 'A':

        start_x = start_y =  0

        end_x = end_y = 10

    elif which == 'B':

        start_x, start_y = 0, 0

        end_x, end_y = 15, 15

    else:

        start_x, start_y = 0, 0

        end_x, end_y = 20, 20 

    userdata = [[image[0,i][start_x:end_x, start_y:end_y] for i in range(16)]for image in X[:size]]

    return np.transpose(np.array(userdata),(0,2,3,1))



def load_data(path):

    mat_data = loadmat(os.path.join(path,'ResampleDegImages (1).mat'))

    data =  mat_data['ResampleDegImages'][0]

    return mat_data, np.array(data)



def inidividuals(INDEX, index, USER, Train, Test, y_Train, y_Test):

    AIC_best  = np.inf

    predicted_best = None

    AIC_optimal = {}

    Rank =[[2,2,3],[3,3,4],[4,4,5],[5,5,6],[6,6,7],[7,7,9],[8,8,9],[9,9,11],[10,10,11]]

    #Rank =[[2,2,2]]

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

    predicted_best.to_csv(r"/share/iselingzhang/sarabi/tensor/MPCA_real_"+str(USER)+".2log."+str(INDEX)+".csv", encoding='utf-8', index=False, mode='a')

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

    mat_data, data = load_data(path)

    y = np.array([i[0] for i in mat_data['ResampleTTF']])

    size1, size2, size3 = sizes[INDEX]

    sample = np.arange(len(data))



    np.random.shuffle(sample)

    #pd.DataFrame(sample).to_csv(r"/share/iselingzhang/sarabi/tensor_subspace/users.csv", encoding='utf-8', index=False, mode='a') 

    user1 = prepare_data(data[sample] ,size1, 'A')

    y_A = (y[sample])[:size1]



    np.random.shuffle(sample)

    #pd.DataFrame(sample).to_csv(r"/share/iselingzhang/sarabi/tensor_subspace/users.csv", encoding='utf-8', index=False, mode='a')   

    user2 = prepare_data(data[sample] ,size2, 'B')

    y_B = (y[sample])[:size2]



    np.random.shuffle(sample)

    #pd.DataFrame(sample).to_csv(r"/share/iselingzhang/sarabi/tensor_subspace/users.csv", encoding='utf-8', index=False, mode='a') 

    user3 = prepare_data(data[sample] ,size3, 'C')

    y_C = (y[sample])[:size3]

    

    Rank =[[2,2,3],[3,3,4],[4,4,5],[5,5,6],[6,6,7],[7,7,9],[8,8,9],[9,9,11],[10,10,11]]

    #Rank =[[5,5,6]]

    Abar, A_test, y_A_train, y_A_test = train_test(user1,y_A,20)

    Bbar, B_test, y_B_train, y_B_test = train_test(user2,y_B,20) 

    Cbar, C_test, y_C_train, y_C_test = train_test(user3,y_C,20)

    Abar_deepcopy = copy.deepcopy(Abar)

    AIC_optimal_A = inidividuals(INDEX, index, 'A', Abar, A_test, y_A_train, y_A_test)

    AIC_optimal_B = inidividuals(INDEX, index, 'B', Bbar, B_test, y_B_train, y_B_test)

    AIC_optimal_C = inidividuals(INDEX, index, 'C', Cbar, C_test, y_C_train, y_C_test)

    print('if they are the same before fedearted', np.array_equal(Abar, Abar_deepcopy))

    logger.info('After Individuals. Are they the same?{}'.format(np.array_equal(Abar, Abar_deepcopy)))

    for rank in Rank:

        logger.info('Federated Rank: {}'.format(rank))

        P1,P2,P3 = rank

        #try:

        mpca_FD = MPCA_FD([10,10,16],rank)

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

    predicted_best_FD.to_csv(r"/share/iselingzhang/sarabi/tensor/MPCA_FD_real.2log."+str(INDEX)+".csv", encoding='utf-8', index=False,mode='a')

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

        

sizes = [[35,40,45],[40,45,50],[45,50,55]]

if __name__ == '__main__':

    # Create a pool of processes

    pool = multiprocessing.Pool(processes=10)  # Use 4 CPU cores

    

    np.random.seed(23420)

    random.seed(1234)

    rands = np.random.randint(1,100000,200)

    List = []

    for i in range(200):

        List.append((i,2,int(rands[i])))

    AIC_optimal = pool.map(paralizer, List)

    print('I am done')

    

    








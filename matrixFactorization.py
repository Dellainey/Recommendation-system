# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 22:35:16 2018

@author: Dellainey
"""

import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#from sklearn.neighbors import NearestNeighbors
#from sklearn.metrics.pairwise import cosine_similarity
#from scipy import sparse
from scipy.sparse import csr_matrix
#from pandas import pivot_table
from pandas.api.types import CategoricalDtype


rating = pd.read_csv('rating.csv')
#keeping only userId, movieId and ratings
rating.drop(['timestamp'],axis = 1, inplace =True)
rv = np.random.rand(rating.shape[0])
scaledRating = rating.loc[rv<0.000002]
print(scaledRating.shape)
print(scaledRating)
user = pd.Series([sorted(scaledRating.userId.unique())],dtype="category")
movie = pd.Series([sorted(scaledRating.movieId.unique())],dtype="category")
#user = pd.Categorical(sorted(scaledRating.userId.unique())).all()
#movie = pd.Categorical(sorted(scaledRating.movieId.unique())).all()
rating = scaledRating['rating'].tolist()
row = scaledRating.userId.astype(user).cat.codes
col = scaledRating.movieId.astype(movie).cat.codes
utility_matrix = csr_matrix((rating, (row, col))).toarray()
print(utility_matrix)
print(utility_matrix.shape)

test =[]
train_set = utility_matrix
m = len(train_set)
n = train_set.shape[1]
for i in range (m):
    for j in range (n):
        if train_set[i][j]>0:
            test.append((i,j,train_set[i][j]))
            train_set[i][j]=0
            break 
        else:
            pass
print(test)
print(train_set)

# M stores the number of rows and N stores number of columns
M = len(train_set)
N = train_set.shape[0]
#print(M)
#print(N)
K = 2
P = np.random.rand(M,K)
Q = np.random.rand(N,K)
print("P = " + str(P))
print("Q = " + str(Q))


def matrixFactorization (train_set,P,Q):
    mval = len(train_set)
    nval = train_set.shape[1]
    print(mval)
    print(nval)
    #rmse = 0
    beta = 0.02
    alpha = 0.002
    iterations = 1000
    Q = Q.T
    baseline_matrix = np.dot(P,Q)
    #print("utility Matrx")
    #print(utility_matrix)
    #print("Baseline Matrix")
    #print(baseline_matrix)
    
    for i in range(iterations):
        if (i % 100 == 0):
            print(i)
        rmse=0
        for m in range(mval):
            for n in range(nval):
                # finding the error in rating only for the ones that have been given a rating
                if(train_set[m][n] > 0):
                    err = train_set[m][n] - baseline_matrix[m][n]
                    print(err)
                    for k in range(K):
                        P[m][k] = P[m][k] + alpha*((2*err*Q[k][n])-beta*(P[m][k]))
                        Q[k][n] = Q[k][n] + alpha*((2*err*P[m][k])-beta*(Q[k][n]))
        baseline_matrix = np.dot(P,Q)
        #print(baseline_matrix)
        #break
        counter = 0
        for m in range(mval):
            for n in range(nval):
                if (train_set[m][n] > 0):
                    counter += 1
                    err = train_set[m][n] - baseline_matrix[m][n]
                    #Psum = P[m,:].sum()
                    #Qsum = Q[:,n].sum()
                    rmse = rmse + (err*err)
                    #rmse = rmse + np.sqrt((err*err) + ((beta/2)* ((Psum*Psum)+(Qsum*Qsum))))
        rmse = np.sqrt(rmse/counter)
        print(rmse)
        if(rmse<0.008):
            break
    return P, Q.T, rmse
            #print("err= " +str(squared_error) + " : m = " + str(m) + " n = " + str(n))
            
P, Q , rmse = matrixFactorization(train_set, P, Q)
print("rmse = " + str(rmse))
#print("P = ")
#print(P)
#print("Q = ")
#print(Q)
print(np.dot(P,Q.T))
#rmse = np.sqrt(rmse)
#print("RMSE = " + str(rmse))


baseline_matrix = np.dot(P,Q)
print(test)
length = len(test)
rating_error = 0
for i in test:
    user_index = i[0]
    item_index = i[1]
    actual_rating = i[2]
    print("actual rating= " + str(actual_rating))
    print("Predicted rating = " + str(baseline_matrix[user_index][item_index]))
    rating_error = rating_error + ((baseline_matrix[user_index][item_index]-rating)*(baseline_matrix[user_index][item_index]-rating))
print(rating_error)
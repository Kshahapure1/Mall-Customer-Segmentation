# -*- coding: utf-8 -*-

# kMeans clustering
# dataset: Customers at mall

# libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import preprocessing

# load dataset
path = "C:/Users/HP/Desktop/All program files/stat/mall.csv"
mall = pd.read_csv(path)
print(mall.head(20))

# (optional) rename the columns
mall=mall.rename(columns={'Annual Income (k$)':'income',
                          'Spending Score (1-100)':'score'})

mall.head()

# take the relevant columns for clustering
data_x = mall[['income','score']]
print(data_x)

# standardize the dataset
data_std = data_x.copy()
ss = preprocessing.StandardScaler()
data_std.iloc[:,:] = ss.fit_transform(data_std.iloc[:,:])

data_std.head()
data_x.head()

'''
buildModel: function to calculate the best K for K-Means
            and using this K to build model to create clusters
'''
def buildModel(data):
    # list to store WCSS
    wcss = []
    
    # models (to be used later to calculate silhouette score)
    models = []
    
    # create a list of K
    list_k = range(2,11)
    
    # for every value of list_k, build model and calc WCSS
    for k in list_k:
        # build the model
        m = KMeans(n_clusters=k).fit(data)
        
        # wcss -> m.inertia_
        wcss.append(m.inertia_)
        
        # append models for silhouette scores
        models.append(m)
    
    return(wcss,models)

# build the model
wcss,models = buildModel(data_mm)    

# view the output
print(wcss)    
print(models)

# elbow plot to view the WCSS-K
plt.plot(range(2,11), wcss)
plt.title("WCSS - K")
plt.xlabel("K")
plt.ylabel("WCSS")

# based on the elbow chart, the best K = 5

# silhouette scores to determine the best K
s_scores=[silhouette_score(data_std,m.predict(data_std)) for m in models]
print(s_scores)

# to get the best K, find the maximum s_scores and its corresponding K
best_k = list(range(2,11))[s_scores.index(max(s_scores))]
print("best K = ", best_k)

# create the clusters using the best K
m1 = KMeans(n_clusters = best_k)
clusters = m1.fit_predict(data_std)
print(clusters)

# cluster numbers
np.unique(clusters)

# add the clusters to the actual dataset
mall['cluster'] = clusters

mall.head(30)

'income','score'
# visualise the clusters
plt.scatter(mall.income[mall.cluster==0],mall.score[mall.cluster==0],s=100,c='violet',label="C1" )
plt.scatter(mall.income[mall.cluster==1],mall.score[mall.cluster==1],s=100,c='green',label="C2" )
plt.scatter(mall.income[mall.cluster==2],mall.score[mall.cluster==2],s=100,c='blue',label="C3" )
plt.scatter(mall.income[mall.cluster==3],mall.score[mall.cluster==3],s=100,c='yellow',label="C4" )
plt.scatter(mall.income[mall.cluster==4],mall.score[mall.cluster==4],s=100,c='red',label="C5" )
plt.legend()
plt.title("Customer Clusters")
plt.xlabel("Income")
plt.ylabel("Credit Score")

# distribution of customers across clusters
mall.cluster.value_counts()

# save data as a CSV
mall.to_csv("cluster_customers.csv",index=False)

#!/usr/bin/env python
# coding: utf-8

# In[ ]:



## loading libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

 

class data_analysis(object):

    ## data

    def __init__(self,data):

        self.data = data

    def kmeans_avg(self,cols,n):

        df_kmeans = self.data.loc[:,cols]

        kmeans = KMeans(n)

        kmeans.fit(df_kmeans)

        self.data['cluster'] = kmeans.labels_

        return self.data

    
    def weighted_mean(self,cols):
        wtavg = lambda x: np.average(x, weights = self.data.loc[x.index, cols],axis=0)
        
        return wtavg
    
    def target_encoder(self, column_trans, column_groupby, target, index=None, method='mean'):

        index = self.data.index if index is None else index # Encode the entire input df if no specific indices is supplied
            
        if method == 'mean':
            encoded_tmp = self.data.iloc[index].groupby(column_groupby)[target].mean().reset_index()
        elif method == 'median':
            encoded_tmp = self.data.iloc[index].groupby(column_groupby)[target].median().reset_index()
        elif method == 'std':
            encoded_tmp = self.data.iloc[index].groupby(column_groupby)[target].std().reset_index()
        else:
            raise ValueError("Incorrect method supplied: '{}'. Must be one of 'mean', 'median', 'std'".format(method))
        
        df_tmp = pd.DataFrame(self.data[column_trans])
        encoded_column=df_tmp.merge(encoded_tmp,on=column_trans,how='left').drop(columns=column_groupby).rename(columns={'unit_price':column_trans})
        
        ## add noise 
        np.random.seed(1)
        noise = np.random.normal(loc = 1, scale = 0.05, size = (len(encoded_column),1))
        noise = pd.DataFrame(noise)
        column_tran = encoded_column.multiply(noise[0], axis=0)
       
        return column_tran
    
    

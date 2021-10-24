#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


class Model(object):

#Get the necessary metrics (in this case for a severely imbalanced dataset)
    def __init__(self,data):

          self.data = data
        
    def dataX(self,X_cols):
        X = self.data.loc[:,X_cols]
        X = X.to_numpy()
        return X

    def dataY(self,y_cols):
        y = self.data.loc[:,y_cols]
        return y

    def evaluate(self, model, X, y):
        predictions = model.predict(X)
        rmse = mean_squared_error(y, predictions,squared=False)
        r2 = r2_score(y,predictions)
        print('Model Performance')
        print('RMSE: {:0.4f}'.format(rmse))
        print('R2 = {:0.2f}.'.format(r2))
    
        return rmse,r2
    

    def train_model_grid(self,algorithm,hp_candidates,X,y):
        # Train random forest with KFold cross-validation
        rmse_list,r2_list=[],[]
        # configure the cross-validation procedure
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
        
        for train_ix, test_ix in cv_outer.split(X):
            # split data
                X_train, X_test = X[train_ix, :], X[test_ix, :]
                y_train, y_test = y[train_ix], y[test_ix]
            # configure the cross-validation procedure
                cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
            # define the model
                model = algorithm
             # define search
                search = GridSearchCV(model, hp_candidates, scoring='r2', cv=cv_inner, refit=True)
                result = search.fit(X_train, y_train)
            # get the best performing model fit on the whole training set
                best_model = result.best_estimator_
           # evaluate model on the hold out dataset
                rmse,r2 = self.evaluate(result,X_test,y_test)
                rmse_list.append(rmse)
                r2_list.append(r2)
         
        # summarize the estimated performance of the model
   
        print(f'Training RSME: {np.mean(rmse_list):.3f}')
        print(f'Training R2: {np.mean(r2_list):.3f}')
  
        return best_model


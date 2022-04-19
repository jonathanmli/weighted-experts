# Load libraries
import numpy as np
import pandas as pd
import timeit
import os
import copy
import math
from sklearn.linear_model import LinearRegression


# squared arctan loss centered around 1/sqrt(400) = 0.05 scaled to have range [-1,1)
def avg_squared_arctan(yhat, y, c=400):
    return 4/math.pi * np.mean(np.arctan(c*(yhat-y)**2))-1

def pearson_correlation(yhat,y):
    # note that yhat and y must have at least 2 elements
    # pearson correlation is related to cosine similarity

    # print("yhat", yhat)
    # print("y", y)
    return -np.corrcoef(yhat,y)[0,1]

def soft_max(w):
    return np.exp(w)

class HistoryLog:
    '''
    Provides experts with access to shared resources (mainly predictors X and predicted Y)
    Masks observations beyond end
    '''

    def __init__(self, X=None, y = None):
        '''
        supports np arrays and pd dataframes
        '''
        self.X = X
        self.y = y
        self.current_start = 0
        self.current_end = None

    def set_end(self, end):
        self.current_end = end

    def get_X(self, start = None, end = None):
        '''
        returns np array of all X
        '''
        if self.X is None:
            return None
        if end is None:
            end = self.current_end
        if start is None:
            start = self.current_start

        if end is None:
            return self.X[start:]
        else:
            return self.X[start:end]

    def get_y(self, start = None, end = None):
        '''
        returns np array of all X
        '''
        if self.y is None:
            return None
        if end is None:
            end = self.current_end
        if start is None:
            start = self.current_start

        if end is None:
            return self.y[start:]
        else:
            return self.y[start:end]

    def add_data(self, X, y):
        # make sure shapes are compatible
        # if X.shape[1] != y.shape[0]:
        #     raise Exception('Shapes do not match')
        if self.X is None and self.y is None:
            self.X = X
            self.y = y
        else:
            # print(self.X.shape)
            # print(X.shape)
            # print(type(self.X))
            # print(type(X))
            self.X = np.vstack((self.X, X))
            self.y = np.vstack((self.y, y))
        


class Expert:
    '''
    abstract expert class
    has history attached -- uses history to update itself
    '''
    
    def __init__(self, history: HistoryLog =None ):
        self.history = history
        pass

    # updates the expert with *NEW* information. expert might also refer to its historical log
    def update(self, history=None):
        pass

    # makes a prediction yhats (T*1) for a batch observations x (T*Z)
    def predict(self, x, y = None):
        pass

    # makes predictions based on history
    def predict_from_history(self, start, end, **kwargs):
        return self.predict(self.history.get_X(start, end), self.history.get_y(start, end), **kwargs)
    
    def set_history(self, his):
        self.history = his

class WeightedExpert(Expert):
    '''
    The OG meta algorithm
    '''

    def __init__(self, experts, history: HistoryLog= None, weights = None, update_period = 5, eta = 0.1, weighted_average=True, exponential_update=True, cost_f = pearson_correlation, weight_f = lambda x: x, **kwargs):
        '''
        Update

        cost_f: takes in z, y, where z is T*N array of T predictions from N experts, and y is the true predicted. should be negative for similarity, positive otherwise, between [-1,1]
        '''
        super().__init__()
        # print(cost_f)
        if weights is None:
            weights = np.ones(len(experts))
        self.update_period = update_period
        self.eta = eta # note probably best if we weight eta by batch size
        self.weighted_average = weighted_average
        self.exponential_update = exponential_update
        self.cost_f = cost_f
        self.weight_f = weight_f
        self.weights = weights
        self.experts = experts
        self.update_counter = 0

        # note we set the histories of experts and updates them upon initialization
        self.set_history(history)
        self.update()
        
        

        # maintains combined history for all of its sub experts?
        # note: add replay buffer

    def set_history(self, his):
        self.history = his
        for _ in self.experts:
            _.set_history(his)

    def get_weights(self):
        return self.weights
    
    def set_weights(self, w):
        self.weights = w
        
    def update(self, history=None):
        for _ in self.experts:
            _.update()
        # self.update_counter += 1
        # if self.update_counter % self.update_period == 0:
        #     for _ in self.experts:
        #         _.update()
        #     self.update_counter = 0
        #     self.history.get_X()
        


    def train(self, z, y):
        '''
        trains on z, y, where z is T*N array of time periods and predictions and y (T*1) is array of true values
        '''
        # print("howdy")
        costs_expert = np.zeros(len(self.experts))
        for i in range(len(self.experts)):
            # print(i)
            costs_expert[i] = self.cost_f(z[:,i], y.reshape(-1))
        # print('ex preds', z)
        # print('true', y)
        # print('costse', costs_expert)

        # update weights
        if self.exponential_update:
            # update weights using exponential update (hedge algorithm)
            for i in range(len(self.experts)):
                self.weights[i] *= np.exp(-self.eta * costs_expert[i])
        else:
            # alternatively update weights using multiplicative weights
            for i in range(len(self.experts)):
                self.weights[i] *= 1 - self.eta * costs_expert[i]

        # print('d')

    def get_predictions(self, x):
        '''
        gets predictions from sub experts based on x, where x is T*Z array of time periods and predictors
        returns T*N array of predictions from the N experts 
        '''
        out = np.zeros((len(x),len(self.experts)))
        for i in range(len(self.experts)):
            out[:,i] = self.experts[i].predict(x)
        # print("experts", out)
        return out

    def predict(self, x, y=None, update_experts = True):
        '''
        predicts based on x, where x is T*Z array of time periods and predictors
        if y (T*1) is provided, also trains based on y
        returns T*1 array of predictions
        '''
        # update experts first
        if update_experts:
            self.update()

        predictions = self.get_predictions(x)
        adj_w = self.weight_f(self.weights.copy())

        # normalize weights
        adj_w = adj_w / np.sum(adj_w)

        if self.weighted_average:
            pred_y = np.matmul(predictions, adj_w)
        else:
            pred_y = np.zeros(len(x))
            for i in range(len(x)):
                pred_y[i] = np.random.choice(predictions[i,:], p=adj_w / np.sum(adj_w))

        if y is not None:
            self.train(predictions, y)

        return pred_y.reshape(-1, 1)

    

        

        




class SingleFactorOLS(Expert):
    '''
    Uses OLS on single predictor as prediction 
    '''

    def __init__(self, factor, history=None):
        self.alpha = 0
        self.beta = 0
        self.factor = factor
        Expert.__init__(self, history)

    def predict(self, x):
        # print(type(x[self.factor]))
    
        if len(x.shape) == 1:
            return self.beta * x[self.factor]+ self.alpha
        else:
            return self.beta * x[:,self.factor]+ self.alpha
            #
        #float(

    #this takes the longest
    def update(self):
        # print(self.factor, self.history.get_X()[:, self.factor].reshape(-1, 1))
        # print(self.history.get_y())
        reg = LinearRegression().fit(self.history.get_X()[:, self.factor].reshape(-1, 1), self.history.get_y())
        self.beta = reg.coef_[0].item()
        self.alpha = reg.intercept_.item()
        # print(type(self.alpha))
        # print(type(self.beta))

class SingleFactorCorr(Expert):

    def __init__(self, factor, history=None):
        self.alpha = 0
        self.beta = 0
        self.factor = factor
        Expert.__init__(self, history)

    def predict(self, x):
        # print(type(x[self.factor]))
        return self.beta * x[self.factor]+ self.alpha
            #
        #float(

    #this takes the longest
    def update(self):
        # print(self.factor, self.history.get_X()[:, self.factor].reshape(-1, 1))
        # print(self.history.get_y())
        reg = LinearRegression().fit(self.history.get_X()[:, self.factor].reshape(-1, 1), self.history.get_y())
        self.beta = reg.coef_[0].item()
        self.alpha = reg.intercept_.item()
        # print(type(self.alpha))
        # print(type(self.beta))



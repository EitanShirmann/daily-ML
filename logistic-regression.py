# logistic regression classifier

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


costs = []
set = datasets.load_breast_cancer()
X, y = set.data, set.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def cost(A, Y):
    return np.sum(((- np.log(A))*Y + (-np.log(1-A))*(1-Y)))/len(A)


class LogisticRegessor:
    def __init__(self, lr = 0.001, n_iters = 100):
        self.lr = lr
        self.n_iters = n_iters

        self.weights = None
        self.bias = None


    def fit(self, X, y):
        
        # initialize weights 
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        

        # gd
        for _ in range(self.n_iters):
            yhat = np.dot(X, self.weights)+self.bias
            yhat = sigmoid(yhat)
            dw   = np.dot(X.T, (yhat - y))/m
            db   = np.sum(yhat-y)/m
            if _%10==0:
                costs.append(cost(yhat, y))
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        plt.plot(costs)
        plt.title("Costs")
        plt.show()
        
        

    
    def predict(self, X):
        preds = sigmoid(np.dot(X, self.weights)+self.bias)
        
        return [1 if i>0.5 else 0 for i in preds]
        


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


regressor = LogisticRegessor(lr=0.00001, n_iters=10000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print("LogisticRegessor classification accuracy:", accuracy(y_test, predictions))
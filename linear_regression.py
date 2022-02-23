# loading data
import numpy as np 
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn import datasets
import matplotlib.pyplot as plt

X,y = datasets.make_regression(n_samples=100, n_features=1, noise = 20, random_state=0)
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=20, random_state=0)



class LinearRegressor():
    
    def __init__(self, n_features, lr = 0.1, n_iters =  40):
        self.n = n_features
        self.lr = lr

        self.iters = n_iters
        self.w = np.zeros((self.n))
        self.b = 0
        self.losses = []

    def fit(self, X, y):
        for i in range(self.iters):
            yhats = np.dot(X, self.w)+self.b 
            
            
            loss = mean_squared_error(yhats,y)
            
            
            self.w-= self.lr * -2/len(y_train)*np.dot(X_train.T, (y_train-yhats))
            self.b-= self.lr * -2/len(y_train)*np.sum((y_train-yhats))
            self.losses.append(loss)

    def showProgress(self):
        plt.plot(self.losses)
        plt.show()


    def predict(self, X):
        return self.w.T*X+self.b




line = LinearRegressor(1)

line.fit(X_train, y_train)
line.showProgress()

preds = line.predict(X_test)
preds = np.argmax(preds, axis = 1)
print(preds.shape, y_test.shape)
print(mean_squared_error(preds, y_test))


preds = line.predict(X_train)
preds = np.argmax(preds, axis = 1)
print(preds.shape, y_test.shape)
print(mean_squared_error(preds, y_train))
    


print(X_train.shape, "n", y_train.shape)
plt.scatter(X_train,y_train)
plt.plot()
plt.show()
        

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

def axaline(m,y0, ax=None, **kwargs):
    if not ax:
        ax = plt.gca()
    tr = mtransforms.BboxTransformTo(
            mtransforms.TransformedBbox(ax.viewLim, ax.transScale))  + \
         ax.transScale.inverted()
    aff = mtransforms.Affine2D.from_values(1,m,0,0,0,y0)
    trinv = ax.transData
    line = plt.Line2D([0,1],[0,0],transform=tr+aff+trinv, **kwargs)
    ax.add_line(line)

x = np.random.rand(20)*6-0.7
y = (np.random.rand(20)-.5)*4
c = (x > 3).astype(int)

fig, ax = plt.subplots()
ax.scatter(X_train,y_train, cmap="bwr")

# draw y=m*x+y0 into the plot

print("w", line.w)
print("b", line.b)
m = line.w; y0 = line.b
axaline(m,y0, ax=ax, color="limegreen", linewidth=5)

plt.show()
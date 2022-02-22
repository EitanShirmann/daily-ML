# loading data
from collections import Counter
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
X,y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=1234)

# distance metrics
def l1Distance(a,b):
    return np.sum(abs(a-b))


def l2Distance(a,b):
    return np.sqrt(np.sum((a-b)**2))
    

# knn class
class KNN():
    def __init__(self, k = 3):
        self.k = k


    def fit(self, X, y):
        self.X_train = X
        self.y_train = y


    # prediction for m samples
    def predict(self, X):
        preds = np.array([self._predict(x) for x in X])
        return np.array(preds)

    #prediction for one sample
    def _predict(self, x):
        #compute the distances to each point
        distances = [l1Distance(x, x_train) for x_train in self.X_train]
        #sorting by thew lowest distance
        k_indeces = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indeces]
        #voting among the k nearest
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]



knn = KNN()

knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

correct = 0
total = 0


for i in range(len(X_test)):
    if predictions[i]==y_test[i]:
        correct +=1
    total+=1

print("Accuracy is ",correct/total)




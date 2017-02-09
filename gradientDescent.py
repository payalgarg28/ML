import numpy as np
#from sklearn import datasets, cross_validation
#from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#dataset = datasets.load_boston()

#X = dataset.data
#Y = dataset.target

#X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(X, Y, test_size=0.2)

def calc(a0, a1, X, Y):
	n = len(X)
	temp0 = 0;
	temp1 = 0;
	for i in X:
		temp0 += 2*(a0 + a1*X[i] - Y[i])
		temp1 += 2*(a0 + a1*X[i] - Y[i])*X[i]
		return temp0, temp1
	

def best_fit(X, Y, a0, a1, alpha):
	temp0, temp1 = calc(a0, a1, X, Y)
	a0new = a0 - alpha*temp0
	a1new = a1 - alpha*temp1
	if((abs(a0-a0new)<=0.0000000001) and (abs(a1-a1new)<=0.0000000001)):
		return a0, a1
	else:
		return best_fit(X, Y, a0new, a1new, alpha)

def gradDesc(X,Y):
	return best_fit(X, Y, 0,  1, 0.01)

X = np.array([1,2,3,4,5,6])
Y = np.array([1.5, 4, 4, 6, 7, 8])

a0, a1 = gradDesc(X,Y)

print(a0, a1, X, Y)
#predicted = np.array([])

#for l in X:
#	predicted.append(a0 + a1*X[l




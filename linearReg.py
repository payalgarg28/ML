import numpy as nnp
from sklearn import datasets,cross_validation

from sklearn.linear_model import LinearRegression
dataset = datasets.load_Boston()
X=dataset.data
Y=dataset.target

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size=0.2)

model = LinearRegression()
model.fit(X_train,Y_train)
score = model.score(X_test,Y_test)
print(score)

b=model.predict(X_test)
print(b)
print(Y_test)


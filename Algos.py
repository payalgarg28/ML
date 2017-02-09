import numpy as np
from sklearn import preprocessing, datasets, cross_validation, svm, tree, neighbors
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

dataset = datasets.load_breast_cancer()
X = dataset.data
Y = dataset.target

X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(X,Y,test_size=0.2)


model = LinearRegression()
model.fit(X_Train,Y_Train)
b=model.predict(X_Test)
predicted=np.round(b)
score=model.score(X_Test,Y_Test)
print(score)

svmLinear = svm.SVR(kernel = 'linear')
svmRbf = svm.SVR(kernel = 'rbf')
DecisionTreeClf = tree.DecisionTreeClassifier()
KNN = neighbors.KNeighborsClassifier()
naiveBayes = GaussianNB()

for clf, name in [(svmLinear, 'Linear SVM'), (svmRbf, 'RBF SVM'), (naiveBayes, 'Naive Bayes'),
		   (DecisionTreeClf, 'Decision Tree'), (KNN, 'KNN')]:
	clf.fit(X_Train, Y_Train)
	print(name, clf.score(X_Test, Y_Test), metrics.classification_report(Y_Test, np.round(model.predict(X_Test))))




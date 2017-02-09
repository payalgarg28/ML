import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

d = pd.read_csv("autos.csv",encoding="CP1252")

y = d["price"]
x = d;
x.drop(["price"], 1, inplace=True)

x.fillna("nkn", inplace=True)

con = preprocessing.LabelEncoder()

for i in ['dateCrawled', 'name', 'seller', 'offerType', 'abtest', 'vehicleType', 'gearbox', 'model', 'fuelType', 'brand', 'notRepairedDamage', 'dateCreated', 'lastSeen']:
	x[i] = con.fit_transform(x[i])

model = SVR(kernel="linear")
selector = RFE(model, 5, step=2)
selector.fit(x,y)
print(selector.support_)
print(selector.ranking_)

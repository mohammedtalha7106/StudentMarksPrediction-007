import pandas as pd
import numpy as np
# import dataset
myData = pd.read_csv('finaldataset/updatedataset.csv')
print(myData.head())

# split input and output and make in array format
x = np.array(myData.drop(['student_marks'], axis=1))
y = np.array((myData['study_hours']))

# check x, y
print(x)
print(y)

from sklearn.model_selection import train_test_split
# split test and train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression    ##for linear_regression
from sklearn.linear_model import LogisticRegression    ##for logistic regression
from sklearn.neighbors import KNeighborsRegressor       ##for knn regression
from sklearn.tree import DecisionTreeRegressor          ##for DecisionTreeRegression
from sklearn.ensemble import RandomForestRegressor       ##RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR







# check train and test data length
print("x_test :",  len(x_test))
print("y_test :",  len(y_test))
print("x_trian :", len(x_train))
print("y_trian :", len(y_train))


# applay algorithm
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)
acc =  linear_regression.score(x_test, y_test)
print("Linar Regression : ", acc)

##apply logistic regression:

logistic_regression = LinearRegression()
logistic_regression.fit(x_train , y_train)
acc = logistic_regression.score(x_test , y_test)
print("logistic regression acc  :" , acc )

##apply k-nearest neighbours:
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(x_train , y_train)
acc = knn.score(x_test , y_test)
print("knn acc : " , acc)


##apply DecisionTreeRegressor
dtregression = DecisionTreeRegressor()
dtregression.fit(x_train , y_train)
acc = dtregression.score(x_test , y_test)
print("decision tree regression acc :" , acc)

##apply Random forest regression:
rf_regression = RandomForestRegressor()
rf_regression.fit(x_train , y_train)
acc = rf_regression.score(x_test, y_test)
print("random forest regression acc :" , acc)


##apply support vector machine regression:
svm_regression = SVR()
svm_regression.fit(x_train , y_train)
acc = svm_regression.score(x_test , y_test)
print("svm_regression acc :" , acc)


##svm by linear regressipn :
svmlinear_regression = LinearSVR()
svmlinear_regression.fit(x_train , y_train)
acc = svmlinear_regression.score(x_test , y_test)
print("svmlinear_regression acc :" , acc)








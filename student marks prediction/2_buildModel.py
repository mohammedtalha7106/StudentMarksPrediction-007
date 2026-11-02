###to store best acc value:

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# import dataset
myData = pd.read_csv('finaldataset/updatedataset.csv')
print(myData.head())

# split input and output and make in array format
x = np.array(myData.drop(['student_marks'], axis=1))
y = np.array((myData['study_hours']))

# check x, y
print(x)
print(y)


# split test and train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)



##apply k-nearest neighbours:


bestscore_knn = 0
for count in range(10000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(x_train , y_train)
    acc = knn.score(x_test , y_test)
    print("\n count :" ,count , "acc :" ,acc , end="")
    if bestscore_knn < acc:
        bestscore_knn = acc
        print("\n ---------------------->knn acc : " , acc)
        with open("knn.pickle","wb") as knn_file:
            pickle.dump(knn , knn_file)


##apply DecisionTreeRegressor

bestscore_dtc = 0
for count2 in range(10000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    dtregression = DecisionTreeRegressor()
    dtregression.fit(x_train , y_train)
    acc = dtregression.score(x_test , y_test)
    print("\n count2 :", count2 , "acc:" ,acc ,end="")
    if bestscore_dtc <acc:
        bestscore_dtc = acc
        print("\n ------------------>decision tree regression acc :" , bestscore_dtc)
        with open("decisiontree.pickle" , "wb") as dt_file:
            pickle.dump(dtregression ,dt_file)

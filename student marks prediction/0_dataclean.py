###import pandas to read dataset
import pandas as pd
import numpy as np

##import data to read:
data = pd.read_csv("rawdataset/StudentsPerformance.csv")
print("\n\n display data :\n" , data)


# data set Info
print("\n\nData set info:")
print(data.info())

# shape of data set
print("\n\n(row, col):\n", data.shape)

# find null values
print("sum of null data in each col")
print(data.isnull().sum())

# find mean value
print(data.describe())

###fill null valus :
data['study_hours'] = data['study_hours'].fillna(data['study_hours'].mean())

###check null values :
print("\n\n  check updated null values : \n",data.isnull().sum())



# display columns names
columns_data = data.columns
print(columns_data)

# Data visualization
import matplotlib.pyplot as plt
for name in columns_data:
    x = data[columns_data[9]]
    y = data[name]
    plt.xlabel("student_marks")
    plt.ylabel(name)
    plt.scatter(x, y)
    plt.grid()
    plt.show()

###to drop columns:
data.drop(columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch',
                   'test preparation course', 'math score', 'reading score',
                   'writing score'] , axis=1 ,inplace=True)
print(data.columns)


###export data to csv
from pandas import DataFrame
DataFrame(data.to_csv("finaldataset/updatedataset.csv" ,index=False,header=True))

print("data successfully updated")


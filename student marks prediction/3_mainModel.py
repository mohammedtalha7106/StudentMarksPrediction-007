###import pickle file to load:
import pickle

###input value for prediction:
x = float(input("enter the study hours : "))

###applying by decision tree:
pickle_dt = open("decisiontree.pickle" , "rb")
dt_model = pickle.load(pickle_dt)

print("\n predicting marks(%) by Decision tree acc : ", dt_model.predict([[x]])*10)


import pandas
import numpy
import joblib

from sklearn.linear_model import LinearRegression

ds = pandas.read_csv("Salary_Data.csv")

X = ds["YearsExperience"].values.reshape(30,1)

Y = ds["Salary"]

model = LinearRegression()
model.fit(X ,Y )

'''Predicts the output using this file'''
print("|||||||||| Welcome ||||||||||")

print(" ")

print(" ")

exp = float(input("Enter your exprerience : "))

result = model.predict([[exp]])
print(" ")

print("|||||||||| The predicted Salary is about ||||||||||")
print(result)
print(" ")

joblib.dump( model ,"SalaryModel.pkl")
